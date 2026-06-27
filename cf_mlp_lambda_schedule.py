import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from cf_mlp_layer_mechanistic import layer_diagnostic_rows
from cf_mlp_representation import (
    collect_cf_state,
    layer_indices_for_choice,
    make_features,
    pca_probe_rows,
    supervised_layer_importance_rows,
)
from cf_mlp_scalability import SweepPoint, write_jsonl


def geometric_schedule(base, depth, factor):
    return [float(base * (factor**idx)) for idx in range(depth)]


def parse_schedule(name, base, depth):
    if name in {"constant", "constant_1", "flat"}:
        return [float(base)] * depth
    if name.startswith("constant_"):
        return [float(base * float(name.split("_", 1)[1]))] * depth
    if name.startswith("decay_"):
        return geometric_schedule(base, depth, float(name.split("_", 1)[1]))
    if name.startswith("grow_"):
        return geometric_schedule(base, depth, float(name.split("_", 1)[1]))
    raise ValueError(f"Unknown schedule name: {name}")


def schedule_direction(name):
    if name.startswith("decay_"):
        return "lambda_decay_stronger_invariance"
    if name.startswith("grow_"):
        return "lambda_growth_weaker_invariance"
    if name.startswith("constant_0."):
        return "constant_stronger_invariance"
    if name.startswith("constant_") and name not in {"constant_1"}:
        return "constant_weaker_invariance"
    return "constant_baseline"


def aggregate(rows, key_fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in key_fields)].append(row)
    numeric_keys = sorted(
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float, np.integer, np.floating)) and key not in key_fields
    )
    out = []
    for key, items in sorted(grouped.items()):
        rec = {field: value for field, value in zip(key_fields, key)}
        rec["runs"] = len(items)
        for numeric_key in numeric_keys:
            vals = [float(item[numeric_key]) for item in items if numeric_key in item]
            if vals:
                rec[f"mean_{numeric_key}"] = float(np.mean(vals))
                rec[f"std_{numeric_key}"] = float(np.std(vals, ddof=0))
        out.append(rec)
    return out


def enrich_layer_rows(schedule_name, schedule, rows):
    out = []
    for row in rows:
        layer = int(row["layer"])
        out.append(
            {
                "schedule": schedule_name,
                "schedule_direction": schedule_direction(schedule_name),
                "lambda_reg": float(schedule[layer - 1]),
                **row,
            }
        )
    return out


def representation_rows(state, schedule_name, pca_dims, probe_reg, seed):
    ytr = state["ytr"]
    yte = state["yte"]
    depth = len(state["pathnorm_train"])
    choices = ["all", "early_half", "late_half", "last"] + [f"layer{idx}" for idx in range(1, depth + 1)]
    rows = []
    for choice in choices:
        layer_idx = layer_indices_for_choice(choice, depth)
        xtr, xte = make_features(state, "pathnorm", layer_idx)
        name = f"cf_pathnorm_{choice}"
        print(f"  PCA {schedule_name} {name} shape={xtr.shape}", flush=True)
        for row in pca_probe_rows(name, xtr, xte, ytr, yte, pca_dims, seed=seed, reg=probe_reg):
            rows.append(
                {
                    "schedule": schedule_name,
                    "schedule_direction": schedule_direction(schedule_name),
                    "seed": seed,
                    **row,
                }
            )
        del xtr, xte
        gc.collect()
    return rows


def summarize_schedule(schedule_name, schedule, state, diagnostic_rows, pca_rows, importance_rows):
    stream_rows = state["layer_rows"]
    final_layer = stream_rows[-1]
    best_stream = max(stream_rows, key=lambda row: row["cumulative_accuracy"])
    best_probe = max(diagnostic_rows, key=lambda row: row["probe_test_accuracy"])
    layer_by_idx = {int(row["layer"]): row for row in diagnostic_rows}
    pca_lookup = {
        (row["representation"], int(row["pca_dim"])): row
        for row in pca_rows
        if row["representation"].startswith("cf_pathnorm_")
    }
    max_dim = max((int(row["pca_dim"]) for row in pca_rows), default=0)
    ablate = {
        int(row["layer"]): row["corrected_drop"]
        for row in importance_rows
        if row.get("description") == "layer_shrunk" and row.get("alpha") == 0.0
    }
    last_diag = layer_by_idx[len(stream_rows)]
    layer3_diag = layer_by_idx.get(min(3, len(stream_rows)), last_diag)
    return {
        "schedule": schedule_name,
        "schedule_direction": schedule_direction(schedule_name),
        "seed": int(state["point"]["seed"]),
        "lambda_schedule_json": json.dumps([round(value, 8) for value in schedule]),
        "final_supervised_accuracy": final_layer["cumulative_accuracy"],
        "best_supervised_accuracy": best_stream["cumulative_accuracy"],
        "best_supervised_layer": int(best_stream["layer"]),
        "best_layer_probe_accuracy": best_probe["probe_test_accuracy"],
        "best_layer_probe_layer": int(best_probe["layer"]),
        "last_layer_probe_accuracy": last_diag["probe_test_accuracy"],
        "last_layer_effective_rank": last_diag["effective_rank"],
        "last_layer_top10_var": last_diag["top10_var"],
        "last_layer_view_mse_ratio": last_diag["same_over_shuffled_mse"],
        "last_layer_view_cosine": last_diag["same_view_cosine"],
        "layer3_probe_accuracy": layer3_diag["probe_test_accuracy"],
        "layer3_effective_rank": layer3_diag["effective_rank"],
        "layer3_view_mse_ratio": layer3_diag["same_over_shuffled_mse"],
        "layer3_view_cosine": layer3_diag["same_view_cosine"],
        "layer1_corrected_drop": ablate.get(1, 0.0),
        "last_layer_corrected_drop": ablate.get(len(stream_rows), 0.0),
        "pca_dim": int(max_dim),
        "pca_all_accuracy": pca_lookup.get(("cf_pathnorm_all", max_dim), {}).get("accuracy", float("nan")),
        "pca_early_half_accuracy": pca_lookup.get(("cf_pathnorm_early_half", max_dim), {}).get("accuracy", float("nan")),
        "pca_late_half_accuracy": pca_lookup.get(("cf_pathnorm_late_half", max_dim), {}).get("accuracy", float("nan")),
        "pca_last_accuracy": pca_lookup.get(("cf_pathnorm_last", max_dim), {}).get("accuracy", float("nan")),
        "fit_time_sec": float(state["fit_time_sec"]),
    }


def build_report(out_dir, point, summary_rows, summary_aggregate):
    order_key = lambda row: (
        row["schedule_direction"],
        row["schedule"],
    )
    lines = [
        "# CF-MLP Lambda Schedule Sweep",
        "",
        f"`input_dim={point.input_dim}`, `width={point.width}`, `depth={point.depth}`, "
        f"`n_train={point.n_train}`, `n_test={point.n_test}`.",
        "",
        "The CF gain is `lambda / (eig + lambda)`, so smaller lambda means stronger suppression of positive-pair displacement modes. "
        "A decaying lambda schedule therefore increases invariance with depth; a growing lambda schedule relaxes invariance with depth.",
        "",
        "## Single-Run Schedule Summary",
        "",
        "| Seed | Schedule | Direction | Lambdas | Final supervised | Best layer | Last probe | Last rank | Last view ratio | PCA all | PCA early | PCA late |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summary_rows, key=order_key):
        lines.append(
            f"| {row['seed']} | {row['schedule']} | {row['schedule_direction']} | `{row['lambda_schedule_json']}` | "
            f"{row['final_supervised_accuracy']:.4f} | {row['best_supervised_layer']} | "
            f"{row['last_layer_probe_accuracy']:.4f} | {row['last_layer_effective_rank']:.1f} | "
            f"{row['last_layer_view_mse_ratio']:.3f} | {row['pca_all_accuracy']:.4f} | "
            f"{row['pca_early_half_accuracy']:.4f} | {row['pca_late_half_accuracy']:.4f} |"
        )
    if summary_aggregate:
        lines.extend(
            [
                "",
                "## Aggregate Summary",
                "",
                "| Schedule | Runs | Final supervised | Last probe | Last rank | Last view ratio | PCA all | PCA early | PCA late |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in sorted(summary_aggregate, key=lambda rec: rec["schedule"]):
            lines.append(
                f"| {row['schedule']} | {row['runs']} | "
                f"{row.get('mean_final_supervised_accuracy', float('nan')):.4f} +/- {row.get('std_final_supervised_accuracy', float('nan')):.4f} | "
                f"{row.get('mean_last_layer_probe_accuracy', float('nan')):.4f} +/- {row.get('std_last_layer_probe_accuracy', float('nan')):.4f} | "
                f"{row.get('mean_last_layer_effective_rank', float('nan')):.1f} +/- {row.get('std_last_layer_effective_rank', float('nan')):.1f} | "
                f"{row.get('mean_last_layer_view_mse_ratio', float('nan')):.3f} +/- {row.get('std_last_layer_view_mse_ratio', float('nan')):.3f} | "
                f"{row.get('mean_pca_all_accuracy', float('nan')):.4f} +/- {row.get('std_pca_all_accuracy', float('nan')):.4f} | "
                f"{row.get('mean_pca_early_half_accuracy', float('nan')):.4f} +/- {row.get('std_pca_early_half_accuracy', float('nan')):.4f} | "
                f"{row.get('mean_pca_late_half_accuracy', float('nan')):.4f} +/- {row.get('std_pca_late_half_accuracy', float('nan')):.4f} |"
            )
    lines.append("")
    report = "\n".join(lines)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    return report


def run(args):
    torch.set_num_threads(args.torch_threads)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_stream_rows = []
    all_diagnostic_rows = []
    all_importance_rows = []
    all_pca_rows = []
    all_summary_rows = []
    point = None

    for seed in args.seeds:
        point = SweepPoint(
            dataset="cifar100_fullres_width",
            axis="lambda_schedule",
            scale_value=args.width,
            seed=seed,
            n_train=args.n_train,
            n_test=args.n_test,
            input_dim=args.input_dim,
            width=args.width,
            depth=args.depth,
            num_classes=100,
            lambda_reg=args.lambda_reg,
        )
        for schedule_name in args.schedules:
            schedule = parse_schedule(schedule_name, args.lambda_reg, args.depth)
            print(
                f"seed={seed} schedule={schedule_name} lambdas={[round(v, 5) for v in schedule]} "
                f"input_dim={args.input_dim} width={args.width} depth={args.depth} device={device_name}",
                flush=True,
            )
            state = collect_cf_state(point, device, device_name, lambda_schedule=schedule)
            diagnostics = enrich_layer_rows(
                schedule_name,
                schedule,
                layer_diagnostic_rows("cf", state, seed, args.probe_reg),
            )
            stream = enrich_layer_rows(schedule_name, schedule, state["layer_rows"])
            importance = [
                {
                    "schedule": schedule_name,
                    "schedule_direction": schedule_direction(schedule_name),
                    "seed": seed,
                    **row,
                }
                for row in supervised_layer_importance_rows(state)
            ]
            pca_rows = representation_rows(state, schedule_name, args.pca_dims, args.probe_reg, seed)
            summary = summarize_schedule(schedule_name, schedule, state, diagnostics, pca_rows, importance)

            all_stream_rows.extend({"seed": seed, **row} for row in stream)
            all_diagnostic_rows.extend(diagnostics)
            all_importance_rows.extend(importance)
            all_pca_rows.extend(pca_rows)
            all_summary_rows.append(summary)

            write_jsonl(args.out_dir / "stream_rows.partial.jsonl", all_stream_rows)
            write_jsonl(args.out_dir / "layer_diagnostics.partial.jsonl", all_diagnostic_rows)
            write_jsonl(args.out_dir / "importance_rows.partial.jsonl", all_importance_rows)
            write_jsonl(args.out_dir / "pca_probe_rows.partial.jsonl", all_pca_rows)
            write_jsonl(args.out_dir / "schedule_summary.partial.jsonl", all_summary_rows)

            del state, diagnostics, stream, importance, pca_rows, summary
            torch.cuda.empty_cache()
            gc.collect()

    summary_aggregate = aggregate(all_summary_rows, ["schedule", "schedule_direction"])
    diagnostic_aggregate = aggregate(all_diagnostic_rows, ["schedule", "schedule_direction", "layer"])
    pca_aggregate = aggregate(all_pca_rows, ["schedule", "schedule_direction", "representation", "pca_dim", "source_dim"])

    write_jsonl(args.out_dir / "stream_rows.jsonl", all_stream_rows)
    write_jsonl(args.out_dir / "layer_diagnostics.jsonl", all_diagnostic_rows)
    write_jsonl(args.out_dir / "importance_rows.jsonl", all_importance_rows)
    write_jsonl(args.out_dir / "pca_probe_rows.jsonl", all_pca_rows)
    write_jsonl(args.out_dir / "schedule_summary.jsonl", all_summary_rows)
    write_jsonl(args.out_dir / "schedule_summary_aggregate.jsonl", summary_aggregate)
    write_jsonl(args.out_dir / "layer_diagnostics_aggregate.jsonl", diagnostic_aggregate)
    write_jsonl(args.out_dir / "pca_probe_aggregate.jsonl", pca_aggregate)
    report = build_report(args.out_dir, point, all_summary_rows, summary_aggregate)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Sweep per-layer CF lambda schedules.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_lambda_schedule_seed7"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.40)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--probe-reg", type=float, default=100.0)
    parser.add_argument("--pca-dims", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument(
        "--schedules",
        nargs="+",
        default=[
            "constant_1",
            "constant_0.1",
            "constant_10",
            "decay_0.75",
            "decay_0.5",
            "decay_0.25",
            "grow_1.25",
            "grow_1.5",
            "grow_2.0",
            "grow_4.0",
        ],
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
