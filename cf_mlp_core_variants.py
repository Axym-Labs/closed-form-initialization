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
            vals = [float(item[numeric_key]) for item in items if numeric_key in item and np.isfinite(float(item[numeric_key]))]
            if vals:
                rec[f"mean_{numeric_key}"] = float(np.mean(vals))
                rec[f"std_{numeric_key}"] = float(np.std(vals, ddof=0))
        out.append(rec)
    return out


def geometric(base, depth, factor):
    return [float(base * (factor**idx)) for idx in range(depth)]


def variant_config(name, depth):
    if name == "cf_base_relu":
        return {
            "model_family": "cf",
            "transform_kind": "cf",
            "activation": "relu",
            "activation_alpha": 0.0,
            "invariance_schedule": geometric(1.0, depth, 1.0),
        }
    if name == "cf_relax4_relu":
        return {
            "model_family": "cf",
            "transform_kind": "cf",
            "activation": "relu",
            "activation_alpha": 0.0,
            "invariance_schedule": geometric(1.0, depth, 0.25),
        }
    if name == "cf_relax2_relu":
        return {
            "model_family": "cf",
            "transform_kind": "cf",
            "activation": "relu",
            "activation_alpha": 0.0,
            "invariance_schedule": geometric(1.0, depth, 0.5),
        }
    if name == "whiten_relu":
        return {
            "model_family": "whiten",
            "transform_kind": "whiten",
            "activation": "relu",
            "activation_alpha": 0.0,
            "invariance_schedule": [float("nan")] * depth,
        }
    if name.startswith("cf_relax4_leakygelu"):
        alpha = float(name.replace("cf_relax4_leakygelu", ""))
        return {
            "model_family": "cf",
            "transform_kind": "cf",
            "activation": "leaky_gelu",
            "activation_alpha": alpha,
            "invariance_schedule": geometric(1.0, depth, 0.25),
        }
    if name.startswith("cf_relax4_leaky"):
        alpha = float(name.replace("cf_relax4_leaky", ""))
        return {
            "model_family": "cf",
            "transform_kind": "cf",
            "activation": "leaky_relu",
            "activation_alpha": alpha,
            "invariance_schedule": geometric(1.0, depth, 0.25),
        }
    if name == "cf_relax4_gelu":
        return {
            "model_family": "cf",
            "transform_kind": "cf",
            "activation": "gelu",
            "activation_alpha": 0.0,
            "invariance_schedule": geometric(1.0, depth, 0.25),
        }
    if name == "cf_relax4_identity":
        return {
            "model_family": "cf",
            "transform_kind": "cf",
            "activation": "identity",
            "activation_alpha": 0.0,
            "invariance_schedule": geometric(1.0, depth, 0.25),
        }
    raise ValueError(f"Unknown variant: {name}")


def pca_rows_for_state(state, variant, depth, pca_dims, probe_reg, seed, choices):
    ytr = state["ytr"]
    yte = state["yte"]
    rows = []
    for choice in choices:
        indices = layer_indices_for_choice(choice, depth)
        xtr, xte = make_features(state, "pathnorm", indices)
        representation = f"{variant}_pathnorm_{choice}"
        print(f"  PCA {representation} shape={xtr.shape}", flush=True)
        rows.extend(
            {
                "variant": variant,
                "seed": seed,
                "depth": depth,
                "layer_choice": choice,
                **row,
            }
            for row in pca_probe_rows(representation, xtr, xte, ytr, yte, pca_dims, seed=seed, reg=probe_reg)
        )
        del xtr, xte
        gc.collect()
    return rows


def summarize_variant(variant, depth, config, state, diagnostics, pca_rows, importance_rows):
    stream_rows = state["layer_rows"]
    final_stream = stream_rows[-1]
    best_stream = max(stream_rows, key=lambda row: row["cumulative_accuracy"])
    best_single_stream = max(stream_rows, key=lambda row: row["single_layer_accuracy"])
    best_probe = max(diagnostics, key=lambda row: row["probe_test_accuracy"])
    layer_by_idx = {int(row["layer"]): row for row in diagnostics}
    pca_dim = max((int(row["pca_dim"]) for row in pca_rows), default=0)
    pca_by_choice = {
        row["layer_choice"]: row
        for row in pca_rows
        if int(row["pca_dim"]) == pca_dim
    }
    ablate = {
        int(row["layer"]): row["corrected_drop"]
        for row in importance_rows
        if row.get("description") == "layer_shrunk" and row.get("alpha") == 0.0
    }
    last_diag = layer_by_idx[depth]
    first_diag = layer_by_idx[1]
    return {
        "variant": variant,
        "model_family": config["model_family"],
        "transform_kind": config["transform_kind"],
        "activation": config["activation"],
        "activation_alpha": float(config["activation_alpha"]),
        "seed": int(state["point"]["seed"]),
        "depth": depth,
        "width": int(state["point"]["width"]),
        "input_dim": int(state["point"]["input_dim"]),
        "invariance_schedule_json": json.dumps(
            [None if not np.isfinite(value) else round(float(value), 8) for value in config["invariance_schedule"]]
        ),
        "lambda_schedule_json": json.dumps([round(float(value), 8) for value in state["lambda_schedule"]]),
        "final_supervised_accuracy": final_stream["cumulative_accuracy"],
        "last_residual_head_accuracy": final_stream["single_layer_accuracy"],
        "best_residual_head_accuracy": best_single_stream["single_layer_accuracy"],
        "best_residual_head_layer": int(best_single_stream["layer"]),
        "best_supervised_accuracy": best_stream["cumulative_accuracy"],
        "best_supervised_layer": int(best_stream["layer"]),
        "first_layer_probe_accuracy": first_diag["probe_test_accuracy"],
        "last_layer_probe_accuracy": last_diag["probe_test_accuracy"],
        "best_layer_probe_accuracy": best_probe["probe_test_accuracy"],
        "best_layer_probe_layer": int(best_probe["layer"]),
        "probe_last_minus_first": last_diag["probe_test_accuracy"] - first_diag["probe_test_accuracy"],
        "last_layer_effective_rank": last_diag["effective_rank"],
        "last_layer_top10_var": last_diag["top10_var"],
        "last_layer_view_mse_ratio": last_diag["same_over_shuffled_mse"],
        "last_layer_view_cosine": last_diag["same_view_cosine"],
        "layer1_corrected_drop": ablate.get(1, 0.0),
        "last_layer_corrected_drop": ablate.get(depth, 0.0),
        "pca_dim": pca_dim,
        "pca_all_accuracy": pca_by_choice.get("all", {}).get("accuracy", float("nan")),
        "pca_last_accuracy": pca_by_choice.get("last", {}).get("accuracy", float("nan")),
        "pca_early_half_accuracy": pca_by_choice.get("early_half", {}).get("accuracy", float("nan")),
        "pca_late_half_accuracy": pca_by_choice.get("late_half", {}).get("accuracy", float("nan")),
        "fit_time_sec": float(state["fit_time_sec"]),
    }


def fmt_mean(row, key):
    return f"{row.get(f'mean_{key}', float('nan')):.4f} +/- {row.get(f'std_{key}', float('nan')):.4f}"


def build_report(out_dir, summary_rows, summary_aggregate):
    lines = [
        "# CF-MLP Core Variant Sweep",
        "",
        "## Aggregate Summary",
        "",
        "| Variant | Depth | Runs | Final supervised | Last residual head | Last probe | Last-first probe | Last rank | Last view ratio | PCA all | PCA last | PCA late |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summary_aggregate, key=lambda rec: (rec["depth"], rec["variant"])):
        lines.append(
            f"| {row['variant']} | {row['depth']} | {row['runs']} | "
            f"{fmt_mean(row, 'final_supervised_accuracy')} | "
            f"{fmt_mean(row, 'last_residual_head_accuracy')} | "
            f"{fmt_mean(row, 'last_layer_probe_accuracy')} | "
            f"{fmt_mean(row, 'probe_last_minus_first')} | "
            f"{row.get('mean_last_layer_effective_rank', float('nan')):.1f} +/- {row.get('std_last_layer_effective_rank', float('nan')):.1f} | "
            f"{row.get('mean_last_layer_view_mse_ratio', float('nan')):.3f} +/- {row.get('std_last_layer_view_mse_ratio', float('nan')):.3f} | "
            f"{fmt_mean(row, 'pca_all_accuracy')} | "
            f"{fmt_mean(row, 'pca_last_accuracy')} | "
            f"{fmt_mean(row, 'pca_late_half_accuracy')} |"
        )
    lines.extend(
        [
            "",
            "## Single Runs",
            "",
            "| Seed | Variant | Depth | Final supervised | Last residual head | First probe | Last probe | Best probe layer | PCA all | PCA last | PCA late |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(summary_rows, key=lambda rec: (rec["depth"], rec["variant"], rec["seed"])):
        lines.append(
            f"| {row['seed']} | {row['variant']} | {row['depth']} | {row['final_supervised_accuracy']:.4f} | "
            f"{row.get('last_residual_head_accuracy', float('nan')):.4f} | "
            f"{row['first_layer_probe_accuracy']:.4f} | {row['last_layer_probe_accuracy']:.4f} | "
            f"{row['best_layer_probe_layer']} | {row['pca_all_accuracy']:.4f} | "
            f"{row['pca_last_accuracy']:.4f} | {row['pca_late_half_accuracy']:.4f} |"
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
    all_diagnostics = []
    all_importance = []
    all_pca_rows = []
    all_summary = []

    for depth in args.depths:
        pca_choices = list(args.pca_choices)
        for seed in args.seeds:
            point = SweepPoint(
                dataset="cifar100_fullres_width",
                axis="core_variants",
                scale_value=args.width,
                seed=seed,
                n_train=args.n_train,
                n_test=args.n_test,
                input_dim=args.input_dim,
                width=args.width,
                depth=depth,
                num_classes=100,
            )
            for variant in args.variants:
                config = variant_config(variant, depth)
                print(
                    f"seed={seed} depth={depth} variant={variant} transform={config['transform_kind']} "
                    f"activation={config['activation']} alpha={config['activation_alpha']} device={device_name}",
                    flush=True,
                )
                state = collect_cf_state(
                    point,
                    device,
                    device_name,
                    invariance_schedule=None if config["transform_kind"] == "whiten" else config["invariance_schedule"],
                    transform_kind=config["transform_kind"],
                    activation=config["activation"],
                    activation_alpha=config["activation_alpha"],
                )
                diagnostics = [
                    {"variant": variant, "depth": depth, **row}
                    for row in layer_diagnostic_rows(variant, state, seed, args.probe_reg)
                ]
                importance = [
                    {"variant": variant, "seed": seed, "depth": depth, **row}
                    for row in supervised_layer_importance_rows(state)
                ]
                pca_rows = pca_rows_for_state(state, variant, depth, args.pca_dims, args.probe_reg, seed, pca_choices)
                summary = summarize_variant(variant, depth, config, state, diagnostics, pca_rows, importance)

                all_stream_rows.extend({"variant": variant, "seed": seed, "depth": depth, **row} for row in state["layer_rows"])
                all_diagnostics.extend(diagnostics)
                all_importance.extend(importance)
                all_pca_rows.extend(pca_rows)
                all_summary.append(summary)

                write_jsonl(args.out_dir / "stream_rows.partial.jsonl", all_stream_rows)
                write_jsonl(args.out_dir / "layer_diagnostics.partial.jsonl", all_diagnostics)
                write_jsonl(args.out_dir / "importance_rows.partial.jsonl", all_importance)
                write_jsonl(args.out_dir / "pca_probe_rows.partial.jsonl", all_pca_rows)
                write_jsonl(args.out_dir / "summary.partial.jsonl", all_summary)

                del state, diagnostics, importance, pca_rows, summary
                torch.cuda.empty_cache()
                gc.collect()

    summary_aggregate = aggregate(all_summary, ["variant", "model_family", "transform_kind", "activation", "activation_alpha", "depth"])
    diagnostic_aggregate = aggregate(all_diagnostics, ["variant", "depth", "layer"])
    pca_aggregate = aggregate(all_pca_rows, ["variant", "depth", "layer_choice", "pca_dim", "source_dim"])

    write_jsonl(args.out_dir / "stream_rows.jsonl", all_stream_rows)
    write_jsonl(args.out_dir / "layer_diagnostics.jsonl", all_diagnostics)
    write_jsonl(args.out_dir / "importance_rows.jsonl", all_importance)
    write_jsonl(args.out_dir / "pca_probe_rows.jsonl", all_pca_rows)
    write_jsonl(args.out_dir / "summary.jsonl", all_summary)
    write_jsonl(args.out_dir / "summary_aggregate.jsonl", summary_aggregate)
    write_jsonl(args.out_dir / "layer_diagnostics_aggregate.jsonl", diagnostic_aggregate)
    write_jsonl(args.out_dir / "pca_probe_aggregate.jsonl", pca_aggregate)
    print(build_report(args.out_dir, all_summary, summary_aggregate), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Core CF-MLP representation variants.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_core_variants"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.40)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depths", type=int, nargs="+", default=[6])
    parser.add_argument("--probe-reg", type=float, default=100.0)
    parser.add_argument("--pca-dims", type=int, nargs="+", default=[512])
    parser.add_argument("--pca-choices", nargs="+", default=["all", "last", "early_half", "late_half"])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["cf_base_relu", "cf_relax4_relu", "whiten_relu", "cf_relax4_leaky0.5", "cf_relax4_leaky0.8"],
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
