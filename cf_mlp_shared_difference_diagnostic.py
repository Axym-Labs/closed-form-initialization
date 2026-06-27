import argparse
import csv
import gc
from pathlib import Path

import numpy as np
import torch

from cf_mlp_cf_mech_debug import corr_metrics, fit_operator
from cf_mlp_representation import tensors_from_arrays
from cf_mlp_residual_bt_variants import (
    apply_activation_torch,
    apply_postnorm_linear_if_needed,
    normalize_hidden_for_spec_torch,
    variant_spec,
)
from cf_mlp_scalability import SweepPoint, load_point_data, write_jsonl


def shared_difference_metrics(view1, view2):
    s = 0.5 * (view1 + view2)
    d = 0.5 * (view1 - view2)
    s = s - s.mean(dim=0, keepdim=True)
    d = d - d.mean(dim=0, keepdim=True)
    dim = view1.shape[1]
    shared = torch.sum(s * s) / float(s.shape[0] * dim)
    diff = torch.sum(d * d) / float(d.shape[0] * dim)
    total = shared + diff
    return {
        "shared_trace_per_dim": float(shared.detach().cpu().item()),
        "diff_trace_per_dim": float(diff.detach().cpu().item()),
        "shared_diff_ratio": float((shared / torch.clamp(diff, min=1e-12)).detach().cpu().item()),
        "diff_fraction": float((diff / torch.clamp(total, min=1e-12)).detach().cpu().item()),
    }


def add_prefix(row, prefix, metrics):
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value


def stage_metrics(args, point, variant, spec, layer_idx, stage, view1, view2):
    row = {
        "variant": variant,
        "kind": spec["kind"],
        "activation": spec["activation"],
        "activation_alpha": float(spec["alpha"]),
        "norm_kind": spec.get("norm_kind", "feature"),
        "dataset": point.dataset,
        "seed": point.seed,
        "input_dim": point.input_dim,
        "width": point.width,
        "depth": point.depth,
        "layer": layer_idx + 1,
        "stage": stage,
        "tf32_enabled": bool(args.device.startswith("cuda") and not args.no_tf32),
    }
    row.update(shared_difference_metrics(view1, view2))
    add_prefix(row, "bt", corr_metrics(view1, view2, args.bt_lambda))
    return row


def transition_metrics(prev, cur):
    return {
        "shared_retention": cur["shared_trace_per_dim"] / max(prev["shared_trace_per_dim"], 1e-12),
        "diff_retention": cur["diff_trace_per_dim"] / max(prev["diff_trace_per_dim"], 1e-12),
        "ratio_gain": cur["shared_diff_ratio"] / max(prev["shared_diff_ratio"], 1e-12),
        "diff_fraction_delta": cur["diff_fraction"] - prev["diff_fraction"],
    }


def transition_row(point, variant, layer_idx, transition, prev, cur):
    row = {
        "variant": variant,
        "dataset": point.dataset,
        "seed": point.seed,
        "depth": point.depth,
        "layer": layer_idx + 1,
        "transition": transition,
    }
    row.update(transition_metrics(prev, cur))
    return row


def run_variant(point, variant, args, device):
    spec = variant_spec(variant, point.depth)
    if not spec["kind"].startswith("plain_cf"):
        raise ValueError(f"This diagnostic supports non-residual CF variants, got {spec['kind']}")

    arrays = load_point_data(point)
    tensors = tensors_from_arrays(arrays, device)
    xtr, _, xte, _, view1_tr, view2_tr, view1_te, view2_te = tensors
    train_arrays, test_arrays, _, _ = normalize_hidden_for_spec_torch(
        spec,
        [xtr, view1_tr, view2_tr],
        [xte, view1_te, view2_te],
    )
    base_tr, view1_tr, view2_tr = train_arrays
    base_te, view1_te, view2_te = test_arrays

    stage_rows = []
    transition_rows = []
    for layer_idx in range(point.depth):
        input_row = stage_metrics(args, point, variant, spec, layer_idx, "input", view1_tr, view2_tr)
        stage_rows.append(input_row)

        transform, scale, bias, _ = fit_operator(spec, layer_idx, view1_tr, view2_tr, point.width)
        pre_base = base_tr @ transform
        pre_v1 = view1_tr @ transform
        pre_v2 = view2_tr @ transform
        pre_base_te = base_te @ transform
        pre_v1_te = view1_te @ transform
        pre_v2_te = view2_te @ transform
        if scale is not None:
            pre_base = pre_base * scale
            pre_v1 = pre_v1 * scale
            pre_v2 = pre_v2 * scale
            pre_base_te = pre_base_te * scale
            pre_v1_te = pre_v1_te * scale
            pre_v2_te = pre_v2_te * scale
        if bias is not None:
            pre_base = pre_base + bias
            pre_v1 = pre_v1 + bias
            pre_v2 = pre_v2 + bias
            pre_base_te = pre_base_te + bias
            pre_v1_te = pre_v1_te + bias
            pre_v2_te = pre_v2_te + bias
        pre_row = stage_metrics(args, point, variant, spec, layer_idx, "prelinear", pre_v1, pre_v2)
        stage_rows.append(pre_row)
        transition_rows.append(transition_row(point, variant, layer_idx, "input_to_prelinear", input_row, pre_row))

        act_base = apply_activation_torch(pre_base, spec["activation"], spec["alpha"])
        act_v1 = apply_activation_torch(pre_v1, spec["activation"], spec["alpha"])
        act_v2 = apply_activation_torch(pre_v2, spec["activation"], spec["alpha"])
        act_base_te = apply_activation_torch(pre_base_te, spec["activation"], spec["alpha"])
        act_v1_te = apply_activation_torch(pre_v1_te, spec["activation"], spec["alpha"])
        act_v2_te = apply_activation_torch(pre_v2_te, spec["activation"], spec["alpha"])
        act_row = stage_metrics(args, point, variant, spec, layer_idx, "postact", act_v1, act_v2)
        stage_rows.append(act_row)
        transition_rows.append(transition_row(point, variant, layer_idx, "prelinear_to_postact", pre_row, act_row))

        norm_arrays, norm_test_arrays, _, _ = normalize_hidden_for_spec_torch(
            spec,
            [act_base, act_v1, act_v2],
            [act_base_te, act_v1_te, act_v2_te],
        )
        norm_row = stage_metrics(args, point, variant, spec, layer_idx, "postnorm_before_linear", norm_arrays[1], norm_arrays[2])
        stage_rows.append(norm_row)
        transition_rows.append(transition_row(point, variant, layer_idx, "postact_to_norm", act_row, norm_row))

        final_arrays, final_test_arrays, _ = apply_postnorm_linear_if_needed(
            spec,
            args,
            norm_arrays,
            norm_test_arrays,
            layer_idx,
        )
        final_row = stage_metrics(args, point, variant, spec, layer_idx, "postnorm", final_arrays[1], final_arrays[2])
        stage_rows.append(final_row)
        transition_rows.append(transition_row(point, variant, layer_idx, "norm_to_postnorm", norm_row, final_row))
        transition_rows.append(transition_row(point, variant, layer_idx, "input_to_postnorm", input_row, final_row))

        base_tr, view1_tr, view2_tr = final_arrays
        base_te, view1_te, view2_te = final_test_arrays

    del tensors
    torch.cuda.empty_cache()
    gc.collect()
    return stage_rows, transition_rows


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(stage_rows, transition_rows):
    summaries = []
    for variant, depth in sorted({(row["variant"], row["depth"]) for row in stage_rows}):
        stages = [row for row in stage_rows if row["variant"] == variant and row["depth"] == depth]
        postnorm = sorted([row for row in stages if row["stage"] == "postnorm"], key=lambda row: row["layer"])
        postact = sorted([row for row in stages if row["stage"] == "postact"], key=lambda row: row["layer"])
        prelinear = sorted([row for row in stages if row["stage"] == "prelinear"], key=lambda row: row["layer"])
        trans = [row for row in transition_rows if row["variant"] == variant and row["depth"] == depth]

        def mean_for(transition, key):
            vals = [row[key] for row in trans if row["transition"] == transition]
            return float(np.mean(vals)) if vals else float("nan")

        final = postnorm[-1]
        best = max(postnorm, key=lambda row: row["shared_diff_ratio"])
        summaries.append(
            {
                "variant": variant,
                "depth": depth,
                "final_stage": "postnorm",
                "final_bt_total_per_dim": final["bt_bt_total_per_dim"],
                "final_corr_diag_mean": final["bt_corr_diag_mean"],
                "final_shared_trace_per_dim": final["shared_trace_per_dim"],
                "final_diff_trace_per_dim": final["diff_trace_per_dim"],
                "final_shared_diff_ratio": final["shared_diff_ratio"],
                "final_diff_fraction": final["diff_fraction"],
                "best_postnorm_shared_diff_ratio": best["shared_diff_ratio"],
                "best_postnorm_layer": best["layer"],
                "mean_preact_shared_diff_ratio": float(np.mean([row["shared_diff_ratio"] for row in prelinear])),
                "mean_postact_shared_diff_ratio": float(np.mean([row["shared_diff_ratio"] for row in postact])),
                "mean_postnorm_shared_diff_ratio": float(np.mean([row["shared_diff_ratio"] for row in postnorm])),
                "mean_pre_to_act_shared_retention": mean_for("prelinear_to_postact", "shared_retention"),
                "mean_pre_to_act_diff_retention": mean_for("prelinear_to_postact", "diff_retention"),
                "mean_pre_to_act_ratio_gain": mean_for("prelinear_to_postact", "ratio_gain"),
                "mean_input_to_postnorm_ratio_gain": mean_for("input_to_postnorm", "ratio_gain"),
            }
        )
    return summaries


def fmt(value):
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def write_report(path, summaries):
    lines = [
        "# Shared/Difference CF-BT Diagnostic",
        "",
        "For paired views, `shared = (view1 + view2) / 2` and `diff = (view1 - view2) / 2`.",
        "The central math-first criterion is high shared/diff trace ratio, especially after the nonlinearity and postnorm stage.",
        "",
        "| Variant | Depth | Final BT/dim | Corr diag | Final shared/diff | Final diff frac | Best ratio | Best layer | Pre->act shared retention | Pre->act diff retention | Pre->act ratio gain | Input->postnorm ratio gain |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summaries, key=lambda rec: (rec["depth"], rec["variant"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    str(row["depth"]),
                    fmt(row["final_bt_total_per_dim"]),
                    fmt(row["final_corr_diag_mean"]),
                    fmt(row["final_shared_diff_ratio"]),
                    fmt(row["final_diff_fraction"]),
                    fmt(row["best_postnorm_shared_diff_ratio"]),
                    str(row["best_postnorm_layer"]),
                    fmt(row["mean_pre_to_act_shared_retention"]),
                    fmt(row["mean_pre_to_act_diff_retention"]),
                    fmt(row["mean_pre_to_act_ratio_gain"]),
                    fmt(row["mean_input_to_postnorm_ratio_gain"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "Files: `stage_rows.csv`, `transition_rows.csv`, `summary.csv`."])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args):
    torch.set_num_threads(args.torch_threads)
    if args.bt_offdiag_lambda is None:
        args.bt_offdiag_lambda = args.bt_lambda
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
        torch.backends.cudnn.allow_tf32 = not args.no_tf32
    device = torch.device(args.device)
    fit_operator.args = args
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_stage_rows = []
    all_transition_rows = []
    for depth in args.depths:
        for seed in args.seeds:
            point = SweepPoint(
                dataset=args.dataset,
                axis="cf_shared_difference",
                scale_value=args.width,
                seed=seed,
                n_train=args.n_train,
                n_test=args.n_test,
                input_dim=args.input_dim,
                width=args.width,
                depth=depth,
                num_classes=args.num_classes,
            )
            for variant in args.variants:
                print(f"shared-diff depth={depth} seed={seed} variant={variant}", flush=True)
                stage_rows, transition_rows = run_variant(point, variant, args, device)
                all_stage_rows.extend(stage_rows)
                all_transition_rows.extend(transition_rows)
                write_jsonl(args.out_dir / "stage_rows.partial.jsonl", all_stage_rows)
                write_jsonl(args.out_dir / "transition_rows.partial.jsonl", all_transition_rows)

    summaries = summarize(all_stage_rows, all_transition_rows)
    write_jsonl(args.out_dir / "stage_rows.jsonl", all_stage_rows)
    write_jsonl(args.out_dir / "transition_rows.jsonl", all_transition_rows)
    write_jsonl(args.out_dir / "summary.jsonl", summaries)
    write_csv(args.out_dir / "stage_rows.csv", all_stage_rows)
    write_csv(args.out_dir / "transition_rows.csv", all_transition_rows)
    write_csv(args.out_dir / "summary.csv", summaries)
    write_report(args.out_dir / "report.md", summaries)
    print((args.out_dir / "report.md").read_text(encoding="utf-8"), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Shared/difference trace diagnostic for CF-BT layers.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_shared_difference_seed7"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.35)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--dataset", default="cifar100_simclr")
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depths", type=int, nargs="+", default=[24])
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--bt-lambda", type=float, default=0.005)
    parser.add_argument("--bt-offdiag-lambda", type=float, default=None)
    parser.add_argument("--postrelu-fit-samples", type=int, default=2048)
    parser.add_argument("--postrelu-steps", type=int, default=60)
    parser.add_argument("--postrelu-lr", type=float, default=0.05)
    parser.add_argument("--postrelu-scale-ridge", type=float, default=1e-4)
    parser.add_argument("--postrelu-bias-ridge", type=float, default=1e-4)
    parser.add_argument("--postrelu-grad-clip", type=float, default=10.0)
    parser.add_argument("--postnorm-linear-fit-samples", type=int, default=2048)
    parser.add_argument("--postnorm-linear-steps", type=int, default=60)
    parser.add_argument("--postnorm-linear-lr", type=float, default=0.03)
    parser.add_argument("--postnorm-linear-ridge", type=float, default=1e-4)
    parser.add_argument("--postnorm-linear-grad-clip", type=float, default=10.0)
    parser.add_argument("--postnorm-linear-cca-eps", type=float, default=1e-4)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "plain_cf_relu",
            "plain_cf_agreement_biasopt_relu",
            "plain_cf_agreement_activerank_relu_lo0.05_hi0.55",
            "plain_cf_agreement_corrbias_relu_b-0.25",
            "plain_cf_agreement_biasopt_ccalinear_relu",
            "plain_cf_agreement_activerank_ccalinear_relu_lo0.05_hi0.55",
            "plain_cf_agreement_corrbias_ccalinear_relu_b-1.0",
        ],
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
