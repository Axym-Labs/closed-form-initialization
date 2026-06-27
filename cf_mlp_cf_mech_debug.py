import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch

from cf_mlp_residual_bt_variants import (
    apply_activation_torch,
    apply_postnorm_linear_if_needed,
    fit_agreement_subspace_expand_transform_torch,
    fit_agreement_mode_transform_torch,
    fit_cf_transform_row_correct_torch,
    fit_cf_transform_shared_metric_torch,
    fit_postrelu_active_bias_torch,
    fit_postrelu_active_bias_targets_torch,
    fit_postrelu_affine_torch,
    fit_postrelu_corrbias_torch,
    normalized_hybrid_transform,
    normalize_hidden_for_spec_torch,
    scheduled_spec_value,
    variant_spec,
)
from cf_mlp_scalability import SweepPoint, load_point_data, write_jsonl
from cf_mlp_scalability_gpu import fit_cf_transform_torch
from cf_mlp_representation import tensors_from_arrays


def offdiag_sum_sq(matrix):
    return torch.sum(matrix * matrix) - torch.sum(torch.diagonal(matrix) ** 2)


def corr_metrics(view1, view2, bt_lambda):
    x1 = (view1 - view1.mean(dim=0, keepdim=True)) / torch.clamp(view1.std(dim=0, keepdim=True), min=1e-4)
    x2 = (view2 - view2.mean(dim=0, keepdim=True)) / torch.clamp(view2.std(dim=0, keepdim=True), min=1e-4)
    corr = (x1.T @ x2) / x1.shape[0]
    diag = torch.diagonal(corr)
    on_diag = torch.sum((diag - 1.0) ** 2)
    off_diag = offdiag_sum_sq(corr)
    total = on_diag + float(bt_lambda) * off_diag
    dim = corr.shape[0]
    delta = view1 - view2
    denom = torch.linalg.vector_norm(view1, dim=1) * torch.linalg.vector_norm(view2, dim=1)
    cosine = torch.mean(torch.sum(view1 * view2, dim=1) / torch.clamp(denom, min=1e-12))
    return {
        "bt_total_per_dim": float((total / dim).detach().cpu().item()),
        "bt_on_diag_per_dim": float((on_diag / dim).detach().cpu().item()),
        "bt_weighted_offdiag_per_dim": float((float(bt_lambda) * off_diag / dim).detach().cpu().item()),
        "corr_diag_mean": float(diag.mean().detach().cpu().item()),
        "offdiag_rms": float(torch.sqrt(off_diag / max(dim * (dim - 1), 1)).detach().cpu().item()),
        "pair_delta_mse": float(torch.mean(delta * delta).detach().cpu().item()),
        "same_view_cosine": float(cosine.detach().cpu().item()),
    }


def delta_axis_metrics(pre1, pre2):
    delta = pre1 - pre2
    cov = (delta.T @ delta) / float(delta.shape[0])
    fro2 = torch.sum(cov * cov)
    diag2 = torch.sum(torch.diagonal(cov) ** 2)
    axis_concentration = diag2 / torch.clamp(fro2, min=1e-12)
    per_dim_delta = torch.mean(delta * delta, dim=0)
    active = 0.5 * ((pre1 > 0).to(pre1.dtype).mean(dim=0) + (pre2 > 0).to(pre2.dtype).mean(dim=0))
    k = max(1, int(0.2 * per_dim_delta.numel()))
    order = torch.argsort(per_dim_delta)
    low = order[:k]
    high = order[-k:]
    return {
        "delta_axis_concentration": float(axis_concentration.detach().cpu().item()),
        "low_delta_active_rate": float(active[low].mean().detach().cpu().item()),
        "high_delta_active_rate": float(active[high].mean().detach().cpu().item()),
        "active_rate_gap_high_minus_low": float((active[high].mean() - active[low].mean()).detach().cpu().item()),
        "mean_active_rate": float(active.mean().detach().cpu().item()),
        "pre_delta_top20_mean": float(per_dim_delta[high].mean().detach().cpu().item()),
        "pre_delta_bottom20_mean": float(per_dim_delta[low].mean().detach().cpu().item()),
    }


def fit_operator(spec, layer_idx, view1, view2, width):
    kind = spec["kind"]
    if kind == "plain_cf":
        fitted = fit_cf_transform_torch(view1, view2, width, invariance_strength=spec["schedule"][layer_idx])
        return fitted["transform"], None, None, fitted
    if kind == "plain_cf_postrelu_affineopt":
        fitted = fit_cf_transform_torch(view1, view2, width, invariance_strength=spec["schedule"][layer_idx])
        pre1 = view1 @ fitted["transform"]
        pre2 = view2 @ fitted["transform"]
        scale, bias, opt_stats = fit_postrelu_affine_torch(
            pre1,
            pre2,
            fit_operator.args,
            optimize_scale=spec.get("postrelu_optimize_scale", True),
            objective=spec.get("postrelu_objective", "bt"),
        )
        fitted.update(opt_stats)
        return fitted["transform"], scale, bias, fitted
    if kind == "plain_cf_postrelu_active":
        fitted = fit_cf_transform_torch(view1, view2, width, invariance_strength=spec["schedule"][layer_idx])
        pre1 = view1 @ fitted["transform"]
        pre2 = view2 @ fitted["transform"]
        bias, heuristic_stats = fit_postrelu_active_bias_torch(
            pre1,
            pre2,
            scheduled_spec_value(spec, "active_target", layer_idx),
            fit_operator.args,
        )
        fitted.update(heuristic_stats)
        return fitted["transform"], None, bias, fitted
    if kind == "plain_cf_postrelu_corrbias":
        fitted = fit_cf_transform_torch(view1, view2, width, invariance_strength=spec["schedule"][layer_idx])
        pre1 = view1 @ fitted["transform"]
        pre2 = view2 @ fitted["transform"]
        bias, heuristic_stats = fit_postrelu_corrbias_torch(pre1, pre2, spec["corrbias_beta"], fit_operator.args)
        fitted.update(heuristic_stats)
        return fitted["transform"], None, bias, fitted
    if kind == "plain_cf_rowcorrect":
        fitted = fit_cf_transform_row_correct_torch(view1, view2, width, spec["schedule"][layer_idx])
        fitted["row_corrected"] = True
        return fitted["transform"], None, None, fitted
    if kind == "plain_cf_sharedmetric":
        fitted = fit_cf_transform_shared_metric_torch(view1, view2, width, spec["schedule"][layer_idx])
        return fitted["transform"], None, None, fitted
    if kind == "plain_cf_agreement_mode":
        fitted = fit_agreement_mode_transform_torch(
            view1,
            view2,
            width,
            spec["schedule"][layer_idx],
            spec["mode_gain_used"],
            spec["gate_beta"],
        )
        return fitted["transform"], None, fitted["bias"], fitted
    if kind == "plain_cf_agreement_expand":
        fitted = fit_agreement_subspace_expand_transform_torch(
            view1,
            view2,
            width,
            spec["agreement_expand_keep_dim"],
        )
        return fitted["transform"], None, None, fitted
    if kind == "plain_cf_agreement_postrelu_affineopt":
        fitted = fit_agreement_mode_transform_torch(
            view1,
            view2,
            width,
            spec["schedule"][layer_idx],
            spec["mode_gain_used"],
            gate_beta=0.0,
        )
        pre1 = view1 @ fitted["transform"]
        pre2 = view2 @ fitted["transform"]
        scale, bias, opt_stats = fit_postrelu_affine_torch(
            pre1,
            pre2,
            fit_operator.args,
            optimize_scale=spec.get("postrelu_optimize_scale", False),
            objective=spec.get("postrelu_objective", "bt"),
        )
        fitted.update(opt_stats)
        fitted["agreement_postrelu_fit"] = True
        return fitted["transform"], scale, bias, fitted
    if kind == "plain_cf_agreement_corrbias":
        fitted = fit_agreement_mode_transform_torch(
            view1,
            view2,
            width,
            spec["schedule"][layer_idx],
            spec["mode_gain_used"],
            gate_beta=0.0,
        )
        pre1 = view1 @ fitted["transform"]
        pre2 = view2 @ fitted["transform"]
        bias, heuristic_stats = fit_postrelu_corrbias_torch(
            pre1,
            pre2,
            spec["corrbias_beta"],
            fit_operator.args,
        )
        fitted.update(heuristic_stats)
        fitted["agreement_corrbias"] = True
        return fitted["transform"], None, bias, fitted
    if kind == "plain_cf_agreement_activegain":
        fitted = fit_agreement_mode_transform_torch(
            view1,
            view2,
            width,
            spec["schedule"][layer_idx],
            spec["mode_gain_used"],
            gate_beta=0.0,
        )
        pre1 = view1 @ fitted["transform"]
        pre2 = view2 @ fitted["transform"]
        lo = float(spec["postrelu_active_lo"])
        hi = float(spec["postrelu_active_hi"])
        active_targets = lo + (hi - lo) * fitted["gains"]
        bias, active_stats = fit_postrelu_active_bias_targets_torch(
            pre1,
            pre2,
            active_targets,
            fit_operator.args,
        )
        fitted.update(active_stats)
        fitted["postrelu_active_lo"] = lo
        fitted["postrelu_active_hi"] = hi
        fitted["agreement_activegain"] = True
        return fitted["transform"], None, bias, fitted
    if kind == "plain_cf_agreement_activerank":
        fitted = fit_agreement_mode_transform_torch(
            view1,
            view2,
            width,
            spec["schedule"][layer_idx],
            spec["mode_gain_used"],
            gate_beta=0.0,
        )
        pre1 = view1 @ fitted["transform"]
        pre2 = view2 @ fitted["transform"]
        lo = float(spec["postrelu_active_lo"])
        hi = float(spec["postrelu_active_hi"])
        dim = fitted["transform"].shape[1]
        rank_agreement = torch.linspace(
            1.0,
            0.0,
            dim,
            dtype=fitted["transform"].dtype,
            device=fitted["transform"].device,
        )
        active_targets = lo + (hi - lo) * rank_agreement
        bias, active_stats = fit_postrelu_active_bias_targets_torch(
            pre1,
            pre2,
            active_targets,
            fit_operator.args,
        )
        fitted.update(active_stats)
        fitted["postrelu_active_lo"] = lo
        fitted["postrelu_active_hi"] = hi
        fitted["agreement_activerank"] = True
        return fitted["transform"], None, bias, fitted
    if kind == "plain_cf_hybrid_agreement_postrelu_affineopt":
        cf_fitted = fit_cf_transform_torch(view1, view2, width, invariance_strength=spec["schedule"][layer_idx])
        ag_fitted = fit_agreement_mode_transform_torch(
            view1,
            view2,
            width,
            spec["schedule"][layer_idx],
            spec["mode_gain_used"],
            gate_beta=0.0,
        )
        transform, base_bias, hybrid_stats = normalized_hybrid_transform(
            view1,
            view2,
            cf_fitted["transform"],
            ag_fitted["transform"],
            spec["hybrid_mix"],
        )
        pre1 = view1 @ transform + base_bias
        pre2 = view2 @ transform + base_bias
        scale, bias, opt_stats = fit_postrelu_affine_torch(
            pre1,
            pre2,
            fit_operator.args,
            optimize_scale=spec.get("postrelu_optimize_scale", False),
            objective=spec.get("postrelu_objective", "bt"),
        )
        fitted = dict(cf_fitted)
        fitted.update(hybrid_stats)
        fitted.update(opt_stats)
        fitted["hybrid_agreement_postrelu_fit"] = True
        return transform, scale, base_bias + bias, fitted
    if kind == "plain_cf_switch_bias_agreement":
        if layer_idx < int(spec["switch_layer"]):
            fitted = fit_cf_transform_torch(view1, view2, width, invariance_strength=spec["schedule"][layer_idx])
            basis = "ordinary_cf"
        else:
            fitted = fit_agreement_mode_transform_torch(
                view1,
                view2,
                width,
                spec["schedule"][layer_idx],
                use_gain=False,
                gate_beta=0.0,
            )
            basis = "agreement_mode"
        pre1 = view1 @ fitted["transform"]
        pre2 = view2 @ fitted["transform"]
        scale, bias, opt_stats = fit_postrelu_affine_torch(
            pre1,
            pre2,
            fit_operator.args,
            optimize_scale=spec.get("postrelu_optimize_scale", False),
            objective=spec.get("postrelu_objective", "bt"),
        )
        fitted.update(opt_stats)
        fitted["switch_layer"] = int(spec["switch_layer"])
        fitted["switch_basis"] = basis
        return fitted["transform"], scale, bias, fitted
    raise ValueError(f"Mechanistic debug supports non-residual plain CF variants, got kind={kind}")


def add_prefixed(row, prefix, metrics):
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value


def metadata_without_tensors(fitted):
    out = {}
    for key, value in fitted.items():
        if key in {"transform", "bias", "scale"}:
            continue
        if isinstance(value, (bool, int, float, str)):
            out[key] = value
    return out


def run_variant(point, variant, args, device):
    spec = variant_spec(variant, point.depth)
    if not spec["kind"].startswith("plain_cf"):
        raise ValueError("This diagnostic intentionally isolates non-residual CF variants.")
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

    rows = []
    for layer_idx in range(point.depth):
        transform, scale, bias, fitted = fit_operator(spec, layer_idx, view1_tr, view2_tr, point.width)
        before = corr_metrics(view1_tr, view2_tr, args.bt_lambda)
        pre_v1 = view1_tr @ transform
        pre_v2 = view2_tr @ transform
        pre_base = base_tr @ transform
        pre_base_te = base_te @ transform
        pre_v1_te = view1_te @ transform
        pre_v2_te = view2_te @ transform
        if scale is not None:
            pre_v1 = pre_v1 * scale
            pre_v2 = pre_v2 * scale
            pre_base = pre_base * scale
            pre_base_te = pre_base_te * scale
            pre_v1_te = pre_v1_te * scale
            pre_v2_te = pre_v2_te * scale
        if bias is not None:
            pre_v1 = pre_v1 + bias
            pre_v2 = pre_v2 + bias
            pre_base = pre_base + bias
            pre_base_te = pre_base_te + bias
            pre_v1_te = pre_v1_te + bias
            pre_v2_te = pre_v2_te + bias
        pre = corr_metrics(pre_v1, pre_v2, args.bt_lambda)
        axis = delta_axis_metrics(pre_v1, pre_v2)

        act_base = apply_activation_torch(pre_base, spec["activation"], spec["alpha"])
        act_base_te = apply_activation_torch(pre_base_te, spec["activation"], spec["alpha"])
        act_v1 = apply_activation_torch(pre_v1, spec["activation"], spec["alpha"])
        act_v2 = apply_activation_torch(pre_v2, spec["activation"], spec["alpha"])
        act_v1_te = apply_activation_torch(pre_v1_te, spec["activation"], spec["alpha"])
        act_v2_te = apply_activation_torch(pre_v2_te, spec["activation"], spec["alpha"])
        after_activation = corr_metrics(act_v1, act_v2, args.bt_lambda)

        train_arrays, test_arrays, _, _ = normalize_hidden_for_spec_torch(
            spec,
            [act_base, act_v1, act_v2],
            [act_base_te, act_v1_te, act_v2_te],
        )
        train_arrays, test_arrays, postnorm_linear_stats = apply_postnorm_linear_if_needed(
            spec,
            args,
            train_arrays,
            test_arrays,
            layer_idx,
        )
        base_tr, view1_tr, view2_tr = train_arrays
        base_te, view1_te, view2_te = test_arrays
        after_norm = corr_metrics(view1_tr, view2_tr, args.bt_lambda)

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
        }
        row.update(metadata_without_tensors(fitted))
        row.update(postnorm_linear_stats)
        add_prefixed(row, "before", before)
        add_prefixed(row, "prelinear", pre)
        add_prefixed(row, "postact", after_activation)
        add_prefixed(row, "postnorm", after_norm)
        row.update(axis)
        row["linear_delta_mse_ratio"] = row["prelinear_pair_delta_mse"] / max(row["before_pair_delta_mse"], 1e-12)
        row["activation_delta_mse_ratio"] = row["postact_pair_delta_mse"] / max(row["prelinear_pair_delta_mse"], 1e-12)
        row["norm_delta_mse_ratio"] = row["postnorm_pair_delta_mse"] / max(row["postact_pair_delta_mse"], 1e-12)
        row["norm_minus_postact_bt_total_per_dim"] = row["postnorm_bt_total_per_dim"] - row["postact_bt_total_per_dim"]
        rows.append(row)

    del tensors
    torch.cuda.empty_cache()
    gc.collect()
    return rows


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_report(args, rows, out_dir):
    lines = [
        "# CF-BT Mechanistic Debug",
        "",
        "Non-residual CF variants only. Metrics are computed on paired training views at each layer.",
        "",
        "| Variant | Depth | Final postnorm BT/dim | Min postnorm BT/dim | Min layer | Final prelinear BT/dim | Final postact BT/dim | Final norm-postact delta | Final axis conc. | Final high-low active gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in args.variants:
        for depth in args.depths:
            subset = [row for row in rows if row["variant"] == variant and row["depth"] == depth]
            if not subset:
                continue
            subset = sorted(subset, key=lambda row: row["layer"])
            final = subset[-1]
            best = min(subset, key=lambda row: row["postnorm_bt_total_per_dim"])
            lines.append(
                f"| {variant} | {depth} | {final['postnorm_bt_total_per_dim']:.4g} | "
                f"{best['postnorm_bt_total_per_dim']:.4g} | {best['layer']} | "
                f"{final['prelinear_bt_total_per_dim']:.4g} | {final['postact_bt_total_per_dim']:.4g} | "
                f"{final['norm_minus_postact_bt_total_per_dim']:+.4g} | "
                f"{final['delta_axis_concentration']:.3f} | {final['active_rate_gap_high_minus_low']:+.3f} |"
            )
    lines.extend(
        [
            "",
            "Files: `mech_debug_rows.jsonl`, `mech_debug_rows.csv`.",
            "",
        ]
    )
    report = "\n".join(lines)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    return report


def run(args):
    torch.set_num_threads(args.torch_threads)
    if args.bt_offdiag_lambda is None:
        args.bt_offdiag_lambda = args.bt_lambda
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    fit_operator.args = args
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for depth in args.depths:
        for seed in args.seeds:
            point = SweepPoint(
                dataset=args.dataset,
                axis="cf_mech_debug",
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
                print(f"mech-debug depth={depth} seed={seed} variant={variant}", flush=True)
                rows.extend(run_variant(point, variant, args, device))
                write_jsonl(args.out_dir / "mech_debug_rows.partial.jsonl", rows)

    write_jsonl(args.out_dir / "mech_debug_rows.jsonl", rows)
    write_csv(args.out_dir / "mech_debug_rows.csv", rows)
    print(build_report(args, rows, args.out_dir), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Mechanistic CF-BT layer-stage diagnostics.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_cf_mech_debug_seed7"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.38)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--dataset", default="cifar100_simclr")
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depths", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--bt-lambda", type=float, default=0.005)
    parser.add_argument("--bt-offdiag-lambda", type=float, default=None)
    parser.add_argument("--postrelu-fit-samples", type=int, default=1024)
    parser.add_argument("--postrelu-steps", type=int, default=40)
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
            "plain_cf_rowcorrect_relu",
            "plain_cf_eigen_shrink_relu",
            "plain_cf_agreement_gate_relu_b1.0",
        ],
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
