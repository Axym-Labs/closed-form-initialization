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


def subsample_np(x, max_samples):
    x = np.asarray(x, dtype=np.float64)
    if max_samples <= 0 or x.shape[0] <= max_samples:
        return x
    idx = np.linspace(0, x.shape[0] - 1, max_samples).astype(np.int64)
    return x[idx]


def subsample_torch(x, max_samples):
    if max_samples <= 0 or x.shape[0] <= max_samples:
        return x
    idx = torch.linspace(0, x.shape[0] - 1, max_samples, device=x.device).long()
    return x.index_select(0, idx)


def standardize_torch(x):
    x = x.to(dtype=torch.float32)
    return (x - x.mean(dim=0, keepdim=True)) / torch.clamp(x.std(dim=0, keepdim=True), min=1e-6)


def linear_cka_torch(x, y):
    x0 = x.to(dtype=torch.float32) - x.to(dtype=torch.float32).mean(dim=0, keepdim=True)
    y0 = y.to(dtype=torch.float32) - y.to(dtype=torch.float32).mean(dim=0, keepdim=True)
    xy = x0.T @ y0
    xx = x0.T @ x0
    yy = y0.T @ y0
    num = torch.sum(xy * xy)
    den = torch.sqrt(torch.sum(xx * xx) * torch.sum(yy * yy))
    return float((num / torch.clamp(den, min=1e-12)).detach().cpu().item())


def ridge_predict_r2_torch(x, y, reg):
    xz = standardize_torch(x)
    yz = standardize_torch(y)
    gram = xz.T @ xz
    rhs = xz.T @ yz
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    weight = torch.linalg.solve(gram + float(reg) * xz.shape[0] * eye, rhs)
    pred = xz @ weight
    mse = torch.mean((pred - yz) ** 2)
    baseline = torch.mean(yz * yz)
    r2 = 1.0 - mse / torch.clamp(baseline, min=1e-12)
    return float(r2.detach().cpu().item()), float(mse.detach().cpu().item())


def transition_metrics(prev, cur, args):
    prev_t = subsample_torch(prev, args.max_metric_samples)
    cur_t = subsample_torch(cur, args.max_metric_samples)
    forward_r2, forward_mse = ridge_predict_r2_torch(prev_t, cur_t, args.ridge_reg)
    reverse_r2, reverse_mse = ridge_predict_r2_torch(cur_t, prev_t, args.ridge_reg)
    sym_r2 = 0.5 * (forward_r2 + reverse_r2)
    return {
        "cka": linear_cka_torch(prev_t, cur_t),
        "forward_linear_r2": float(forward_r2),
        "reverse_linear_r2": float(reverse_r2),
        "sym_linear_r2": float(sym_r2),
        "linear_novelty": float(1.0 - sym_r2),
        "forward_linear_mse": float(forward_mse),
        "reverse_linear_mse": float(reverse_mse),
    }


def onehot(y, num_classes):
    eye = np.eye(num_classes, dtype=np.float32)
    return eye[np.asarray(y, dtype=np.int64)]


def cka_to_reference(x, ref, max_samples):
    return linear_cka_torch(subsample_torch(x, max_samples), subsample_torch(ref, max_samples))


def add_prefixed(row, prefix, metrics):
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value


def stage_row(args, point, variant, spec, layer_idx, stage, arrays, raw_np, label_np):
    base, view1, view2 = arrays
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
    add_prefixed(row, "bt", corr_metrics(view1, view2, args.bt_lambda))
    row["base_cka_to_raw"] = cka_to_reference(base, raw_np, args.max_metric_samples)
    row["base_cka_to_labels"] = cka_to_reference(base, label_np, args.max_metric_samples)
    return row


def transition_row(args, point, variant, spec, layer_idx, transition, stream, prev, cur):
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
        "transition": transition,
        "stream": stream,
        "tf32_enabled": bool(args.device.startswith("cuda") and not args.no_tf32),
    }
    row.update(transition_metrics(prev, cur, args))
    return row


def append_transition_rows(rows, args, point, variant, spec, layer_idx, transition, prev_arrays, cur_arrays):
    for stream, prev, cur in zip(("base", "view1", "view2"), prev_arrays, cur_arrays):
        rows.append(transition_row(args, point, variant, spec, layer_idx, transition, stream, prev, cur))


def run_variant(point, variant, args, device):
    spec = variant_spec(variant, point.depth)
    if not spec["kind"].startswith("plain_cf"):
        raise ValueError(f"Stage geometry diagnostic supports non-residual CF variants, got {spec['kind']}")
    arrays = load_point_data(point)
    xtr_np, ytr_np, *_ = arrays
    raw_t = torch.as_tensor(xtr_np.astype(np.float32), device=device)
    label_t = torch.as_tensor(onehot(ytr_np, point.num_classes), device=device)
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
        input_arrays = [base_tr, view1_tr, view2_tr]
        stage_rows.append(stage_row(args, point, variant, spec, layer_idx, "input", input_arrays, raw_t, label_t))

        transform, scale, bias, _ = fit_operator(spec, layer_idx, view1_tr, view2_tr, point.width)
        pre_arrays = [base_tr @ transform, view1_tr @ transform, view2_tr @ transform]
        pre_test_arrays = [base_te @ transform, view1_te @ transform, view2_te @ transform]
        if scale is not None:
            pre_arrays = [arr * scale for arr in pre_arrays]
            pre_test_arrays = [arr * scale for arr in pre_test_arrays]
        if bias is not None:
            pre_arrays = [arr + bias for arr in pre_arrays]
            pre_test_arrays = [arr + bias for arr in pre_test_arrays]
        stage_rows.append(stage_row(args, point, variant, spec, layer_idx, "prelinear", pre_arrays, raw_t, label_t))
        append_transition_rows(transition_rows, args, point, variant, spec, layer_idx, "input_to_prelinear", input_arrays, pre_arrays)

        act_arrays = [apply_activation_torch(arr, spec["activation"], spec["alpha"]) for arr in pre_arrays]
        act_test_arrays = [apply_activation_torch(arr, spec["activation"], spec["alpha"]) for arr in pre_test_arrays]
        stage_rows.append(stage_row(args, point, variant, spec, layer_idx, "postact", act_arrays, raw_t, label_t))
        append_transition_rows(transition_rows, args, point, variant, spec, layer_idx, "prelinear_to_postact", pre_arrays, act_arrays)
        append_transition_rows(transition_rows, args, point, variant, spec, layer_idx, "input_to_postact", input_arrays, act_arrays)

        norm_arrays, norm_test_arrays, _, _ = normalize_hidden_for_spec_torch(spec, act_arrays, act_test_arrays)
        stage_rows.append(stage_row(args, point, variant, spec, layer_idx, "postnorm_before_linear", norm_arrays, raw_t, label_t))
        append_transition_rows(
            transition_rows,
            args,
            point,
            variant,
            spec,
            layer_idx,
            "postact_to_postnorm_before_linear",
            act_arrays,
            norm_arrays,
        )

        final_arrays, final_test_arrays, postnorm_linear_stats = apply_postnorm_linear_if_needed(
            spec,
            args,
            norm_arrays,
            norm_test_arrays,
            layer_idx,
        )
        stage_rows.append(stage_row(args, point, variant, spec, layer_idx, "postnorm", final_arrays, raw_t, label_t))
        append_transition_rows(
            transition_rows,
            args,
            point,
            variant,
            spec,
            layer_idx,
            "postnorm_before_linear_to_postnorm",
            norm_arrays,
            final_arrays,
        )
        append_transition_rows(transition_rows, args, point, variant, spec, layer_idx, "input_to_postnorm", input_arrays, final_arrays)
        for row in stage_rows[-1:]:
            row.update(postnorm_linear_stats)

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


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def summarize(stage_rows, transition_rows):
    summaries = []
    keys = sorted({(row["variant"], row["depth"]) for row in stage_rows})
    for variant, depth in keys:
        stages = [row for row in stage_rows if row["variant"] == variant and row["depth"] == depth]
        postnorm = sorted([row for row in stages if row["stage"] == "postnorm"], key=lambda row: row["layer"])
        if not postnorm:
            continue
        final = postnorm[-1]
        best = min(postnorm, key=lambda row: row["bt_bt_total_per_dim"])
        trans = [row for row in transition_rows if row["variant"] == variant and row["depth"] == depth and row["stream"] == "base"]
        def mean_novelty(name):
            vals = [row["linear_novelty"] for row in trans if row["transition"] == name]
            return float(np.mean(vals)) if vals else float("nan")
        summaries.append(
            {
                "variant": variant,
                "depth": depth,
                "final_postnorm_bt_per_dim": final["bt_bt_total_per_dim"],
                "best_postnorm_bt_per_dim": best["bt_bt_total_per_dim"],
                "best_layer": int(best["layer"]),
                "final_corr_diag_mean": final["bt_corr_diag_mean"],
                "final_weighted_offdiag_per_dim": final["bt_bt_weighted_offdiag_per_dim"],
                "final_base_cka_to_labels": final["base_cka_to_labels"],
                "mean_input_to_prelinear_novelty": mean_novelty("input_to_prelinear"),
                "mean_prelinear_to_postact_novelty": mean_novelty("prelinear_to_postact"),
                "mean_input_to_postact_novelty": mean_novelty("input_to_postact"),
                "mean_postact_to_norm_novelty": mean_novelty("postact_to_postnorm_before_linear"),
                "mean_postlinear_novelty": mean_novelty("postnorm_before_linear_to_postnorm"),
                "mean_input_to_postnorm_novelty": mean_novelty("input_to_postnorm"),
            }
        )
    return summaries


def write_report(path, summaries):
    lines = [
        "# CF Stage Geometry",
        "",
        "Stage-level BT and geometry diagnostics for non-residual CF variants. Novelty is `1 - mean bidirectional ridge R2` for the base stream; higher values mean the stage is less linearly recoverable from the previous stage.",
        "",
        "| Variant | Depth | Final BT/dim | Best BT/dim | Best layer | Corr diag | Weighted off/dim | Label CKA | Input->pre novelty | Pre->act novelty | Input->act novelty | Act->norm novelty | Post-linear novelty | Input->postnorm novelty |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    str(row["depth"]),
                    fmt(row["final_postnorm_bt_per_dim"]),
                    fmt(row["best_postnorm_bt_per_dim"]),
                    str(row["best_layer"]),
                    fmt(row["final_corr_diag_mean"]),
                    fmt(row["final_weighted_offdiag_per_dim"]),
                    fmt(row["final_base_cka_to_labels"]),
                    fmt(row["mean_input_to_prelinear_novelty"]),
                    fmt(row["mean_prelinear_to_postact_novelty"]),
                    fmt(row["mean_input_to_postact_novelty"]),
                    fmt(row["mean_postact_to_norm_novelty"]),
                    fmt(row["mean_postlinear_novelty"]),
                    fmt(row["mean_input_to_postnorm_novelty"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Files: `stage_rows.csv`, `transition_rows.csv`, `summary.csv`.")
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
                axis="cf_stage_geometry",
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
                print(f"stage-geometry depth={depth} seed={seed} variant={variant}", flush=True)
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
    parser = argparse.ArgumentParser(description="Stage-level CF-BT geometry diagnostics.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_cf_stage_geometry_seed7"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.35)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--dataset", default="cifar100_simclr")
    parser.add_argument("--n-train", type=int, default=12000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depths", type=int, nargs="+", default=[12])
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--bt-lambda", type=float, default=0.005)
    parser.add_argument("--bt-offdiag-lambda", type=float, default=None)
    parser.add_argument("--ridge-reg", type=float, default=1e-3)
    parser.add_argument("--max-metric-samples", type=int, default=12000)
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
            "plain_cf_bpbt_nonlinearity",
            "plain_cf_relu_fullwhiten",
            "plain_cf_agreement_biasopt_ccalinear_relu",
        ],
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
