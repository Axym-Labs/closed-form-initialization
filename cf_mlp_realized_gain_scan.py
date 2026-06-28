import argparse
import copy
import csv
import gc
from pathlib import Path

import numpy as np
import torch

from cf_mlp_barlow_clean import load_tensors
from cf_mlp_bt_objective_by_layer import bt_hidden_metrics
from cf_mlp_clean_readouts import linear_classifier_readout
from cf_mlp_layer_mechanistic import covariance_spectrum
from cf_mlp_moment_ols_residual import (
    apply_term_operator,
    branch_features,
    branch_matrix,
    bt_corr_and_gradient,
    build_branch_dictionary,
    cg_solve_moment_ols_terms,
    center_torch,
    deterministic_orthogonal_matrix,
    fit_index_list,
    moment_delta_target,
    normalize_hidden_with_stats_torch,
    numpy_path,
    operator_context,
    point_for,
    rms_torch,
    run_variant,
    self_corr_offdiag_per_dim_torch,
)
from cf_mlp_scalability import write_jsonl


def candidate_args(base_args, spec):
    args = copy.copy(base_args)
    variant = "cf_shrink"
    args.branch_random_blend = 0.0
    args.branch_shared_power = 0.0
    args.branch_novelty_mode = "mix"
    args.branch_novelty_mix = 0.0
    args.branch_novelty_scale = 1.0
    args.branch_mode_balance_power = 0.0
    args.branch_mode_balance_side = "input"
    args.branch_mode_balance_eps = 1e-3
    args.branch_mode_balance_min_gain = 0.0
    args.branch_mode_balance_max_gain = 0.0
    if spec in {"plain", "concat_cf_rand", "concat_cf_rand2"}:
        pass
    elif spec == "randorth":
        variant = "random_orth"
    elif spec == "nov025":
        args.branch_novelty_mix = 0.25
    elif spec == "nov05":
        args.branch_novelty_mix = 0.5
    elif spec == "modein01":
        args.branch_mode_balance_power = 0.1
        args.branch_mode_balance_side = "input"
        args.branch_mode_balance_max_gain = 4.0
    elif spec == "modeout025":
        args.branch_mode_balance_power = 0.25
        args.branch_mode_balance_side = "output"
        args.branch_mode_balance_max_gain = 4.0
    elif spec == "randomblend01":
        args.branch_random_blend = 0.1
    elif spec == "sharedcross":
        variant = "shared_cross"
    else:
        raise ValueError(f"Unknown candidate spec: {spec}")
    return args, variant


def candidate_branch(args, variant, spec, layer_idx, view1, view2, seed):
    if spec == "concat_cf_rand":
        cf_branch = branch_matrix(args, "cf_shrink", layer_idx, view1, view2, seed)
        rand_branch = deterministic_orthogonal_matrix(
            view1.shape[1],
            view1.shape[1],
            seed + 9001 * (layer_idx + 1),
            view1.dtype,
            view1.device,
        )
        return torch.cat([cf_branch, rand_branch], dim=1)
    if spec == "concat_cf_rand2":
        cf_branch = branch_matrix(args, "cf_shrink", layer_idx, view1, view2, seed)
        rand_branch1 = deterministic_orthogonal_matrix(
            view1.shape[1],
            view1.shape[1],
            seed + 9001 * (layer_idx + 1),
            view1.dtype,
            view1.device,
        )
        rand_branch2 = deterministic_orthogonal_matrix(
            view1.shape[1],
            view1.shape[1],
            seed + 17011 * (layer_idx + 1),
            view1.dtype,
            view1.device,
        )
        return torch.cat([cf_branch, rand_branch1, rand_branch2], dim=1)
    return branch_matrix(args, variant, layer_idx, view1, view2, seed)


def score_row(row, metric):
    if metric == "test_bt":
        return (row["after_test_bt"], -row["after_test_corr_nuclear_per_dim"])
    if metric == "train_bt":
        return (row["after_train_bt"], -row["after_corr_nuclear_per_dim"])
    if metric == "test_nuclear":
        return (-row["after_test_corr_nuclear_per_dim"], row["after_test_bt"])
    raise ValueError(f"Unknown scan select metric: {metric}")


def solve_candidate(args, point, device, state, spec, ytr, yte, return_state=False):
    cand_args, variant = candidate_args(args, spec)
    base_tr, view1_tr, view2_tr, base_te, view1_te, view2_te = state
    before_train_np, _, before_v1_np, before_v2_np = numpy_path(base_tr, base_te, view1_tr, view2_tr)
    _, _, before_v1_test_np, before_v2_test_np = numpy_path(base_tr, base_te, view1_te, view2_te)
    before_bt = bt_hidden_metrics(before_v1_np, before_v2_np, args.bt_lambda)
    before_test_bt = bt_hidden_metrics(before_v1_test_np, before_v2_test_np, args.bt_lambda)

    branch = candidate_branch(cand_args, variant, spec, int(args.scan_layer), view1_tr, view2_tr, point.seed)
    branch_train, branch_test, _, _ = branch_features(
        cand_args,
        [base_tr, view1_tr, view2_tr, base_te, view1_te, view2_te],
        branch,
    )
    branch_train, branch_test, branch_metrics = build_branch_dictionary(
        cand_args,
        [base_tr, view1_tr, view2_tr],
        [base_te, view1_te, view2_te],
        branch_train,
        branch_test,
    )
    phi_base, phi_v1, phi_v2 = branch_train
    phi_base_te, phi_v1_te, phi_v2_te = branch_test
    idx_fits = fit_index_list(cand_args, int(args.scan_layer), point.seed, view1_tr.shape[0], device)
    terms = []
    fit_weight = 1.0 / float(len(idx_fits))
    first_diag = {}
    eta_layer = float(args.eta_total) / float(point.depth)
    for fit_idx, idx_fit in enumerate(idx_fits):
        if idx_fit is None:
            view1_fit = view1_tr
            view2_fit = view2_tr
            phi_v1_fit = phi_v1
            phi_v2_fit = phi_v2
        else:
            view1_fit = view1_tr[idx_fit]
            view2_fit = view2_tr[idx_fit]
            phi_v1_fit = phi_v1[idx_fit]
            phi_v2_fit = phi_v2[idx_fit]
        z1, z2, corr, grad, scale1, scale2 = bt_corr_and_gradient(view1_fit, view2_fit, cand_args.bt_lambda)
        target = moment_delta_target(cand_args, corr, grad, eta_layer)
        phi1 = center_torch(phi_v1_fit)
        phi2 = center_torch(phi_v2_fit)
        context = operator_context(cand_args, phi1, phi2, z1, z2, corr)
        terms.append(
            {
                "type": "moment",
                "name": "bt_cross",
                "weight": fit_weight * float(cand_args.moment_target_weight),
                "m1": (phi1.T @ z2) / float(z1.shape[0]),
                "m2": (z1.T @ phi2) / float(z1.shape[0]),
                "inv_scale1": 1.0 / torch.clamp(scale1, min=1e-12),
                "inv_scale2": 1.0 / torch.clamp(scale2, min=1e-12),
                "context": context,
                "target": target,
            }
        )
        if fit_idx == 0:
            first_diag = {"target": target, "corr": corr}
    b, solve = cg_solve_moment_ols_terms(
        terms,
        (phi_v1.shape[1], view1_tr.shape[1]),
        view1_tr.dtype,
        view1_tr.device,
        cand_args.ols_ridge,
        cand_args.cg_iters,
        cand_args.cg_tol,
    )
    delta_base = phi_base @ b
    base_rms_before = rms_torch(base_tr)
    update_ratio = rms_torch(delta_base) / torch.clamp(base_rms_before, min=1e-12)
    applied_scale = 1.0
    if cand_args.max_update_ratio > 0 and update_ratio > float(cand_args.max_update_ratio):
        applied_scale = float((float(cand_args.max_update_ratio) / update_ratio).detach().cpu().item())
        b = b * applied_scale
        delta_base = delta_base * applied_scale
    delta_v1 = phi_v1 @ b
    delta_v2 = phi_v2 @ b
    delta_base_te = phi_base_te @ b
    delta_v1_te = phi_v1_te @ b
    delta_v2_te = phi_v2_te @ b
    scale_rows = []
    best_row = None
    best_state = None
    for scan_scale in args.scan_scales:
        scan_scale = float(scan_scale)
        train_arrays, test_arrays, _, _ = normalize_hidden_with_stats_torch(
            [
                base_tr + scan_scale * delta_base,
                view1_tr + scan_scale * delta_v1,
                view2_tr + scan_scale * delta_v2,
            ],
            [
                base_te + scan_scale * delta_base_te,
                view1_te + scan_scale * delta_v1_te,
                view2_te + scan_scale * delta_v2_te,
            ],
        )
        after_base, after_v1, after_v2 = train_arrays
        after_base_te, after_v1_te, after_v2_te = test_arrays
        train_np, test_np, v1_np, v2_np = numpy_path(after_base, after_base_te, after_v1, after_v2)
        _, _, v1_test_np, v2_test_np = numpy_path(after_base, after_base_te, after_v1_te, after_v2_te)
        after_bt = bt_hidden_metrics(v1_np, v2_np, args.bt_lambda)
        after_test_bt = bt_hidden_metrics(v1_test_np, v2_test_np, args.bt_lambda)
        cov = covariance_spectrum(train_np)
        if getattr(args, "scan_skip_readout", False):
            readout = {"test_accuracy": float("nan")}
        else:
            readout = linear_classifier_readout(train_np, test_np, ytr, yte, args.probe_reg)
        row = {
            "candidate": spec,
            "variant": variant,
            "prefix_layers": int(args.prefix_layers),
            "scan_layer": int(args.scan_layer) + 1,
            "scan_scale": scan_scale,
            "before_train_bt": before_bt["bt_total_per_dim"],
            "before_test_bt": before_test_bt["bt_total_per_dim"],
            "after_train_bt": after_bt["bt_total_per_dim"],
            "after_test_bt": after_test_bt["bt_total_per_dim"],
            "delta_train_bt": after_bt["bt_total_per_dim"] - before_bt["bt_total_per_dim"],
            "delta_test_bt": after_test_bt["bt_total_per_dim"] - before_test_bt["bt_total_per_dim"],
            "after_corr_diag": after_bt["corr_diag_mean"],
            "after_test_corr_diag": after_test_bt["corr_diag_mean"],
            "after_corr_nuclear_per_dim": after_bt["corr_nuclear_per_dim"],
            "after_test_corr_nuclear_per_dim": after_test_bt["corr_nuclear_per_dim"],
            "after_self_corr_offdiag_per_dim": float(
                (0.5 * (self_corr_offdiag_per_dim_torch(after_v1) + self_corr_offdiag_per_dim_torch(after_v2)))
                .detach()
                .cpu()
                .item()
            ),
            "applied_scale": applied_scale,
            "target_norm": float(torch.linalg.vector_norm(first_diag["target"]).detach().cpu().item()),
            "corr_norm": float(torch.linalg.vector_norm(first_diag["corr"]).detach().cpu().item()),
            "branch_residual_energy_fraction": branch_metrics["branch_residual_energy_fraction"],
            "branch_projection_r2": branch_metrics["branch_projection_r2"],
            "test_accuracy": readout["test_accuracy"],
        }
        row.update({f"after_{key}": value for key, value in cov.items()})
        row.update(solve)
        scale_rows.append(row)
        if return_state and (best_row is None or score_row(row, args.scan_select_metric) < score_row(best_row, args.scan_select_metric)):
            best_row = dict(row)
            best_state = (after_base, after_v1, after_v2, after_base_te, after_v1_te, after_v2_te)
    if not return_state:
        best_row = min(scale_rows, key=lambda item: score_row(item, args.scan_select_metric))
        best_row = dict(best_row)
    best_row["selected_by"] = args.scan_select_metric
    if return_state:
        return best_row, scale_rows, best_state
    return best_row, scale_rows


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def ensure_solver_defaults(args):
    defaults = {
        "sample_gradient_weight": 0.0,
        "sample_gradient_scale": 1.0,
        "sample_mode_balance_power": 0.0,
        "sample_mode_balance_eps": 1e-3,
        "sample_mode_balance_min_gain": 0.0,
        "sample_mode_balance_max_gain": 0.0,
        "stable_mode_penalty": 0.0,
        "stable_mode_kind": "agreement",
        "stable_mode_tangent": "raw",
        "stable_mode_diagnostic": False,
        "stable_mode_count": 128,
        "stable_mode_ridge": 1e-3,
        "stable_mode_max_delta": 0.0,
        "stable_mode_normalization": "operator",
        "stable_mode_weight_min": 0.0,
        "stable_mode_weight_max": 0.0,
        "self_cov_weight": 0.0,
        "bt_quadratic_scale": False,
        "bt_quadratic_eval_size": 0,
        "bt_quadratic_scale_max": 1.0,
        "layernorm_kinetic_weight": 0.0,
        "layernorm_kinetic_normalization": "operator",
        "layernorm_kinetic_include_base": False,
        "old_span_update_penalty": 0.0,
        "old_span_update_ridge": 1e-3,
        "old_span_update_normalization": "none",
        "old_span_update_tangent": "raw",
        "old_span_update_weight_min": 0.0,
        "old_span_update_weight_max": 0.0,
        "old_span_adaptive_path": [],
        "old_span_adaptive_rule": "fraction",
        "old_span_adaptive_metric": "bt",
        "old_span_adaptive_eval_size": 0,
        "old_span_adaptive_bt_fraction": 0.95,
        "old_span_adaptive_nuclear_weight": 1.0,
        "line_search_scales": [1.0],
        "line_search_include_zero": False,
        "line_search_mode": "score",
        "line_search_self_cov_weight": -1.0,
        "line_search_self_cov_rel_tol": 0.0,
        "line_search_self_cov_abs_tol": 0.0,
        "line_search_rank_rel_tol": 0.0,
        "line_search_rank_abs_tol": 0.0,
        "line_search_min_bt_gain": 0.0,
        "branch_random_blend": 0.0,
        "branch_shared_power": 0.0,
        "branch_post_transform": "none",
        "branch_post_invariance": 1.0,
        "branch_post_dim": 0,
        "branch_post_cov_ridge": 1e-4,
        "branch_mode_balance_power": 0.0,
        "branch_mode_balance_side": "input",
        "branch_mode_balance_eps": 1e-3,
        "branch_mode_balance_min_gain": 0.0,
        "branch_mode_balance_max_gain": 0.0,
        "branch_novelty_mode": "mix",
        "branch_novelty_mix": 0.0,
        "branch_novelty_scale": 1.0,
        "branch_novelty_filter": "none",
        "branch_novelty_filter_invariance": 1.0,
        "branch_novelty_filter_count": 128,
        "branch_novelty_filter_ridge": 1e-3,
        "branch_novelty_filter_max_delta": 0.0,
        "branch_novelty_filter_projection_ridge": 1e-5,
        "fd_scale": 1e-3,
        "pca_dim": 512,
        "scan_skip_readout": False,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def run(args):
    ensure_solver_defaults(args)
    torch.set_num_threads(args.torch_threads)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
        torch.backends.cudnn.allow_tf32 = not args.no_tf32
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    point = point_for(args, int(args.depth), int(args.seed))
    tensors = load_tensors(point, device)
    if int(args.prefix_layers) > 0:
        prefix_args = copy.copy(args)
        prefix_args.depths = [int(args.depth)]
        prefix_args.variants = ["cf_shrink"]
        prefix_args.seeds = [int(args.seed)]
        prefix_state = run_variant(prefix_args, point, tensors, "cf_shrink", device)
        idx = int(args.prefix_layers) - 1
        state = (
            torch.from_numpy(prefix_state["pathnorm_train"][idx]).to(device),
            torch.from_numpy(prefix_state["pathnorm_view1_train"][idx]).to(device),
            torch.from_numpy(prefix_state["pathnorm_view2_train"][idx]).to(device),
            torch.from_numpy(prefix_state["pathnorm_test"][idx]).to(device),
            torch.from_numpy(prefix_state["pathnorm_view1_test"][idx]).to(device),
            torch.from_numpy(prefix_state["pathnorm_view2_test"][idx]).to(device),
        )
        del prefix_state
        gc.collect()
        torch.cuda.empty_cache()
    else:
        train_arrays, test_arrays, _, _ = normalize_hidden_with_stats_torch(
            [tensors["xtr"], tensors["view1_tr"], tensors["view2_tr"]],
            [tensors["xte"], tensors["view1_te"], tensors["view2_te"]],
        )
        state = (*train_arrays, *test_arrays)
    rows = []
    scale_rows = []
    for spec in args.candidates:
        best_row, spec_scale_rows = solve_candidate(args, point, device, state, spec, tensors["ytr_np"], tensors["yte_np"])
        rows.append(best_row)
        scale_rows.extend(spec_scale_rows)
    write_jsonl(args.out_dir / "realized_gain_rows.jsonl", rows)
    write_jsonl(args.out_dir / "realized_gain_scale_rows.jsonl", scale_rows)
    write_csv(args.out_dir / "realized_gain_rows.csv", rows)
    write_csv(args.out_dir / "realized_gain_scale_rows.csv", scale_rows)
    for row in sorted(rows, key=lambda item: (item["after_test_bt"], -item["after_test_corr_nuclear_per_dim"])):
        print(
            f"{row['candidate']}: testBT {row['before_test_bt']:.4f}->{row['after_test_bt']:.4f} "
            f"nuc {row['after_test_corr_nuclear_per_dim']:.4f} rank {row['after_effective_rank']:.1f} "
            f"self {row['after_self_corr_offdiag_per_dim']:.1f} acc {row['test_accuracy']:.4f} "
            f"scale {row['scan_scale']:.4g}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Scan realized one-step gains for CF-BT residual branch candidates.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_realized_gain_scan"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.7)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--dataset", default="tinyimagenet200_barlow")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--prefix-layers", type=int, default=0)
    parser.add_argument("--scan-layer", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=200)
    parser.add_argument("--bt-lambda", type=float, default=0.005)
    parser.add_argument("--eta-total", type=float, default=32.0)
    parser.add_argument("--moment-target-kind", choices=["bt", "polar", "bt_plus_polar"], default="bt")
    parser.add_argument("--moment-target-weight", type=float, default=1.0)
    parser.add_argument("--polar-target-weight", type=float, default=1.0)
    parser.add_argument("--diag-gradient-multiplier", type=float, default=1.0)
    parser.add_argument("--ols-ridge", type=float, default=1e-5)
    parser.add_argument("--cg-iters", type=int, default=120)
    parser.add_argument("--cg-tol", type=float, default=1e-4)
    parser.add_argument("--standardization-jacobian", choices=["frozen", "projected"], default="projected")
    parser.add_argument("--moment-batch-size", type=int, default=1024)
    parser.add_argument("--moment-ensembles", type=int, default=1)
    parser.add_argument("--branch-dim", type=int, default=512)
    parser.add_argument("--branch-residual-ridge", type=float, default=1e-3)
    parser.add_argument("--activation-alpha", type=float, default=0.5)
    parser.add_argument("--cf-invariance", type=float, default=1.0)
    parser.add_argument("--max-update-ratio", type=float, default=0.35)
    parser.add_argument("--probe-reg", type=float, default=100.0)
    parser.add_argument("--ridge-reg", type=float, default=1e-3)
    parser.add_argument("--spectrum-eps", type=float, default=1e-6)
    parser.add_argument("--max-spectrum-samples", type=int, default=12000)
    parser.add_argument("--max-transition-samples", type=int, default=6000)
    parser.add_argument("--cut-thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--soft-lambdas", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.25, 1.0])
    parser.add_argument("--scan-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--scan-select-metric", choices=["test_bt", "train_bt", "test_nuclear"], default="test_bt")
    parser.add_argument("--scan-skip-readout", action="store_true")
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=[
            "plain",
            "randorth",
            "concat_cf_rand",
            "concat_cf_rand2",
            "nov025",
            "nov05",
            "modein01",
            "modeout025",
            "randomblend01",
            "sharedcross",
        ],
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
