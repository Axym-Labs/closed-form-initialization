import argparse
import csv
import gc
from pathlib import Path

import numpy as np
import torch

from cf_mlp_barlow_clean import load_tensors
from cf_mlp_moment_ols_residual import (
    normalize_hidden_with_stats_torch,
    numpy_path,
    point_for,
    readout_rows,
)
from cf_mlp_realized_gain_scan import ensure_solver_defaults, score_row, solve_candidate
from cf_mlp_scalability import write_jsonl


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def selector_score(row, metric):
    return score_row(row, metric)


def summarize_selector(args, variant, point, selected_rows, readout_summary):
    final = selected_rows[-1]
    return {
        "variant": variant,
        "seed": point.seed,
        "dataset": point.dataset,
        "depth": point.depth,
        "selector_metric": args.scan_select_metric,
        "final_train_bt_total_per_dim": final["after_train_bt"],
        "final_test_bt_total_per_dim": final["after_test_bt"],
        "final_corr_diag_mean": final["after_corr_diag"],
        "final_test_corr_diag_mean": final["after_test_corr_diag"],
        "final_corr_nuclear_per_dim": final["after_corr_nuclear_per_dim"],
        "final_test_corr_nuclear_per_dim": final["after_test_corr_nuclear_per_dim"],
        "final_effective_rank": final["after_effective_rank"],
        "final_self_corr_offdiag_per_dim": final["after_self_corr_offdiag_per_dim"],
        "bt_improving_step_fraction": float(np.mean([row["delta_train_bt"] < 0.0 for row in selected_rows])),
        "mean_scan_scale": float(np.mean([row["scan_scale"] for row in selected_rows])),
        "last_layer_accuracy": readout_summary["last_layer_accuracy"],
        "all_pca_accuracy": readout_summary["all_pca_accuracy"],
        "best_layer_accuracy": readout_summary["best_layer_accuracy"],
        "best_layer": readout_summary["best_layer"],
        "candidate_path": " ".join(row["candidate"] for row in selected_rows),
        "scale_path": " ".join(f"{row['scan_scale']:.4g}" for row in selected_rows),
    }


def write_report(path, summaries):
    lines = [
        "# Realized-Gain CF-BT Selector",
        "",
        "Each layer solves closed-form projected BT-gradient OLS for a menu of residual branches,",
        "then applies the candidate/scale with the best realized selector statistic.",
        "",
        "| Variant | Depth | Metric | Train BT | Test BT | Corr train/test | Nuclear train/test | Rank | Self-cov off | Last acc | All PCA | Best layer | Candidates | Scales |",
        "| --- | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    str(row["depth"]),
                    row["selector_metric"],
                    f"{row['final_train_bt_total_per_dim']:.4g}",
                    f"{row['final_test_bt_total_per_dim']:.4g}",
                    f"{row['final_corr_diag_mean']:.4g}/{row['final_test_corr_diag_mean']:.4g}",
                    f"{row['final_corr_nuclear_per_dim']:.4g}/{row['final_test_corr_nuclear_per_dim']:.4g}",
                    f"{row['final_effective_rank']:.4g}",
                    f"{row['final_self_corr_offdiag_per_dim']:.4g}",
                    f"{row['last_layer_accuracy']:.4g}",
                    f"{row['all_pca_accuracy']:.4g}",
                    f"{row['best_layer_accuracy']:.4g} @ {row['best_layer']}",
                    row["candidate_path"],
                    row["scale_path"],
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args):
    ensure_solver_defaults(args)
    args.scan_skip_readout = True
    torch.set_num_threads(args.torch_threads)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
        torch.backends.cudnn.allow_tf32 = not args.no_tf32
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_selected_rows = []
    all_scale_rows = []
    all_layer_readouts = []
    all_setup_readouts = []
    summaries = []

    for depth in args.depths:
        for seed in args.seeds:
            point = point_for(args, int(depth), int(seed))
            tensors = load_tensors(point, device)
            train_arrays, test_arrays, _, _ = normalize_hidden_with_stats_torch(
                [tensors["xtr"], tensors["view1_tr"], tensors["view2_tr"]],
                [tensors["xte"], tensors["view1_te"], tensors["view2_te"]],
            )
            state = (*train_arrays, *test_arrays)
            variant = f"realized_selector_{args.scan_select_metric}"
            path_train = []
            path_test = []
            path_view1 = []
            path_view2 = []
            path_view1_test = []
            path_view2_test = []
            selected_rows = []

            for layer_idx in range(point.depth):
                args.scan_layer = int(layer_idx)
                args.prefix_layers = int(layer_idx)
                best_row = None
                best_state = None
                for spec in args.candidates:
                    cand_row, scale_rows, cand_state = solve_candidate(
                        args,
                        point,
                        device,
                        state,
                        spec,
                        tensors["ytr_np"],
                        tensors["yte_np"],
                        return_state=True,
                    )
                    cand_row = dict(cand_row)
                    cand_row["selector_layer"] = layer_idx + 1
                    for row in scale_rows:
                        row["selector_layer"] = layer_idx + 1
                    all_scale_rows.extend(scale_rows)
                    if best_row is None or selector_score(cand_row, args.scan_select_metric) < selector_score(
                        best_row, args.scan_select_metric
                    ):
                        best_row = cand_row
                        best_state = cand_state
                    else:
                        del cand_state
                    gc.collect()

                if best_state is None:
                    raise RuntimeError(f"No selected candidate at layer {layer_idx + 1}")
                state = best_state
                best_row["variant"] = variant
                selected_rows.append(best_row)
                all_selected_rows.append(best_row)

                base_tr, view1_tr, view2_tr, base_te, view1_te, view2_te = state
                train_np, test_np, v1_np, v2_np = numpy_path(base_tr, base_te, view1_tr, view2_tr)
                _, _, v1_test_np, v2_test_np = numpy_path(base_tr, base_te, view1_te, view2_te)
                path_train.append(train_np)
                path_test.append(test_np)
                path_view1.append(v1_np)
                path_view2.append(v2_np)
                path_view1_test.append(v1_test_np)
                path_view2_test.append(v2_test_np)
                print(
                    f"layer {layer_idx + 1}: {best_row['candidate']} scale={best_row['scan_scale']:.4g} "
                    f"trainBT={best_row['after_train_bt']:.4f} testBT={best_row['after_test_bt']:.4f} "
                    f"rank={best_row['after_effective_rank']:.1f}",
                    flush=True,
                )

            state_np = {
                "pathnorm_train": path_train,
                "pathnorm_test": path_test,
                "pathnorm_view1_train": path_view1,
                "pathnorm_view2_train": path_view2,
                "pathnorm_view1_test": path_view1_test,
                "pathnorm_view2_test": path_view2_test,
            }
            layer_rows, setup_rows, readout_summary = readout_rows(
                args,
                point,
                variant,
                state_np,
                tensors["ytr_np"],
                tensors["yte_np"],
            )
            all_layer_readouts.extend(layer_rows)
            all_setup_readouts.extend(setup_rows)
            summaries.append(summarize_selector(args, variant, point, selected_rows, readout_summary))
            del tensors, state
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_jsonl(args.out_dir / "selected_rows.jsonl", all_selected_rows)
    write_jsonl(args.out_dir / "candidate_scale_rows.jsonl", all_scale_rows)
    write_jsonl(args.out_dir / "layer_readouts.jsonl", all_layer_readouts)
    write_jsonl(args.out_dir / "setup_readouts.jsonl", all_setup_readouts)
    write_jsonl(args.out_dir / "summary.jsonl", summaries)
    write_csv(args.out_dir / "selected_rows.csv", all_selected_rows)
    write_csv(args.out_dir / "candidate_scale_rows.csv", all_scale_rows)
    write_csv(args.out_dir / "layer_readouts.csv", all_layer_readouts)
    write_csv(args.out_dir / "summary.csv", summaries)
    write_report(args.out_dir / "report.md", summaries)
    print((args.out_dir / "report.md").read_text(encoding="utf-8"), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Greedy realized-gain selector for CF-BT residual gradient steps.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_realized_selector"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.7)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--dataset", default="tinyimagenet200_barlow")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--depths", type=int, nargs="+", default=[12])
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
    parser.add_argument("--pca-dim", type=int, default=512)
    parser.add_argument("--ridge-reg", type=float, default=1e-3)
    parser.add_argument("--spectrum-eps", type=float, default=1e-6)
    parser.add_argument("--max-spectrum-samples", type=int, default=12000)
    parser.add_argument("--max-transition-samples", type=int, default=6000)
    parser.add_argument("--cut-thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--soft-lambdas", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.25, 1.0])
    parser.add_argument("--scan-scales", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.125, 0.0625, 0.0])
    parser.add_argument("--scan-select-metric", choices=["test_bt", "train_bt", "test_nuclear"], default="test_bt")
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["plain", "nov025", "modeout025", "randomblend01", "randorth", "sharedcross"],
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
