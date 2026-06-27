import argparse
import csv
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cf_mlp_barlow_clean import collect_barlow_representations, load_tensors
from cf_mlp_bt_objective_by_layer import (
    bt_hidden_metrics,
    find_nonres_bt_model,
    find_residual_bt_model,
    residual_cf_args,
)
from cf_mlp_layer_mechanistic import covariance_spectrum
from cf_mlp_residual_barlow import collect_residual_representations
from cf_mlp_residual_bt_variants import collect_variant_state
from cf_mlp_scalability import SweepPoint, write_jsonl


PAPER_STYLE = Path("/home/davwis/.codex/local-plugins/plugins/local-reusability-assets/plotting/paper.mplstyle")


def point_for(args, depth, seed):
    return SweepPoint(
        dataset=args.dataset,
        axis="bpbt_spectral_diagnostic",
        scale_value=args.width,
        seed=seed,
        n_train=args.n_train,
        n_test=args.n_test,
        input_dim=args.input_dim,
        width=args.width,
        depth=depth,
        num_classes=args.num_classes,
    )


def subsample_np(x, max_samples):
    x = np.asarray(x, dtype=np.float32)
    if max_samples <= 0 or x.shape[0] <= max_samples:
        return x
    idx = np.linspace(0, x.shape[0] - 1, max_samples).astype(np.int64)
    return x[idx]


def centered_torch(x):
    x = x.to(dtype=torch.float32)
    return x - x.mean(dim=0, keepdim=True)


def linear_cka_torch(x, y):
    x0 = centered_torch(x)
    y0 = centered_torch(y)
    xy = x0.T @ y0
    xx = x0.T @ x0
    yy = y0.T @ y0
    num = torch.sum(xy * xy)
    den = torch.sqrt(torch.sum(xx * xx) * torch.sum(yy * yy))
    return float((num / torch.clamp(den, min=1e-12)).detach().cpu().item())


def standardize_torch(x):
    x = x.to(dtype=torch.float32)
    return (x - x.mean(dim=0, keepdim=True)) / torch.clamp(x.std(dim=0, keepdim=True), min=1e-6)


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
    return float((1.0 - mse / torch.clamp(baseline, min=1e-12)).detach().cpu().item())


def transition_metrics(prev, cur, args, device):
    prev_np = subsample_np(prev, args.max_transition_samples)
    cur_np = subsample_np(cur, args.max_transition_samples)
    prev_t = torch.from_numpy(prev_np).to(device)
    cur_t = torch.from_numpy(cur_np).to(device)
    forward_r2 = ridge_predict_r2_torch(prev_t, cur_t, args.ridge_reg)
    reverse_r2 = ridge_predict_r2_torch(cur_t, prev_t, args.ridge_reg)
    sym_r2 = 0.5 * (forward_r2 + reverse_r2)
    return {
        "prev_to_cur_cka": linear_cka_torch(prev_t, cur_t),
        "prev_to_cur_forward_r2": forward_r2,
        "prev_to_cur_reverse_r2": reverse_r2,
        "prev_to_cur_sym_r2": sym_r2,
        "prev_to_cur_linear_novelty": 1.0 - sym_r2,
    }


def shared_difference_metrics(view1, view2):
    x1 = np.asarray(view1, dtype=np.float64)
    x2 = np.asarray(view2, dtype=np.float64)
    shared = 0.5 * (x1 + x2)
    diff = 0.5 * (x1 - x2)
    shared = shared - shared.mean(axis=0, keepdims=True)
    diff = diff - diff.mean(axis=0, keepdims=True)
    shared_trace = float(np.mean(shared * shared))
    diff_trace = float(np.mean(diff * diff))
    total = shared_trace + diff_trace
    return {
        "shared_trace_per_dim": shared_trace,
        "diff_trace_per_dim": diff_trace,
        "shared_diff_ratio": shared_trace / max(diff_trace, 1e-12),
        "diff_fraction": diff_trace / max(total, 1e-12),
    }


def agreement_spectrum_metrics(view1, view2, args, device):
    v1_np = subsample_np(view1, args.max_spectrum_samples)
    v2_np = subsample_np(view2, args.max_spectrum_samples)
    v1 = torch.from_numpy(v1_np).to(device)
    v2 = torch.from_numpy(v2_np).to(device)
    dim = v1.shape[1]
    mean = 0.5 * (v1.mean(dim=0, keepdim=True) + v2.mean(dim=0, keepdim=True))
    h1 = v1 - mean
    h2 = v2 - mean
    n = float(v1.shape[0])
    sigma1 = (h1.T @ h1) / n
    sigma2 = (h2.T @ h2) / n
    sigma = 0.5 * (sigma1 + sigma2)
    sigma = 0.5 * (sigma + sigma.T)
    delta_h = h1 - h2
    delta = (delta_h.T @ delta_h) / n
    delta = 0.5 * (delta + delta.T)

    evals_sigma, evecs_sigma = torch.linalg.eigh(sigma)
    evals_sigma = torch.clamp(evals_sigma, min=args.spectrum_eps)
    sigma_inv_sqrt = (evecs_sigma / torch.sqrt(evals_sigma).unsqueeze(0)) @ evecs_sigma.T
    m_matrix = sigma_inv_sqrt @ delta @ sigma_inv_sqrt
    m_matrix = 0.5 * (m_matrix + m_matrix.T)
    eigvals = torch.sort(torch.linalg.eigvalsh(m_matrix)).values.detach().cpu().numpy()
    eigvals = np.maximum(eigvals.astype(np.float64), 0.0)
    total = float(np.sum(eigvals))
    probs = eigvals / max(total, 1e-12)
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12)))) if total > 1e-12 else 0.0

    row = {
        "agreement_delta_min": float(eigvals[0]),
        "agreement_delta_q10": float(np.quantile(eigvals, 0.10)),
        "agreement_delta_q25": float(np.quantile(eigvals, 0.25)),
        "agreement_delta_median": float(np.quantile(eigvals, 0.50)),
        "agreement_delta_q75": float(np.quantile(eigvals, 0.75)),
        "agreement_delta_q90": float(np.quantile(eigvals, 0.90)),
        "agreement_delta_max": float(eigvals[-1]),
        "agreement_delta_mean": float(np.mean(eigvals)),
        "agreement_delta_effective_rank": float(np.exp(entropy)) if total > 1e-12 else 0.0,
    }
    for threshold in args.cut_thresholds:
        label = str(threshold).replace(".", "p")
        count = int(np.sum(eigvals <= threshold))
        row[f"agreement_cut_count_le_{label}"] = count
        row[f"agreement_cut_fraction_le_{label}"] = count / dim
    for lambd in args.soft_lambdas:
        label = str(lambd).replace(".", "p")
        gains = float(lambd) / (eigvals + float(lambd))
        row[f"agreement_soft_keep_sum_lam_{label}"] = float(np.sum(gains))
        row[f"agreement_soft_keep_mean_lam_{label}"] = float(np.mean(gains))
    return row


def row_for_layer(args, point, model, variant, checkpoint, layer_idx, train, view1, view2, prev_train, device):
    row = {
        "model": model,
        "variant": variant,
        "checkpoint": checkpoint,
        "dataset": point.dataset,
        "seed": point.seed,
        "input_dim": point.input_dim,
        "width": point.width,
        "depth": point.depth,
        "layer": layer_idx + 1,
    }
    row.update(bt_hidden_metrics(view1, view2, args.bt_lambda))
    row.update(shared_difference_metrics(view1, view2))
    row.update(agreement_spectrum_metrics(view1, view2, args, device))
    row.update(covariance_spectrum(train))
    if prev_train is None:
        row.update(
            {
                "prev_to_cur_cka": float("nan"),
                "prev_to_cur_forward_r2": float("nan"),
                "prev_to_cur_reverse_r2": float("nan"),
                "prev_to_cur_sym_r2": float("nan"),
                "prev_to_cur_linear_novelty": float("nan"),
            }
        )
    else:
        row.update(transition_metrics(prev_train, train, args, device))
    return row


def rows_from_reps(args, point, model, variant, checkpoint, reps, device):
    rows = []
    prev_train = None
    for layer_idx, (train, view1, view2) in enumerate(
        zip(reps["pathnorm_train"], reps["pathnorm_view1_train"], reps["pathnorm_view2_train"])
    ):
        rows.append(row_for_layer(args, point, model, variant, checkpoint, layer_idx, train, view1, view2, prev_train, device))
        prev_train = train
    return rows


def collect_residual_backprop_bt(args, depth, seed, device):
    point = point_for(args, depth, seed)
    tensors = load_tensors(point, device)
    model_path = find_residual_bt_model(args.bt_model_dir, depth)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    reps = collect_residual_representations(point, tensors, state, device)
    rows = rows_from_reps(args, point, "residual_backprop_bt", "residual_backprop_bt", str(model_path), reps, device)
    del tensors, state, reps
    torch.cuda.empty_cache()
    gc.collect()
    return rows


def collect_nonres_backprop_bt(args, depth, seed, device):
    point = point_for(args, depth, seed)
    tensors = load_tensors(point, device)
    model_path = find_nonres_bt_model(args.nonres_bt_model_dir, depth)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    reps = collect_barlow_representations(point, tensors, state, device)
    rows = rows_from_reps(args, point, "nonres_backprop_bt", "nonres_backprop_bt", str(model_path), reps, device)
    del tensors, state, reps
    torch.cuda.empty_cache()
    gc.collect()
    return rows


def collect_nonres_cf_bt(args, depth, seed, variant, device, device_name):
    point = point_for(args, depth, seed)
    state = collect_variant_state(point, variant, residual_cf_args(args), device, device_name)
    rows = rows_from_reps(args, point, "nonres_cf_bt", variant, "", state, device)
    del state
    torch.cuda.empty_cache()
    gc.collect()
    return rows


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def grouped(rows):
    keys = sorted({(row["model"], row["variant"], row["depth"], row["seed"]) for row in rows})
    for key in keys:
        items = [row for row in rows if (row["model"], row["variant"], row["depth"], row["seed"]) == key]
        yield key, sorted(items, key=lambda row: row["layer"])


def summarize(rows):
    summaries = []
    for (model, variant, depth, seed), items in grouped(rows):
        first = items[0]
        final = items[-1]
        best = min(items, key=lambda row: row["bt_total_per_dim"])
        bt_steps = [b["bt_total_per_dim"] - a["bt_total_per_dim"] for a, b in zip(items[:-1], items[1:])]
        summaries.append(
            {
                "model": model,
                "variant": variant,
                "depth": depth,
                "seed": seed,
                "first_bt_per_dim": first["bt_total_per_dim"],
                "final_bt_per_dim": final["bt_total_per_dim"],
                "best_bt_per_dim": best["bt_total_per_dim"],
                "best_bt_layer": best["layer"],
                "bt_improving_step_fraction": float(np.mean([step <= 0.0 for step in bt_steps])) if bt_steps else float("nan"),
                "first_corr_diag": first["corr_diag_mean"],
                "final_corr_diag": final["corr_diag_mean"],
                "first_shared_diff_ratio": first["shared_diff_ratio"],
                "final_shared_diff_ratio": final["shared_diff_ratio"],
                "first_cut_count_le_0p25": first.get("agreement_cut_count_le_0p25", float("nan")),
                "final_cut_count_le_0p25": final.get("agreement_cut_count_le_0p25", float("nan")),
                "first_soft_keep_lam_0p1": first.get("agreement_soft_keep_sum_lam_0p1", float("nan")),
                "final_soft_keep_lam_0p1": final.get("agreement_soft_keep_sum_lam_0p1", float("nan")),
                "first_effective_rank": first["effective_rank"],
                "final_effective_rank": final["effective_rank"],
                "mean_linear_novelty": float(
                    np.nanmean([row["prev_to_cur_linear_novelty"] for row in items])
                ),
                "final_linear_novelty": final["prev_to_cur_linear_novelty"],
            }
        )
    return summaries


def fmt(value):
    if isinstance(value, float) and np.isnan(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def label_for(model, variant):
    if model == "nonres_cf_bt":
        return variant.replace("plain_cf_", "CF ")
    return model


def plot_depth24(rows, out_dir):
    if PAPER_STYLE.exists():
        plt.style.use(str(PAPER_STYLE))
    metrics = [
        ("bt_total_per_dim", "BT / dim", "log"),
        ("corr_diag_mean", "corr diag", "linear"),
        ("shared_diff_ratio", "shared/diff", "log"),
        ("agreement_soft_keep_sum_lam_0p1", "soft keep sum, lambda=0.1", "linear"),
        ("effective_rank", "effective rank", "linear"),
        ("prev_to_cur_linear_novelty", "layer novelty", "linear"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.2), sharex=False)
    axes = axes.reshape(-1)
    for ax, (metric, ylabel, scale) in zip(axes, metrics):
        for (model, variant, depth, _seed), items in grouped(rows):
            if depth != 24:
                continue
            ax.plot(
                [row["layer"] for row in items],
                [row[metric] for row in items],
                marker="o",
                markersize=2.2,
                linewidth=1.1,
                label=label_for(model, variant),
            )
        ax.set_xlabel("layer")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        if scale == "log":
            ax.set_yscale("log")
    axes[0].legend(loc="best", fontsize=6, frameon=False)
    fig.suptitle("Depth-24 BP-BT and CF spectral diagnostics")
    fig.tight_layout()
    pdf = out_dir / "depth24_spectral_diagnostics.pdf"
    png = out_dir / "depth24_spectral_diagnostics.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def write_report(args, summaries, rows, out_dir, figure_paths):
    lines = [
        "# BP-BT Spectral Diagnostic",
        "",
        "Layerwise diagnostics reuse saved backprop-BT checkpoints and selected non-residual CF states.",
        "The agreement spectrum is the eigenvalue spectrum of the paired-view difference covariance whitened by average view covariance, i.e. the object used by the CF agreement/cutting rules.",
        "",
        "| Model | Variant | Depth | Final BT/dim | Best BT/dim | Best layer | Corr diag first->final | Shared/diff first->final | Cut count <=0.25 first->final | Soft keep lambda=0.1 first->final | Eff. rank first->final | Mean novelty |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summaries, key=lambda rec: (rec["depth"], rec["model"], rec["variant"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["variant"],
                    str(row["depth"]),
                    fmt(row["final_bt_per_dim"]),
                    fmt(row["best_bt_per_dim"]),
                    str(row["best_bt_layer"]),
                    f"{fmt(row['first_corr_diag'])}->{fmt(row['final_corr_diag'])}",
                    f"{fmt(row['first_shared_diff_ratio'])}->{fmt(row['final_shared_diff_ratio'])}",
                    f"{fmt(row['first_cut_count_le_0p25'])}->{fmt(row['final_cut_count_le_0p25'])}",
                    f"{fmt(row['first_soft_keep_lam_0p1'])}->{fmt(row['final_soft_keep_lam_0p1'])}",
                    f"{fmt(row['first_effective_rank'])}->{fmt(row['final_effective_rank'])}",
                    fmt(row["mean_linear_novelty"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Hooks",
            "",
            "- If BP-BT improves BT while the cut-count/soft-keep dimensionality stays broad, then a hard high-agreement subspace rule is too destructive.",
            "- If BP-BT's shared/diff ratio improves with only mild rank loss, the CF target should be a residual redistribution/conditioning map rather than a selector.",
            "- If BP-BT's novelty is small but nonzero, the useful nonlinearity is likely an incremental residual correction, not a fresh decomposition into new linear regions every layer.",
            "",
            "## Files",
            "",
            f"- Plot: `{figure_paths[0].name}` / `{figure_paths[1].name}`",
            "- Rows: `spectral_rows.jsonl`, `spectral_rows.csv`",
            "- Summary: `summary.jsonl`, `summary.csv`",
            "",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    torch.set_num_threads(args.torch_threads)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
        torch.backends.cudnn.allow_tf32 = not args.no_tf32
    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    selected = set(args.models)
    for depth in args.depths:
        for seed in args.seeds:
            if "residual_backprop_bt" in selected:
                print(f"spectral diagnostic depth={depth} seed={seed} model=residual_backprop_bt", flush=True)
                rows.extend(collect_residual_backprop_bt(args, depth, seed, device))
                write_jsonl(args.out_dir / "spectral_rows.partial.jsonl", rows)
            if "nonres_backprop_bt" in selected:
                print(f"spectral diagnostic depth={depth} seed={seed} model=nonres_backprop_bt", flush=True)
                rows.extend(collect_nonres_backprop_bt(args, depth, seed, device))
                write_jsonl(args.out_dir / "spectral_rows.partial.jsonl", rows)
            if "nonres_cf_bt" in selected:
                for variant in args.cf_variants:
                    print(f"spectral diagnostic depth={depth} seed={seed} model=nonres_cf_bt variant={variant}", flush=True)
                    rows.extend(collect_nonres_cf_bt(args, depth, seed, variant, device, device_name))
                    write_jsonl(args.out_dir / "spectral_rows.partial.jsonl", rows)

    write_jsonl(args.out_dir / "spectral_rows.jsonl", rows)
    write_csv(args.out_dir / "spectral_rows.csv", rows)
    summaries = summarize(rows)
    write_jsonl(args.out_dir / "summary.jsonl", summaries)
    write_csv(args.out_dir / "summary.csv", summaries)
    figure_paths = plot_depth24(rows, args.out_dir)
    write_report(args, summaries, rows, args.out_dir, figure_paths)
    print((args.out_dir / "report.md").read_text(encoding="utf-8"), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Compare BP-BT and CF-BT layer trajectories in the spectral objects used by CF rules.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_bpbt_spectral_diagnostic_seed7"))
    parser.add_argument("--bt-model-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_residual_barlow_seed7"))
    parser.add_argument("--nonres-bt-model-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_barlow_layer_only_cifar100_simclr_seed7"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.55)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--models", nargs="+", default=["residual_backprop_bt", "nonres_cf_bt"], choices=["residual_backprop_bt", "nonres_backprop_bt", "nonres_cf_bt"])
    parser.add_argument("--cf-variants", nargs="+", default=["plain_cf_relu", "plain_cf_agreement_expand_fullwhiten_relu_k192"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--dataset", default="cifar100_simclr")
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depths", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--bt-lambda", type=float, default=0.005)
    parser.add_argument("--max-spectrum-samples", type=int, default=50000)
    parser.add_argument("--max-transition-samples", type=int, default=12000)
    parser.add_argument("--ridge-reg", type=float, default=1e-3)
    parser.add_argument("--spectrum-eps", type=float, default=1e-6)
    parser.add_argument("--cut-thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--soft-lambdas", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.25, 1.0])
    parser.add_argument("--cf-residual-scale", type=float, default=1.0)
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
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
