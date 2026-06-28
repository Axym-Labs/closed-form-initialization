import argparse
import csv
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import torch

from cf_mlp_residual_barlow import collect_residual_representations
from cf_mlp_residual_bt_variants import collect_variant_state
from cf_mlp_barlow_clean import collect_barlow_representations, load_tensors
from cf_mlp_scalability import SweepPoint, write_jsonl


PAPER_STYLE = Path("/home/davwis/.codex/local-plugins/plugins/local-reusability-assets/plotting/paper.mplstyle")
MODEL_ORDER = ["residual_backprop_bt", "nonres_backprop_bt", "residual_cf_bt", "nonres_cf_bt"]
MODEL_LABELS = {
    "residual_backprop_bt": "residual backprop-BT",
    "nonres_backprop_bt": "non-residual backprop-BT",
    "residual_cf_bt": "residual CF-BT",
    "nonres_cf_bt": "non-residual CF-BT",
}
MODEL_COLORS = {
    "residual_backprop_bt": "#0072B2",
    "nonres_backprop_bt": "#009E73",
    "residual_cf_bt": "#D55E00",
    "nonres_cf_bt": "#CC79A7",
}
MODEL_LINESTYLES = {
    "residual_backprop_bt": "-",
    "nonres_backprop_bt": "--",
    "residual_cf_bt": "-",
    "nonres_cf_bt": "--",
}


def activation_suffix(variant):
    if "agreement_biasopt_ccaridge_relu_r" in variant:
        return f"agreement bias-opt + CCA ridge r={variant.rsplit('_r', 1)[1]}"
    if "postrelu_biasopt_ccaridge_relu_r" in variant:
        return f"bias-opt + CCA ridge r={variant.rsplit('_r', 1)[1]}"
    if "agreement_biasopt_ccablend_relu_m" in variant:
        return f"agreement bias-opt + CCA blend m={variant.rsplit('_m', 1)[1]}"
    if "postrelu_biasopt_ccablend_relu_m" in variant:
        return f"bias-opt + CCA blend m={variant.rsplit('_m', 1)[1]}"
    if "agreement_biasopt_ccalinear_relu" in variant:
        return "agreement bias-opt + shared-CCA"
    if "postrelu_biasopt_ccalinear_relu" in variant:
        return "bias-opt + shared-CCA"
    if "agreement_biasopt_crosseig_relu" in variant:
        return "agreement bias-opt + cross eig-rotate"
    if "agreement_biasopt_ccarotate_relu" in variant:
        return "agreement bias-opt + CCA eig-rotate"
    if "agreement_biasopt_ccapower_relu_p" in variant:
        return f"agreement bias-opt + CCA power p={variant.rsplit('_p', 1)[1]}"
    if "agreement_biasopt_ccapowersched_relu_p" in variant:
        tail = variant.rsplit("_p", 1)[1]
        start, end = tail.split("_to", 1)
        return f"agreement bias-opt + CCA power {start}->{end}"
    if "agreement_biasopt_alignls_relu_r" in variant:
        return f"agreement bias-opt + align-LS r={variant.rsplit('_r', 1)[1]}"
    if "agreement_biasopt_linearopt_relu" in variant:
        return "agreement bias-opt + postnorm linear"
    if "postrelu_biasopt_linearopt_relu" in variant:
        return "bias-opt + postnorm linear"
    if "agreement_corrbias_ccalinear_relu_b" in variant:
        return f"agreement corr-bias + shared-CCA b={variant.rsplit('_b', 1)[1]}"
    if "agreement_corrbias_relu_b" in variant:
        return f"agreement corr-bias b={variant.rsplit('_b', 1)[1]}"
    if "agreement_activegain_relu_lo" in variant:
        tail = variant.rsplit("_lo", 1)[1]
        lo, hi = tail.split("_hi", 1)
        return f"agreement active-gain lo={lo} hi={hi}"
    if "agreement_activerank_ccalinear_relu_lo" in variant:
        tail = variant.rsplit("_lo", 1)[1]
        lo, hi = tail.split("_hi", 1)
        return f"agreement active-rank + shared-CCA lo={lo} hi={hi}"
    if "agreement_activerank_relu_lo" in variant:
        tail = variant.rsplit("_lo", 1)[1]
        lo, hi = tail.split("_hi", 1)
        return f"agreement active-rank lo={lo} hi={hi}"
    if "hybrid_agreement_biasopt_relu_m" in variant:
        return f"hybrid agreement bias-opt m={variant.rsplit('_m', 1)[1]}"
    if "hybrid_agreement_diagopt_relu_m" in variant:
        return f"hybrid agreement diag-opt m={variant.rsplit('_m', 1)[1]}"
    if "agreement_biasopt_relu" in variant:
        return "agreement-space bias-opt"
    if "agreement_diagopt_relu" in variant:
        return "agreement-space diag-opt"
    if "eigenshrink_biasopt_relu" in variant:
        return "eigenshrink-space bias-opt"
    if "eigenshrink_diagopt_relu" in variant:
        return "eigenshrink-space diag-opt"
    if "postrelu_biasdiagopt_relu" in variant:
        return "post-ReLU bias diag-opt"
    if "postrelu_diagopt_relu" in variant:
        return "post-ReLU diag-opt"
    if "postrelu_biasopt_relu" in variant:
        return "post-ReLU bias-opt"
    if "postrelu_affineopt_relu" in variant:
        return "post-ReLU affine-opt"
    if "postrelu_active_ramp_relu" in variant:
        return "post-ReLU active ramp"
    if "postrelu_active_relu_a" in variant:
        return f"post-ReLU active target a={variant.rsplit('_a', 1)[1]}"
    if "postrelu_corrbias_relu_b" in variant:
        return f"post-ReLU corr-bias b={variant.rsplit('_b', 1)[1]}"
    if "agreement_gate_relu_b" in variant:
        return f"agreement-gate ReLU b={variant.rsplit('_b', 1)[1]}"
    if "agreement_expand_fullwhiten_relu_k" in variant:
        return f"agreement expand fullwhiten ReLU k={variant.rsplit('_k', 1)[1]}"
    if "agreement_expand_adaptwhiten_relu_k" in variant:
        prefix, threshold = variant.rsplit("_t", 1)
        return f"agreement expand adaptive whiten ReLU k={prefix.rsplit('_k', 1)[1]} t={threshold}"
    if "agreement_expand_relu_k" in variant:
        return f"agreement expand ReLU k={variant.rsplit('_k', 1)[1]}"
    if "sharedmetric_relu" in variant:
        return "shared-metric ReLU"
    if "eigen_shrink_relu" in variant:
        return "eigen-shrink ReLU"
    if "rowcorrect_relu" in variant:
        return "row-correct ReLU"
    if "relu" in variant and "leakygelu" not in variant:
        return "ReLU"
    if "bpbt_nonlinearity" in variant or "leakygelu0.5" in variant:
        return "leaky-GELU alpha=0.5"
    return variant


def model_label(args, model):
    label = MODEL_LABELS[model]
    if model == "residual_cf_bt":
        return f"{label} ({activation_suffix(args.residual_cf_variant)})"
    if model == "nonres_cf_bt":
        return f"{label} ({activation_suffix(args.nonres_cf_variant)})"
    return label


def offdiag_values(matrix):
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask]


def bt_hidden_metrics(view1, view2, lambd):
    x1 = np.asarray(view1, dtype=np.float64)
    x2 = np.asarray(view2, dtype=np.float64)
    x1 = (x1 - x1.mean(axis=0, keepdims=True)) / np.maximum(x1.std(axis=0, keepdims=True), 1e-4)
    x2 = (x2 - x2.mean(axis=0, keepdims=True)) / np.maximum(x2.std(axis=0, keepdims=True), 1e-4)
    corr = (x1.T @ x2) / x1.shape[0]
    diag = np.diag(corr)
    off = offdiag_values(corr)
    on_diag = float(np.sum((diag - 1.0) ** 2))
    off_diag = float(np.sum(off * off))
    weighted_off = float(lambd * off_diag)
    total = on_diag + weighted_off
    dim = int(corr.shape[0])
    singular = np.linalg.svd(corr, compute_uv=False)
    singular = np.maximum(singular.astype(np.float64), 0.0)
    singular_sum = float(np.sum(singular))
    singular_probs = singular / max(singular_sum, 1e-12)
    singular_entropy = -float(np.sum(singular_probs * np.log(np.maximum(singular_probs, 1e-12))))
    return {
        "hidden_dim": dim,
        "bt_lambda": float(lambd),
        "bt_total": total,
        "bt_total_per_dim": total / dim,
        "bt_on_diag": on_diag,
        "bt_on_diag_per_dim": on_diag / dim,
        "bt_weighted_off_diag": weighted_off,
        "bt_weighted_off_diag_per_dim": weighted_off / dim,
        "bt_off_diag": off_diag,
        "bt_off_diag_mean_sq": off_diag / max(dim * (dim - 1), 1),
        "corr_diag_mean": float(np.mean(diag)),
        "corr_diag_min": float(np.min(diag)),
        "corr_diag_max": float(np.max(diag)),
        "corr_singular_mean": float(np.mean(singular)),
        "corr_singular_max": float(np.max(singular)),
        "corr_nuclear_per_dim": singular_sum / dim,
        "corr_singular_effective_rank": float(np.exp(singular_entropy)),
        "corr_trace_to_nuclear": float(np.trace(corr) / max(singular_sum, 1e-12)),
        "offdiag_rms": float(np.sqrt(np.mean(off * off))),
    }


def residual_cf_args(args):
    return SimpleNamespace(
        width=args.width,
        inverse_reg=1e-2,
        residual_scale=args.cf_residual_scale,
        align_weight=1.0,
        bt_weight=0.05,
        bt_offdiag_lambda=args.bt_lambda,
        linearized_ridge=1.0,
        linearized_lr=0.05,
        linearized_steps=30,
        linearized_grad_clip=10.0,
        linearized_max_norm=16.0,
        linearized_residual_scale=0.25,
        postrelu_fit_samples=args.postrelu_fit_samples,
        postrelu_steps=args.postrelu_steps,
        postrelu_lr=args.postrelu_lr,
        postrelu_scale_ridge=args.postrelu_scale_ridge,
        postrelu_bias_ridge=args.postrelu_bias_ridge,
        postrelu_grad_clip=args.postrelu_grad_clip,
        postnorm_linear_fit_samples=args.postnorm_linear_fit_samples,
        postnorm_linear_steps=args.postnorm_linear_steps,
        postnorm_linear_lr=args.postnorm_linear_lr,
        postnorm_linear_ridge=args.postnorm_linear_ridge,
        postnorm_linear_grad_clip=args.postnorm_linear_grad_clip,
        postnorm_linear_cca_eps=args.postnorm_linear_cca_eps,
    )


def point_for(args, depth, seed):
    return SweepPoint(
        dataset=args.dataset,
        axis="bt_objective_by_layer",
        scale_value=args.width,
        seed=seed,
        n_train=args.n_train,
        n_test=args.n_test,
        input_dim=args.input_dim,
        width=args.width,
        depth=depth,
        num_classes=args.num_classes,
    )


def find_residual_bt_model(model_dir, depth):
    matches = sorted(model_dir.glob(f"residual_bt_*_d{depth}_*.pt"))
    if not matches:
        raise FileNotFoundError(f"No residual BT model found for depth {depth} in {model_dir}")
    if len(matches) > 1:
        print(f"Using {matches[-1]} among {len(matches)} depth-{depth} residual BT checkpoints", flush=True)
    return matches[-1]


def find_nonres_bt_model(model_dir, depth):
    matches = sorted(model_dir.glob(f"bt_*_d{depth}_*.pt"))
    if not matches:
        raise FileNotFoundError(f"No non-residual BT model found for depth {depth} in {model_dir}")
    preferred = [path for path in matches if "l0.005" in path.name]
    selected = preferred[-1] if preferred else matches[-1]
    if len(matches) > 1:
        print(f"Using {selected} among {len(matches)} depth-{depth} non-residual BT checkpoints", flush=True)
    return selected


def collect_backprop_bt_rows(args, depth, seed, device):
    point = point_for(args, depth, seed)
    tensors = load_tensors(point, device)
    model_path = find_residual_bt_model(args.bt_model_dir, depth)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    reps = collect_residual_representations(point, tensors, state, device)
    rows = []
    for idx, (view1, view2) in enumerate(zip(reps["pathnorm_view1_train"], reps["pathnorm_view2_train"])):
        row = {
            "model": "residual_backprop_bt",
            "representation": "hidden_512_no_projector",
            "dataset": args.dataset,
            "seed": seed,
            "depth": depth,
            "layer": idx + 1,
            "optimized_layer": depth,
            "actively_optimized": idx + 1 == depth,
            "checkpoint": str(model_path),
        }
        row.update(bt_hidden_metrics(view1, view2, args.bt_lambda))
        rows.append(row)
    del tensors, state, reps
    torch.cuda.empty_cache()
    gc.collect()
    return rows


def collect_nonres_backprop_bt_rows(args, depth, seed, device):
    point = point_for(args, depth, seed)
    tensors = load_tensors(point, device)
    model_path = find_nonres_bt_model(args.nonres_bt_model_dir, depth)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    reps = collect_barlow_representations(point, tensors, state, device)
    rows = []
    for idx, (view1, view2) in enumerate(zip(reps["pathnorm_view1_train"], reps["pathnorm_view2_train"])):
        row = {
            "model": "nonres_backprop_bt",
            "representation": "hidden_512_no_projector",
            "dataset": args.dataset,
            "seed": seed,
            "depth": depth,
            "layer": idx + 1,
            "optimized_layer": depth,
            "actively_optimized": idx + 1 == depth,
            "checkpoint": str(model_path),
        }
        row.update(bt_hidden_metrics(view1, view2, args.bt_lambda))
        rows.append(row)
    del tensors, state, reps
    torch.cuda.empty_cache()
    gc.collect()
    return rows


def collect_cf_bt_rows(args, depth, seed, device, device_name, variant, model_name):
    point = point_for(args, depth, seed)
    state = collect_variant_state(
        point,
        variant,
        residual_cf_args(args),
        device,
        device_name,
    )
    rows = []
    for idx, (view1, view2) in enumerate(zip(state["pathnorm_view1_train"], state["pathnorm_view2_train"])):
        row = {
            "model": model_name,
            "cf_variant": variant,
            "activation": state["activation"],
            "activation_alpha": state["activation_alpha"],
            "representation": "hidden_512_no_projector",
            "dataset": args.dataset,
            "seed": seed,
            "depth": depth,
            "layer": idx + 1,
            "optimized_layer": idx + 1,
            "actively_optimized": True,
            "checkpoint": "",
        }
        row.update(bt_hidden_metrics(view1, view2, args.bt_lambda))
        rows.append(row)
    del state
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


def rows_for(rows, model, depth):
    return sorted((row for row in rows if row["model"] == model and row["depth"] == depth), key=lambda row: row["layer"])


def plot_metric_grid(args, rows, out_dir):
    if PAPER_STYLE.exists():
        plt.style.use(str(PAPER_STYLE))
    metrics = [
        ("bt_total_per_dim", "BT total / dim"),
        ("bt_on_diag_per_dim", "on-diag error / dim"),
        ("bt_weighted_off_diag_per_dim", "weighted off-diag / dim"),
    ]
    fig, axes = plt.subplots(len(metrics), len(args.depths), figsize=(8.6, 5.6), sharex=False)
    if len(args.depths) == 1:
        axes = np.asarray(axes).reshape(len(metrics), 1)
    for col, depth in enumerate(args.depths):
        for row_idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col]
            for model in MODEL_ORDER:
                model_rows = rows_for(rows, model, depth)
                if not model_rows:
                    continue
                ax.plot(
                    [item["layer"] for item in model_rows],
                    [item[metric] for item in model_rows],
                    marker="o",
                    markersize=2.4,
                    linewidth=1.2,
                    linestyle=MODEL_LINESTYLES[model],
                    color=MODEL_COLORS[model],
                    label=model_label(args, model) if row_idx == 0 and col == 0 else None,
                )
            ax.set_yscale("log")
            ax.set_title(f"depth {depth}")
            ax.set_xlabel("layer")
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
            ax.set_xticks(sorted({1, depth} | set(range(0, depth + 1, max(1, depth // 4)))))
    fig.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Hidden-state Barlow Twins objective by layer", y=1.06)
    fig.tight_layout()
    pdf = out_dir / "bt_objective_by_layer.pdf"
    png = out_dir / "bt_objective_by_layer.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_component_tradeoff(args, rows, out_dir):
    if PAPER_STYLE.exists():
        plt.style.use(str(PAPER_STYLE))
    fig, axes = plt.subplots(1, len(args.depths), figsize=(9.2, 2.6), sharey=True)
    if len(args.depths) == 1:
        axes = [axes]
    for ax, depth in zip(axes, args.depths):
        for model in MODEL_ORDER:
            model_rows = rows_for(rows, model, depth)
            if not model_rows:
                continue
            ax.scatter(
                [item["bt_on_diag_per_dim"] for item in model_rows],
                [item["bt_weighted_off_diag_per_dim"] for item in model_rows],
                s=18 if "residual" in model else 16,
                marker="o" if "residual" in model else "s",
                color=MODEL_COLORS[model],
                label=model_label(args, model),
                alpha=0.9,
            )
            for item in model_rows:
                if item["layer"] in {1, depth}:
                    ax.annotate(str(item["layer"]), (item["bt_on_diag_per_dim"], item["bt_weighted_off_diag_per_dim"]), fontsize=6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(f"depth {depth}")
        ax.set_xlabel("on-diag / dim")
        ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("weighted off-diag / dim")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles[:4], labels_[:4], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    pdf = out_dir / "bt_objective_component_tradeoff.pdf"
    png = out_dir / "bt_objective_component_tradeoff.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def build_report(args, rows, out_dir, figure_paths):
    lines = [
        "# BT Objective By Layer",
        "",
        "Computed on hidden 512D paired-view activations without the backprop projector.",
        f"BT lambda for the weighted off-diagonal term: `{args.bt_lambda}`.",
        f"TF32 enabled: `{not args.no_tf32}`.",
        f"Included models: `{', '.join(args.models)}`.",
        f"Residual CF variant: `{args.residual_cf_variant}`.",
        f"Non-residual CF variant: `{args.nonres_cf_variant}`.",
        "",
        "| Depth | Model | Final total/dim | Min total/dim | Min layer | Final on/dim | Final weighted off/dim | Final diag mean | Final offdiag RMS |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for depth in args.depths:
        for model in MODEL_ORDER:
            model_rows = rows_for(rows, model, depth)
            if not model_rows:
                continue
            final = model_rows[-1]
            best = min(model_rows, key=lambda row: row["bt_total_per_dim"])
            lines.append(
                f"| {depth} | {model_label(args, model)} | {final['bt_total_per_dim']:.4g} | {best['bt_total_per_dim']:.4g} | "
                f"{best['layer']} | {final['bt_on_diag_per_dim']:.4g} | "
                f"{final['bt_weighted_off_diag_per_dim']:.4g} | {final['corr_diag_mean']:.3f} | "
                f"{final['offdiag_rms']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Main plot: `{figure_paths[0][0].name}` / `{figure_paths[0][1].name}`",
            f"- Component plot: `{figure_paths[1][0].name}` / `{figure_paths[1][1].name}`",
            "- Data: `bt_objective_by_layer.jsonl` and `bt_objective_by_layer.csv`",
            "",
        ]
    )
    report = "\n".join(lines)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    return report


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
    selected_models = set(args.models)
    for depth in args.depths:
        for seed in args.seeds:
            if "residual_backprop_bt" in selected_models:
                print(f"computing BT objective rows depth={depth} seed={seed} model=residual_backprop_bt", flush=True)
                rows.extend(collect_backprop_bt_rows(args, depth, seed, device))
                write_jsonl(args.out_dir / "bt_objective_by_layer.partial.jsonl", rows)
            if "nonres_backprop_bt" in selected_models:
                print(f"computing BT objective rows depth={depth} seed={seed} model=nonres_backprop_bt", flush=True)
                rows.extend(collect_nonres_backprop_bt_rows(args, depth, seed, device))
                write_jsonl(args.out_dir / "bt_objective_by_layer.partial.jsonl", rows)
            if "residual_cf_bt" in selected_models:
                print(f"computing BT objective rows depth={depth} seed={seed} model=residual_cf_bt device={device_name}", flush=True)
                rows.extend(collect_cf_bt_rows(args, depth, seed, device, device_name, args.residual_cf_variant, "residual_cf_bt"))
                write_jsonl(args.out_dir / "bt_objective_by_layer.partial.jsonl", rows)
            if "nonres_cf_bt" in selected_models:
                print(f"computing BT objective rows depth={depth} seed={seed} model=nonres_cf_bt device={device_name}", flush=True)
                rows.extend(collect_cf_bt_rows(args, depth, seed, device, device_name, args.nonres_cf_variant, "nonres_cf_bt"))
                write_jsonl(args.out_dir / "bt_objective_by_layer.partial.jsonl", rows)

    write_jsonl(args.out_dir / "bt_objective_by_layer.jsonl", rows)
    write_csv(args.out_dir / "bt_objective_by_layer.csv", rows)
    figure_paths = [
        plot_metric_grid(args, rows, args.out_dir),
        plot_component_tradeoff(args, rows, args.out_dir),
    ]
    print(build_report(args, rows, args.out_dir, figure_paths), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Plot Barlow Twins objective components by layer for residual/non-residual BP-BT and CF-BT.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_all_variants_seed7"))
    parser.add_argument("--bt-model-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_residual_barlow_seed7"))
    parser.add_argument("--nonres-bt-model-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_barlow_layer_only_cifar100_simclr_seed7"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.38)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=MODEL_ORDER)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--dataset", default="cifar100_simclr")
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depths", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--bt-lambda", type=float, default=0.005)
    parser.add_argument("--cf-residual-scale", type=float, default=1.0)
    parser.add_argument("--residual-cf-variant", default="residual_cf_branch_bpbt_nonlinearity")
    parser.add_argument("--nonres-cf-variant", default="plain_cf_bpbt_nonlinearity")
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
