import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SOURCES = [
    (
        "bp_and_cf_leaky_gelu",
        Path("docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_all_variants_seed7/bt_objective_by_layer.csv"),
        {"residual_backprop_bt", "nonres_backprop_bt", "residual_cf_bt", "nonres_cf_bt"},
    ),
    (
        "cf_relu",
        Path("docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_relu_seed7/bt_objective_by_layer.csv"),
        {"residual_cf_bt", "nonres_cf_bt"},
    ),
    (
        "cf_postrelu_affineopt",
        Path(
            "docs/cf_mlp_representation_learning/"
            "artifacts_bt_objective_by_layer_postrelu_affine_s2048_steps80_seed7/"
            "bt_objective_by_layer.csv"
        ),
        {"residual_cf_bt", "nonres_cf_bt"},
    ),
    (
        "cf_postrelu_biasopt",
        Path(
            "docs/cf_mlp_representation_learning/"
            "artifacts_bt_objective_by_layer_postrelu_biasopt_seed7/"
            "bt_objective_by_layer.csv"
        ),
        {"residual_cf_bt", "nonres_cf_bt"},
    ),
    (
        "cf_agreement_linearopt",
        Path(
            "docs/cf_mlp_representation_learning/"
            "artifacts_bt_objective_by_layer_agreement_linearopt_seed7/"
            "bt_objective_by_layer.csv"
        ),
        {"nonres_cf_bt"},
    ),
    (
        "cf_agreement_shared_cca",
        Path(
            "docs/cf_mlp_representation_learning/"
            "artifacts_bt_objective_by_layer_agreement_ccalinear_seed7/"
            "bt_objective_by_layer.csv"
        ),
        {"nonres_cf_bt"},
    ),
    (
        "cf_agreement_active_rank",
        Path(
            "docs/cf_mlp_representation_learning/"
            "artifacts_bt_objective_by_layer_activerank_lo005_hi055_seed7/"
            "bt_objective_by_layer.csv"
        ),
        {"nonres_cf_bt"},
    ),
    (
        "cf_postrelu_active_ramp",
        Path(
            "docs/cf_mlp_representation_learning/"
            "artifacts_bt_objective_by_layer_active_ramp_lo025_hi08_seed7/"
            "bt_objective_by_layer.csv"
        ),
        {"residual_cf_bt", "nonres_cf_bt"},
    ),
    (
        "cf_agreement_gate_b4",
        Path("docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_agreement_gate_b4_seed7/bt_objective_by_layer.csv"),
        {"residual_cf_bt", "nonres_cf_bt"},
    ),
]


def load_rows():
    rows = []
    for source_label, path, models in SOURCES:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["model"] not in models:
                    continue
                out = dict(row)
                out["source_label"] = source_label
                for key in [
                    "depth",
                    "layer",
                    "bt_total_per_dim",
                    "bt_on_diag_per_dim",
                    "bt_weighted_off_diag_per_dim",
                    "corr_diag_mean",
                    "offdiag_rms",
                ]:
                    out[key] = float(out[key])
                out["depth"] = int(out["depth"])
                out["layer"] = int(out["layer"])
                rows.append(out)
    return rows


def summarize_group(rows, eps=1e-4):
    ordered = sorted(rows, key=lambda row: row["layer"])
    total = np.asarray([row["bt_total_per_dim"] for row in ordered], dtype=float)
    on = np.asarray([row["bt_on_diag_per_dim"] for row in ordered], dtype=float)
    off = np.asarray([row["bt_weighted_off_diag_per_dim"] for row in ordered], dtype=float)
    corr = np.asarray([row["corr_diag_mean"] for row in ordered], dtype=float)
    off_rms = np.asarray([row["offdiag_rms"] for row in ordered], dtype=float)
    layers = np.asarray([row["layer"] for row in ordered], dtype=float)
    diffs = np.diff(total)
    best_idx = int(np.argmin(total))
    if len(total) > 1:
        slope = float(np.polyfit(layers, total, deg=1)[0])
        decrease_frac = float(np.mean(diffs <= eps))
    else:
        slope = float("nan")
        decrease_frac = float("nan")
    first = float(total[0])
    last = float(total[-1])
    improvement = first - last
    return {
        "source_label": ordered[0]["source_label"],
        "model": ordered[0]["model"],
        "depth": int(ordered[0]["depth"]),
        "first_total": first,
        "last_total": last,
        "best_total": float(total[best_idx]),
        "best_layer": int(ordered[best_idx]["layer"]),
        "improvement_abs": improvement,
        "improvement_frac": improvement / max(first, 1e-12),
        "monotone_decrease_frac": decrease_frac,
        "linear_slope_per_layer": slope,
        "on_improvement_abs": float(on[0] - on[-1]),
        "off_improvement_abs": float(off[0] - off[-1]),
        "corr_diag_gain": float(corr[-1] - corr[0]),
        "offdiag_rms_reduction": float(off_rms[0] - off_rms[-1]),
        "first_on": float(on[0]),
        "last_on": float(on[-1]),
        "first_off": float(off[0]),
        "last_off": float(off[-1]),
        "first_corr_diag_mean": float(corr[0]),
        "last_corr_diag_mean": float(corr[-1]),
    }


def write_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_report(path, summaries):
    lines = [
        "# BT Trajectory Diagnostics",
        "",
        "Primary metric is layerwise trajectory shape, not final readout. Positive improvement means BT total/dim decreases from layer 1 to the final layer.",
        "",
        "| Source | Model | Depth | First | Last | Best | Best layer | Improvement | Step decrease frac | Slope/layer | On improvement | Off improvement | Corr diag gain |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summaries, key=lambda r: (r["model"], r["depth"], r["source_label"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["source_label"],
                    row["model"],
                    str(row["depth"]),
                    fmt(row["first_total"]),
                    fmt(row["last_total"]),
                    fmt(row["best_total"]),
                    str(row["best_layer"]),
                    fmt(row["improvement_frac"]),
                    fmt(row["monotone_decrease_frac"]),
                    fmt(row["linear_slope_per_layer"]),
                    fmt(row["on_improvement_abs"]),
                    fmt(row["off_improvement_abs"]),
                    fmt(row["corr_diag_gain"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Files: `trajectory_summary.csv`, `trajectory_total.png`, `trajectory_components.png`.")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_total(rows, out_dir):
    plt.style.use("/home/davwis/.codex/local-plugins/plugins/local-reusability-assets/plotting/paper.mplstyle")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharey="row")
    models = ["residual_backprop_bt", "residual_cf_bt", "nonres_backprop_bt", "nonres_cf_bt"]
    colors = {
        "bp_and_cf_leaky_gelu": "#0072B2",
        "cf_relu": "#D55E00",
        "cf_postrelu_affineopt": "#009E73",
        "cf_postrelu_biasopt": "#56B4E9",
        "cf_agreement_linearopt": "#CC79A7",
        "cf_agreement_shared_cca": "#E69F00",
        "cf_agreement_active_rank": "#999999",
        "cf_postrelu_active_ramp": "#CC79A7",
        "cf_agreement_gate_b4": "#999999",
    }
    styles = {
        "residual_backprop_bt": "-",
        "nonres_backprop_bt": "--",
        "residual_cf_bt": "-",
        "nonres_cf_bt": "--",
    }
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source_label"], row["model"], row["depth"])].append(row)
    for col, depth in enumerate([6, 12, 24]):
        for row_idx, model_set in enumerate([models[:2], models[2:]]):
            ax = axes[row_idx, col]
            for model in model_set:
                for source in [item[0] for item in SOURCES]:
                    key = (source, model, depth)
                    if key not in grouped:
                        continue
                    curve = sorted(grouped[key], key=lambda rec: rec["layer"])
                    if model.endswith("backprop_bt") and source != "bp_and_cf_leaky_gelu":
                        continue
                    label = f"{source.replace('bp_and_cf_', '')} / {model.replace('_bt', '')}"
                    ax.plot(
                        [rec["layer"] for rec in curve],
                        [rec["bt_total_per_dim"] for rec in curve],
                        marker="o",
                        markersize=2.4,
                        linewidth=1.2,
                        color=colors[source],
                        linestyle=styles[model],
                        label=label,
                    )
            ax.set_title(f"depth {depth}")
            ax.set_xlabel("layer")
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel("BT total / dim")
            ax.set_yscale("log")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out_dir / "trajectory_total.png", dpi=200)
    fig.savefig(out_dir / "trajectory_total.pdf")
    plt.close(fig)


def plot_components(rows, out_dir):
    plt.style.use("/home/davwis/.codex/local-plugins/plugins/local-reusability-assets/plotting/paper.mplstyle")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharey=False)
    selected_sources = [
        "bp_and_cf_leaky_gelu",
        "cf_relu",
        "cf_postrelu_affineopt",
        "cf_postrelu_biasopt",
        "cf_agreement_linearopt",
        "cf_agreement_shared_cca",
        "cf_agreement_active_rank",
        "cf_postrelu_active_ramp",
    ]
    colors = {
        "bp_and_cf_leaky_gelu": "#0072B2",
        "cf_relu": "#D55E00",
        "cf_postrelu_affineopt": "#009E73",
        "cf_postrelu_biasopt": "#56B4E9",
        "cf_agreement_linearopt": "#CC79A7",
        "cf_agreement_shared_cca": "#E69F00",
        "cf_agreement_active_rank": "#999999",
        "cf_postrelu_active_ramp": "#CC79A7",
    }
    grouped = defaultdict(list)
    for row in rows:
        if row["model"] != "nonres_cf_bt":
            continue
        grouped[(row["source_label"], row["depth"])].append(row)
    for col, depth in enumerate([6, 12, 24]):
        for source in selected_sources:
            key = (source, depth)
            if key not in grouped:
                continue
            curve = sorted(grouped[key], key=lambda rec: rec["layer"])
            label = source.replace("bp_and_cf_", "")
            axes[0, col].plot(
                [rec["layer"] for rec in curve],
                [rec["bt_on_diag_per_dim"] for rec in curve],
                marker="o",
                markersize=2.4,
                linewidth=1.2,
                color=colors[source],
                label=label,
            )
            axes[1, col].plot(
                [rec["layer"] for rec in curve],
                [rec["bt_weighted_off_diag_per_dim"] for rec in curve],
                marker="o",
                markersize=2.4,
                linewidth=1.2,
                color=colors[source],
                label=label,
            )
        axes[0, col].set_title(f"depth {depth}")
        axes[1, col].set_xlabel("layer")
        axes[0, col].grid(True, alpha=0.3)
        axes[1, col].grid(True, alpha=0.3)
        axes[0, col].set_yscale("log")
        axes[1, col].set_yscale("log")
    axes[0, 0].set_ylabel("on-diag error / dim")
    axes[1, 0].set_ylabel("weighted off-diag / dim")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out_dir / "trajectory_components.png", dpi=200)
    fig.savefig(out_dir / "trajectory_components.pdf")
    plt.close(fig)


def run(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source_label"], row["model"], row["depth"])].append(row)
    summaries = [summarize_group(group) for group in grouped.values()]
    summaries = sorted(summaries, key=lambda row: (row["model"], row["depth"], row["source_label"]))
    write_csv(args.out_dir / "trajectory_summary.csv", summaries)
    write_report(args.out_dir / "report.md", summaries)
    plot_total(rows, args.out_dir)
    plot_components(rows, args.out_dir)
    print((args.out_dir / "report.md").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Layerwise BT trajectory diagnostics for CF-BT variants.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/cf_mlp_representation_learning/artifacts_bt_trajectory_diagnostics_seed7"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
