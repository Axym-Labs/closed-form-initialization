import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARC = ROOT / "docs" / "cf_mlp_representation_learning"
OUT_DIR = ARC / "artifacts"

PAPER_STYLE = Path("/home/davwis/.codex/local-plugins/plugins/local-reusability-assets/plotting/paper.mplstyle")


BT_SOURCES = [
    {
        "key": "bp_greedy",
        "label": "BP-BT greedy residual",
        "color": "#000000",
        "linestyle": "-",
        "path": ARC / "artifacts_bp_moment_velocity_seed7" / "moment_velocity_rows.csv",
        "kind": "bp",
        "model": "greedy_residual_bpbt",
    },
    {
        "key": "bp_e2e",
        "label": "BP-BT e2e residual",
        "color": "#666666",
        "linestyle": "--",
        "path": ARC / "artifacts_bp_moment_velocity_seed7" / "moment_velocity_rows.csv",
        "kind": "bp",
        "model": "e2e_residual_bpbt",
    },
    {
        "key": "cf_oldspan",
        "label": "CF old-span adaptive",
        "color": "#0072B2",
        "linestyle": "-",
        "paths": [
            ARC / "artifacts_moment_ols_cifar100_oldspan_adapt095_ls_d6_b1024" / "mech_rows.csv",
            ARC / "artifacts_moment_ols_cifar100_oldspan_adapt095_ls_d12_b1024" / "mech_rows.csv",
            ARC / "artifacts_moment_ols_cifar100_oldspan_adapt095_ls_d24_b1024" / "mech_rows.csv",
        ],
        "kind": "cf",
    },
    {
        "key": "cf_gain_floor",
        "label": "CF gain-floor progress",
        "color": "#009E73",
        "linestyle": "-.",
        "paths": [
            ARC
            / "artifacts_moment_ols_cifar100_layernorm_random_postcf_adaptivefloor_prog05_ext_kinetic1_quad_k4_d12d24_b1024_ridge1e5"
            / "mech_rows.csv",
        ],
        "kind": "cf",
    },
    {
        "key": "cf_bal_identity",
        "label": "CF balanced identity",
        "color": "#CC79A7",
        "linestyle": "-",
        "paths": [
            ARC
            / "artifacts_moment_ols_cifar100_layernorm_balanced_identity_current_cap025_k4_d6d24_b1024_ridge1e5"
            / "mech_rows.csv",
            ARC
            / "artifacts_moment_ols_cifar100_layernorm_balanced_identity_current_cap025_k4_d12_b1024_ridge1e5"
            / "mech_rows.csv",
        ],
        "kind": "cf",
    },
    {
        "key": "cf_identity_metric",
        "label": "CF identity + diag metric",
        "color": "#D55E00",
        "linestyle": "-",
        "paths": [
            ARC
            / "artifacts_moment_ols_cifar100_layernorm_identity_diagmetric_current_lawdiag_d12_b1024_ridge1e5"
            / "mech_rows.csv",
        ],
        "kind": "cf",
    },
]

MECH_SOURCES = [
    {
        "key": "bp_greedy",
        "label": "BP-BT greedy",
        "color": "#000000",
        "linestyle": "-",
        "path": ARC / "artifacts_bp_moment_velocity_seed7" / "moment_velocity_rows.csv",
        "kind": "bp",
        "model": "greedy_residual_bpbt",
    },
    {
        "key": "bp_e2e",
        "label": "BP-BT e2e",
        "color": "#666666",
        "linestyle": "--",
        "path": ARC / "artifacts_bp_moment_velocity_seed7" / "moment_velocity_rows.csv",
        "kind": "bp",
        "model": "e2e_residual_bpbt",
    },
    {
        "key": "cf_bal_identity",
        "label": "CF balanced identity",
        "color": "#CC79A7",
        "linestyle": "-",
        "path": ARC / "artifacts_moment_ols_cifar100_layernorm_balanced_identity_current_lawdiag_d12_b1024_ridge1e5" / "mech_rows.csv",
        "kind": "cf",
    },
    {
        "key": "cf_identity_metric",
        "label": "CF identity + diag metric",
        "color": "#D55E00",
        "linestyle": "-",
        "path": ARC / "artifacts_moment_ols_cifar100_layernorm_identity_diagmetric_current_lawdiag_d12_b1024_ridge1e5" / "mech_rows.csv",
        "kind": "cf",
    },
]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_bt_rows():
    rows = []
    for source in BT_SOURCES:
        if source["kind"] == "bp":
            for row in read_csv(source["path"]):
                if row["model"] != source["model"]:
                    continue
                rows.append(
                    {
                        "key": source["key"],
                        "label": source["label"],
                        "color": source["color"],
                        "linestyle": source["linestyle"],
                        "depth": int(row["depth"]),
                        "layer": int(row["layer"]),
                        "train_bt": float(row["bt_after_per_dim"]),
                        "test_bt": np.nan,
                    }
                )
        else:
            for path in source["paths"]:
                for row in read_csv(path):
                    rows.append(
                        {
                            "key": source["key"],
                            "label": source["label"],
                            "color": source["color"],
                            "linestyle": source["linestyle"],
                            "depth": int(row["depth"]),
                            "layer": int(row["layer"]),
                            "train_bt": float(row["after_bt_total_per_dim"]),
                            "test_bt": float(row["after_test_bt_total_per_dim"]),
                        }
                    )
    return rows


def load_mechanism_rows():
    rows = []
    for source in MECH_SOURCES:
        for row in read_csv(source["path"]):
            if int(row["depth"]) != 12:
                continue
            if source["kind"] == "bp":
                if row["model"] != source["model"]:
                    continue
                rows.append(
                    {
                        "key": source["key"],
                        "label": source["label"],
                        "color": source["color"],
                        "linestyle": source["linestyle"],
                        "layer": int(row["layer"]),
                        "cos_identity": float(row["delta_cos_identity_minus_c"]),
                        "cos_polar": float(row["delta_cos_polar_minus_c"]),
                        "diag_frac": float(row["corr_delta_diag_norm_frac"]),
                        "delta_norm": float(row["corr_delta_norm"]),
                    }
                )
            else:
                rows.append(
                    {
                        "key": source["key"],
                        "label": source["label"],
                        "color": source["color"],
                        "linestyle": source["linestyle"],
                        "layer": int(row["layer"]),
                        "cos_identity": float(row["actual_delta_cos_identity_minus_c"]),
                        "cos_polar": float(row["actual_delta_cos_polar_minus_c"]),
                        "diag_frac": float(row["actual_delta_diag_norm_frac"]),
                        "delta_norm": float(row["actual_corr_delta_norm"]),
                    }
                )
    return rows


def grouped(rows, key_fields):
    out = defaultdict(list)
    for row in rows:
        out[tuple(row[field] for field in key_fields)].append(row)
    return out


def plot_bt_trajectories(rows):
    plt.style.use(str(PAPER_STYLE))
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=False, sharey="row")
    by_curve = grouped(rows, ["key", "depth"])
    source_by_key = {source["key"]: source for source in BT_SOURCES}
    for col, depth in enumerate([6, 12, 24]):
        ax_train = axes[0, col]
        ax_test = axes[1, col]
        for source in BT_SOURCES:
            key = (source["key"], depth)
            if key not in by_curve:
                continue
            curve = sorted(by_curve[key], key=lambda item: item["layer"])
            layers = [item["layer"] for item in curve]
            train = [item["train_bt"] for item in curve]
            ax_train.plot(
                layers,
                train,
                marker="o",
                markersize=2.4,
                linewidth=1.25,
                color=source["color"],
                linestyle=source["linestyle"],
                label=source["label"],
            )
            test = [item["test_bt"] for item in curve]
            if np.isfinite(test).any():
                ax_test.plot(
                    layers,
                    test,
                    marker="o",
                    markersize=2.4,
                    linewidth=1.25,
                    color=source["color"],
                    linestyle=source["linestyle"],
                    label=source["label"],
                )
        ax_train.set_title(f"depth {depth}")
        ax_test.set_xlabel("layer")
        for ax in [ax_train, ax_test]:
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
        if col == 0:
            ax_train.set_ylabel("train BT total / dim")
            ax_test.set_ylabel("held-out BT total / dim")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT_DIR / "strong_residual_bt_trajectory.png", dpi=220)
    fig.savefig(OUT_DIR / "strong_residual_bt_trajectory.pdf")
    plt.close(fig)


def plot_mechanism_trajectory(rows):
    plt.style.use(str(PAPER_STYLE))
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), sharex=True)
    metrics = [
        ("cos_identity", r"cos($\Delta C$, $I-C$)"),
        ("cos_polar", r"cos($\Delta C$, polar$(C)-C$)"),
        ("diag_frac", r"diag norm fraction"),
        ("delta_norm", r"$\|\Delta C\|_F$"),
    ]
    by_curve = grouped(rows, ["key"])
    source_by_key = {source["key"]: source for source in MECH_SOURCES}
    for ax, (metric, ylabel) in zip(axes.reshape(-1), metrics):
        for key, curve in by_curve.items():
            key = key[0]
            source = source_by_key[key]
            curve = sorted(curve, key=lambda item: item["layer"])
            ax.plot(
                [item["layer"] for item in curve],
                [item[metric] for item in curve],
                marker="o",
                markersize=2.4,
                linewidth=1.25,
                color=source["color"],
                linestyle=source["linestyle"],
                label=source["label"],
            )
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(ylabel)
        if metric == "delta_norm":
            ax.set_yscale("log")
    axes[1, 0].set_xlabel("layer")
    axes[1, 1].set_xlabel("layer")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.text(0.5, 0.925, "Depth-12 realized moment-update trajectory", ha="center", va="center", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(OUT_DIR / "strong_residual_moment_law_d12.png", dpi=220)
    fig.savefig(OUT_DIR / "strong_residual_moment_law_d12.pdf")
    plt.close(fig)


def summarize_bt(rows):
    summaries = []
    for (key, depth), curve in grouped(rows, ["key", "depth"]).items():
        curve = sorted(curve, key=lambda item: item["layer"])
        source = next(source for source in BT_SOURCES if source["key"] == key)
        summaries.append(
            {
                "label": source["label"],
                "depth": depth,
                "first_train_bt": curve[0]["train_bt"],
                "final_train_bt": curve[-1]["train_bt"],
                "best_train_bt": min(item["train_bt"] for item in curve),
                "best_train_layer": min(curve, key=lambda item: item["train_bt"])["layer"],
                "final_test_bt": curve[-1]["test_bt"],
                "improving_step_fraction": float(
                    np.mean(np.diff([item["train_bt"] for item in curve]) <= 1e-8)
                )
                if len(curve) > 1
                else np.nan,
            }
        )
    return sorted(summaries, key=lambda item: (item["depth"], item["label"]))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if isinstance(value, float) and np.isnan(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_report(summaries):
    lines = [
        "# Strong Residual CF-BT Trajectory Plots",
        "",
        "Plots compare the strongest saved residual CF-BT variants against residual BP-BT controls.",
        "",
        "Files:",
        "",
        "- `strong_residual_bt_trajectory.png/pdf`: old-style BT trajectory plot by depth.",
        "- `strong_residual_moment_law_d12.png/pdf`: depth-12 realized moment-law trajectory.",
        "- `strong_residual_bt_trajectory_summary.csv`: numeric summary used by the plot.",
        "",
        "| Setup | Depth | First train BT | Final train BT | Best train BT | Best layer | Final held-out BT | Step decrease frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    str(row["depth"]),
                    fmt(row["first_train_bt"]),
                    fmt(row["final_train_bt"]),
                    fmt(row["best_train_bt"]),
                    str(row["best_train_layer"]),
                    fmt(row["final_test_bt"]),
                    fmt(row["improving_step_fraction"]),
                ]
            )
            + " |"
        )
    (OUT_DIR / "strong_residual_trajectory_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bt_rows = load_bt_rows()
    mechanism_rows = load_mechanism_rows()
    plot_bt_trajectories(bt_rows)
    plot_mechanism_trajectory(mechanism_rows)
    summaries = summarize_bt(bt_rows)
    write_csv(OUT_DIR / "strong_residual_bt_trajectory_summary.csv", summaries)
    write_report(summaries)
    print((OUT_DIR / "strong_residual_trajectory_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
