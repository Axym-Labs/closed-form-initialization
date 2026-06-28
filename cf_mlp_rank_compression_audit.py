import csv
from pathlib import Path


OUT_DIR = Path("docs/cf_mlp_representation_learning/artifacts_rank_compression_audit_seed7")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row, key, default=float("nan")):
    value = row.get(key, "")
    if value == "" or value == "nan":
        return default
    return float(value)


def fmt(value):
    if isinstance(value, float):
        if value != value:
            return "n/a"
        return f"{value:.4g}"
    return str(value)


def route3_rows():
    summary = {
        (row["model"], int(row["depth"])): row
        for row in read_csv("docs/cf_mlp_representation_learning/artifacts_route3_residual_bt_seed7/summary.csv")
    }
    stages = read_csv("docs/cf_mlp_representation_learning/artifacts_route3_residual_bt_seed7/stage_rows.csv")
    rows = []
    for model in ["e2e_residual_bpbt", "greedy_residual_bpbt"]:
        final = next(row for row in stages if row["model"] == model and row["depth"] == "24" and row["layer"] == "24")
        summ = summary[(model, 24)]
        rows.append(
            {
                "family": "residual_bpbt",
                "model": model,
                "source": "route3_residual_bt_seed7",
                "depth": 24,
                "train_bt": as_float(summ, "final_output_bt_total_per_dim"),
                "test_bt": float("nan"),
                "corr_diag": as_float(summ, "final_output_corr_diag_mean"),
                "shared_diff": as_float(summ, "final_output_shared_diff_ratio"),
                "self_cov_offdiag": float("nan"),
                "stage_cov_rank": as_float(final, "output_effective_rank"),
                "stage_top10_var": as_float(final, "output_top10_var"),
                "readout_cov_rank": as_float(summ, "last_effective_rank"),
                "agreement_eff_rank": as_float(final, "agreement_agreement_delta_effective_rank"),
                "soft_keep_0p1": as_float(final, "agreement_agreement_soft_keep_sum_lam_0p1"),
                "all_pca": as_float(summ, "all_pca_accuracy"),
                "best_acc": as_float(summ, "best_layer_accuracy"),
                "best_layer": int(float(summ["best_layer"])),
            }
        )
    return rows


def projected_moment_rows():
    paths = [
        ("moment_ols_eta4", "docs/cf_mlp_representation_learning/artifacts_moment_ols_projected_full_eta4_cfshrink/summary.csv"),
        ("moment_ols_eta8", "docs/cf_mlp_representation_learning/artifacts_moment_ols_projected_full_eta8_cfshrink/summary.csv"),
        ("moment_ols_eta16_ls", "docs/cf_mlp_representation_learning/artifacts_moment_ols_projected_depth24_eta16_ls_cfshrink/summary.csv"),
        ("moment_ols_eta32_ls", "docs/cf_mlp_representation_learning/artifacts_moment_ols_projected_depth24_eta32_ls_cfshrink/summary.csv"),
        ("moment_ols_eta64_ls", "docs/cf_mlp_representation_learning/artifacts_moment_ols_projected_depth24_eta64_ls_cfshrink/summary.csv"),
        ("moment_ols_mb4096_eta32", "docs/cf_mlp_representation_learning/artifacts_moment_ols_minibatch_depth24_b4096/summary.csv"),
        ("moment_ols_mb1024_eta32", "docs/cf_mlp_representation_learning/artifacts_moment_ols_minibatch_depth24_b1024/summary.csv"),
        ("moment_ols_mb512_eta32", "docs/cf_mlp_representation_learning/artifacts_moment_ols_minibatch_depth24_b512/summary.csv"),
    ]
    rows = []
    for model, path in paths:
        for row in read_csv(path):
            if row["depth"] != "24":
                continue
            rows.append(
                {
                    "family": "cf_projected_moment_ols",
                    "model": model,
                    "source": Path(path).parent.name,
                    "depth": 24,
                    "train_bt": as_float(row, "final_train_bt_total_per_dim"),
                    "test_bt": as_float(row, "final_test_bt_total_per_dim"),
                    "corr_diag": as_float(row, "final_corr_diag_mean"),
                    "shared_diff": as_float(row, "final_shared_diff_ratio"),
                    "self_cov_offdiag": as_float(row, "final_self_corr_offdiag_per_dim"),
                    "stage_cov_rank": as_float(row, "final_effective_rank"),
                    "stage_top10_var": float("nan"),
                    "readout_cov_rank": as_float(row, "final_effective_rank"),
                    "agreement_eff_rank": float("nan"),
                    "soft_keep_0p1": float("nan"),
                    "all_pca": as_float(row, "all_pca_accuracy"),
                    "best_acc": as_float(row, "best_layer_accuracy"),
                    "best_layer": int(float(row["best_layer"])),
                }
            )
    return rows


def spectral_rows():
    paths = [
        "docs/cf_mlp_representation_learning/artifacts_bpbt_spectral_diagnostic_seed7/summary.csv",
        "docs/cf_mlp_representation_learning/artifacts_bpbt_spectral_diagnostic_nonres_seed7/summary.csv",
    ]
    rows = []
    for path in paths:
        for row in read_csv(path):
            if row["depth"] != "24":
                continue
            rows.append(
                {
                    "family": "spectral_existing",
                    "model": row["model"] + ":" + row["variant"],
                    "source": Path(path).parent.name,
                    "depth": 24,
                    "train_bt": as_float(row, "final_bt_per_dim"),
                    "test_bt": float("nan"),
                    "corr_diag": as_float(row, "final_corr_diag"),
                    "shared_diff": as_float(row, "final_shared_diff_ratio"),
                    "self_cov_offdiag": float("nan"),
                    "stage_cov_rank": as_float(row, "final_effective_rank"),
                    "stage_top10_var": float("nan"),
                    "readout_cov_rank": as_float(row, "final_effective_rank"),
                    "agreement_eff_rank": float("nan"),
                    "soft_keep_0p1": as_float(row, "final_soft_keep_lam_0p1"),
                    "all_pca": float("nan"),
                    "best_acc": float("nan"),
                    "best_layer": int(float(row["best_bt_layer"])),
                }
            )
    return rows


def write_csv(path, rows):
    keys = [
        "family",
        "model",
        "source",
        "depth",
        "train_bt",
        "test_bt",
        "corr_diag",
        "shared_diff",
        "self_cov_offdiag",
        "stage_cov_rank",
        "stage_top10_var",
        "readout_cov_rank",
        "agreement_eff_rank",
        "soft_keep_0p1",
        "all_pca",
        "best_acc",
        "best_layer",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, rows):
    lines = [
        "# Rank And Compression Audit",
        "",
        "Depth-24 CIFAR100 SimCLR-positive comparison. Ranks are not interchangeable:",
        "`stage_cov_rank` is hidden self-covariance effective rank, `agreement_eff_rank` is paired-difference spectral rank, and `soft_keep_0p1` is the CF agreement soft-keep mass.",
        "",
        "| Family | Model | BT | Test BT | Corr | Shared/diff | Self-cov off | Cov rank | Top10 var | Agreement rank | Soft keep 0.1 | All PCA | Best layer |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["family"],
                    row["model"],
                    fmt(row["train_bt"]),
                    fmt(row["test_bt"]),
                    fmt(row["corr_diag"]),
                    fmt(row["shared_diff"]),
                    fmt(row["self_cov_offdiag"]),
                    fmt(row["stage_cov_rank"]),
                    fmt(row["stage_top10_var"]),
                    fmt(row["agreement_eff_rank"]),
                    fmt(row["soft_keep_0p1"]),
                    fmt(row["all_pca"]),
                    f"{fmt(row['best_acc'])} @ {row['best_layer']}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Greedy residual BP-BT does not share the corrected CF moment-OLS low-rank failure: it improves BT more strongly while maintaining much broader covariance/agreement spectra.",
            "- Non-residual BP-BT at depth 24 does collapse, but it also fails the BT objective; this points to architecture/optimization mismatch rather than a generic SGD property.",
            "- Corrected full-dataset CF moment-OLS has a real monotone BT trajectory, but its self-covariance rank stays near 9-12 at useful strengths, while greedy residual BP-BT has much broader spectra.",
            "- Minibatch moment estimation partially fixes this: batch 1024 improves BT, lowers self-covariance off-mass, raises covariance rank, and improves PCA/readout quality relative to full-dataset moment OLS.",
            "- Naive self-covariance target terms were less promising in calibration: they raised rank only by suppressing BT descent. The useful mechanism so far is stochastic mode assembly, not direct covariance penalty.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = route3_rows() + projected_moment_rows() + spectral_rows()
    rows = sorted(rows, key=lambda row: (row["family"], row["model"]))
    write_csv(OUT_DIR / "rank_compression_audit.csv", rows)
    write_report(OUT_DIR / "report.md", rows)
    print((OUT_DIR / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
