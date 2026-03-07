import json
from pathlib import Path

import numpy as np

import spectral_gap_study as sgs


PAIR_NAME = "same-class-random-pair"
OUTPUT_PATH = Path("same_class_pair_results.json")


def build_same_class_view(X, y, seed):
    rng = np.random.default_rng(seed)
    paired = np.empty_like(X)
    classes = np.unique(y)

    for cls in classes:
        cls_idx = np.flatnonzero(y == cls)
        if cls_idx.size == 1:
            paired[cls_idx] = X[cls_idx]
            continue

        choices = rng.integers(0, cls_idx.size, size=cls_idx.size)
        same = cls_idx[choices] == cls_idx
        if np.any(same):
            choices[same] = (choices[same] + 1) % cls_idx.size
        paired[cls_idx] = X[cls_idx[choices]]

    return paired


def run_same_class_study():
    xtr, ytr, xte, yte = sgs.load_mnist_numpy()
    family_tr = [build_same_class_view(xtr, ytr, seed=sgs.SEED + 211)]
    stats = sgs.compute_pair_statistics(xtr, family_tr)

    rows = []
    for method_name in sgs.METHODS:
        model = sgs.fit_method(method_name, stats, sgs.SHALLOW_D)
        metrics = sgs.evaluate_projection(model, stats, xtr, ytr, xte, yte)
        row = {
            "pair_name": PAIR_NAME,
            "method": method_name,
            "d": sgs.SHALLOW_D,
            "delta_trace_ratio": stats["delta_trace_ratio"],
            "commutator_ratio": stats["commutator_ratio"],
            "supervised_positive_pairs": True,
        }
        row.update(metrics)
        if "floor" in model:
            row["floor"] = model["floor"]
        rows.append(row)

    rows.sort(key=lambda row: row["probe_accuracy"], reverse=True)
    payload = {
        "config": {
            "pair_name": PAIR_NAME,
            "seed": sgs.SEED,
            "n_train": sgs.N_TRAIN,
            "n_test": sgs.N_TEST,
            "d": sgs.SHALLOW_D,
            "methods": sgs.METHODS,
            "supervised_positive_pairs": True,
        },
        "rows": rows,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[pair={PAIR_NAME}, d={sgs.SHALLOW_D}]")
    print(
        f"delta_trace_ratio={stats['delta_trace_ratio']:.4f} | "
        f"commutator_ratio={stats['commutator_ratio']:.4f}"
    )
    for row in rows:
        print(
            f"{row['method']:>24} | acc={row['probe_accuracy']:.4f} | "
            f"retained_vs_pca={row['retained_vs_pca']:.3f} | "
            f"shared={row['shared_energy']:.4f} | delta={row['delta_energy']:.4f}"
        )
    print(f"Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_same_class_study()
