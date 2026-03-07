import argparse
import json
from inspect import signature
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from torchvision import datasets
from torchvision.transforms import ToTensor

import test2


SEED = 7
N_TRAIN = 12000
N_TEST = 3000
DEPTH = 3
FINAL_D = 32
PROBE_MAX_ITER = 2000
REG_EPS = 1e-4

INV_WEIGHT = 1.0
ORTH_WEIGHT = 1.0
IDENTITY_WEIGHT = 1.0


def covariance(X):
    return (X.T @ X) / X.shape[0]


def cross_covariance(X, Y):
    return (X.T @ Y) / X.shape[0]


def load_mnist_numpy():
    rng = np.random.default_rng(SEED)
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=ToTensor())
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=ToTensor())

    xtr = train_ds.data.numpy().reshape(-1, 28 * 28).astype(np.float64) / 255.0
    ytr = train_ds.targets.numpy()
    xte = test_ds.data.numpy().reshape(-1, 28 * 28).astype(np.float64) / 255.0
    yte = test_ds.targets.numpy()

    idx_tr = rng.choice(len(xtr), size=N_TRAIN, replace=False)
    idx_te = rng.choice(len(xte), size=N_TEST, replace=False)
    xtr, ytr = xtr[idx_tr], ytr[idx_tr]
    xte, yte = xte[idx_te], yte[idx_te]

    mu = xtr.mean(axis=0, keepdims=True)
    xtr = xtr - mu
    xte = xte - mu
    return xtr, ytr, xte, yte


def standardize_train_test(ztr, zte):
    mu = ztr.mean(axis=0, keepdims=True)
    std = ztr.std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    return (ztr - mu) / std, (zte - mu) / std


def fit_linear_probe(ztr, ytr, zte, yte):
    kwargs = {
        "max_iter": PROBE_MAX_ITER,
        "solver": "lbfgs",
        "n_jobs": None,
    }
    if "multi_class" in signature(LogisticRegression).parameters:
        kwargs["multi_class"] = "multinomial"
    clf = LogisticRegression(**kwargs)
    clf.fit(ztr, ytr)
    return float((clf.predict(zte) == yte).mean())


def sample_pair_views(X, suite_name, seed):
    rng = np.random.default_rng(seed)
    p = X.shape[1]
    mats = [np.eye(p, dtype=np.float64)] + test2.build_augmentation_suite(
        suite_name, h=28, w=28, rng=np.random.default_rng(seed)
    )
    idx1 = rng.integers(len(mats), size=X.shape[0])
    idx2 = rng.integers(len(mats), size=X.shape[0])

    x1 = np.empty_like(X)
    x2 = np.empty_like(X)
    for mat_idx, A in enumerate(mats):
        mask1 = idx1 == mat_idx
        mask2 = idx2 == mat_idx
        if np.any(mask1):
            x1[mask1] = X[mask1] @ A.T
        if np.any(mask2):
            x2[mask2] = X[mask2] @ A.T
    return x1, x2


def center_pair(H1, H2):
    mu = 0.5 * (H1.mean(axis=0, keepdims=True) + H2.mean(axis=0, keepdims=True))
    return H1 - mu, H2 - mu


def compute_paired_stats(H1, H2):
    H1c, H2c = center_pair(H1, H2)
    sigma1 = covariance(H1c)
    sigma2 = covariance(H2c)
    sigma_bar = 0.5 * (sigma1 + sigma2)
    shared = 0.5 * (cross_covariance(H1c, H2c) + cross_covariance(H2c, H1c))
    delta = covariance(H1c - H2c)
    return {
        "sigma1": sigma1,
        "sigma2": sigma2,
        "sigma_bar": sigma_bar,
        "shared": shared,
        "delta": delta,
    }


def sqrt_and_inv_sqrt_psd(matrix, reg_eps):
    evals, evecs = eigh(0.5 * (matrix + matrix.T))
    evals = np.maximum(evals, reg_eps)
    sqrt = (evecs * np.sqrt(evals)) @ evecs.T
    inv_sqrt = (evecs / np.sqrt(evals)) @ evecs.T
    return sqrt, inv_sqrt, evals


def scalar_objective(scale, lam, inv_weight, orth_weight, identity_weight):
    return (
        inv_weight * lam * scale * scale
        + orth_weight * (scale * scale - 1.0) * (scale * scale - 1.0)
        + identity_weight * (scale - 1.0) * (scale - 1.0)
    )


def solve_scalar_scale(lam, inv_weight, orth_weight, identity_weight):
    coeffs = [
        4.0 * orth_weight,
        0.0,
        2.0 * (inv_weight * lam - 2.0 * orth_weight + identity_weight),
        -2.0 * identity_weight,
    ]
    roots = np.roots(coeffs)
    candidates = [0.0, 1.0]
    for root in roots:
        if abs(root.imag) < 1e-8 and root.real >= 0.0:
            candidates.append(float(root.real))

    best_scale = 1.0
    best_value = scalar_objective(best_scale, lam, inv_weight, orth_weight, identity_weight)
    for candidate in candidates:
        value = scalar_objective(candidate, lam, inv_weight, orth_weight, identity_weight)
        if value < best_value:
            best_scale = candidate
            best_value = value
    return best_scale


def fit_fullwidth_layer(H1, H2, inv_weight, orth_weight, identity_weight):
    stats = compute_paired_stats(H1, H2)
    sigma_bar = 0.5 * (stats["sigma_bar"] + stats["sigma_bar"].T)
    delta = 0.5 * (stats["delta"] + stats["delta"].T)

    sigma_sqrt, sigma_inv_sqrt, sigma_evals = sqrt_and_inv_sqrt_psd(sigma_bar, REG_EPS)
    M = sigma_inv_sqrt @ delta @ sigma_inv_sqrt
    M = 0.5 * (M + M.T)
    evals_M, evecs_M = eigh(M)
    order = np.argsort(evals_M)
    evals_M = np.maximum(evals_M[order], 0.0)
    basis = evecs_M[:, order]

    scales = np.array(
        [
            solve_scalar_scale(lam, inv_weight, orth_weight, identity_weight)
            for lam in evals_M
        ],
        dtype=np.float64,
    )

    shaper = (basis * scales) @ basis.T
    row_transform = sigma_inv_sqrt @ shaper @ sigma_sqrt
    shaped_cov = shaper @ shaper.T
    return {
        "transform": row_transform,
        "basis": basis,
        "scales": scales,
        "sigma_evals": sigma_evals,
        "whitened_evals": evals_M,
        "stats": stats,
        "orth_penalty": float(np.linalg.norm(shaped_cov - np.eye(shaped_cov.shape[0]), ord="fro") ** 2),
    }


def apply_layer(X, transform):
    return np.maximum(X @ transform, 0.0)


def normalize_hidden(train_arrays, test_arrays):
    mean = sum(arr.mean(axis=0, keepdims=True) for arr in train_arrays) / len(train_arrays)
    centered_train = [arr - mean for arr in train_arrays]
    centered_test = [arr - mean for arr in test_arrays]

    avg_var = sum(np.mean(arr * arr, axis=0, keepdims=True) for arr in centered_train) / len(centered_train)
    scale = np.sqrt(np.maximum(avg_var, 1e-6))
    scaled_train = [arr / scale for arr in centered_train]
    scaled_test = [arr / scale for arr in centered_test]
    return scaled_train, scaled_test


def top_pca_projection(Xtr, d):
    sigma = covariance(Xtr - Xtr.mean(axis=0, keepdims=True))
    evals, evecs = eigh(sigma)
    return evecs[:, -d:]


def run_experiment(suite_name, depth, final_dim, inv_weight, orth_weight, identity_weight):
    xtr, ytr, xte, yte = load_mnist_numpy()
    base_tr = xtr.copy()
    base_te = xte.copy()
    view1_tr, view2_tr = sample_pair_views(xtr, suite_name, seed=SEED + 13)
    view1_te, view2_te = sample_pair_views(xte, suite_name, seed=SEED + 31)

    layers = []
    for layer_idx in range(depth):
        model = fit_fullwidth_layer(
            view1_tr,
            view2_tr,
            inv_weight=inv_weight,
            orth_weight=orth_weight,
            identity_weight=identity_weight,
        )

        base_tr = apply_layer(base_tr, model["transform"])
        base_te = apply_layer(base_te, model["transform"])
        view1_tr = apply_layer(view1_tr, model["transform"])
        view2_tr = apply_layer(view2_tr, model["transform"])
        view1_te = apply_layer(view1_te, model["transform"])
        view2_te = apply_layer(view2_te, model["transform"])

        train_arrays, test_arrays = normalize_hidden(
            [base_tr, view1_tr, view2_tr],
            [base_te, view1_te, view2_te],
        )
        base_tr, view1_tr, view2_tr = train_arrays
        base_te, view1_te, view2_te = test_arrays

        post_stats = compute_paired_stats(view1_tr, view2_tr)
        layers.append(
            {
                "layer": layer_idx + 1,
                "mean_scale": float(model["scales"].mean()),
                "min_scale": float(model["scales"].min()),
                "max_scale": float(model["scales"].max()),
                "orth_penalty": model["orth_penalty"],
                "delta_trace": float(np.trace(post_stats["delta"])),
                "shared_trace": float(np.trace(post_stats["shared"])),
                "base_trace": float(np.trace(covariance(base_tr))),
            }
        )

    ztr_full, zte_full = standardize_train_test(base_tr, base_te)
    acc_full = fit_linear_probe(ztr_full, ytr, zte_full, yte)

    pca_basis = top_pca_projection(base_tr, final_dim)
    ztr_pca = base_tr @ pca_basis
    zte_pca = base_te @ pca_basis
    ztr_pca, zte_pca = standardize_train_test(ztr_pca, zte_pca)
    acc_pca32 = fit_linear_probe(ztr_pca, ytr, zte_pca, yte)

    raw_basis = top_pca_projection(xtr, final_dim)
    raw_tr = xtr @ raw_basis
    raw_te = xte @ raw_basis
    raw_tr, raw_te = standardize_train_test(raw_tr, raw_te)
    acc_raw_pca32 = fit_linear_probe(raw_tr, ytr, raw_te, yte)

    return {
        "suite": suite_name,
        "depth": depth,
        "final_dim": final_dim,
        "inv_weight": inv_weight,
        "orth_weight": orth_weight,
        "identity_weight": identity_weight,
        "full_probe_accuracy": acc_full,
        "final_pca_probe_accuracy": acc_pca32,
        "raw_input_pca_probe_accuracy": acc_raw_pca32,
        "layers": layers,
        "note": (
            "Closed-form full-width pairwise layer: whiten current paired covariance, "
            "diagonalize the whitened disagreement matrix, solve one cubic per mode, "
            "apply a square operator, and compress only at the end."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Greedy full-width closed-form pairwise layer on MNIST.")
    parser.add_argument(
        "--suite",
        default="single-translation",
        choices=["image", "translation", "single-translation", "random-masking", "block-masking", "blurring"],
    )
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--final-d", type=int, default=FINAL_D)
    parser.add_argument("--inv-weight", type=float, default=INV_WEIGHT)
    parser.add_argument("--orth-weight", type=float, default=ORTH_WEIGHT)
    parser.add_argument("--identity-weight", type=float, default=IDENTITY_WEIGHT)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    result = run_experiment(
        suite_name=args.suite,
        depth=args.depth,
        final_dim=args.final_d,
        inv_weight=args.inv_weight,
        orth_weight=args.orth_weight,
        identity_weight=args.identity_weight,
    )

    print(
        f"Greedy full-width closed-form layer  |  suite={result['suite']}  |  depth={result['depth']}  |  final_d={result['final_dim']}"
    )
    print(f"full probe accuracy      : {result['full_probe_accuracy']:.4f}")
    print(f"final PCA-{result['final_dim']} probe : {result['final_pca_probe_accuracy']:.4f}")
    print(f"raw input PCA-{result['final_dim']} probe : {result['raw_input_pca_probe_accuracy']:.4f}")
    for layer in result["layers"]:
        print(
            f"layer {layer['layer']:>2d} | scale(mean/min/max)=({layer['mean_scale']:.3f}, "
            f"{layer['min_scale']:.3f}, {layer['max_scale']:.3f}) | "
            f"orth_penalty={layer['orth_penalty']:.3f} | "
            f"delta_trace={layer['delta_trace']:.3f} | shared_trace={layer['shared_trace']:.3f}"
        )
    print(result["note"])

    if args.save_json is not None:
        args.save_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved json to {args.save_json}")


if __name__ == "__main__":
    main()
