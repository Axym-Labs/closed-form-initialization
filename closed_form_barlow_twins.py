import argparse
import json
from inspect import signature
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from torchvision import datasets
from torchvision.transforms import ToTensor

from project_paths import resolve_json_path
import mnist_linear_augmentation_suites as aug_suites


SEED = 7
N_TRAIN = 12000
N_TEST = 3000
DEPTH = 3
FINAL_D = 32
PROBE_MAX_ITER = 2000
REG_EPS = 1e-4

LAMBDA_REG = 1.0


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
    mats = [np.eye(p, dtype=np.float64)] + aug_suites.build_augmentation_suite(
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
        "sigma_bar": 0.5 * (sigma_bar + sigma_bar.T),
        "shared": 0.5 * (shared + shared.T),
        "delta": 0.5 * (delta + delta.T),
    }


def sqrt_and_inv_sqrt_psd(matrix, reg_eps):
    evals, evecs = eigh(0.5 * (matrix + matrix.T))
    evals = np.maximum(evals, reg_eps)
    sqrt = (evecs * np.sqrt(evals)) @ evecs.T
    inv_sqrt = (evecs / np.sqrt(evals)) @ evecs.T
    return sqrt, inv_sqrt


def fit_whitened_cov_layer(stats, lambda_reg):
    if lambda_reg <= 0.0:
        raise ValueError("lambda must be strictly positive.")

    sigma_bar = stats["sigma_bar"]
    delta = stats["delta"]
    sigma_sqrt, sigma_inv_sqrt = sqrt_and_inv_sqrt_psd(sigma_bar, REG_EPS)
    m_matrix = sigma_inv_sqrt @ delta @ sigma_inv_sqrt
    m_matrix = 0.5 * (m_matrix + m_matrix.T)

    dim_r = sigma_bar.shape[0]
    eigvals_m, eigvecs_m = eigh(m_matrix)
    gains = lambda_reg / (np.maximum(eigvals_m, 0.0) + lambda_reg)
    g_matrix = (eigvecs_m * gains) @ eigvecs_m.T
    transform = sigma_sqrt @ g_matrix @ sigma_inv_sqrt
    return {
        "transform": transform,
        "g_matrix": g_matrix,
        "m_matrix": m_matrix,
        "gains": gains,
        "distance_to_whitened_identity": float(np.linalg.norm(g_matrix - np.eye(dim_r), ord="fro")),
        "max_m_eigenvalue": float(np.max(eigvals_m)),
        "min_m_eigenvalue": float(np.min(eigvals_m)),
    }


def fit_layer(H1, H2, lambda_reg):
    stats = compute_paired_stats(H1, H2)
    model = fit_whitened_cov_layer(
        stats,
        lambda_reg=lambda_reg,
    )
    whitened_delta = model["m_matrix"]
    return {
        "transform": model["transform"],
        "stats": stats,
        "transform_fro": float(np.linalg.norm(model["transform"], ord="fro")),
        "distance_to_identity": float(np.linalg.norm(model["transform"] - np.eye(model["transform"].shape[0]), ord="fro")),
        "distance_to_whitened_identity": model["distance_to_whitened_identity"],
        "total_delta_trace": float(np.trace(stats["delta"])),
        "min_whitened_delta": float(np.min(np.linalg.eigvalsh(whitened_delta))),
        "max_whitened_delta": model["max_m_eigenvalue"],
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


def run_experiment(suite_name, depth, final_dim, lambda_reg):
    xtr, ytr, xte, yte = load_mnist_numpy()
    base_tr = xtr.copy()
    base_te = xte.copy()
    view1_tr, view2_tr = sample_pair_views(xtr, suite_name, seed=SEED + 13)
    view1_te, view2_te = sample_pair_views(xte, suite_name, seed=SEED + 31)

    layers = []
    for layer_idx in range(depth):
        model = fit_layer(
            view1_tr,
            view2_tr,
            lambda_reg=lambda_reg,
        )
        transform = model["transform"]

        base_tr = apply_layer(base_tr, transform)
        base_te = apply_layer(base_te, transform)
        view1_tr = apply_layer(view1_tr, transform)
        view2_tr = apply_layer(view2_tr, transform)
        view1_te = apply_layer(view1_te, transform)
        view2_te = apply_layer(view2_te, transform)

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
                "transform_fro": model["transform_fro"],
                "distance_to_identity": model["distance_to_identity"],
                "distance_to_whitened_identity": model["distance_to_whitened_identity"],
                "total_delta_trace": model["total_delta_trace"],
                "post_delta_trace": float(np.trace(post_stats["delta"])),
                "post_shared_trace": float(np.trace(post_stats["shared"])),
                "base_trace": float(np.trace(covariance(base_tr))),
                "max_whitened_delta": model["max_whitened_delta"],
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
        "lambda": lambda_reg,
        "full_probe_accuracy": acc_full,
        "final_pca_probe_accuracy": acc_pca32,
        "raw_input_pca_probe_accuracy": acc_raw_pca32,
        "layers": layers,
        "note": (
            "Closed-form Barlow Twins layer with objective "
            "tr(G^T M G) + lambda ||G - I||_F^2, whose exact solution is "
            "G = lambda (M + lambda I)^(-1)."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Closed-form Barlow Twins layer on MNIST.")
    parser.add_argument(
        "--suite",
        default="single-translation",
        choices=["image", "translation", "single-translation", "random-masking", "block-masking", "blurring", "rotation", "random-crops"],
    )
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--final-d", type=int, default=FINAL_D)
    parser.add_argument("--lambda", dest="lambda_reg", type=float, default=LAMBDA_REG)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    if args.lambda_reg <= 0.0:
        raise ValueError("--lambda must be strictly positive.")

    result = run_experiment(
        suite_name=args.suite,
        depth=args.depth,
        final_dim=args.final_d,
        lambda_reg=args.lambda_reg,
    )

    print(
        f"Closed-form Barlow Twins  |  suite={result['suite']}  |  depth={result['depth']}  |  final_d={result['final_dim']}"
    )
    print(f"full probe accuracy      : {result['full_probe_accuracy']:.4f}")
    print(f"final PCA-{result['final_dim']} probe : {result['final_pca_probe_accuracy']:.4f}")
    print(f"raw input PCA-{result['final_dim']} probe : {result['raw_input_pca_probe_accuracy']:.4f}")
    for layer in result["layers"]:
        print(
            f"layer {layer['layer']:>2d} | ||T-I||_F={layer['distance_to_identity']:.3f} | "
            f"||G-I||_F={layer['distance_to_whitened_identity']:.3f} | "
            f"post_delta={layer['post_delta_trace']:.3f} | post_shared={layer['post_shared_trace']:.3f} | "
            f"max_M={layer['max_whitened_delta']:.3f}"
        )
    print(result["note"])

    if args.save_json is not None:
        output_path = resolve_json_path(args.save_json)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved json to {output_path}")


if __name__ == "__main__":
    main()
