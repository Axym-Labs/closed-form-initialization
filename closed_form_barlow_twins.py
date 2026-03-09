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


SEED = 7
N_TRAIN = 12000
N_TEST = 3000
DEPTH = 3
FINAL_D = 32
PROBE_MAX_ITER = 2000
REG_EPS = 1e-4

LAMBDA_REG = 1.0
LEAKY_RELU_SLOPE = 0.1
LAYER_METHODS = [
    "closed-form-barlow",
    "iterref-old",
    "iterref-symcca",
    "residual-barlow",
    "whitened-shared-pca",
    "paper-cca",
    "paper-cca-shared",
]


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
    import mnist_linear_augmentation_suites as aug_suites

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
    cross12 = cross_covariance(H1c, H2c)
    cross21 = cross_covariance(H2c, H1c)
    shared = 0.5 * (cross12 + cross21)
    delta = covariance(H1c - H2c)
    return {
        "sigma1": 0.5 * (sigma1 + sigma1.T),
        "sigma2": 0.5 * (sigma2 + sigma2.T),
        "sigma_bar": 0.5 * (sigma_bar + sigma_bar.T),
        "cross12": cross12,
        "cross21": cross21,
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


def fit_iterref_old_layer(stats, lambda_reg):
    if lambda_reg <= 0.0:
        raise ValueError("lambda must be strictly positive.")

    sigma_bar = stats["sigma_bar"]
    delta = stats["delta"]
    sigma_sqrt, sigma_inv_sqrt = sqrt_and_inv_sqrt_psd(sigma_bar, REG_EPS)
    m_matrix = sigma_inv_sqrt @ delta @ sigma_inv_sqrt
    m_matrix = 0.5 * (m_matrix + m_matrix.T)

    eigvals_m, eigvecs_m = eigh(m_matrix)

    # First-order residual refinement of the old disagreement objective:
    # choose R to reduce ||M + RM + MR||_F^2 + lambda ||R||_F^2.
    residual_steps = -(2.0 * eigvals_m * eigvals_m) / (4.0 * eigvals_m * eigvals_m + lambda_reg)
    residual_gains = 1.0 + residual_steps
    g_matrix = (eigvecs_m * residual_gains) @ eigvecs_m.T
    transform = sigma_sqrt @ g_matrix @ sigma_inv_sqrt

    return {
        "transform": transform,
        "g_matrix": g_matrix,
        "m_matrix": m_matrix,
        "residual_steps": residual_steps,
        "residual_gains": residual_gains,
        "distance_to_whitened_identity": float(np.linalg.norm(g_matrix - np.eye(g_matrix.shape[0]), ord="fro")),
        "max_m_eigenvalue": float(np.max(eigvals_m)),
        "min_m_eigenvalue": float(np.min(eigvals_m)),
        "max_residual_gain": float(np.max(residual_gains)),
        "min_residual_gain": float(np.min(residual_gains)),
    }


def fit_residual_barlow_layer(stats, lambda_reg):
    if lambda_reg <= 0.0:
        raise ValueError("lambda must be strictly positive.")

    sigma_bar = stats["sigma_bar"]
    shared = stats["shared"]
    sigma_sqrt, sigma_inv_sqrt = sqrt_and_inv_sqrt_psd(sigma_bar, REG_EPS)
    s_matrix = sigma_inv_sqrt @ shared @ sigma_inv_sqrt
    s_matrix = 0.5 * (s_matrix + s_matrix.T)

    eigvals_s, eigvecs_s = eigh(s_matrix)

    # First-order residual refinement in the whitened space:
    # minimize ||I - (S + RS + SR)||_F^2 + lambda ||R||_F^2.
    residual_steps = (2.0 * eigvals_s * (1.0 - eigvals_s)) / (4.0 * eigvals_s * eigvals_s + lambda_reg)
    residual_gains = 1.0 + residual_steps
    g_matrix = (eigvecs_s * residual_gains) @ eigvecs_s.T
    transform = sigma_sqrt @ g_matrix @ sigma_inv_sqrt

    return {
        "transform": transform,
        "g_matrix": g_matrix,
        "s_matrix": s_matrix,
        "residual_steps": residual_steps,
        "residual_gains": residual_gains,
        "distance_to_whitened_identity": float(np.linalg.norm(g_matrix - np.eye(g_matrix.shape[0]), ord="fro")),
        "max_shared_eigenvalue": float(np.max(eigvals_s)),
        "min_shared_eigenvalue": float(np.min(eigvals_s)),
        "max_residual_gain": float(np.max(residual_gains)),
        "min_residual_gain": float(np.min(residual_gains)),
    }


def fit_whitened_shared_pca_layer(stats):
    sigma_bar = stats["sigma_bar"]
    shared = stats["shared"]
    _, sigma_inv_sqrt = sqrt_and_inv_sqrt_psd(sigma_bar, REG_EPS)
    s_matrix = sigma_inv_sqrt @ shared @ sigma_inv_sqrt
    s_matrix = 0.5 * (s_matrix + s_matrix.T)

    eigvals_s, eigvecs_s = eigh(s_matrix)
    transform = eigvecs_s.T @ sigma_inv_sqrt
    return {
        "transform": transform,
        "s_matrix": s_matrix,
        "shared_eigenvalues": eigvals_s,
        "max_shared_eigenvalue": float(np.max(eigvals_s)),
        "min_shared_eigenvalue": float(np.min(eigvals_s)),
    }


def fit_paper_cca_shared_layer(stats):
    sigma_bar = stats["sigma_bar"] + REG_EPS * np.eye(stats["sigma_bar"].shape[0], dtype=np.float64)
    shared = 0.5 * (stats["shared"] + stats["shared"].T)
    eigvals, eigvecs = eigh(shared, sigma_bar)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    transform = eigvecs.T
    return {
        "transform": transform,
        "generalized_eigenvalues": eigvals,
        "max_shared_eigenvalue": float(np.max(eigvals)),
        "min_shared_eigenvalue": float(np.min(eigvals)),
    }


def fit_paper_cca_layer(stats):
    sigma1 = stats["sigma1"] + REG_EPS * np.eye(stats["sigma1"].shape[0], dtype=np.float64)
    sigma2 = stats["sigma2"] + REG_EPS * np.eye(stats["sigma2"].shape[0], dtype=np.float64)
    cross12 = stats["cross12"]
    cross21 = stats["cross21"]

    matrix_a = np.linalg.solve(sigma1, cross12 @ np.linalg.solve(sigma2, cross21))
    matrix_a = 0.5 * (matrix_a + matrix_a.T)
    eigvals_a, eigvecs_a = eigh(matrix_a)
    order = np.argsort(eigvals_a)[::-1]
    eigvals_a = np.maximum(eigvals_a[order], 0.0)
    transform_a = eigvecs_a[:, order]

    sigma1_norms = np.sqrt(np.maximum(np.sum(transform_a * (sigma1 @ transform_a), axis=0), REG_EPS))
    transform_a = transform_a / sigma1_norms[None, :]

    canonical_corrs = np.sqrt(eigvals_a)
    transform_b = np.linalg.solve(sigma2, cross21 @ transform_a)
    transform_b = transform_b / np.maximum(canonical_corrs[None, :], np.sqrt(REG_EPS))
    sigma2_norms = np.sqrt(np.maximum(np.sum(transform_b * (sigma2 @ transform_b), axis=0), REG_EPS))
    transform_b = transform_b / sigma2_norms[None, :]

    transform_base = 0.5 * (transform_a + transform_b)
    return {
        "transform_a": transform_a,
        "transform_b": transform_b,
        "transform_base": transform_base,
        "canonical_correlations": canonical_corrs,
        "max_canonical_correlation": float(np.max(canonical_corrs)),
        "min_canonical_correlation": float(np.min(canonical_corrs)),
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
        "transform_base": model["transform"],
        "transform_view1": model["transform"],
        "transform_view2": model["transform"],
        "stats": stats,
        "transform_fro": float(np.linalg.norm(model["transform"], ord="fro")),
        "distance_to_identity": float(np.linalg.norm(model["transform"] - np.eye(model["transform"].shape[0]), ord="fro")),
        "distance_to_whitened_identity": model["distance_to_whitened_identity"],
        "total_delta_trace": float(np.trace(stats["delta"])),
        "min_whitened_delta": float(np.min(np.linalg.eigvalsh(whitened_delta))),
        "max_whitened_delta": model["max_m_eigenvalue"],
    }


def fit_residual_barlow_from_pairs(H1, H2, lambda_reg):
    stats = compute_paired_stats(H1, H2)
    model = fit_residual_barlow_layer(stats, lambda_reg=lambda_reg)
    transform = model["transform"]
    return {
        "transform": transform,
        "transform_base": transform,
        "transform_view1": transform,
        "transform_view2": transform,
        "stats": stats,
        "transform_fro": float(np.linalg.norm(transform, ord="fro")),
        "distance_to_identity": float(np.linalg.norm(transform - np.eye(transform.shape[0]), ord="fro")),
        "distance_to_whitened_identity": model["distance_to_whitened_identity"],
        "total_delta_trace": float(np.trace(stats["delta"])),
        "min_whitened_delta": float("nan"),
        "max_whitened_delta": float("nan"),
        "max_shared_eigenvalue": model["max_shared_eigenvalue"],
        "min_shared_eigenvalue": model["min_shared_eigenvalue"],
        "max_residual_gain": model["max_residual_gain"],
        "min_residual_gain": model["min_residual_gain"],
    }


def fit_iterref_old_from_pairs(H1, H2, lambda_reg):
    stats = compute_paired_stats(H1, H2)
    model = fit_iterref_old_layer(stats, lambda_reg=lambda_reg)
    transform = model["transform"]
    return {
        "transform": transform,
        "transform_base": transform,
        "transform_view1": transform,
        "transform_view2": transform,
        "stats": stats,
        "transform_fro": float(np.linalg.norm(transform, ord="fro")),
        "distance_to_identity": float(np.linalg.norm(transform - np.eye(transform.shape[0]), ord="fro")),
        "distance_to_whitened_identity": model["distance_to_whitened_identity"],
        "total_delta_trace": float(np.trace(stats["delta"])),
        "min_whitened_delta": model["min_m_eigenvalue"],
        "max_whitened_delta": model["max_m_eigenvalue"],
        "max_residual_gain": model["max_residual_gain"],
        "min_residual_gain": model["min_residual_gain"],
    }


def fit_whitened_shared_pca_from_pairs(H1, H2):
    stats = compute_paired_stats(H1, H2)
    model = fit_whitened_shared_pca_layer(stats)
    return {
        "transform": model["transform"],
        "transform_base": model["transform"],
        "transform_view1": model["transform"],
        "transform_view2": model["transform"],
        "stats": stats,
        "transform_fro": float(np.linalg.norm(model["transform"], ord="fro")),
        "distance_to_identity": float(np.linalg.norm(model["transform"] - np.eye(model["transform"].shape[0]), ord="fro")),
        "distance_to_whitened_identity": float("nan"),
        "total_delta_trace": float(np.trace(stats["delta"])),
        "min_whitened_delta": float("nan"),
        "max_whitened_delta": float("nan"),
        "max_shared_eigenvalue": model["max_shared_eigenvalue"],
        "min_shared_eigenvalue": model["min_shared_eigenvalue"],
    }


def fit_paper_cca_from_pairs(H1, H2):
    stats = compute_paired_stats(H1, H2)
    model = fit_paper_cca_layer(stats)
    transform_base = model["transform_base"]
    return {
        "transform": transform_base,
        "transform_base": transform_base,
        "transform_view1": model["transform_a"],
        "transform_view2": model["transform_b"],
        "stats": stats,
        "transform_fro": float(np.linalg.norm(transform_base, ord="fro")),
        "distance_to_identity": float(np.linalg.norm(transform_base - np.eye(transform_base.shape[0]), ord="fro")),
        "distance_to_whitened_identity": float("nan"),
        "total_delta_trace": float(np.trace(stats["delta"])),
        "min_whitened_delta": float("nan"),
        "max_whitened_delta": float("nan"),
        "max_canonical_correlation": model["max_canonical_correlation"],
        "min_canonical_correlation": model["min_canonical_correlation"],
    }


def fit_paper_cca_shared_from_pairs(H1, H2):
    stats = compute_paired_stats(H1, H2)
    model = fit_paper_cca_shared_layer(stats)
    transform = model["transform"]
    return {
        "transform": transform,
        "transform_base": transform,
        "transform_view1": transform,
        "transform_view2": transform,
        "stats": stats,
        "transform_fro": float(np.linalg.norm(transform, ord="fro")),
        "distance_to_identity": float(np.linalg.norm(transform - np.eye(transform.shape[0]), ord="fro")),
        "distance_to_whitened_identity": float("nan"),
        "total_delta_trace": float(np.trace(stats["delta"])),
        "min_whitened_delta": float("nan"),
        "max_whitened_delta": float("nan"),
        "max_shared_eigenvalue": model["max_shared_eigenvalue"],
        "min_shared_eigenvalue": model["min_shared_eigenvalue"],
    }


def apply_activation(X, activation):
    if activation == "relu":
        return np.maximum(X, 0.0)
    if activation == "tanh":
        return np.tanh(X)
    if activation == "leaky-relu":
        return np.where(X >= 0.0, X, LEAKY_RELU_SLOPE * X)
    if activation == "identity":
        return X
    raise ValueError(f"Unknown activation: {activation}")


def apply_layer(X, transform, activation="relu"):
    return apply_activation(X @ transform, activation)


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


def fit_layer_by_method(method_name, view1_tr, view2_tr, lambda_reg):
    if method_name == "closed-form-barlow":
        return fit_layer(view1_tr, view2_tr, lambda_reg=lambda_reg)
    if method_name == "iterref-old":
        return fit_iterref_old_from_pairs(view1_tr, view2_tr, lambda_reg=lambda_reg)
    if method_name == "iterref-symcca":
        return fit_residual_barlow_from_pairs(view1_tr, view2_tr, lambda_reg=lambda_reg)
    if method_name == "residual-barlow":
        return fit_residual_barlow_from_pairs(view1_tr, view2_tr, lambda_reg=lambda_reg)
    if method_name == "whitened-shared-pca":
        return fit_whitened_shared_pca_from_pairs(view1_tr, view2_tr)
    if method_name == "paper-cca":
        return fit_paper_cca_from_pairs(view1_tr, view2_tr)
    if method_name == "paper-cca-shared":
        return fit_paper_cca_shared_from_pairs(view1_tr, view2_tr)
    raise ValueError(f"Unknown layer method: {method_name}")


def method_note(method_name):
    if method_name == "closed-form-barlow":
        return (
            "Closed-form Barlow Twins layer with objective "
            "tr(G^T M G) + lambda ||G - I||_F^2, whose exact solution is "
            "G = lambda (M + lambda I)^(-1)."
        )
    if method_name == "iterref-old":
        return (
            "Iterative-refinement version of the old disagreement formulation: "
            "reduce the current whitened disagreement matrix via a local residual solve "
            "for ||M + RM + MR||_F^2 + lambda ||R||_F^2."
        )
    if method_name == "iterref-symcca":
        return (
            "Iterative-refinement version of the symmetric CCA/shared-correlation "
            "formulation: reduce the current mismatch to identity via "
            "||I - (S + RS + SR)||_F^2 + lambda ||R||_F^2."
        )
    if method_name == "residual-barlow":
        return (
            "Residual Barlow layer using a first-order refinement of the whitened "
            "shared-correlation target: choose a near-identity residual update that "
            "reduces ||I - (S + RS + SR)||_F^2 + lambda ||R||_F^2."
        )
    if method_name == "whitened-shared-pca":
        return (
            "Whitened shared-covariance PCA layer: diagonalize the whitened shared "
            "cross-covariance and rotate into its eigenbasis."
        )
    if method_name == "paper-cca":
        return "Untied linear CCA layer from the paper-equivalent Barlow/CCA solution."
    if method_name == "paper-cca-shared":
        return "Shared-weight generalized-eigenvalue layer corresponding to the symmetric paper CCA solution."
    raise ValueError(f"Unknown layer method: {method_name}")


def run_experiment(suite_name, depth, final_dim, lambda_reg, activation="relu", layer_method="closed-form-barlow"):
    xtr, ytr, xte, yte = load_mnist_numpy()
    base_tr = xtr.copy()
    base_te = xte.copy()
    view1_tr, view2_tr = sample_pair_views(xtr, suite_name, seed=SEED + 13)
    view1_te, view2_te = sample_pair_views(xte, suite_name, seed=SEED + 31)

    layers = []
    for layer_idx in range(depth):
        model = fit_layer_by_method(layer_method, view1_tr, view2_tr, lambda_reg=lambda_reg)
        transform_base = model["transform_base"]
        transform_view1 = model["transform_view1"]
        transform_view2 = model["transform_view2"]

        base_tr = apply_layer(base_tr, transform_base, activation=activation)
        base_te = apply_layer(base_te, transform_base, activation=activation)
        view1_tr = apply_layer(view1_tr, transform_view1, activation=activation)
        view2_tr = apply_layer(view2_tr, transform_view2, activation=activation)
        view1_te = apply_layer(view1_te, transform_view1, activation=activation)
        view2_te = apply_layer(view2_te, transform_view2, activation=activation)

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
                "max_shared_eigenvalue": model.get("max_shared_eigenvalue"),
                "min_shared_eigenvalue": model.get("min_shared_eigenvalue"),
                "max_residual_gain": model.get("max_residual_gain"),
                "min_residual_gain": model.get("min_residual_gain"),
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
        "activation": activation,
        "layer_method": layer_method,
        "full_probe_accuracy": acc_full,
        "final_pca_probe_accuracy": acc_pca32,
        "raw_input_pca_probe_accuracy": acc_raw_pca32,
        "layers": layers,
        "note": method_note(layer_method),
    }


def main():
    parser = argparse.ArgumentParser(description="Closed-form Barlow Twins layer on MNIST.")
    parser.add_argument(
        "--suite",
        default="single-translation",
        choices=["image", "translation", "single-translation", "random-masking", "block-masking", "blurring", "rotation", "random-crops"],
    )
    parser.add_argument("--layer-method", choices=LAYER_METHODS, default="closed-form-barlow")
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--final-d", type=int, default=FINAL_D)
    parser.add_argument("--lambda", dest="lambda_reg", type=float, default=LAMBDA_REG)
    parser.add_argument("--activation", choices=["relu", "tanh", "leaky-relu", "identity"], default="relu")
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    if args.lambda_reg <= 0.0:
        raise ValueError("--lambda must be strictly positive.")

    result = run_experiment(
        suite_name=args.suite,
        depth=args.depth,
        final_dim=args.final_d,
        lambda_reg=args.lambda_reg,
        activation=args.activation,
        layer_method=args.layer_method,
    )

    print(
        f"Analytic MNIST DNN  |  method={result['layer_method']}  |  suite={result['suite']}  |  depth={result['depth']}  |  activation={result['activation']}  |  final_d={result['final_dim']}"
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
