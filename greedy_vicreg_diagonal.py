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

GAMMA = 1.0
INV_WEIGHT = 25.0
VAR_WEIGHT = 25.0
IDENTITY_WEIGHT = 25.0


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
    return H1 - mu, H2 - mu, mu


def compute_paired_stats(H1, H2):
    H1c, H2c, _ = center_pair(H1, H2)
    sigma1 = covariance(H1c)
    sigma2 = covariance(H2c)
    sigma_bar = 0.5 * (sigma1 + sigma2)
    delta = covariance(H1c - H2c)
    shared = 0.5 * (cross_covariance(H1c, H2c) + cross_covariance(H2c, H1c))
    return {
        "sigma1": sigma1,
        "sigma2": sigma2,
        "sigma_bar": sigma_bar,
        "delta": delta,
        "shared": shared,
    }


def residual_vicreg_objective(scale, std, delta_coord, gamma, inv_weight, var_weight, identity_weight):
    return (
        inv_weight * delta_coord * scale * scale
        + var_weight * max(0.0, gamma - scale * std)
        + identity_weight * (scale - 1.0) * (scale - 1.0)
    )


def solve_residual_scale(std, delta_coord, gamma, inv_weight, var_weight, identity_weight):
    a = inv_weight * delta_coord
    r = identity_weight
    boundary = gamma / max(std, 1e-12)

    candidates = [0.0, 1.0, boundary]
    denom = 2.0 * (a + r)
    if denom > 1e-12:
        low_region = (var_weight * std + 2.0 * r) / denom
        candidates.append(np.clip(low_region, 0.0, boundary))
        high_region = r / max(a + r, 1e-12)
        if high_region >= boundary:
            candidates.append(high_region)

    best = 1.0
    best_obj = residual_vicreg_objective(best, std, delta_coord, gamma, inv_weight, var_weight, identity_weight)
    for candidate in candidates:
        value = residual_vicreg_objective(
            float(candidate), std, delta_coord, gamma, inv_weight, var_weight, identity_weight
        )
        if value < best_obj:
            best = float(candidate)
            best_obj = value
    return best


def fit_diagonal_vicreg_layer(H1, H2, gamma, inv_weight, var_weight, identity_weight):
    stats = compute_paired_stats(H1, H2)
    sigma_bar = 0.5 * (stats["sigma_bar"] + stats["sigma_bar"].T)
    delta = 0.5 * (stats["delta"] + stats["delta"].T)

    evals, evecs = eigh(sigma_bar)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    basis = evecs[:, order]
    delta_diag = np.maximum(np.sum(basis * (delta @ basis), axis=0), 0.0)
    stds = np.sqrt(np.maximum(evals, 1e-12))
    scales = np.array(
        [
            solve_residual_scale(std, delta_coord, gamma, inv_weight, var_weight, identity_weight)
            for std, delta_coord in zip(stds, delta_diag)
        ],
        dtype=np.float64,
    )
    operator = (basis * scales) @ basis.T

    view_cov = operator @ sigma_bar @ operator.T
    offdiag = view_cov - np.diag(np.diag(view_cov))
    active_var_fraction = float(np.mean(scales * stds < gamma))
    return {
        "operator": operator,
        "basis": basis,
        "scales": scales,
        "stds": stds,
        "delta_diag": delta_diag,
        "stats": stats,
        "offdiag_energy": float(np.linalg.norm(offdiag, ord="fro") ** 2),
        "active_var_fraction": active_var_fraction,
    }


def apply_layer(X, operator):
    return np.maximum(X @ operator.T, 0.0)


def recenter_all(train_arrays, test_arrays):
    mu = sum(arr.mean(axis=0, keepdims=True) for arr in train_arrays) / len(train_arrays)
    centered_train = [arr - mu for arr in train_arrays]
    centered_test = [arr - mu for arr in test_arrays]
    return centered_train, centered_test, mu


def top_pca_projection(Xtr, d):
    sigma = covariance(Xtr - Xtr.mean(axis=0, keepdims=True))
    evals, evecs = eigh(sigma)
    basis = evecs[:, -d:]
    return basis


def run_experiment(suite_name, depth, final_dim, gamma, inv_weight, var_weight, identity_weight):
    xtr, ytr, xte, yte = load_mnist_numpy()
    base_tr = xtr.copy()
    base_te = xte.copy()
    view1_tr, view2_tr = sample_pair_views(xtr, suite_name, seed=SEED + 11)
    view1_te, view2_te = sample_pair_views(xte, suite_name, seed=SEED + 29)

    layers = []
    for layer_idx in range(depth):
        model = fit_diagonal_vicreg_layer(
            view1_tr,
            view2_tr,
            gamma=gamma,
            inv_weight=inv_weight,
            var_weight=var_weight,
            identity_weight=identity_weight,
        )

        base_tr = apply_layer(base_tr, model["operator"])
        base_te = apply_layer(base_te, model["operator"])
        view1_tr = apply_layer(view1_tr, model["operator"])
        view2_tr = apply_layer(view2_tr, model["operator"])
        view1_te = apply_layer(view1_te, model["operator"])
        view2_te = apply_layer(view2_te, model["operator"])

        train_arrays, test_arrays, _ = recenter_all(
            [base_tr, view1_tr, view2_tr],
            [base_te, view1_te, view2_te],
        )
        base_tr, view1_tr, view2_tr = train_arrays
        base_te, view1_te, view2_te = test_arrays

        post_stats = compute_paired_stats(view1_tr, view2_tr)
        pair_delta_trace = float(np.trace(post_stats["delta"]))
        pair_shared_trace = float(np.trace(post_stats["shared"]))
        base_cov = covariance(base_tr - base_tr.mean(axis=0, keepdims=True))
        layers.append(
            {
                "layer": layer_idx + 1,
                "mean_scale": float(model["scales"].mean()),
                "min_scale": float(model["scales"].min()),
                "max_scale": float(model["scales"].max()),
                "active_var_fraction": model["active_var_fraction"],
                "offdiag_energy": model["offdiag_energy"],
                "pair_delta_trace": pair_delta_trace,
                "pair_shared_trace": pair_shared_trace,
                "base_trace": float(np.trace(base_cov)),
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

    raw_full_tr, raw_full_te = standardize_train_test(xtr, xte)
    acc_raw_full = fit_linear_probe(raw_full_tr, ytr, raw_full_te, yte)

    return {
        "suite": suite_name,
        "depth": depth,
        "final_dim": final_dim,
        "gamma": gamma,
        "inv_weight": inv_weight,
        "var_weight": var_weight,
        "identity_weight": identity_weight,
        "full_probe_accuracy": acc_full,
        "pca32_probe_accuracy": acc_pca32,
        "raw_input_full_probe_accuracy": acc_raw_full,
        "raw_input_pca32_probe_accuracy": acc_raw_pca32,
        "layers": layers,
        "note": (
            "Closed-form diagonal relaxation of greedy VICReg on paired hidden activations. "
            "This is not an exact solver for the full linear VICReg loss."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Greedy closed-form diagonal VICReg relaxation on MNIST.")
    parser.add_argument(
        "--suite",
        default="single-translation",
        choices=["image", "translation", "single-translation", "random-masking", "block-masking", "blurring"],
    )
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--final-d", type=int, default=FINAL_D)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--inv-weight", type=float, default=INV_WEIGHT)
    parser.add_argument("--var-weight", type=float, default=VAR_WEIGHT)
    parser.add_argument("--identity-weight", type=float, default=IDENTITY_WEIGHT)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    result = run_experiment(
        suite_name=args.suite,
        depth=args.depth,
        final_dim=args.final_d,
        gamma=args.gamma,
        inv_weight=args.inv_weight,
        var_weight=args.var_weight,
        identity_weight=args.identity_weight,
    )

    print(
        f"Greedy diagonal VICReg relaxation  |  suite={result['suite']}  |  depth={result['depth']}  |  final_d={result['final_dim']}"
    )
    print(f"full probe accuracy       : {result['full_probe_accuracy']:.4f}")
    print(f"final PCA-{result['final_dim']} probe: {result['pca32_probe_accuracy']:.4f}")
    print(f"raw input full probe      : {result['raw_input_full_probe_accuracy']:.4f}")
    print(f"raw input PCA-{result['final_dim']} probe : {result['raw_input_pca32_probe_accuracy']:.4f}")
    for layer in result["layers"]:
        print(
            f"layer {layer['layer']:>2d} | scale(mean/min/max)=({layer['mean_scale']:.3f}, "
            f"{layer['min_scale']:.3f}, {layer['max_scale']:.3f}) | "
            f"active_var={layer['active_var_fraction']:.3f} | "
            f"delta_trace={layer['pair_delta_trace']:.3f} | shared_trace={layer['pair_shared_trace']:.3f}"
        )
    print(result["note"])

    if args.save_json is not None:
        args.save_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved json to {args.save_json}")


if __name__ == "__main__":
    main()
