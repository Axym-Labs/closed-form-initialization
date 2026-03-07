import argparse
import json
from inspect import signature
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, solve_sylvester
from sklearn.linear_model import LogisticRegression
from torchvision import datasets


SEED = 7
N_TRAIN = 6000
N_TEST = 1000
DEPTH = 3
FINAL_D = 32
RESIDUAL_RANK = 256
PROBE_MAX_ITER = 1000
REG_EPS = 1e-4
IMAGE_SIZE = 32
CHANNELS = 3
TRANSLATION_SHIFT_X = 3
TRANSLATION_SHIFT_Y = 3
BLOCK_GRID_SPLITS = 3

INV_WEIGHT = 1.0
PRES_WEIGHT = 1.0
SHARED_WEIGHT = 0.5
IDENTITY_WEIGHT = 1.0


def covariance(X):
    return (X.T @ X) / X.shape[0]


def cross_covariance(X, Y):
    return (X.T @ Y) / X.shape[0]


def orthonormal_basis(columns):
    q, _ = np.linalg.qr(columns, mode="reduced")
    return q


def flatten_images(images):
    return images.reshape(images.shape[0], -1)


def standardize_train_test(ztr, zte):
    mu = ztr.mean(axis=0, keepdims=True)
    std = ztr.std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    return (ztr - mu) / std, (zte - mu) / std


def fit_linear_probe(ztr, ytr, zte, yte):
    kwargs = {
        "max_iter": PROBE_MAX_ITER,
        "solver": "lbfgs",
    }
    if "multi_class" in signature(LogisticRegression).parameters:
        kwargs["multi_class"] = "multinomial"
    clf = LogisticRegression(**kwargs)
    clf.fit(ztr, ytr)
    return float((clf.predict(zte) == yte).mean())


def load_cifar_numpy(dataset_name):
    rng = np.random.default_rng(SEED)
    if dataset_name == "cifar10":
        train_ds = datasets.CIFAR10(root="./data", train=True, download=True)
        test_ds = datasets.CIFAR10(root="./data", train=False, download=True)
    elif dataset_name == "cifar100":
        train_ds = datasets.CIFAR100(root="./data", train=True, download=True)
        test_ds = datasets.CIFAR100(root="./data", train=False, download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    xtr = train_ds.data.astype(np.float64) / 255.0
    ytr = np.asarray(train_ds.targets)
    xte = test_ds.data.astype(np.float64) / 255.0
    yte = np.asarray(test_ds.targets)

    idx_tr = rng.choice(len(xtr), size=N_TRAIN, replace=False)
    idx_te = rng.choice(len(xte), size=N_TEST, replace=False)
    xtr, ytr = xtr[idx_tr], ytr[idx_tr]
    xte, yte = xte[idx_te], yte[idx_te]

    # HWC -> CHW so each channel occupies a contiguous block when flattened.
    xtr = np.transpose(xtr, (0, 3, 1, 2))
    xte = np.transpose(xte, (0, 3, 1, 2))

    mu = xtr.mean(axis=0, keepdims=True)
    xtr = xtr - mu
    xte = xte - mu
    return xtr, ytr, xte, yte


def grid_block_specs(num_splits, h, w):
    row_edges = np.linspace(0, h, num_splits + 1, dtype=int)
    col_edges = np.linspace(0, w, num_splits + 1, dtype=int)
    specs = []
    for i in range(num_splits):
        for j in range(num_splits):
            top = row_edges[i]
            left = col_edges[j]
            height = row_edges[i + 1] - row_edges[i]
            width = col_edges[j + 1] - col_edges[j]
            specs.append((top, left, height, width))
    return specs


def apply_translation(images, dx, dy):
    out = np.zeros_like(images)
    h = images.shape[2]
    w = images.shape[3]

    src_top = max(0, -dy)
    src_bottom = min(h, h - dy) if dy >= 0 else h
    dst_top = max(0, dy)
    dst_bottom = dst_top + (src_bottom - src_top)

    src_left = max(0, -dx)
    src_right = min(w, w - dx) if dx >= 0 else w
    dst_left = max(0, dx)
    dst_right = dst_left + (src_right - src_left)

    out[:, :, dst_top:dst_bottom, dst_left:dst_right] = images[:, :, src_top:src_bottom, src_left:src_right]
    return out


def apply_block_mask(images, spec):
    top, left, height, width = spec
    out = images.copy()
    out[:, :, top : top + height, left : left + width] = 0.0
    return out


def build_suite_ops(suite_name):
    if suite_name == "single-translation":
        return [("translate", (TRANSLATION_SHIFT_X, TRANSLATION_SHIFT_Y))]
    if suite_name == "block-masking":
        return [("block-mask", spec) for spec in grid_block_specs(BLOCK_GRID_SPLITS, IMAGE_SIZE, IMAGE_SIZE)]
    raise ValueError(f"Unknown suite: {suite_name}")


def apply_op(images, op):
    kind, payload = op
    if kind == "identity":
        return images.copy()
    if kind == "translate":
        dx, dy = payload
        return apply_translation(images, dx=dx, dy=dy)
    if kind == "block-mask":
        return apply_block_mask(images, payload)
    raise ValueError(f"Unknown op kind: {kind}")


def sample_pair_views(images, suite_name, seed):
    rng = np.random.default_rng(seed)
    ops = [("identity", None)] + build_suite_ops(suite_name)
    idx1 = rng.integers(len(ops), size=images.shape[0])
    idx2 = rng.integers(len(ops), size=images.shape[0])

    view1 = np.empty_like(images)
    view2 = np.empty_like(images)
    for op_idx, op in enumerate(ops):
        mask1 = idx1 == op_idx
        mask2 = idx2 == op_idx
        if np.any(mask1):
            view1[mask1] = apply_op(images[mask1], op)
        if np.any(mask2):
            view2[mask2] = apply_op(images[mask2], op)
    return view1, view2


def sample_same_class_pairs(images, labels, seed):
    rng = np.random.default_rng(seed)
    view1 = images.copy()
    view2 = np.empty_like(images)
    classes = np.unique(labels)

    for cls in classes:
        cls_idx = np.flatnonzero(labels == cls)
        if cls_idx.size == 1:
            view2[cls_idx] = images[cls_idx]
            continue
        choices = rng.integers(0, cls_idx.size, size=cls_idx.size)
        same = cls_idx[choices] == cls_idx
        if np.any(same):
            choices[same] = (choices[same] + 1) % cls_idx.size
        view2[cls_idx] = images[cls_idx[choices]]
    return view1, view2


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


def fit_rank_deflated_layer(H1, H2, residual_rank, inv_weight, pres_weight, shared_weight, identity_weight):
    stats = compute_paired_stats(H1, H2)
    sigma_bar = stats["sigma_bar"]
    delta = stats["delta"]
    shared = stats["shared"]

    p = sigma_bar.shape[0]
    rank = min(residual_rank, p)
    sigma_scale = float(np.trace(sigma_bar) / p)
    sigma_reg = sigma_bar + (REG_EPS + 1e-6 * sigma_scale) * np.eye(p, dtype=np.float64)

    if rank == p:
        evals, evecs = eigh(delta, sigma_reg)
        residual_basis = orthonormal_basis(evecs[:, -rank:])
        top_evals = evals[-rank:]
    else:
        evals, evecs = eigh(delta, sigma_reg, subset_by_index=[p - rank, p - 1])
        residual_basis = orthonormal_basis(evecs)
        top_evals = evals

    sigma_r = residual_basis.T @ sigma_bar @ residual_basis
    shared_r = residual_basis.T @ shared @ residual_basis
    delta_r = residual_basis.T @ delta @ residual_basis

    A = inv_weight * delta_r + identity_weight * np.eye(rank, dtype=np.float64)
    B = pres_weight * (sigma_r @ sigma_r)
    C = shared_weight * shared_r + pres_weight * (sigma_r @ sigma_r) + identity_weight * np.eye(
        rank, dtype=np.float64
    )
    transform_r = solve_sylvester(A, B, C)
    return {
        "basis": residual_basis,
        "transform_r": transform_r,
        "stats": stats,
        "distance_to_identity_r": float(np.linalg.norm(transform_r - np.eye(rank), ord="fro")),
        "top_generalized_delta": float(np.max(top_evals)),
    }


def apply_layer(X, basis, transform_r):
    proj = X @ basis
    updated = X + proj @ (transform_r - np.eye(transform_r.shape[0], dtype=np.float64)) @ basis.T
    return np.maximum(updated, 0.0)


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
    p = sigma.shape[0]
    evals, evecs = eigh(sigma, subset_by_index=[p - d, p - 1])
    order = np.argsort(evals)
    return evecs[:, order]


def run_experiment(dataset_name, suite_name, depth, final_dim, residual_rank, pair_source):
    base_tr_img, ytr, base_te_img, yte = load_cifar_numpy(dataset_name)
    if pair_source == "same-class":
        view1_tr_img, view2_tr_img = sample_same_class_pairs(base_tr_img, ytr, seed=SEED + 11)
        view1_te_img, view2_te_img = sample_same_class_pairs(base_te_img, yte, seed=SEED + 29)
    else:
        view1_tr_img, view2_tr_img = sample_pair_views(base_tr_img, suite_name, seed=SEED + 11)
        view1_te_img, view2_te_img = sample_pair_views(base_te_img, suite_name, seed=SEED + 29)

    base_tr = flatten_images(base_tr_img)
    base_te = flatten_images(base_te_img)
    view1_tr = flatten_images(view1_tr_img)
    view2_tr = flatten_images(view2_tr_img)
    view1_te = flatten_images(view1_te_img)
    view2_te = flatten_images(view2_te_img)

    layers = []
    for layer_idx in range(depth):
        model = fit_rank_deflated_layer(
            view1_tr,
            view2_tr,
            residual_rank=residual_rank,
            inv_weight=INV_WEIGHT,
            pres_weight=PRES_WEIGHT,
            shared_weight=SHARED_WEIGHT,
            identity_weight=IDENTITY_WEIGHT,
        )

        base_tr = apply_layer(base_tr, model["basis"], model["transform_r"])
        base_te = apply_layer(base_te, model["basis"], model["transform_r"])
        view1_tr = apply_layer(view1_tr, model["basis"], model["transform_r"])
        view2_tr = apply_layer(view2_tr, model["basis"], model["transform_r"])
        view1_te = apply_layer(view1_te, model["basis"], model["transform_r"])
        view2_te = apply_layer(view2_te, model["basis"], model["transform_r"])

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
                "residual_rank": residual_rank,
                "distance_to_identity_r": model["distance_to_identity_r"],
                "top_generalized_delta": model["top_generalized_delta"],
                "post_delta_trace": float(np.trace(post_stats["delta"])),
                "post_shared_trace": float(np.trace(post_stats["shared"])),
            }
        )

    ztr_full, zte_full = standardize_train_test(base_tr, base_te)
    full_acc = fit_linear_probe(ztr_full, ytr, zte_full, yte)

    pca_basis = top_pca_projection(base_tr, final_dim)
    ztr_pca = base_tr @ pca_basis
    zte_pca = base_te @ pca_basis
    ztr_pca, zte_pca = standardize_train_test(ztr_pca, zte_pca)
    final_pca_acc = fit_linear_probe(ztr_pca, ytr, zte_pca, yte)

    return {
        "dataset": dataset_name,
        "suite": suite_name,
        "pair_source": pair_source,
        "depth": depth,
        "final_dim": final_dim,
        "residual_rank": residual_rank,
        "full_probe_accuracy": full_acc,
        "final_pca_probe_accuracy": final_pca_acc,
        "layers": layers,
        "note": (
            "Scalable CIFAR version of the deflated full-width Sylvester DNN. "
            "Each layer updates only a rank-r residual subspace of the current full representation."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="CIFAR evaluation for the deflated full-width Sylvester DNN.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default=None)
    parser.add_argument("--suite", choices=["single-translation", "block-masking"], default=None)
    parser.add_argument("--pair-source", choices=["suite", "same-class"], default="suite")
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--final-d", type=int, default=FINAL_D)
    parser.add_argument("--residual-rank", type=int, default=RESIDUAL_RANK)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    datasets_to_run = [args.dataset] if args.dataset is not None else ["cifar10", "cifar100"]
    if args.pair_source == "same-class":
        suites_to_run = [args.suite] if args.suite is not None else [None]
    else:
        suites_to_run = [args.suite] if args.suite is not None else ["single-translation", "block-masking"]

    results = []
    for dataset_name in datasets_to_run:
        for suite_name in suites_to_run:
            result = run_experiment(
                dataset_name=dataset_name,
                suite_name=suite_name,
                depth=args.depth,
                final_dim=args.final_d,
                residual_rank=args.residual_rank,
                pair_source=args.pair_source,
            )
            results.append(result)
            suite_label = suite_name if suite_name is not None else "same-class"
            print(
                f"{dataset_name:>8s} | {suite_label:>18s} | pair_source={result['pair_source']:<10s} | full={result['full_probe_accuracy']:.4f} | "
                f"pca{result['final_dim']}={result['final_pca_probe_accuracy']:.4f}"
            )

    if args.save_json is not None:
        args.save_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"saved json to {args.save_json}")


if __name__ == "__main__":
    main()
