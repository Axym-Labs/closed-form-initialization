import argparse
import json
from inspect import signature
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from torchvision import datasets

import closed_form_barlow_twins as cfbt
from project_paths import resolve_json_path


SEED = 7
N_TRAIN = 6000
N_TEST = 1000
DEPTH = 3
FINAL_D = 32
PROBE_MAX_ITER = 2000

SINGLE_TRANSLATION_DX = 3
SINGLE_TRANSLATION_DY = 3
BLOCK_GRID_SPLITS = 3
LAYER_METHODS = [
    "closed-form-barlow",
    "iterref-old",
    "iterref-symcca",
    "residual-barlow",
    "whitened-shared-pca",
    "paper-cca",
    "paper-cca-shared",
]
ACTIVATIONS = ["relu", "tanh", "leaky-relu", "identity"]


def images_to_flat(images):
    return images.reshape(images.shape[0], -1)


def flat_to_images(flat, channels, h, w):
    return flat.reshape(flat.shape[0], channels, h, w)


def load_cifar_numpy(dataset_name, n_train, n_test, seed):
    rng = np.random.default_rng(seed)
    dataset_cls = {
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
    }[dataset_name]

    train_ds = dataset_cls(root="./data", train=True, download=True)
    test_ds = dataset_cls(root="./data", train=False, download=True)

    xtr = train_ds.data.astype(np.float64) / 255.0
    xte = test_ds.data.astype(np.float64) / 255.0
    ytr = np.asarray(train_ds.targets)
    yte = np.asarray(test_ds.targets)

    idx_tr = rng.choice(len(xtr), size=n_train, replace=False)
    idx_te = rng.choice(len(xte), size=n_test, replace=False)
    xtr = xtr[idx_tr]
    ytr = ytr[idx_tr]
    xte = xte[idx_te]
    yte = yte[idx_te]

    xtr = np.transpose(xtr, (0, 3, 1, 2))
    xte = np.transpose(xte, (0, 3, 1, 2))

    xtr_flat = images_to_flat(xtr)
    xte_flat = images_to_flat(xte)
    mu = xtr_flat.mean(axis=0, keepdims=True)
    xtr = flat_to_images(xtr_flat - mu, channels=3, h=32, w=32)
    xte = flat_to_images(xte_flat - mu, channels=3, h=32, w=32)
    return xtr, ytr, xte, yte


def shift_images(images, dx, dy):
    shifted = np.zeros_like(images)
    src_r0 = max(0, -dy)
    src_r1 = images.shape[2] - max(0, dy)
    dst_r0 = max(0, dy)
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    src_c0 = max(0, -dx)
    src_c1 = images.shape[3] - max(0, dx)
    dst_c0 = max(0, dx)
    dst_c1 = dst_c0 + (src_c1 - src_c0)
    shifted[:, :, dst_r0:dst_r1, dst_c0:dst_c1] = images[:, :, src_r0:src_r1, src_c0:src_c1]
    return shifted


def grid_block_specs(num_splits, h, w):
    row_edges = np.linspace(0, h, num_splits + 1, dtype=int)
    col_edges = np.linspace(0, w, num_splits + 1, dtype=int)
    specs = []
    for i in range(num_splits):
        for j in range(num_splits):
            specs.append(
                (
                    row_edges[i],
                    col_edges[j],
                    row_edges[i + 1] - row_edges[i],
                    col_edges[j + 1] - col_edges[j],
                )
            )
    return specs


def block_mask_images(images, top, left, height, width):
    masked = images.copy()
    masked[:, :, top : top + height, left : left + width] = 0.0
    return masked


def build_transform_specs(suite_name, h=32, w=32):
    if suite_name == "single-translation":
        return [("shift", (SINGLE_TRANSLATION_DX, SINGLE_TRANSLATION_DY))]
    if suite_name == "block-masking":
        return [("block-mask", spec) for spec in grid_block_specs(BLOCK_GRID_SPLITS, h=h, w=w)]
    raise ValueError(f"Unsupported CIFAR suite: {suite_name}")


def apply_spec(images, spec):
    kind, params = spec
    if kind == "identity":
        return images.copy()
    if kind == "shift":
        dx, dy = params
        return shift_images(images, dx=dx, dy=dy)
    if kind == "block-mask":
        top, left, height, width = params
        return block_mask_images(images, top, left, height, width)
    raise ValueError(f"Unknown transform kind: {kind}")


def sample_pair_views(images, suite_name, seed):
    rng = np.random.default_rng(seed)
    specs = [("identity", None)] + build_transform_specs(suite_name, h=images.shape[2], w=images.shape[3])
    idx1 = rng.integers(len(specs), size=images.shape[0])
    idx2 = rng.integers(len(specs), size=images.shape[0])
    x1 = np.empty_like(images)
    x2 = np.empty_like(images)
    for spec_idx, spec in enumerate(specs):
        mask1 = idx1 == spec_idx
        mask2 = idx2 == spec_idx
        if np.any(mask1):
            x1[mask1] = apply_spec(images[mask1], spec)
        if np.any(mask2):
            x2[mask2] = apply_spec(images[mask2], spec)
    return images_to_flat(x1), images_to_flat(x2)


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


def fit_layer_by_method(method_name, view1_tr, view2_tr, lambda_reg):
    if method_name == "closed-form-barlow":
        return cfbt.fit_layer(view1_tr, view2_tr, lambda_reg=lambda_reg)
    if method_name == "iterref-old":
        return cfbt.fit_iterref_old_from_pairs(view1_tr, view2_tr, lambda_reg=lambda_reg)
    if method_name == "iterref-symcca":
        return cfbt.fit_residual_barlow_from_pairs(view1_tr, view2_tr, lambda_reg=lambda_reg)
    if method_name == "residual-barlow":
        return cfbt.fit_residual_barlow_from_pairs(view1_tr, view2_tr, lambda_reg=lambda_reg)
    if method_name == "whitened-shared-pca":
        return cfbt.fit_whitened_shared_pca_from_pairs(view1_tr, view2_tr)
    if method_name == "paper-cca":
        return cfbt.fit_paper_cca_from_pairs(view1_tr, view2_tr)
    if method_name == "paper-cca-shared":
        return cfbt.fit_paper_cca_shared_from_pairs(view1_tr, view2_tr)
    raise ValueError(f"Unknown layer method: {method_name}")


def run_experiment(dataset_name, suite_name, lambda_reg, depth, final_dim, n_train, n_test, layer_method, activation):
    xtr_img, ytr, xte_img, yte = load_cifar_numpy(dataset_name, n_train=n_train, n_test=n_test, seed=SEED)
    xtr = images_to_flat(xtr_img)
    xte = images_to_flat(xte_img)

    base_tr = xtr.copy()
    base_te = xte.copy()
    view1_tr, view2_tr = sample_pair_views(xtr_img, suite_name, seed=SEED + 13)
    view1_te, view2_te = sample_pair_views(xte_img, suite_name, seed=SEED + 31)

    layers = []
    for layer_idx in range(depth):
        model = fit_layer_by_method(layer_method, view1_tr, view2_tr, lambda_reg=lambda_reg)
        transform_base = model["transform_base"]
        transform_view1 = model["transform_view1"]
        transform_view2 = model["transform_view2"]

        base_tr = cfbt.apply_layer(base_tr, transform_base, activation=activation)
        base_te = cfbt.apply_layer(base_te, transform_base, activation=activation)
        view1_tr = cfbt.apply_layer(view1_tr, transform_view1, activation=activation)
        view2_tr = cfbt.apply_layer(view2_tr, transform_view2, activation=activation)
        view1_te = cfbt.apply_layer(view1_te, transform_view1, activation=activation)
        view2_te = cfbt.apply_layer(view2_te, transform_view2, activation=activation)

        train_arrays, test_arrays = cfbt.normalize_hidden(
            [base_tr, view1_tr, view2_tr],
            [base_te, view1_te, view2_te],
        )
        base_tr, view1_tr, view2_tr = train_arrays
        base_te, view1_te, view2_te = test_arrays

        post_stats = cfbt.compute_paired_stats(view1_tr, view2_tr)
        layers.append(
            {
                "layer": layer_idx + 1,
                "distance_to_identity": model["distance_to_identity"],
                "distance_to_whitened_identity": model["distance_to_whitened_identity"],
                "total_delta_trace": model["total_delta_trace"],
                "post_delta_trace": float(np.trace(post_stats["delta"])),
                "post_shared_trace": float(np.trace(post_stats["shared"])),
                "base_trace": float(np.trace(cfbt.covariance(base_tr))),
                "max_whitened_delta": model["max_whitened_delta"],
                "max_shared_eigenvalue": model.get("max_shared_eigenvalue"),
                "min_shared_eigenvalue": model.get("min_shared_eigenvalue"),
                "max_residual_gain": model.get("max_residual_gain"),
                "min_residual_gain": model.get("min_residual_gain"),
                "max_canonical_correlation": model.get("max_canonical_correlation"),
                "min_canonical_correlation": model.get("min_canonical_correlation"),
            }
        )

    ztr_full, zte_full = cfbt.standardize_train_test(base_tr, base_te)
    acc_full = fit_linear_probe(ztr_full, ytr, zte_full, yte)

    final_basis = cfbt.top_pca_projection(base_tr, final_dim)
    ztr_pca = base_tr @ final_basis
    zte_pca = base_te @ final_basis
    ztr_pca, zte_pca = cfbt.standardize_train_test(ztr_pca, zte_pca)
    acc_final_pca = fit_linear_probe(ztr_pca, ytr, zte_pca, yte)

    raw_basis = cfbt.top_pca_projection(xtr, final_dim)
    raw_tr = xtr @ raw_basis
    raw_te = xte @ raw_basis
    raw_tr, raw_te = cfbt.standardize_train_test(raw_tr, raw_te)
    acc_raw_pca = fit_linear_probe(raw_tr, ytr, raw_te, yte)

    return {
        "dataset": dataset_name,
        "suite": suite_name,
        "layer_method": layer_method,
        "activation": activation,
        "lambda": lambda_reg,
        "depth": depth,
        "final_dim": final_dim,
        "n_train": n_train,
        "n_test": n_test,
        "full_probe_accuracy": acc_full,
        "final_pca_probe_accuracy": acc_final_pca,
        "raw_input_pca_probe_accuracy": acc_raw_pca,
        "layers": layers,
    }


def main():
    parser = argparse.ArgumentParser(description="Closed-form Barlow Twins on CIFAR.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--suite", choices=["single-translation", "block-masking"], default="single-translation")
    parser.add_argument("--layer-method", choices=LAYER_METHODS, default="closed-form-barlow")
    parser.add_argument("--activation", choices=ACTIVATIONS, default="relu")
    parser.add_argument("--lambda", dest="lambda_reg", type=float, default=1.0)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--final-d", type=int, default=FINAL_D)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    result = run_experiment(
        dataset_name=args.dataset,
        suite_name=args.suite,
        lambda_reg=args.lambda_reg,
        depth=args.depth,
        final_dim=args.final_d,
        n_train=args.n_train,
        n_test=args.n_test,
        layer_method=args.layer_method,
        activation=args.activation,
    )

    print(
        f"CIFAR analytic DNN  |  method={result['layer_method']}  |  activation={result['activation']}  |  dataset={result['dataset']}  |  suite={result['suite']}  |  lambda={result['lambda']:.3f}"
    )
    print(f"full probe accuracy      : {result['full_probe_accuracy']:.4f}")
    print(f"final PCA-{result['final_dim']} probe : {result['final_pca_probe_accuracy']:.4f}")
    print(f"raw input PCA-{result['final_dim']} probe : {result['raw_input_pca_probe_accuracy']:.4f}")

    if args.save_json is not None:
        output_path = resolve_json_path(args.save_json)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved json to {output_path}")


if __name__ == "__main__":
    main()
