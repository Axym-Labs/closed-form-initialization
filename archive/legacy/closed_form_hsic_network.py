import argparse
import json
from pathlib import Path

import numpy as np

import closed_form_barlow_twins as cfbt
import closed_form_barlow_twins_cifar as cfbt_cifar
from project_paths import resolve_json_path


SEED = 7
DEPTH = 3
RFF_DIM = 256
HEAD_REG = 1e-2
MAX_BANDWIDTH_POINTS = 512


def one_hot(y, num_classes):
    eye = np.eye(num_classes, dtype=np.float64)
    return eye[y]


def ridge_regression(X, Y, reg):
    gram = X.T @ X
    rhs = X.T @ Y
    if reg <= 0.0:
        return np.linalg.pinv(gram, rcond=1e-10) @ rhs
    return np.linalg.solve(gram + reg * np.eye(gram.shape[0], dtype=np.float64), rhs)


def ridge_head_accuracy(Htr, ytr, Hte, yte, reg):
    num_classes = int(max(np.max(ytr), np.max(yte)) + 1)
    Ytr = one_hot(ytr, num_classes)
    head = ridge_regression(Htr, Ytr, reg=reg)
    logits = Hte @ head
    pred = np.argmax(logits, axis=1)
    return float((pred == yte).mean()), head


def standardize_train_test(Htr, Hte):
    mu = Htr.mean(axis=0, keepdims=True)
    std = Htr.std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    return (Htr - mu) / std, (Hte - mu) / std


def load_dataset(dataset_name, n_train, n_test):
    if dataset_name == "mnist":
        xtr, ytr, xte, yte = cfbt.load_mnist_numpy()
        return xtr, ytr, xte, yte
    if dataset_name in {"cifar10", "cifar100"}:
        xtr_img, ytr, xte_img, yte = cfbt_cifar.load_cifar_numpy(
            dataset_name, n_train=n_train, n_test=n_test, seed=SEED
        )
        return cfbt_cifar.images_to_flat(xtr_img), ytr, cfbt_cifar.images_to_flat(xte_img), yte
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def estimate_bandwidth(X, seed, max_points=MAX_BANDWIDTH_POINTS):
    rng = np.random.default_rng(seed)
    if X.shape[0] > max_points:
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        X = X[idx]
    norms = np.sum(X * X, axis=1, keepdims=True)
    sq_dists = np.maximum(norms + norms.T - 2.0 * (X @ X.T), 0.0)
    tri = sq_dists[np.triu_indices(sq_dists.shape[0], k=1)]
    tri = tri[tri > 1e-12]
    if tri.size == 0:
        return 1.0
    # Median heuristic for exp(-||x-y||^2 / (2 sigma^2)).
    return float(np.sqrt(0.5 * np.median(tri)))


def rff_features(X, out_dim, bandwidth, seed):
    rng = np.random.default_rng(seed)
    omega = rng.normal(loc=0.0, scale=1.0 / max(bandwidth, 1e-6), size=(X.shape[1], out_dim))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(out_dim,))
    return np.sqrt(2.0 / out_dim) * np.cos(X @ omega + phase), omega, phase


def rff_features_with_params(X, omega, phase):
    return np.sqrt(2.0 / omega.shape[1]) * np.cos(X @ omega + phase)


def class_mean_embeddings(features, labels, num_classes):
    means = np.zeros((num_classes, features.shape[1]), dtype=np.float64)
    for cls in range(num_classes):
        cls_mask = labels == cls
        means[cls] = features[cls_mask].mean(axis=0)
    return means


def hsic_layer(Htr, ytr, Hte, num_classes, rff_dim, seed):
    bandwidth = estimate_bandwidth(Htr, seed=seed)
    phi_tr, omega, phase = rff_features(Htr, out_dim=rff_dim, bandwidth=bandwidth, seed=seed)
    phi_te = rff_features_with_params(Hte, omega=omega, phase=phase)
    prototypes = class_mean_embeddings(phi_tr, ytr, num_classes=num_classes)
    scores_tr = phi_tr @ prototypes.T
    scores_te = phi_te @ prototypes.T
    direct_pred = np.argmax(scores_te, axis=1)
    return scores_tr, scores_te, {
        "bandwidth": bandwidth,
        "prototype_norm": float(np.linalg.norm(prototypes, ord="fro")),
        "feature_norm": float(np.linalg.norm(phi_tr, ord="fro")),
        "direct_scores_train": scores_tr,
        "direct_scores_test": scores_te,
        "direct_pred": direct_pred,
    }


def run_experiment(dataset_name, depth, rff_dim, head_reg, n_train, n_test):
    xtr, ytr, xte, yte = load_dataset(dataset_name, n_train=n_train, n_test=n_test)
    num_classes = int(max(np.max(ytr), np.max(yte)) + 1)
    Htr = xtr.copy()
    Hte = xte.copy()
    layers = []

    for layer_idx in range(depth):
        raw_tr, raw_te, layer_meta = hsic_layer(
            Htr, ytr, Hte, num_classes=num_classes, rff_dim=rff_dim, seed=SEED + 101 * (layer_idx + 1)
        )
        Htr, Hte = standardize_train_test(raw_tr, raw_te)
        ridge_acc, _ = ridge_head_accuracy(Htr, ytr, Hte, yte, reg=head_reg)
        direct_acc = float((layer_meta["direct_pred"] == yte).mean())
        layers.append(
            {
                "layer": layer_idx + 1,
                "ridge_classifier_accuracy": ridge_acc,
                "direct_prototype_accuracy": direct_acc,
                "bandwidth": layer_meta["bandwidth"],
                "prototype_norm": layer_meta["prototype_norm"],
                "feature_norm": layer_meta["feature_norm"],
            }
        )

    num_params = depth * (rff_dim + num_classes * rff_dim)
    return {
        "dataset": dataset_name,
        "depth": depth,
        "rff_dim": rff_dim,
        "head_reg": head_reg,
        "n_train": int(xtr.shape[0]),
        "n_test": int(xte.shape[0]),
        "ridge_classifier_accuracy": layers[-1]["ridge_classifier_accuracy"],
        "direct_prototype_accuracy": layers[-1]["direct_prototype_accuracy"],
        "layers": layers,
        "num_params": int(num_params),
        "note": (
            "HSIC / kernel-mean-embedding analytic network. "
            "Each layer lifts the current representation with Gaussian random Fourier features, "
            "computes per-class kernel mean embeddings in that feature space, and feeds the resulting "
            "class-similarity scores to the next layer."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Closed-form HSIC / kernel-mean-embedding network.")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100"], default="mnist")
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--rff-dim", type=int, default=RFF_DIM)
    parser.add_argument("--head-reg", type=float, default=HEAD_REG)
    parser.add_argument("--n-train", type=int, default=cfbt_cifar.N_TRAIN)
    parser.add_argument("--n-test", type=int, default=cfbt_cifar.N_TEST)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    result = run_experiment(
        dataset_name=args.dataset,
        depth=args.depth,
        rff_dim=args.rff_dim,
        head_reg=args.head_reg,
        n_train=args.n_train,
        n_test=args.n_test,
    )

    print(
        f"Closed-form HSIC network  |  dataset={result['dataset']}  |  depth={result['depth']}  |  "
        f"rff_dim={result['rff_dim']}"
    )
    print(f"ridge classifier acc  : {result['ridge_classifier_accuracy']:.4f}")
    print(f"direct prototype acc  : {result['direct_prototype_accuracy']:.4f}")
    print(f"parameter count       : {result['num_params']}")
    for layer in result["layers"]:
        print(
            f"layer {layer['layer']:>2d} | ridge={layer['ridge_classifier_accuracy']:.4f} | "
            f"direct={layer['direct_prototype_accuracy']:.4f} | bw={layer['bandwidth']:.4f}"
        )
    print(result["note"])

    if args.save_json is not None:
        output_path = resolve_json_path(args.save_json)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved json to {output_path}")


if __name__ == "__main__":
    main()
