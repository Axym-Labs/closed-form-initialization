import argparse
import json
from pathlib import Path

import numpy as np

import closed_form_barlow_twins as cfbt
import closed_form_barlow_twins_cifar as cfbt_cifar
from project_paths import resolve_json_path


SEED = 7
DEPTH = 3
REG = 1.0
HEAD_REG = 1e-2
ACTIVATION = "relu"


def one_hot(y, num_classes):
    eye = np.eye(num_classes, dtype=np.float64)
    return eye[y]


def ridge_regression(X, Y, reg):
    gram = X.T @ X
    rhs = X.T @ Y
    if reg <= 0.0:
        return np.linalg.pinv(gram, rcond=1e-10) @ rhs
    return np.linalg.solve(gram + reg * np.eye(gram.shape[0], dtype=np.float64), rhs)


def standardize_train_test(Xtr, Xte):
    mu = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    return (Xtr - mu) / std, (Xte - mu) / std


def load_dataset(dataset_name, n_train, n_test):
    if dataset_name == "mnist":
        return cfbt.load_mnist_numpy()
    if dataset_name in {"cifar10", "cifar100"}:
        xtr_img, ytr, xte_img, yte = cfbt_cifar.load_cifar_numpy(
            dataset_name, n_train=n_train, n_test=n_test, seed=SEED
        )
        return cfbt_cifar.images_to_flat(xtr_img), ytr, cfbt_cifar.images_to_flat(xte_img), yte
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def apply_activation(X, activation):
    return cfbt.apply_activation(X, activation)


def ridge_head_accuracy(Xtr, ytr, Xte, yte, reg):
    num_classes = int(max(np.max(ytr), np.max(yte)) + 1)
    Ytr = one_hot(ytr, num_classes)
    head = ridge_regression(Xtr, Ytr, reg=reg)
    logits = Xte @ head
    pred = np.argmax(logits, axis=1)
    return float((pred == yte).mean()), head


def run_experiment(dataset_name, depth, reg, head_reg, activation, n_train, n_test):
    Xtr, ytr, Xte, yte = load_dataset(dataset_name, n_train=n_train, n_test=n_test)
    Xtr, Xte = standardize_train_test(Xtr, Xte)
    num_classes = int(max(np.max(ytr), np.max(yte)) + 1)
    Ytr = one_hot(ytr, num_classes)

    Htr = Xtr
    Hte = Xte
    layers = []
    feature_dims = [Htr.shape[1]]

    for layer_idx in range(depth):
        W = ridge_regression(Htr, Ytr, reg=reg)
        Ptr = Htr @ W
        Pte = Hte @ W
        Qtr = apply_activation(Ptr, activation)
        Qte = apply_activation(Pte, activation)
        Htr = np.concatenate([Htr, Qtr], axis=1)
        Hte = np.concatenate([Hte, Qte], axis=1)
        Htr, Hte = standardize_train_test(Htr, Hte)
        acc, _ = ridge_head_accuracy(Htr, ytr, Hte, yte, reg=head_reg)
        layers.append(
            {
                "layer": layer_idx + 1,
                "classifier_accuracy": acc,
                "feature_dim": int(Htr.shape[1]),
                "predictor_norm": float(np.linalg.norm(W, ord="fro")),
                "relearned_norm": float(np.linalg.norm(Qtr, ord="fro")),
            }
        )
        feature_dims.append(Htr.shape[1])

    num_params = sum(feature_dims[i] * num_classes for i in range(depth))
    return {
        "dataset": dataset_name,
        "depth": depth,
        "reg": reg,
        "head_reg": head_reg,
        "activation": activation,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "classifier_accuracy": layers[-1]["classifier_accuracy"],
        "layers": layers,
        "num_params": int(num_params),
        "note": (
            "Deep Analytic Network style baseline. Each layer solves a closed-form ridge regression "
            "to the labels, passes the prediction features through a nonlinearity, concatenates them "
            "to the running representation, and trains the next layer on the accumulated features."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="DAN-style closed-form analytic network.")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100"], default="mnist")
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--reg", type=float, default=REG)
    parser.add_argument("--head-reg", type=float, default=HEAD_REG)
    parser.add_argument("--activation", choices=["relu", "tanh", "leaky-relu", "identity"], default=ACTIVATION)
    parser.add_argument("--n-train", type=int, default=cfbt_cifar.N_TRAIN)
    parser.add_argument("--n-test", type=int, default=cfbt_cifar.N_TEST)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    result = run_experiment(
        dataset_name=args.dataset,
        depth=args.depth,
        reg=args.reg,
        head_reg=args.head_reg,
        activation=args.activation,
        n_train=args.n_train,
        n_test=args.n_test,
    )

    print(
        f"Deep analytic network  |  dataset={result['dataset']}  |  depth={result['depth']}  |  "
        f"activation={result['activation']}  |  reg={result['reg']}"
    )
    print(f"classifier accuracy : {result['classifier_accuracy']:.4f}")
    print(f"parameter count     : {result['num_params']}")
    for layer in result["layers"]:
        print(
            f"layer {layer['layer']:>2d} | acc={layer['classifier_accuracy']:.4f} | "
            f"feat_dim={layer['feature_dim']} | ||W||_F={layer['predictor_norm']:.3f}"
        )
    print(result["note"])

    if args.save_json is not None:
        output_path = resolve_json_path(args.save_json)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved json to {output_path}")


if __name__ == "__main__":
    main()
