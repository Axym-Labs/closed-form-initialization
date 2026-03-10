import argparse
import json
from pathlib import Path

import numpy as np

import closed_form_barlow_twins as cfbt
import closed_form_barlow_twins_cifar as cfbt_cifar
from project_paths import resolve_json_path


SEED = 7
WIDTH = 256
DEPTH = 3
REG = 1.0
HEAD_REG = 1e-2
UPDATE_MODES = ["residual", "stacked"]
BLOCK_TARGETS = ["embedding", "task-gradient"]
SHRINKAGE = 1.0
TARGET_CODES = ["random", "onehot", "orthogonal"]


def one_hot(y, num_classes):
    eye = np.eye(num_classes, dtype=np.float64)
    return eye[y]


def ridge_regression(X, Y, reg):
    gram = X.T @ X
    rhs = X.T @ Y
    if reg <= 0.0:
        return np.linalg.pinv(gram, rcond=1e-10) @ rhs
    return np.linalg.solve(gram + reg * np.eye(gram.shape[0], dtype=np.float64), rhs)


def normalize_train_test(Htr, Hte):
    mu = Htr.mean(axis=0, keepdims=True)
    std = Htr.std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    return (Htr - mu) / std, (Hte - mu) / std


def class_embedding(num_classes, width, seed, target_code):
    if target_code == "onehot":
        emb = np.zeros((num_classes, width), dtype=np.float64)
        emb[:, :num_classes] = np.eye(num_classes, dtype=np.float64)
        return emb
    if target_code == "orthogonal":
        rng = np.random.default_rng(seed)
        basis = rng.standard_normal((width, width))
        q, _ = np.linalg.qr(basis)
        return q[:num_classes]
    if target_code == "random":
        rng = np.random.default_rng(seed)
        emb = rng.standard_normal((num_classes, width))
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        return emb
    raise ValueError(f"Unknown target code: {target_code}")


def load_dataset(dataset_name, n_train, n_test):
    if dataset_name == "mnist":
        xtr, ytr, xte, yte = cfbt.load_mnist_numpy()
        return xtr, ytr, xte, yte
    if dataset_name in {"cifar10", "cifar100"}:
        xtr_img, ytr, xte_img, yte = cfbt_cifar.load_cifar_numpy(dataset_name, n_train=n_train, n_test=n_test, seed=SEED)
        return cfbt_cifar.images_to_flat(xtr_img), ytr, cfbt_cifar.images_to_flat(xte_img), yte
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def ridge_head_accuracy(Htr, ytr, Hte, yte, reg):
    num_classes = int(max(np.max(ytr), np.max(yte)) + 1)
    Ytr = one_hot(ytr, num_classes)
    head = ridge_regression(Htr, Ytr, reg=reg)
    logits = Hte @ head
    pred = np.argmax(logits, axis=1)
    return float((pred == yte).mean()), head


def apply_activation(X, activation):
    return cfbt.apply_activation(X, activation)


def hidden_for_next_block(G, activation):
    return apply_activation(G, activation)


def activation_derivative_from_preact(G, activation):
    if activation == "relu":
        return (G > 0.0).astype(np.float64)
    if activation == "tanh":
        H = np.tanh(G)
        return 1.0 - H * H
    if activation == "leaky-relu":
        return np.where(G >= 0.0, 1.0, cfbt.LEAKY_RELU_SLOPE)
    if activation == "identity":
        return np.ones_like(G)
    raise ValueError(f"Unknown activation: {activation}")


def maybe_normalize_hidden(Htr, Hte, enabled):
    if not enabled:
        return Htr, Hte
    return normalize_train_test(Htr, Hte)


def run_experiment(dataset_name, width, depth, activation, reg, head_reg, n_train, n_test, update_mode, block_target, shrinkage, target_code):
    Xtr, ytr, Xte, yte = load_dataset(dataset_name, n_train=n_train, n_test=n_test)
    num_classes = int(max(np.max(ytr), np.max(yte)) + 1)
    Ytr = one_hot(ytr, num_classes)

    target_embedding = class_embedding(num_classes, width, seed=SEED + 17, target_code=target_code)
    target_tr = Ytr @ target_embedding
    normalize_hidden_states = block_target != "task-gradient"

    stem = ridge_regression(Xtr, target_tr, reg=reg)
    Gtr = Xtr @ stem
    Gte = Xte @ stem
    Htr = hidden_for_next_block(Gtr, activation)
    Hte = hidden_for_next_block(Gte, activation)
    Htr, Hte = maybe_normalize_hidden(Htr, Hte, enabled=normalize_hidden_states)

    layers = []
    acc, _ = ridge_head_accuracy(Htr, ytr, Hte, yte, reg=head_reg)
    layers.append(
        {
            "layer": 1,
            "stage": "stem",
            "classifier_accuracy": acc,
            "block_target_norm": float(np.linalg.norm(target_tr - Gtr, ord="fro")),
            "update_norm": float(np.linalg.norm(stem, ord="fro")),
        }
    )

    _, head = ridge_head_accuracy(Htr, ytr, Hte, yte, reg=head_reg)
    for layer_idx in range(1, depth):
        if block_target == "embedding":
            block_fit_target = target_tr - Gtr
        elif block_target == "task-gradient":
            logits_tr = Htr @ head
            neg_grad_hidden = (Ytr - logits_tr) @ head.T
            block_fit_target = neg_grad_hidden * activation_derivative_from_preact(Gtr, activation)
        else:
            raise ValueError(f"Unknown block target: {block_target}")

        block = ridge_regression(Htr, block_fit_target, reg=reg)
        next_tr = Htr @ block
        next_te = Hte @ block
        if update_mode == "residual":
            Gtr = Gtr + shrinkage * next_tr
            Gte = Gte + shrinkage * next_te
        elif update_mode == "stacked":
            Gtr = shrinkage * next_tr
            Gte = shrinkage * next_te
        else:
            raise ValueError(f"Unknown update mode: {update_mode}")
        Htr = hidden_for_next_block(Gtr, activation)
        Hte = hidden_for_next_block(Gte, activation)
        Htr, Hte = maybe_normalize_hidden(Htr, Hte, enabled=normalize_hidden_states)
        acc, head = ridge_head_accuracy(Htr, ytr, Hte, yte, reg=head_reg)
        layers.append(
            {
                "layer": layer_idx + 1,
                "stage": "residual",
                "classifier_accuracy": acc,
                "block_target_norm": float(np.linalg.norm(block_fit_target, ord="fro")),
                "update_norm": float(np.linalg.norm(block, ord="fro")),
            }
        )

    num_params = Xtr.shape[1] * width + (depth - 1) * width * width + width * num_classes
    return {
        "dataset": dataset_name,
        "width": width,
        "depth": depth,
        "activation": activation,
        "update_mode": update_mode,
        "block_target": block_target,
        "shrinkage": shrinkage,
        "target_code": target_code,
        "reg": reg,
        "head_reg": head_reg,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "classifier_accuracy": layers[-1]["classifier_accuracy"],
        "layers": layers,
        "num_params": int(num_params),
        "note": (
            "Supervised pre-activation greedy residual network. "
            "The stem and each residual block are fit by closed-form ridge regression, "
            "the block output is linear, and the nonlinear activation is applied only "
            "when the next block consumes that state. "
            f"Block target: {block_target}. Target code: {target_code}. Update mode: {update_mode}. "
            f"Shrinkage: {shrinkage}. Hidden normalization between blocks: {normalize_hidden_states}."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Closed-form supervised residual regression network.")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100"], default="mnist")
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--activation", choices=["relu", "tanh", "leaky-relu", "identity"], default="tanh")
    parser.add_argument("--update-mode", choices=UPDATE_MODES, default="residual")
    parser.add_argument("--block-target", choices=BLOCK_TARGETS, default="embedding")
    parser.add_argument("--target-code", choices=TARGET_CODES, default="random")
    parser.add_argument("--shrinkage", type=float, default=SHRINKAGE)
    parser.add_argument("--reg", type=float, default=REG)
    parser.add_argument("--head-reg", type=float, default=HEAD_REG)
    parser.add_argument("--n-train", type=int, default=cfbt_cifar.N_TRAIN)
    parser.add_argument("--n-test", type=int, default=cfbt_cifar.N_TEST)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    result = run_experiment(
        dataset_name=args.dataset,
        width=args.width,
        depth=args.depth,
        activation=args.activation,
        update_mode=args.update_mode,
        block_target=args.block_target,
        shrinkage=args.shrinkage,
        target_code=args.target_code,
        reg=args.reg,
        head_reg=args.head_reg,
        n_train=args.n_train,
        n_test=args.n_test,
    )

    print(
        f"Closed-form supervised residual  |  dataset={result['dataset']}  |  width={result['width']}  |  "
        f"depth={result['depth']}  |  activation={result['activation']}  |  update={result['update_mode']}  |  "
        f"target={result['block_target']}  |  code={result['target_code']}  |  shrinkage={result['shrinkage']}"
    )
    print(f"classifier accuracy : {result['classifier_accuracy']:.4f}")
    print(f"parameter count     : {result['num_params']}")
    for layer in result["layers"]:
        print(
            f"layer {layer['layer']:>2d} | {layer['stage']:>8s} | acc={layer['classifier_accuracy']:.4f} | "
            f"||block-target||_F={layer['block_target_norm']:.3f} | ||update||_F={layer['update_norm']:.3f}"
        )
    print(result["note"])

    if args.save_json is not None:
        output_path = resolve_json_path(args.save_json)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved json to {output_path}")


if __name__ == "__main__":
    main()
