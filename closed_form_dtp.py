import argparse
import json
from inspect import signature
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import closed_form_barlow_twins as cfbt
import closed_form_barlow_twins_cifar as cfbt_cifar
from project_paths import resolve_json_path


SEED = 7
WIDTH = 512
DEPTH = 3
SWEEPS = 3
REG_FWD = 100.0
REG_INV = 100.0
HEAD_REG = 0.01
PROBE_MAX_ITER = 2000
TANH_CLIP = 0.95


def ridge_regression(X, Y, reg):
    gram = X.T @ X
    eye = np.eye(gram.shape[0], dtype=np.float64)
    if reg > 0.0:
        return np.linalg.solve(gram + reg * eye, X.T @ Y)
    return np.linalg.pinv(X) @ Y


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


def one_hot(y, num_classes):
    out = np.zeros((len(y), num_classes), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


def load_dataset(dataset_name, n_train=None, n_test=None):
    if dataset_name == "mnist":
        xtr, ytr, xte, yte = cfbt.load_mnist_numpy()
        return xtr, ytr, xte, yte
    if dataset_name in {"cifar10", "cifar100"}:
        n_train = cfbt_cifar.N_TRAIN if n_train is None else n_train
        n_test = cfbt_cifar.N_TEST if n_test is None else n_test
        xtr_img, ytr, xte_img, yte = cfbt_cifar.load_cifar_numpy(dataset_name, n_train=n_train, n_test=n_test, seed=SEED)
        return cfbt_cifar.images_to_flat(xtr_img), ytr, cfbt_cifar.images_to_flat(xte_img), yte
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def top_pca_init(X, width):
    cov = cfbt.covariance(X)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    basis = evecs[:, order[:width]]
    proj = X @ basis
    scale = np.mean(np.std(proj, axis=0))
    scale = 1.0 / max(scale, 1e-6)
    return basis * scale


def forward_pass(X, forward_weights, output_weight):
    activations = [X]
    preacts = []
    H = X
    for W in forward_weights:
        A = H @ W
        H = np.tanh(A)
        preacts.append(A)
        activations.append(H)
    logits = activations[-1] @ output_weight
    return {
        "activations": activations,
        "preacts": preacts,
        "logits": logits,
    }


def classifier_accuracy(logits, y):
    return float((np.argmax(logits, axis=1) == y).mean())


def clipped_artanh(X):
    return np.arctanh(np.clip(X, -TANH_CLIP, TANH_CLIP))


def initialize_forward(Xtr, width, depth):
    W0 = top_pca_init(Xtr, width)
    hidden_weights = [W0]
    for _ in range(depth - 1):
        hidden_weights.append(0.5 * np.eye(width, dtype=np.float64))
    return hidden_weights


def evaluate(features_tr, ytr, features_te, yte, logits_te):
    probe_tr, probe_te = cfbt.standardize_train_test(features_tr, features_te)
    probe_acc = fit_linear_probe(probe_tr, ytr, probe_te, yte)
    cls_acc = classifier_accuracy(logits_te, yte)
    return probe_acc, cls_acc


def run_experiment(dataset_name, width, depth, sweeps, reg_fwd, reg_inv, head_reg, n_train, n_test):
    xtr, ytr, xte, yte = load_dataset(dataset_name, n_train=n_train, n_test=n_test)
    num_classes = int(max(np.max(ytr), np.max(yte)) + 1)
    Ytr = one_hot(ytr, num_classes)

    forward_weights = initialize_forward(xtr, width=width, depth=depth)

    init_forward = forward_pass(xtr, forward_weights, np.zeros((width, num_classes), dtype=np.float64))
    output_weight = ridge_regression(init_forward["activations"][-1], Ytr, reg=head_reg)

    sweeps_out = []
    for sweep_idx in range(sweeps):
        train = forward_pass(xtr, forward_weights, output_weight)
        test = forward_pass(xte, forward_weights, output_weight)

        logits_tr = train["logits"]
        logits_te = test["logits"]
        probe_acc, cls_acc = evaluate(train["activations"][-1], ytr, test["activations"][-1], yte, logits_te)

        inverses = []
        for layer_idx in range(depth, 0, -1):
            src = train["activations"][layer_idx]
            dst = train["activations"][layer_idx - 1]
            inverses.append(ridge_regression(src, dst, reg=reg_inv))
        inverses = inverses[::-1]
        output_inverse = ridge_regression(logits_tr, train["activations"][-1], reg=reg_inv)

        targets = [None] * (depth + 1)
        targets[depth] = np.clip(
            train["activations"][-1]
            + (Ytr @ output_inverse) - (logits_tr @ output_inverse),
            -TANH_CLIP,
            TANH_CLIP,
        )
        for layer_idx in range(depth - 1, 0, -1):
            hi = train["activations"][layer_idx]
            h_next = train["activations"][layer_idx + 1]
            G_next = inverses[layer_idx]
            targets[layer_idx] = np.clip(
                hi + (targets[layer_idx + 1] @ G_next) - (h_next @ G_next),
                -TANH_CLIP,
                TANH_CLIP,
            )

        new_weights = []
        for layer_idx in range(1, depth + 1):
            inp = train["activations"][layer_idx - 1]
            tgt = clipped_artanh(targets[layer_idx])
            new_weights.append(ridge_regression(inp, tgt, reg=reg_fwd))
        forward_weights = new_weights

        train_after = forward_pass(xtr, forward_weights, output_weight)
        output_weight = ridge_regression(train_after["activations"][-1], Ytr, reg=head_reg)

        sweeps_out.append(
            {
                "sweep": sweep_idx + 1,
                "classifier_accuracy": cls_acc,
                "probe_accuracy": probe_acc,
                "train_logit_loss": float(0.5 * np.mean(np.sum((Ytr - logits_tr) ** 2, axis=1))),
                "hidden_target_norm": float(np.linalg.norm(targets[1], ord="fro")),
                "output_inverse_norm": float(np.linalg.norm(output_inverse, ord="fro")),
                "mean_inverse_norm": float(np.mean([np.linalg.norm(g, ord="fro") for g in inverses])),
                "mean_forward_norm": float(np.mean([np.linalg.norm(w, ord="fro") for w in forward_weights])),
            }
        )

    final_train = forward_pass(xtr, forward_weights, output_weight)
    final_test = forward_pass(xte, forward_weights, output_weight)
    probe_acc, cls_acc = evaluate(final_train["activations"][-1], ytr, final_test["activations"][-1], yte, final_test["logits"])

    return {
        "dataset": dataset_name,
        "width": width,
        "depth": depth,
        "sweeps": sweeps,
        "reg_fwd": reg_fwd,
        "reg_inv": reg_inv,
        "head_reg": head_reg,
        "n_train": int(xtr.shape[0]),
        "n_test": int(xte.shape[0]),
        "classifier_accuracy": cls_acc,
        "probe_accuracy": probe_acc,
        "num_params": int(
            xtr.shape[1] * width + (depth - 1) * width * width + width * num_classes
        ),
        "sweep_stats": sweeps_out,
        "note": (
            "Closed-form DTP-style surrogate. Hidden layers use tanh so local targets can be inverted with artanh. "
            "At each sweep, inverse maps and forward maps are fit by ridge regression, and hidden targets use the standard "
            "difference correction h_hat_l = h_l + g_{l+1}(h_hat_{l+1}) - g_{l+1}(h_{l+1}). This is not exact standard DTP, "
            "but a closed-form local least-squares analogue."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Closed-form DTP-style local least-squares surrogate.")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100", "all"], default="all")
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--sweeps", type=int, default=SWEEPS)
    parser.add_argument("--reg-fwd", type=float, default=REG_FWD)
    parser.add_argument("--reg-inv", type=float, default=REG_INV)
    parser.add_argument("--head-reg", type=float, default=HEAD_REG)
    parser.add_argument("--save-json", action="store_true")
    args = parser.parse_args()

    datasets_to_run = ["mnist", "cifar10", "cifar100"] if args.dataset == "all" else [args.dataset]
    for dataset_name in datasets_to_run:
        result = run_experiment(
            dataset_name=dataset_name,
            width=args.width,
            depth=args.depth,
            sweeps=args.sweeps,
            reg_fwd=args.reg_fwd,
            reg_inv=args.reg_inv,
            head_reg=args.head_reg,
            n_train=None,
            n_test=None,
        )
        print(
            f"Closed-form DTP  |  dataset={result['dataset']}  |  width={result['width']}  |  depth={result['depth']}  |  sweeps={result['sweeps']}"
        )
        print(f"classifier accuracy : {result['classifier_accuracy']:.4f}")
        print(f"probe accuracy      : {result['probe_accuracy']:.4f}")
        print(f"parameter count     : {result['num_params']}")
        for row in result["sweep_stats"]:
            print(
                f"sweep {row['sweep']:>2d} | cls={row['classifier_accuracy']:.4f} | probe={row['probe_accuracy']:.4f} | "
                f"train-logit-loss={row['train_logit_loss']:.4f} | ||h_hat1||_F={row['hidden_target_norm']:.2f} | "
                f"||g_out||_F={row['output_inverse_norm']:.2f} | mean||g||_F={row['mean_inverse_norm']:.2f} | mean||W||_F={row['mean_forward_norm']:.2f}"
            )
        if args.save_json:
            path = resolve_json_path(Path(f"closed_form_dtp_{dataset_name}_width{args.width}_depth{args.depth}_sweeps{args.sweeps}.json"))
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"saved json to {path}")


if __name__ == "__main__":
    main()
