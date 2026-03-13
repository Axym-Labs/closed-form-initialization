import argparse
import json
from inspect import signature
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import closed_form_barlow_twins as cfbt
import closed_form_barlow_twins_cifar as cfbt_cifar
import closed_form_supervised_residual as cfsr
from project_paths import resolve_json_path


SEED = 7
WIDTH = 512
DEPTH = 3
SWEEPS = 3
REG_FWD = 100.0
HEAD_REG = 0.01
STEP_SIZE = 1.0
PROBE_MAX_ITER = 2000


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


def top_pca_init(X, width):
    cov = cfbt.covariance(X)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    basis = evecs[:, order[:width]]
    proj = X @ basis
    scale = 1.0 / max(np.mean(np.std(proj, axis=0)), 1e-6)
    return basis * scale


def initialize_forward(Xtr, width, depth):
    W0 = top_pca_init(Xtr, width)
    weights = [W0]
    for _ in range(depth - 1):
        weights.append(0.5 * np.eye(width, dtype=np.float64))
    return weights


def initialize_feedback(num_classes, width, depth):
    rng = np.random.default_rng(SEED + 29)
    feedback = []
    for _ in range(depth):
        B = rng.standard_normal((num_classes, width))
        B /= max(np.linalg.norm(B, ord="fro"), 1e-12)
        feedback.append(B)
    return feedback


def forward_pass(X, forward_weights, output_weight, activation):
    activations = [X]
    preacts = []
    H = X
    for W in forward_weights:
        A = H @ W
        H = cfsr.apply_activation(A, activation)
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


def evaluate(final_tr, ytr, final_te, yte, logits_te):
    probe_tr, probe_te = cfbt.standardize_train_test(final_tr, final_te)
    probe_acc = fit_linear_probe(probe_tr, ytr, probe_te, yte)
    cls_acc = classifier_accuracy(logits_te, yte)
    return probe_acc, cls_acc


def run_experiment(dataset_name, width, depth, sweeps, reg_fwd, head_reg, step_size, activation, n_train, n_test):
    if dataset_name == "mnist":
        n_train = cfbt.N_TRAIN if n_train is None else n_train
        n_test = cfbt.N_TEST if n_test is None else n_test
    else:
        n_train = cfbt_cifar.N_TRAIN if n_train is None else n_train
        n_test = cfbt_cifar.N_TEST if n_test is None else n_test
    xtr, ytr, xte, yte = cfsr.load_dataset(dataset_name, n_train=n_train, n_test=n_test)
    num_classes = int(max(np.max(ytr), np.max(yte)) + 1)
    Ytr = one_hot(ytr, num_classes)

    forward_weights = initialize_forward(xtr, width, depth)
    feedback = initialize_feedback(num_classes, width, depth)

    init_forward = forward_pass(xtr, forward_weights, np.zeros((width, num_classes), dtype=np.float64), activation)
    output_weight = cfsr.ridge_regression(init_forward["activations"][-1], Ytr, reg=head_reg)

    sweep_stats = []
    for sweep_idx in range(sweeps):
        train = forward_pass(xtr, forward_weights, output_weight, activation)
        test = forward_pass(xte, forward_weights, output_weight, activation)

        logits_tr = train["logits"]
        logits_te = test["logits"]
        error = Ytr - logits_tr
        probe_acc, cls_acc = evaluate(train["activations"][-1], ytr, test["activations"][-1], yte, logits_te)

        new_weights = []
        delta_norms = []
        weight_norms = []
        for layer_idx in range(depth):
            H_prev = train["activations"][layer_idx]
            A_l = train["preacts"][layer_idx]
            local_delta = (error @ feedback[layer_idx]) * cfsr.activation_derivative_from_preact(A_l, activation)
            target_preact = A_l + step_size * local_delta
            W_new = cfsr.ridge_regression(H_prev, target_preact, reg=reg_fwd)
            new_weights.append(W_new)
            delta_norms.append(float(np.linalg.norm(local_delta, ord="fro")))
            weight_norms.append(float(np.linalg.norm(W_new, ord="fro")))
        forward_weights = new_weights

        train_after = forward_pass(xtr, forward_weights, output_weight, activation)
        output_weight = cfsr.ridge_regression(train_after["activations"][-1], Ytr, reg=head_reg)

        sweep_stats.append(
            {
                "sweep": sweep_idx + 1,
                "classifier_accuracy": cls_acc,
                "probe_accuracy": probe_acc,
                "train_logit_loss": cfsr.squared_prediction_loss(logits_tr, Ytr),
                "mean_delta_norm": float(np.mean(delta_norms)),
                "max_delta_norm": float(np.max(delta_norms)),
                "mean_weight_norm": float(np.mean(weight_norms)),
            }
        )

    final_train = forward_pass(xtr, forward_weights, output_weight, activation)
    final_test = forward_pass(xte, forward_weights, output_weight, activation)
    probe_acc, cls_acc = evaluate(final_train["activations"][-1], ytr, final_test["activations"][-1], yte, final_test["logits"])

    num_params = xtr.shape[1] * width + (depth - 1) * width * width + width * num_classes
    return {
        "dataset": dataset_name,
        "width": width,
        "depth": depth,
        "sweeps": sweeps,
        "activation": activation,
        "reg_fwd": reg_fwd,
        "head_reg": head_reg,
        "step_size": step_size,
        "n_train": int(xtr.shape[0]),
        "n_test": int(xte.shape[0]),
        "classifier_accuracy": cls_acc,
        "probe_accuracy": probe_acc,
        "num_params": int(num_params),
        "sweep_stats": sweep_stats,
        "note": (
            "Closed-form DFA-style surrogate. Output error is projected to each hidden layer through fixed random feedback matrices, "
            "and each forward layer is refit by ridge regression to a one-step preactivation target A_l + eta * delta_l. "
            "This is a closed-form local least-squares analogue of DFA, not exact standard DFA."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Closed-form DFA-style local least-squares surrogate.")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100", "all"], default="all")
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--sweeps", type=int, default=SWEEPS)
    parser.add_argument("--reg-fwd", type=float, default=REG_FWD)
    parser.add_argument("--head-reg", type=float, default=HEAD_REG)
    parser.add_argument("--step-size", type=float, default=STEP_SIZE)
    parser.add_argument("--activation", choices=["relu", "tanh", "leaky-relu", "identity"], default="relu")
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
            head_reg=args.head_reg,
            step_size=args.step_size,
            activation=args.activation,
            n_train=None,
            n_test=None,
        )
        print(
            f"Closed-form DFA  |  dataset={result['dataset']}  |  width={result['width']}  |  depth={result['depth']}  |  sweeps={result['sweeps']}  |  activation={result['activation']}"
        )
        print(f"classifier accuracy : {result['classifier_accuracy']:.4f}")
        print(f"probe accuracy      : {result['probe_accuracy']:.4f}")
        print(f"parameter count     : {result['num_params']}")
        for row in result["sweep_stats"]:
            print(
                f"sweep {row['sweep']:>2d} | cls={row['classifier_accuracy']:.4f} | probe={row['probe_accuracy']:.4f} | "
                f"train-logit-loss={row['train_logit_loss']:.4f} | mean||delta||_F={row['mean_delta_norm']:.2f} | "
                f"max||delta||_F={row['max_delta_norm']:.2f} | mean||W||_F={row['mean_weight_norm']:.2f}"
            )
        if args.save_json:
            path = resolve_json_path(Path(f"closed_form_dfa_{dataset_name}_width{args.width}_depth{args.depth}_sweeps{args.sweeps}_{args.activation}.json"))
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"saved json to {path}")


if __name__ == "__main__":
    main()
