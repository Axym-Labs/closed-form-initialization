import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import backprop_cifar_bt
import cifar_shared
import closed_form_barlow_twins as cfbt
import dual_path_residual_cifar
import greedy_barlow_twins_cifar
from experiment_settings import (
    ACTIVATION,
    ANALYTIC_AUG_REPEATS,
    ANALYTIC_MODELS,
    BACKPROP_EPOCHS,
    BT_USE_PROJECTOR,
    DATASETS,
    DEPTH,
    DUAL_MAPPING,
    GREEDY_BT_EPOCHS,
    HEAD_REG,
    LAMBDA_REG,
    LEARNED_MODELS,
    LINEAR_MODELS,
    N_TEST,
    N_TRAIN,
    SEED,
    SUITES,
    W,
)
from project_paths import default_json_path, resolve_json_path


ALL_MODELS = LINEAR_MODELS + ANALYTIC_MODELS + LEARNED_MODELS


@dataclass(frozen=True)
class ExperimentSpec:
    model: str
    dataset: str
    suite: str
    width: int
    depth: int
    n_train: int
    n_test: int
    augment_repeats: int
    dual_mapping: bool = DUAL_MAPPING
    seed: int = SEED


def one_hot(y, num_classes):
    eye = np.eye(num_classes, dtype=np.float64)
    return eye[y]


def ridge_regression(X, Y, reg):
    gram = X.T @ X
    rhs = X.T @ Y
    return np.linalg.solve(gram + reg * np.eye(gram.shape[0], dtype=np.float64), rhs)


def classify_from_logits(logits, y):
    return float((np.argmax(logits, axis=1) == y).mean())


def mlp_parameter_count(width, depth, num_classes, with_head=True, dual_mapping=False, bt_projector=False):
    hidden = depth * width * width
    total = hidden
    if with_head:
        if dual_mapping:
            total += depth * (width * num_classes + num_classes)
        else:
            total += width * num_classes + num_classes
    if bt_projector:
        total += 2 * width * width + width
    return total


def mlp_forward_macs(width, depth, num_classes, with_head=True, dual_mapping=False, bt_projector=False):
    hidden = depth * width * width
    total = hidden
    if with_head:
        total += depth * width * num_classes if dual_mapping else width * num_classes
    if bt_projector:
        total += 2 * width * width
    return total


def run_linear_regression(spec, dataset):
    ytr = dataset["ytr"]
    yte = dataset["yte"]
    num_classes = int(max(np.max(ytr), np.max(yte)) + 1)
    ytr_onehot = one_hot(ytr, num_classes)
    start = time.perf_counter()
    weights = ridge_regression(dataset["xtr"], ytr_onehot, HEAD_REG)
    logits_te = dataset["xte"] @ weights
    wall = time.perf_counter() - start
    acc = classify_from_logits(logits_te, yte)
    return {
        "model": spec.model,
        "classifier_accuracy": acc,
        "depth_metrics": [{"depth": 0, "classifier_accuracy": acc}],
        "parameter_count": int(weights.size),
        "forward_macs_per_sample": int(dataset["xtr"].shape[1] * num_classes),
        "wall_time_sec": wall,
    }


def run_pca_linear(spec, dataset):
    ytr = dataset["ytr"]
    yte = dataset["yte"]
    num_classes = int(max(np.max(ytr), np.max(yte)) + 1)
    ytr_onehot = one_hot(ytr, num_classes)
    start = time.perf_counter()
    basis, _ = dual_path_residual_cifar.fit_pca_basis(dataset["xtr"], width=spec.width)
    ztr = dataset["xtr"] @ basis
    zte = dataset["xte"] @ basis
    weights = ridge_regression(ztr, ytr_onehot, HEAD_REG)
    logits_te = zte @ weights
    wall = time.perf_counter() - start
    acc = classify_from_logits(logits_te, yte)
    return {
        "model": spec.model,
        "classifier_accuracy": acc,
        "depth_metrics": [{"depth": 1, "classifier_accuracy": acc}],
        "parameter_count": int(basis.size + weights.size),
        "forward_macs_per_sample": int(dataset["xtr"].shape[1] * spec.width + spec.width * num_classes),
        "wall_time_sec": wall,
    }


def run_random_linear(spec, dataset):
    ytr = dataset["ytr"]
    yte = dataset["yte"]
    num_classes = int(max(np.max(ytr), np.max(yte)) + 1)
    ytr_onehot = one_hot(ytr, num_classes)
    start = time.perf_counter()
    basis = dual_path_residual_cifar.fit_random_orthogonal(dataset["xtr"].shape[1], spec.width, seed=spec.seed + 401)
    ztr = dataset["xtr"] @ basis
    zte = dataset["xte"] @ basis
    weights = ridge_regression(ztr, ytr_onehot, HEAD_REG)
    logits_te = zte @ weights
    wall = time.perf_counter() - start
    acc = classify_from_logits(logits_te, yte)
    return {
        "model": spec.model,
        "classifier_accuracy": acc,
        "depth_metrics": [{"depth": 1, "classifier_accuracy": acc}],
        "parameter_count": int(basis.size + weights.size),
        "forward_macs_per_sample": int(dataset["xtr"].shape[1] * spec.width + spec.width * num_classes),
        "wall_time_sec": wall,
    }


def run_experiment(spec, dataset_cache):
    dataset = dataset_cache[(spec.dataset, spec.n_train, spec.n_test, spec.seed, spec.width)]

    if spec.model == "linear-regression":
        result = run_linear_regression(spec, dataset)
    elif spec.model == "pca":
        result = run_pca_linear(spec, dataset)
    elif spec.model == "random":
        result = run_random_linear(spec, dataset)
    elif spec.model in ANALYTIC_MODELS:
        result = dual_path_residual_cifar.run_experiment(
            dataset_name=spec.dataset,
            suite_name=spec.suite,
            layer_method=spec.model,
            width=spec.width,
            depth=spec.depth,
            head_reg=HEAD_REG,
            lambda_reg=LAMBDA_REG,
            activation=ACTIVATION,
            aug_repeats=spec.augment_repeats,
            dual_mapping=spec.dual_mapping,
        )
        result["parameter_count"] = result["total_parameter_count"]
        num_classes = int(max(np.max(dataset["ytr"]), np.max(dataset["yte"])) + 1)
        result["forward_macs_per_sample"] = mlp_forward_macs(spec.width, spec.depth, num_classes, with_head=True, dual_mapping=spec.dual_mapping)
        result["depth_metrics"] = [
            {
                "depth": layer["layer"],
                "classifier_accuracy": layer["classifier_accuracy"],
                "train_prediction_loss": layer["train_prediction_loss"],
                "test_prediction_loss": layer["test_prediction_loss"],
                "hidden_probe_accuracy": layer["hidden_probe_accuracy"],
            }
            for layer in result["layers"]
        ]
    elif spec.model == "supervised-backprop":
        result = backprop_cifar_bt.run_experiment(
            dataset_name=spec.dataset,
            suite_name=spec.suite,
            mode="supervised",
            widths=[spec.width] * spec.depth,
            activation=ACTIVATION,
            dual_mapping=spec.dual_mapping,
            bt_projector=BT_USE_PROJECTOR,
            batch_size=backprop_cifar_bt.BATCH_SIZE,
            epochs=BACKPROP_EPOCHS,
            lr=backprop_cifar_bt.LR,
            momentum=backprop_cifar_bt.MOMENTUM,
            weight_decay=backprop_cifar_bt.WEIGHT_DECAY,
            lambda_offdiag=backprop_cifar_bt.LAMBDA_OFFDIAG,
            n_train=spec.n_train,
            n_test=spec.n_test,
        )
        num_classes = int(max(np.max(dataset["ytr"]), np.max(dataset["yte"])) + 1)
        result["parameter_count"] = mlp_parameter_count(spec.width, spec.depth, num_classes, with_head=True, dual_mapping=spec.dual_mapping)
        result["forward_macs_per_sample"] = mlp_forward_macs(spec.width, spec.depth, num_classes, with_head=True, dual_mapping=spec.dual_mapping)
        result["classifier_accuracy"] = result["classifier_accuracy"]
    elif spec.model == "barlow-twins-backprop":
        result = backprop_cifar_bt.run_experiment(
            dataset_name=spec.dataset,
            suite_name=spec.suite,
            mode="barlow-twins",
            widths=[spec.width] * spec.depth,
            activation=ACTIVATION,
            dual_mapping=spec.dual_mapping,
            bt_projector=BT_USE_PROJECTOR,
            batch_size=backprop_cifar_bt.BATCH_SIZE,
            epochs=BACKPROP_EPOCHS,
            lr=backprop_cifar_bt.LR,
            momentum=backprop_cifar_bt.MOMENTUM,
            weight_decay=backprop_cifar_bt.WEIGHT_DECAY,
            lambda_offdiag=backprop_cifar_bt.LAMBDA_OFFDIAG,
            n_train=spec.n_train,
            n_test=spec.n_test,
        )
        num_classes = int(max(np.max(dataset["ytr"]), np.max(dataset["yte"])) + 1)
        result["parameter_count"] = mlp_parameter_count(spec.width, spec.depth, num_classes, with_head=False, bt_projector=BT_USE_PROJECTOR)
        result["forward_macs_per_sample"] = mlp_forward_macs(spec.width, spec.depth, num_classes, with_head=False, bt_projector=BT_USE_PROJECTOR)
    elif spec.model == "barlow-twins-greedy-post":
        result = greedy_barlow_twins_cifar.run_variant(
            dataset_name=spec.dataset,
            suite_name=spec.suite,
            loss_position="post",
            widths=[spec.width] * spec.depth,
            batch_size=greedy_barlow_twins_cifar.BATCH_SIZE,
            epochs=GREEDY_BT_EPOCHS,
            lr=greedy_barlow_twins_cifar.LR,
            momentum=greedy_barlow_twins_cifar.MOMENTUM,
            weight_decay=greedy_barlow_twins_cifar.WEIGHT_DECAY,
            lambda_offdiag=greedy_barlow_twins_cifar.LAMBDA_OFFDIAG,
            activation=ACTIVATION,
            n_train=spec.n_train,
            n_test=spec.n_test,
        )
        num_classes = int(max(np.max(dataset["ytr"]), np.max(dataset["yte"])) + 1)
        result["parameter_count"] = mlp_parameter_count(spec.width, spec.depth, num_classes, with_head=False)
        result["forward_macs_per_sample"] = mlp_forward_macs(spec.width, spec.depth, num_classes, with_head=False)
    else:
        raise ValueError(f"Unknown model: {spec.model}")

    result["model"] = spec.model
    result["spec"] = asdict(spec)
    primary = result.get("classifier_accuracy")
    if primary is None:
        primary = result.get("probe_accuracy")
    result["primary_score"] = primary
    return result


def run_experiments(specs):
    deduped = list(dict.fromkeys(specs))
    dataset_cache = {}
    for spec in deduped:
        key = (spec.dataset, spec.n_train, spec.n_test, spec.seed, spec.width)
        if key not in dataset_cache:
            dataset_cache[key] = cifar_shared.load_cifar_numpy(spec.dataset, spec.n_train, spec.n_test, spec.seed, spec.width)

    results = []
    for spec in deduped:
        results.append(run_experiment(spec, dataset_cache))
    return results


def default_specs(models, datasets, suites, width, depth, n_train, n_test, augment_repeats):
    specs = []
    for model in models:
        if model in {"linear-regression", "pca", "random"}:
            for dataset in datasets:
                specs.append(ExperimentSpec(model=model, dataset=dataset, suite=suites[0], width=width, depth=1, n_train=n_train, n_test=n_test, augment_repeats=1))
        else:
            for dataset in datasets:
                for suite in suites:
                    specs.append(ExperimentSpec(model=model, dataset=dataset, suite=suite, width=width, depth=depth, n_train=n_train, n_test=n_test, augment_repeats=augment_repeats))
    return specs


def main():
    parser = argparse.ArgumentParser(description="Unified CIFAR experiment sweep for retained models.")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS + ["all"], default=["all"])
    parser.add_argument("--datasets", nargs="+", choices=DATASETS + ["all"], default=["all"])
    parser.add_argument("--suites", nargs="+", choices=SUITES + ["all"], default=["all"])
    parser.add_argument("--width", type=int, default=W)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--augment-repeats", type=int, default=ANALYTIC_AUG_REPEATS)
    parser.add_argument("--dual-mapping", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    models = ALL_MODELS if args.models == ["all"] else args.models
    datasets = DATASETS if args.datasets == ["all"] else args.datasets
    suites = SUITES if args.suites == ["all"] else args.suites

    specs = default_specs(models, datasets, suites, args.width, args.depth, args.n_train, args.n_test, args.augment_repeats)
    if args.dual_mapping:
        specs = [ExperimentSpec(**{**asdict(spec), "dual_mapping": True}) for spec in specs]
    results = run_experiments(specs)

    payload = {
        "config": {
            "models": models,
            "datasets": datasets,
            "suites": suites,
            "width": args.width,
            "depth": args.depth,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "augment_repeats": args.augment_repeats,
        },
        "results": results,
    }

    json_path = default_json_path("cifar_model_sweep.json") if args.json_out is None else resolve_json_path(args.json_out)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved json to {json_path}")
    for result in results:
        score = result["primary_score"]
        print(f"{result['spec']['model']:24s} {result['spec']['dataset']:8s} {result['spec']['suite']:12s} score={score}")


if __name__ == "__main__":
    main()
