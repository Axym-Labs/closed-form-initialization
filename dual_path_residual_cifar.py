import argparse
import json
from pathlib import Path

import numpy as np

import closed_form_barlow_twins as cfbt
import closed_form_barlow_twins_cifar as cfbt_cifar
from closed_form_supervised_residual import one_hot, ridge_regression, squared_prediction_loss
from project_paths import default_json_path, resolve_json_path


SEED = 7
N_TRAIN = 6000
N_TEST = 1000
DEPTH = 3
WIDTH = 512
HEAD_REG = 100.0
LAMBDA_REG = 1.0
ACTIVATION = "relu"

LAYER_METHODS = ["pca", "random"] + cfbt_cifar.LAYER_METHODS
SUITES = ["single-translation", "block-masking"]
DATASETS = ["cifar10", "cifar100"]


def fit_pca_basis(X, width):
    centered = X - X.mean(axis=0, keepdims=True)
    sigma = cfbt.covariance(centered)
    evals, evecs = np.linalg.eigh(sigma)
    order = np.argsort(evals)[::-1]
    k = min(width, X.shape[1])
    return evecs[:, order[:k]], evals[order[:k]]


def fit_random_orthogonal(dim, width, seed):
    rng = np.random.default_rng(seed)
    k = min(width, dim)
    q, _ = np.linalg.qr(rng.standard_normal((dim, k)))
    return q[:, :k]


def topk_columns(matrix, scores, width, descending=True):
    order = np.argsort(scores)
    if descending:
        order = order[::-1]
    keep = order[: min(width, matrix.shape[1])]
    return matrix[:, keep], np.asarray(scores)[keep]


def fit_activation_transforms(method_name, base_tr, view1_tr, view2_tr, width, lambda_reg, layer_seed):
    current_dim = base_tr.shape[1]
    k = min(width, current_dim)

    if method_name == "pca":
        transform, top_evals = fit_pca_basis(base_tr, width=k)
        return {
            "transform_base": transform,
            "transform_view1": transform,
            "transform_view2": transform,
            "method_stats": {
                "method": method_name,
                "rank": int(transform.shape[1]),
                "top_score": float(np.max(top_evals)),
                "bottom_score": float(np.min(top_evals)),
            },
        }

    if method_name == "random":
        transform = fit_random_orthogonal(current_dim, width=k, seed=layer_seed)
        return {
            "transform_base": transform,
            "transform_view1": transform,
            "transform_view2": transform,
            "method_stats": {
                "method": method_name,
                "rank": int(transform.shape[1]),
            },
        }

    stats = cfbt.compute_paired_stats(view1_tr, view2_tr)
    sigma_bar = stats["sigma_bar"]

    if method_name in {"closed-form-barlow", "iterref-old", "iterref-symcca", "residual-barlow", "whitened-shared-pca"}:
        _, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    if method_name == "closed-form-barlow":
        delta = stats["delta"]
        m_matrix = sigma_inv_sqrt @ delta @ sigma_inv_sqrt
        m_matrix = 0.5 * (m_matrix + m_matrix.T)
        eigvals, eigvecs = np.linalg.eigh(m_matrix)
        gains = lambda_reg / (np.maximum(eigvals, 0.0) + lambda_reg)
        modes, kept_gains = topk_columns(eigvecs, gains, width=k, descending=True)
        transform = sigma_inv_sqrt @ (modes * kept_gains)
        return {
            "transform_base": transform,
            "transform_view1": transform,
            "transform_view2": transform,
            "method_stats": {
                "method": method_name,
                "rank": int(transform.shape[1]),
                "top_score": float(np.max(kept_gains)),
                "bottom_score": float(np.min(kept_gains)),
                "max_whitened_delta": float(np.max(eigvals)),
                "min_whitened_delta": float(np.min(eigvals)),
                "rank_variant": "spectral-coordinates",
            },
        }

    if method_name == "iterref-old":
        delta = stats["delta"]
        m_matrix = sigma_inv_sqrt @ delta @ sigma_inv_sqrt
        m_matrix = 0.5 * (m_matrix + m_matrix.T)
        eigvals, eigvecs = np.linalg.eigh(m_matrix)
        residual_gains = (2.0 * eigvals * eigvals + lambda_reg) / (4.0 * eigvals * eigvals + lambda_reg)
        modes, kept_gains = topk_columns(eigvecs, residual_gains, width=k, descending=True)
        transform = sigma_inv_sqrt @ (modes * kept_gains)
        return {
            "transform_base": transform,
            "transform_view1": transform,
            "transform_view2": transform,
            "method_stats": {
                "method": method_name,
                "rank": int(transform.shape[1]),
                "top_score": float(np.max(kept_gains)),
                "bottom_score": float(np.min(kept_gains)),
                "max_whitened_delta": float(np.max(eigvals)),
                "min_whitened_delta": float(np.min(eigvals)),
                "rank_variant": "spectral-coordinates",
            },
        }

    if method_name in {"iterref-symcca", "residual-barlow"}:
        shared = stats["shared"]
        s_matrix = sigma_inv_sqrt @ shared @ sigma_inv_sqrt
        s_matrix = 0.5 * (s_matrix + s_matrix.T)
        eigvals, eigvecs = np.linalg.eigh(s_matrix)
        residual_steps = (2.0 * eigvals * (1.0 - eigvals)) / (4.0 * eigvals * eigvals + lambda_reg)
        residual_gains = 1.0 + residual_steps
        modes, kept_gains = topk_columns(eigvecs, residual_gains, width=k, descending=True)
        transform = sigma_inv_sqrt @ (modes * kept_gains)
        return {
            "transform_base": transform,
            "transform_view1": transform,
            "transform_view2": transform,
            "method_stats": {
                "method": method_name,
                "rank": int(transform.shape[1]),
                "top_score": float(np.max(kept_gains)),
                "bottom_score": float(np.min(kept_gains)),
                "max_shared_eigenvalue": float(np.max(eigvals)),
                "min_shared_eigenvalue": float(np.min(eigvals)),
                "rank_variant": "spectral-coordinates",
            },
        }

    if method_name == "whitened-shared-pca":
        shared = stats["shared"]
        s_matrix = sigma_inv_sqrt @ shared @ sigma_inv_sqrt
        s_matrix = 0.5 * (s_matrix + s_matrix.T)
        eigvals, eigvecs = np.linalg.eigh(s_matrix)
        modes, kept_scores = topk_columns(eigvecs, eigvals, width=k, descending=True)
        transform = sigma_inv_sqrt @ modes
        return {
            "transform_base": transform,
            "transform_view1": transform,
            "transform_view2": transform,
            "method_stats": {
                "method": method_name,
                "rank": int(transform.shape[1]),
                "top_score": float(np.max(kept_scores)),
                "bottom_score": float(np.min(kept_scores)),
                "rank_variant": "top-eigenspace",
            },
        }

    if method_name == "paper-cca-shared":
        sigma_reg = sigma_bar + cfbt.REG_EPS * np.eye(current_dim, dtype=np.float64)
        shared = 0.5 * (stats["shared"] + stats["shared"].T)
        eigvals, eigvecs = cfbt.eigh(shared, sigma_reg)
        modes, kept_scores = topk_columns(eigvecs, eigvals, width=k, descending=True)
        return {
            "transform_base": modes,
            "transform_view1": modes,
            "transform_view2": modes,
            "method_stats": {
                "method": method_name,
                "rank": int(modes.shape[1]),
                "top_score": float(np.max(kept_scores)),
                "bottom_score": float(np.min(kept_scores)),
                "rank_variant": "generalized-eigenspace",
            },
        }

    if method_name == "paper-cca":
        model = cfbt.fit_paper_cca_layer(stats)
        transform_a = model["transform_a"][:, :k]
        transform_b = model["transform_b"][:, :k]
        transform_base = 0.5 * (transform_a + transform_b)
        canonical_corrs = model["canonical_correlations"][:k]
        return {
            "transform_base": transform_base,
            "transform_view1": transform_a,
            "transform_view2": transform_b,
            "method_stats": {
                "method": method_name,
                "rank": int(transform_base.shape[1]),
                "top_score": float(np.max(canonical_corrs)),
                "bottom_score": float(np.min(canonical_corrs)),
                "rank_variant": "canonical-subspace",
            },
        }

    return {
        "transform_base": fit_random_orthogonal(current_dim, width=k, seed=layer_seed),
        "transform_view1": fit_random_orthogonal(current_dim, width=k, seed=layer_seed + 1),
        "transform_view2": fit_random_orthogonal(current_dim, width=k, seed=layer_seed + 2),
        "method_stats": {"method": method_name, "rank": k, "rank_variant": "fallback-random"},
    }


def hidden_probe_accuracy(Htr, ytr, Hte, yte):
    ztr, zte = cfbt.standardize_train_test(Htr, Hte)
    return cfbt_cifar.fit_linear_probe(ztr, ytr, zte, yte)


def run_experiment(dataset_name, suite_name, layer_method, width, depth, head_reg, lambda_reg, activation):
    xtr_img, ytr, xte_img, yte = cfbt_cifar.load_cifar_numpy(
        dataset_name,
        n_train=N_TRAIN,
        n_test=N_TEST,
        seed=SEED,
    )
    base_tr = cfbt_cifar.images_to_flat(xtr_img)
    base_te = cfbt_cifar.images_to_flat(xte_img)
    view1_tr, view2_tr = cfbt_cifar.sample_pair_views(xtr_img, suite_name, seed=SEED + 13)
    view1_te, view2_te = cfbt_cifar.sample_pair_views(xte_img, suite_name, seed=SEED + 31)

    train_arrays, test_arrays = cfbt.normalize_hidden(
        [base_tr, view1_tr, view2_tr],
        [base_te, view1_te, view2_te],
    )
    base_tr, view1_tr, view2_tr = train_arrays
    base_te, view1_te, view2_te = test_arrays

    num_classes = int(max(np.max(ytr), np.max(yte)) + 1)
    ytr_onehot = one_hot(ytr, num_classes)
    yte_onehot = one_hot(yte, num_classes)
    yhat_tr = np.zeros_like(ytr_onehot)
    yhat_te = np.zeros_like(yte_onehot)

    layers = []
    output_param_count = 0
    activation_param_count = 0

    for layer_idx in range(depth):
        output_map = ridge_regression(base_tr, ytr_onehot - yhat_tr, reg=head_reg)
        output_param_count += int(output_map.size)

        yhat_tr = yhat_tr + base_tr @ output_map
        yhat_te = yhat_te + base_te @ output_map

        layer_acc = float((np.argmax(yhat_te, axis=1) == yte).mean())
        layer_train_loss = squared_prediction_loss(yhat_tr, ytr_onehot)
        layer_test_loss = squared_prediction_loss(yhat_te, yte_onehot)
        layer_hidden_probe = hidden_probe_accuracy(base_tr, ytr, base_te, yte)

        fitted = fit_activation_transforms(
            method_name=layer_method,
            base_tr=base_tr,
            view1_tr=view1_tr,
            view2_tr=view2_tr,
            width=width,
            lambda_reg=lambda_reg,
            layer_seed=SEED + 97 * (layer_idx + 1),
        )
        activation_param_count += int(fitted["transform_base"].size)

        base_tr = cfbt.apply_layer(base_tr, fitted["transform_base"], activation=activation)
        base_te = cfbt.apply_layer(base_te, fitted["transform_base"], activation=activation)
        view1_tr = cfbt.apply_layer(view1_tr, fitted["transform_view1"], activation=activation)
        view2_tr = cfbt.apply_layer(view2_tr, fitted["transform_view2"], activation=activation)
        view1_te = cfbt.apply_layer(view1_te, fitted["transform_view1"], activation=activation)
        view2_te = cfbt.apply_layer(view2_te, fitted["transform_view2"], activation=activation)

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
                "classifier_accuracy": layer_acc,
                "train_prediction_loss": layer_train_loss,
                "test_prediction_loss": layer_test_loss,
                "hidden_probe_accuracy": layer_hidden_probe,
                "post_delta_trace": float(np.trace(post_stats["delta"])),
                "post_shared_trace": float(np.trace(post_stats["shared"])),
                "base_trace": float(np.trace(cfbt.covariance(base_tr))),
                **fitted["method_stats"],
            }
        )

    final_hidden_probe = hidden_probe_accuracy(base_tr, ytr, base_te, yte)
    results = {
        "dataset": dataset_name,
        "suite": suite_name,
        "layer_method": layer_method,
        "activation": activation,
        "width": width,
        "depth": depth,
        "head_reg": head_reg,
        "lambda_reg": lambda_reg,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "classifier_accuracy": float((np.argmax(yhat_te, axis=1) == yte).mean()),
        "train_prediction_loss": squared_prediction_loss(yhat_tr, ytr_onehot),
        "test_prediction_loss": squared_prediction_loss(yhat_te, yte_onehot),
        "final_hidden_probe_accuracy": final_hidden_probe,
        "layers": layers,
        "output_param_count": output_param_count,
        "activation_param_count": activation_param_count,
        "total_parameter_count": output_param_count + activation_param_count,
    }
    return results


def run_sweep(datasets, suites, methods, width, depth, head_reg, lambda_reg, activation):
    all_results = []
    for dataset_name in datasets:
        for suite_name in suites:
            for method_name in methods:
                all_results.append(
                    run_experiment(
                        dataset_name=dataset_name,
                        suite_name=suite_name,
                        layer_method=method_name,
                        width=width,
                        depth=depth,
                        head_reg=head_reg,
                        lambda_reg=lambda_reg,
                        activation=activation,
                    )
                )
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    parser.add_argument("--suite", choices=SUITES + ["all"], default="all")
    parser.add_argument("--method", choices=LAYER_METHODS + ["all"], default="all")
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--head-reg", type=float, default=HEAD_REG)
    parser.add_argument("--lambda-reg", type=float, default=LAMBDA_REG)
    parser.add_argument("--activation", type=str, default=ACTIVATION)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    suites = SUITES if args.suite == "all" else [args.suite]
    methods = LAYER_METHODS if args.method == "all" else [args.method]

    results = run_sweep(
        datasets=datasets,
        suites=suites,
        methods=methods,
        width=args.width,
        depth=args.depth,
        head_reg=args.head_reg,
        lambda_reg=args.lambda_reg,
        activation=args.activation,
    )

    summary = {
        "config": {
            "datasets": datasets,
            "suites": suites,
            "methods": methods,
            "width": args.width,
            "depth": args.depth,
            "head_reg": args.head_reg,
            "lambda_reg": args.lambda_reg,
            "activation": args.activation,
        },
        "results": results,
    }

    json_name = f"dual_path_residual_cifar_width{args.width}_sweep.json"
    json_path = default_json_path(json_name) if args.json_out is None else resolve_json_path(args.json_out)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved results to {json_path}")
    for row in results:
        print(
            f"{row['dataset']:8s}  {row['suite']:18s}  {row['layer_method']:20s}  "
            f"acc={row['classifier_accuracy']:.4f}  hidden_probe={row['final_hidden_probe_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
