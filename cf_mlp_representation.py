import argparse
import gc
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from cf_mlp_scalability import SweepPoint, accuracy_from_logits, load_point_data, write_jsonl
from cf_mlp_scalability_gpu import (
    accuracy_from_logits_torch,
    fit_cf_transform_torch,
    fit_whitening_transform_torch,
    lambda_from_invariance_strength,
    normalize_hidden_with_stats_torch,
    ridge_regression_torch,
)


def one_hot_np(y, num_classes):
    eye = np.eye(num_classes, dtype=np.float32)
    return eye[y]


def softmax_ce_np(logits, y):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(np.clip(shifted, -60.0, 60.0))
    probs = exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), 1e-12)
    return float(-np.log(np.maximum(probs[np.arange(y.shape[0]), y], 1e-12)).mean())


def ridge_map_np(x, y, reg, fit_bias=True):
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    if fit_bias:
        x64 = np.concatenate([x64, np.ones((x64.shape[0], 1), dtype=np.float64)], axis=1)
    gram = x64.T @ x64
    penalty = reg * np.eye(gram.shape[0], dtype=np.float64)
    if fit_bias:
        penalty[-1, -1] = 0.0
    rhs = x64.T @ y64
    return np.linalg.solve(gram + penalty, rhs)


def apply_map_np(x, weight, fit_bias=True):
    x64 = np.asarray(x, dtype=np.float64)
    if fit_bias:
        return x64 @ weight[:-1] + weight[-1]
    return x64 @ weight


def standardize_pair(train, test, eps=1e-6):
    mean = train.mean(axis=0, keepdims=True)
    std = np.maximum(train.std(axis=0, keepdims=True), eps)
    return ((train - mean) / std).astype(np.float32), ((test - mean) / std).astype(np.float32)


def tensors_from_arrays(arrays, device):
    xtr, ytr, xte, yte, view1_tr, view2_tr, view1_te, view2_te = arrays

    def x_tensor(arr):
        return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device, non_blocking=False)

    return (
        x_tensor(xtr),
        torch.from_numpy(ytr.astype(np.int64)).to(device),
        x_tensor(xte),
        torch.from_numpy(yte.astype(np.int64)).to(device),
        x_tensor(view1_tr),
        x_tensor(view2_tr),
        x_tensor(view1_te),
        x_tensor(view2_te),
    )


def apply_activation_torch(x, activation, alpha=0.0):
    if activation == "relu":
        return torch.relu(x)
    if activation == "leaky_relu":
        return F.leaky_relu(x, negative_slope=float(alpha))
    if activation == "gelu":
        return F.gelu(x)
    if activation == "leaky_gelu":
        return F.gelu(x) + float(alpha) * torch.minimum(x, torch.zeros((), dtype=x.dtype, device=x.device))
    if activation == "identity":
        return x
    raise ValueError(f"Unknown activation: {activation}")


def collect_cf_state(
    point,
    device,
    device_name,
    lambda_schedule=None,
    invariance_schedule=None,
    transform_kind="cf",
    activation="relu",
    activation_alpha=0.0,
):
    if invariance_schedule is not None:
        if lambda_schedule is not None:
            raise ValueError("Pass either lambda_schedule or invariance_schedule, not both")
        if len(invariance_schedule) != point.depth:
            raise ValueError(f"invariance_schedule length {len(invariance_schedule)} does not match depth {point.depth}")
        invariance_schedule = [float(value) for value in invariance_schedule]
        lambda_schedule = [lambda_from_invariance_strength(value) for value in invariance_schedule]
    if lambda_schedule is None:
        lambda_schedule = [point.lambda_reg] * point.depth
    if len(lambda_schedule) != point.depth:
        raise ValueError(f"lambda_schedule length {len(lambda_schedule)} does not match depth {point.depth}")
    lambda_schedule = [float(value) for value in lambda_schedule]
    if invariance_schedule is None:
        invariance_schedule = [1.0 / max(value, 1e-12) for value in lambda_schedule]
    if transform_kind not in {"cf", "whiten"}:
        raise ValueError(f"Unknown transform_kind: {transform_kind}")

    arrays = load_point_data(point)
    xtr_np, ytr_np, xte_np, yte_np, *_ = arrays
    tensors = tensors_from_arrays(arrays, device)
    xtr, ytr, xte, yte, view1_tr, view2_tr, view1_te, view2_te = tensors

    train_arrays, test_arrays, _, _ = normalize_hidden_with_stats_torch(
        [xtr, view1_tr, view2_tr],
        [xte, view1_te, view2_te],
    )
    base_tr, view1_tr, view2_tr = train_arrays
    base_te, view1_te, view2_te = test_arrays

    y_onehot = F.one_hot(ytr, num_classes=point.num_classes).to(torch.float32)
    yhat_tr = torch.zeros_like(y_onehot)
    yhat_te = torch.zeros((yte.shape[0], point.num_classes), dtype=torch.float32, device=device)

    raw_train = []
    raw_test = []
    raw_view1_train = []
    raw_view2_train = []
    pathnorm_train = []
    pathnorm_test = []
    pathnorm_view1_train = []
    pathnorm_view2_train = []
    heads = []
    contrib_train = []
    contrib_test = []
    layer_rows = []
    transforms = []

    torch.cuda.synchronize()
    start = time.perf_counter()
    for layer_idx in range(point.depth):
        layer_lambda = lambda_schedule[layer_idx]
        layer_invariance = invariance_schedule[layer_idx]
        if transform_kind == "cf":
            fitted = fit_cf_transform_torch(view1_tr, view2_tr, point.width, invariance_strength=layer_invariance)
        else:
            fitted = fit_whitening_transform_torch(view1_tr, view2_tr, point.width)
        transform = fitted["transform"]
        transforms.append(transform.detach().cpu().numpy().astype(np.float32))

        base_tr = apply_activation_torch(base_tr @ transform, activation, activation_alpha)
        base_te = apply_activation_torch(base_te @ transform, activation, activation_alpha)
        view1_tr = apply_activation_torch(view1_tr @ transform, activation, activation_alpha)
        view2_tr = apply_activation_torch(view2_tr @ transform, activation, activation_alpha)
        view1_te = apply_activation_torch(view1_te @ transform, activation, activation_alpha)
        view2_te = apply_activation_torch(view2_te @ transform, activation, activation_alpha)

        raw_train.append(base_tr.detach().cpu().numpy().astype(np.float32))
        raw_test.append(base_te.detach().cpu().numpy().astype(np.float32))
        raw_view1_train.append(view1_tr.detach().cpu().numpy().astype(np.float32))
        raw_view2_train.append(view2_tr.detach().cpu().numpy().astype(np.float32))

        out_map = ridge_regression_torch(base_tr, y_onehot - yhat_tr, point.head_reg)
        heads.append(out_map.detach().cpu().numpy().astype(np.float32))
        layer_train = base_tr @ out_map
        layer_test = base_te @ out_map
        contrib_train.append(layer_train.detach().cpu().numpy().astype(np.float32))
        contrib_test.append(layer_test.detach().cpu().numpy().astype(np.float32))
        yhat_tr = yhat_tr + layer_train
        yhat_te = yhat_te + layer_test

        layer_rows.append(
            {
                "layer": layer_idx + 1,
                "transform_kind": transform_kind,
                "activation": activation,
                "activation_alpha": float(activation_alpha),
                "lambda_reg": layer_lambda,
                "invariance_strength": layer_invariance,
                "cumulative_accuracy": accuracy_from_logits_torch(yhat_te, yte),
                "single_layer_accuracy": accuracy_from_logits_torch(layer_test, yte),
                "max_whitened_delta": fitted["max_whitened_delta"],
                "mean_gain": fitted["mean_gain"],
                "min_gain": fitted["min_gain"],
                "head_fro_norm": float(torch.linalg.matrix_norm(out_map).detach().cpu().item()),
                "test_contribution_rms": float(torch.sqrt(torch.mean(layer_test * layer_test)).detach().cpu().item()),
            }
        )

        train_arrays, test_arrays, _, _ = normalize_hidden_with_stats_torch(
            [base_tr, view1_tr, view2_tr],
            [base_te, view1_te, view2_te],
        )
        base_tr, view1_tr, view2_tr = train_arrays
        base_te, view1_te, view2_te = test_arrays
        pathnorm_train.append(base_tr.detach().cpu().numpy().astype(np.float32))
        pathnorm_test.append(base_te.detach().cpu().numpy().astype(np.float32))
        pathnorm_view1_train.append(view1_tr.detach().cpu().numpy().astype(np.float32))
        pathnorm_view2_train.append(view2_tr.detach().cpu().numpy().astype(np.float32))

    torch.cuda.synchronize()
    fit_time = time.perf_counter() - start
    del tensors
    torch.cuda.empty_cache()

    return {
        "point": asdict(point),
        "device": device_name,
        "transform_kind": transform_kind,
        "activation": activation,
        "activation_alpha": float(activation_alpha),
        "lambda_schedule": lambda_schedule,
        "invariance_schedule": invariance_schedule,
        "fit_time_sec": fit_time,
        "xtr": xtr_np.astype(np.float32),
        "xte": xte_np.astype(np.float32),
        "ytr": ytr_np.astype(np.int64),
        "yte": yte_np.astype(np.int64),
        "raw_train": raw_train,
        "raw_test": raw_test,
        "raw_view1_train": raw_view1_train,
        "raw_view2_train": raw_view2_train,
        "pathnorm_train": pathnorm_train,
        "pathnorm_test": pathnorm_test,
        "pathnorm_view1_train": pathnorm_view1_train,
        "pathnorm_view2_train": pathnorm_view2_train,
        "heads": heads,
        "contrib_train": contrib_train,
        "contrib_test": contrib_test,
        "layer_rows": layer_rows,
        "transforms": transforms,
    }


def collect_random_path(point, arrays, device):
    tensors = tensors_from_arrays(arrays, device)
    xtr, _, xte, _, view1_tr, view2_tr, view1_te, view2_te = tensors
    train_arrays, test_arrays, _, _ = normalize_hidden_with_stats_torch(
        [xtr, view1_tr, view2_tr],
        [xte, view1_te, view2_te],
    )
    base_tr, view1_tr, view2_tr = train_arrays
    base_te, view1_te, view2_te = test_arrays

    rng = torch.Generator(device=device)
    rng.manual_seed(point.seed + 909)
    raw_train = []
    raw_test = []
    pathnorm_train = []
    pathnorm_test = []
    current_dim = point.input_dim
    for _ in range(point.depth):
        out_dim = min(point.width, current_dim)
        weight = torch.randn((current_dim, out_dim), generator=rng, device=device) * math.sqrt(2.0 / current_dim)
        base_tr = torch.relu(base_tr @ weight)
        base_te = torch.relu(base_te @ weight)
        view1_tr = torch.relu(view1_tr @ weight)
        view2_tr = torch.relu(view2_tr @ weight)
        view1_te = torch.relu(view1_te @ weight)
        view2_te = torch.relu(view2_te @ weight)
        raw_train.append(base_tr.detach().cpu().numpy().astype(np.float32))
        raw_test.append(base_te.detach().cpu().numpy().astype(np.float32))
        train_arrays, test_arrays, _, _ = normalize_hidden_with_stats_torch(
            [base_tr, view1_tr, view2_tr],
            [base_te, view1_te, view2_te],
        )
        base_tr, view1_tr, view2_tr = train_arrays
        base_te, view1_te, view2_te = test_arrays
        pathnorm_train.append(base_tr.detach().cpu().numpy().astype(np.float32))
        pathnorm_test.append(base_te.detach().cpu().numpy().astype(np.float32))
        current_dim = out_dim
    del tensors
    torch.cuda.empty_cache()
    return {
        "raw_train": raw_train,
        "raw_test": raw_test,
        "pathnorm_train": pathnorm_train,
        "pathnorm_test": pathnorm_test,
    }


def head_svd_rows(state):
    rows = []
    for idx, head in enumerate(state["heads"]):
        _, s, _ = np.linalg.svd(head.astype(np.float64), full_matrices=False)
        energy = np.cumsum(s * s) / max(float(np.sum(s * s)), 1e-12)
        rows.append(
            {
                "layer": idx + 1,
                "rank90": int(np.searchsorted(energy, 0.90) + 1),
                "rank95": int(np.searchsorted(energy, 0.95) + 1),
                "rank99": int(np.searchsorted(energy, 0.99) + 1),
                "top_singular": float(s[0]),
                "tail_singular": float(s[-1]),
                "fro_norm": float(np.sqrt(np.sum(s * s))),
            }
        )
    return rows


def lowrank_head_accuracy_rows(state, ranks):
    rows = []
    yte = state["yte"]
    for rank in ranks:
        logits = np.zeros((yte.shape[0], state["point"]["num_classes"]), dtype=np.float64)
        for hte, head in zip(state["raw_test"], state["heads"]):
            u, s, vt = np.linalg.svd(head.astype(np.float64), full_matrices=False)
            r = min(rank, s.shape[0])
            approx = (u[:, :r] * s[:r]) @ vt[:r]
            logits += hte @ approx
        rows.append(
            {
                "head_rank_per_layer": rank,
                "accuracy": accuracy_from_logits(logits, yte),
                "cross_entropy": softmax_ce_np(logits, yte),
            }
        )
    return rows


def supervised_layer_importance_rows(state, alphas=(0.0, 0.5)):
    ytr = state["ytr"]
    yte = state["yte"]
    y_onehot = one_hot_np(ytr, state["point"]["num_classes"])
    train_parts = state["contrib_train"]
    test_parts = state["contrib_test"]
    total_train = np.sum(train_parts, axis=0)
    total_test = np.sum(test_parts, axis=0)
    correction = ridge_map_np(total_train, y_onehot, reg=1e-3, fit_bias=True)
    corrected_total = apply_map_np(total_test, correction, fit_bias=True)
    baseline_acc = accuracy_from_logits(total_test, yte)
    corrected_baseline_acc = accuracy_from_logits(corrected_total, yte)
    rows = [
        {
            "layer": 0,
            "alpha": 1.0,
            "raw_accuracy": baseline_acc,
            "corrected_accuracy": corrected_baseline_acc,
            "corrected_drop": 0.0,
            "description": "all_layers",
        }
    ]
    for idx, (ctr, cte) in enumerate(zip(train_parts, test_parts)):
        for alpha in alphas:
            mod_train = total_train - (1.0 - alpha) * ctr
            mod_test = total_test - (1.0 - alpha) * cte
            corr = ridge_map_np(mod_train, y_onehot, reg=1e-3, fit_bias=True)
            corrected = apply_map_np(mod_test, corr, fit_bias=True)
            rows.append(
                {
                    "layer": idx + 1,
                    "alpha": float(alpha),
                    "raw_accuracy": accuracy_from_logits(mod_test, yte),
                    "corrected_accuracy": accuracy_from_logits(corrected, yte),
                    "corrected_drop": corrected_baseline_acc - accuracy_from_logits(corrected, yte),
                    "description": "layer_shrunk",
                }
            )
    return rows


def layer_indices_for_choice(choice, depth, important_layers=None):
    if choice == "all":
        return list(range(depth))
    if choice == "first":
        return [0]
    if choice == "last":
        return [depth - 1]
    if choice == "early_half":
        return list(range(max(1, depth // 2)))
    if choice == "late_half":
        return list(range(depth - max(1, depth // 2), depth))
    if choice.startswith("layer"):
        layer = int(choice.replace("layer", "")) - 1
        return [layer]
    if choice == "important_top2" and important_layers:
        return sorted([layer - 1 for layer in important_layers[:2]])
    if choice == "important_top3" and important_layers:
        return sorted([layer - 1 for layer in important_layers[:3]])
    raise ValueError(f"Unknown layer choice: {choice}")


def make_features(source, mode, indices):
    if source == "raw_input":
        raise ValueError("raw_input is handled separately")
    if mode == "raw_zscore":
        train_layers = source["raw_train"]
        test_layers = source["raw_test"]
    else:
        train_layers = source[f"{mode}_train"]
        test_layers = source[f"{mode}_test"]
    train_parts = []
    test_parts = []
    for idx in indices:
        tr = train_layers[idx]
        te = test_layers[idx]
        if mode == "raw_zscore":
            tr, te = standardize_pair(source["raw_train"][idx], source["raw_test"][idx])
        train_parts.append(tr)
        test_parts.append(te)
    return np.concatenate(train_parts, axis=1), np.concatenate(test_parts, axis=1)


def pca_probe_rows(name, xtr, xte, ytr, yte, dims, seed, reg=100.0, standardize_scores=True):
    max_dim = min(max(dims), xtr.shape[1], xtr.shape[0] - 1)
    dims = [dim for dim in dims if dim <= max_dim]
    if not dims:
        return []
    pca = PCA(n_components=max_dim, svd_solver="randomized", iterated_power=3, random_state=seed)
    start = time.perf_counter()
    ztr = pca.fit_transform(xtr)
    zte = pca.transform(xte)
    pca_time = time.perf_counter() - start
    rows = []
    y_onehot = one_hot_np(ytr, int(np.max(ytr)) + 1)
    for dim in dims:
        ftr = ztr[:, :dim].astype(np.float32)
        fte = zte[:, :dim].astype(np.float32)
        if standardize_scores:
            ftr, fte = standardize_pair(ftr, fte)
        weight = ridge_map_np(ftr, y_onehot, reg=reg, fit_bias=True)
        logits = apply_map_np(fte, weight, fit_bias=True)
        rows.append(
            {
                "representation": name,
                "pca_dim": int(dim),
                "source_dim": int(xtr.shape[1]),
                "accuracy": accuracy_from_logits(logits, yte),
                "cross_entropy": softmax_ce_np(logits, yte),
                "explained_variance": float(np.sum(pca.explained_variance_ratio_[:dim])),
                "pca_time_sec": float(pca_time),
                "probe_reg": float(reg),
                "score_standardized": bool(standardize_scores),
            }
        )
    return rows


def summarize_rows(rows, key_fields, value_field="accuracy"):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in key_fields)].append(row)
    out = []
    for key, items in sorted(grouped.items()):
        vals = np.asarray([item[value_field] for item in items], dtype=np.float64)
        rec = {field: val for field, val in zip(key_fields, key)}
        rec.update(
            {
                "runs": len(items),
                f"mean_{value_field}": float(vals.mean()),
                f"std_{value_field}": float(vals.std(ddof=0)),
            }
        )
        out.append(rec)
    return out


def build_report(out_dir, point, layer_rows, svd_rows, importance_rows, pca_rows, aggregate_rows):
    lines = [
        "# CF-MLP Representation Diagnostic",
        "",
        f"Dataset `{point.dataset}`, `n_train={point.n_train}`, `n_test={point.n_test}`, input dim `{point.input_dim}`, width `{point.width}`, depth `{point.depth}`.",
        "",
        "## Supervised Residual Stream",
        "",
        "| Layer | Cumulative acc | Single-layer acc | Head rank95 | Corrected ablation drop | Contribution RMS |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rank95 = {row["layer"]: row["rank95"] for row in svd_rows}
    ablate_drop = {
        row["layer"]: row["corrected_drop"]
        for row in importance_rows
        if row["description"] == "layer_shrunk" and row["alpha"] == 0.0
    }
    for row in layer_rows:
        lines.append(
            f"| {row['layer']} | {row['cumulative_accuracy']:.4f} | {row['single_layer_accuracy']:.4f} | "
            f"{rank95.get(row['layer'], 0)} | {ablate_drop.get(row['layer'], 0.0):+.4f} | {row['test_contribution_rms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## PCA Linear Probe",
            "",
            "| Representation | PCA dim | Source dim | Runs | Mean acc | Std |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            f"| {row['representation']} | {row['pca_dim']} | {row['source_dim']} | {row['runs']} | "
            f"{row['mean_accuracy']:.4f} | {row['std_accuracy']:.4f} |"
        )
    lines.append("")
    report = "\n".join(lines)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    return report


def run(args):
    torch.set_num_threads(args.torch_threads)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        n_train, n_test, depth, pca_dims = 6000, 1000, 3, [32, 64, 128, 256]
        seeds = args.seeds[:1]
    else:
        n_train, n_test, depth, pca_dims = args.n_train, args.n_test, args.depth, args.pca_dims
        seeds = args.seeds

    all_layer_rows = []
    all_svd_rows = []
    all_lowrank_rows = []
    all_importance_rows = []
    all_pca_rows = []
    report_text = ""

    for seed in seeds:
        point = SweepPoint(
            dataset="cifar100_fullres_width",
            axis="representation",
            scale_value=args.width,
            seed=seed,
            n_train=n_train,
            n_test=n_test,
            input_dim=args.input_dim,
            width=args.width,
            depth=depth,
            num_classes=100,
        )
        print(
            f"seed={seed} n={n_train} test={n_test} input_dim={args.input_dim} width={args.width} depth={depth} device={device_name}",
            flush=True,
        )
        state = collect_cf_state(point, device, device_name)
        arrays = (state["xtr"], state["ytr"], state["xte"], state["yte"])
        ytr, yte = state["ytr"], state["yte"]

        for row in state["layer_rows"]:
            all_layer_rows.append({"seed": seed, **row})
        svd_rows = head_svd_rows(state)
        for row in svd_rows:
            all_svd_rows.append({"seed": seed, **row})
        for row in lowrank_head_accuracy_rows(state, ranks=[1, 2, 4, 8, 16, 32, 64, 100]):
            all_lowrank_rows.append({"seed": seed, **row})
        importance_rows = supervised_layer_importance_rows(state)
        for row in importance_rows:
            all_importance_rows.append({"seed": seed, **row})

        importance_order = [
            row["layer"]
            for row in sorted(
                [r for r in importance_rows if r["description"] == "layer_shrunk" and r["alpha"] == 0.0],
                key=lambda r: r["corrected_drop"],
                reverse=True,
            )
        ]

        representations = []
        representations.append(("raw_input", state["xtr"], state["xte"]))
        random_state = collect_random_path(point, load_point_data(point), device)
        for choice in ["all"]:
            idx = layer_indices_for_choice(choice, depth)
            tr, te = make_features(random_state, "pathnorm", idx)
            representations.append((f"random_pathnorm_{choice}", tr, te))

        for mode in ["pathnorm", "raw_zscore", "raw"]:
            idx = layer_indices_for_choice("all", depth)
            tr, te = make_features(state, mode, idx)
            representations.append((f"cf_{mode}_all", tr, te))

        layer_choices = ["first", "last", "early_half", "late_half"] + [f"layer{i}" for i in range(1, depth + 1)]
        if importance_order:
            layer_choices.extend(["important_top2", "important_top3"])
        for choice in layer_choices:
            idx = layer_indices_for_choice(choice, depth, importance_order)
            tr, te = make_features(state, "pathnorm", idx)
            representations.append((f"cf_pathnorm_{choice}", tr, te))

        for name, tr, te in representations:
            print(f"seed={seed} PCA {name} shape={tr.shape}", flush=True)
            all_pca_rows.extend(
                {
                    "seed": seed,
                    **row,
                }
                for row in pca_probe_rows(name, tr, te, ytr, yte, pca_dims, seed=seed, reg=args.probe_reg)
            )
            del tr, te
            gc.collect()

        write_jsonl(args.out_dir / "layer_rows.partial.jsonl", all_layer_rows)
        write_jsonl(args.out_dir / "head_svd_rows.partial.jsonl", all_svd_rows)
        write_jsonl(args.out_dir / "head_lowrank_rows.partial.jsonl", all_lowrank_rows)
        write_jsonl(args.out_dir / "layer_importance_rows.partial.jsonl", all_importance_rows)
        write_jsonl(args.out_dir / "pca_probe_rows.partial.jsonl", all_pca_rows)

        aggregate_rows = summarize_rows(all_pca_rows, ["representation", "pca_dim", "source_dim"])
        report_text = build_report(args.out_dir, point, all_layer_rows, all_svd_rows, all_importance_rows, all_pca_rows, aggregate_rows)
        del state, random_state
        torch.cuda.empty_cache()
        gc.collect()

    aggregate_rows = summarize_rows(all_pca_rows, ["representation", "pca_dim", "source_dim"])
    write_jsonl(args.out_dir / "layer_rows.jsonl", all_layer_rows)
    write_jsonl(args.out_dir / "head_svd_rows.jsonl", all_svd_rows)
    write_jsonl(args.out_dir / "head_lowrank_rows.jsonl", all_lowrank_rows)
    write_jsonl(args.out_dir / "layer_importance_rows.jsonl", all_importance_rows)
    write_jsonl(args.out_dir / "pca_probe_rows.jsonl", all_pca_rows)
    write_jsonl(args.out_dir / "pca_probe_aggregate.jsonl", aggregate_rows)
    if report_text:
        print(report_text)


def main():
    parser = argparse.ArgumentParser(description="CF-MLP depth-path PCA representation diagnostics.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.40)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=3072)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--pca-dims", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--probe-reg", type=float, default=100.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
