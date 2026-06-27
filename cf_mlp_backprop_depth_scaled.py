import argparse
import gc
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from sklearn.decomposition import PCA

from cf_mlp_clean_readouts import (
    collect_depth_representations,
    linear_classifier_readout,
    pca512_readout,
)
from cf_mlp_last_layer_content import (
    class_knn_metrics,
    class_one_hot,
    linear_cka,
    raw_reconstruction_metrics,
    view_retrieval_metrics,
)
from cf_mlp_layer_mechanistic import (
    covariance_spectrum,
    forward_with_hiddens,
    standardize_many,
    supervised_layer_importance_rows,
    view_alignment,
)
from cf_mlp_representation import apply_map_np, ridge_map_np, standardize_pair, tensors_from_arrays
from cf_mlp_scalability import SweepPoint, accuracy_from_logits, load_cifar100_full, load_point_data, write_jsonl
from cf_mlp_scalability_gpu import init_backprop_params, normalize_hidden_with_stats_torch


def cpu_params(params):
    return [param.detach().cpu().clone() for param in params]


def eval_backprop(x, y, weights, heads):
    with torch.no_grad():
        _, _, logits = forward_with_hiddens(x, weights, heads)
        final = logits[-1]
        return {
            "accuracy": float((final.argmax(dim=1) == y).float().mean().detach().cpu().item()),
            "cross_entropy": float(F.cross_entropy(final, y).detach().cpu().item()),
            "layer_accuracy": [
                float((out.argmax(dim=1) == y).float().mean().detach().cpu().item())
                for out in logits
            ],
        }


def train_backprop_best(point, arrays, device, epochs, eval_every_epochs):
    tensors = tensors_from_arrays(arrays, device)
    xtr, ytr, xte, yte, view1_tr, view2_tr, *_ = tensors
    _, _, norm_mean, norm_scale = normalize_hidden_with_stats_torch([xtr, view1_tr, view2_tr], [xte])
    xtr_norm = (xtr - norm_mean) / norm_scale
    xte_norm = (xte - norm_mean) / norm_scale
    weights, heads = init_backprop_params(point, device)
    params = list(weights.parameters()) + list(heads.parameters())
    optimizer = torch.optim.AdamW(params, lr=point.lr, weight_decay=point.weight_decay)
    steps_per_epoch = max(1, int(math.ceil(point.n_train / point.batch_size)))
    total_steps = int(math.ceil(epochs * steps_per_epoch))
    eval_interval = max(1, int(round(eval_every_epochs * steps_per_epoch)))
    permutation = torch.randperm(point.n_train, device=device)
    cursor = 0
    losses = []
    history = []
    best = None
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize()
    for step in range(1, total_steps + 1):
        if cursor + point.batch_size > point.n_train:
            permutation = torch.randperm(point.n_train, device=device)
            cursor = 0
        batch_idx = permutation[cursor : cursor + point.batch_size]
        cursor += point.batch_size
        _, _, logits = forward_with_hiddens(xtr_norm[batch_idx], weights, heads)
        loss = sum(F.cross_entropy(out, ytr[batch_idx]) for out in logits) / len(logits)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        if step % eval_interval == 0 or step == total_steps:
            metrics = eval_backprop(xte_norm, yte, weights, heads)
            epoch = step / steps_per_epoch
            row = {
                "step": step,
                "epoch": epoch,
                "mean_train_loss": float(np.mean(losses[-min(len(losses), steps_per_epoch) :])),
                **metrics,
            }
            history.append(row)
            print(
                f"bp depth={point.depth} epoch={epoch:.1f} acc={metrics['accuracy']:.4f} loss={row['mean_train_loss']:.3f}",
                flush=True,
            )
            if best is None or metrics["accuracy"] > best["metrics"]["accuracy"]:
                best = {
                    "step": step,
                    "epoch": epoch,
                    "metrics": metrics,
                    "weights": cpu_params(weights),
                    "heads": cpu_params(heads),
                }
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    final_metrics = eval_backprop(xte_norm, yte, weights, heads)
    state = {
        "weights": best["weights"],
        "heads": best["heads"],
        "norm_mean": norm_mean.detach().cpu(),
        "norm_scale": norm_scale.detach().cpu(),
        "history": history,
        "best_epoch": best["epoch"],
        "best_step": best["step"],
        "best_metrics": best["metrics"],
        "final_metrics": final_metrics,
        "fit_time_sec": elapsed,
        "epochs": epochs,
    }
    del tensors
    torch.cuda.empty_cache()
    return state


def collect_backprop_representations(point, arrays, train_state, device):
    tensors = tensors_from_arrays(arrays, device)
    xtr, ytr, xte, yte, view1_tr, view2_tr, *_ = tensors
    norm_mean = train_state["norm_mean"].to(device)
    norm_scale = train_state["norm_scale"].to(device)
    xtr_norm = (xtr - norm_mean) / norm_scale
    xte_norm = (xte - norm_mean) / norm_scale
    view1_norm = (view1_tr - norm_mean) / norm_scale
    view2_norm = (view2_tr - norm_mean) / norm_scale
    weights = [param.to(device) for param in train_state["weights"]]
    heads = [param.to(device) for param in train_state["heads"]]
    with torch.no_grad():
        train_h, train_c, train_logits = forward_with_hiddens(xtr_norm, weights, heads)
        test_h, test_c, test_logits = forward_with_hiddens(xte_norm, weights, heads)
        view1_h, _, _ = forward_with_hiddens(view1_norm, weights, heads)
        view2_h, _, _ = forward_with_hiddens(view2_norm, weights, heads)
    raw_train = [h.detach().cpu().numpy().astype(np.float32) for h in train_h]
    raw_test = [h.detach().cpu().numpy().astype(np.float32) for h in test_h]
    raw_view1 = [h.detach().cpu().numpy().astype(np.float32) for h in view1_h]
    raw_view2 = [h.detach().cpu().numpy().astype(np.float32) for h in view2_h]
    pathnorm_train = []
    pathnorm_test = []
    pathnorm_view1 = []
    pathnorm_view2 = []
    for htr, hte, hv1, hv2 in zip(raw_train, raw_test, raw_view1, raw_view2):
        htr_s, hte_s, hv1_s, hv2_s = standardize_many(htr, hte, hv1, hv2)
        pathnorm_train.append(htr_s)
        pathnorm_test.append(hte_s)
        pathnorm_view1.append(hv1_s)
        pathnorm_view2.append(hv2_s)
    contrib_train = [c.detach().cpu().numpy().astype(np.float32) for c in train_c]
    contrib_test = [c.detach().cpu().numpy().astype(np.float32) for c in test_c]
    head_np = [head.detach().cpu().numpy().astype(np.float32) for head in heads]
    yte_np = arrays[3].astype(np.int64)
    layer_rows = []
    for idx, (cum, part, head) in enumerate(zip(test_logits, test_c, heads)):
        layer_rows.append(
            {
                "layer": idx + 1,
                "cumulative_accuracy": accuracy_from_logits(cum.detach().cpu().numpy(), yte_np),
                "single_layer_accuracy": accuracy_from_logits(part.detach().cpu().numpy(), yte_np),
                "head_fro_norm": float(torch.linalg.matrix_norm(head).detach().cpu().item()),
                "test_contribution_rms": float(torch.sqrt(torch.mean(part * part)).detach().cpu().item()),
            }
        )
    del tensors
    torch.cuda.empty_cache()
    return {
        "point": asdict(point),
        "xtr": arrays[0].astype(np.float32),
        "ytr": arrays[1].astype(np.int64),
        "xte": arrays[2].astype(np.float32),
        "yte": arrays[3].astype(np.int64),
        "raw_train": raw_train,
        "raw_test": raw_test,
        "raw_view1_train": raw_view1,
        "raw_view2_train": raw_view2,
        "pathnorm_train": pathnorm_train,
        "pathnorm_test": pathnorm_test,
        "pathnorm_view1_train": pathnorm_view1,
        "pathnorm_view2_train": pathnorm_view2,
        "contrib_train": contrib_train,
        "contrib_test": contrib_test,
        "heads": head_np,
        "layer_rows": layer_rows,
    }


def backprop_readout_rows(state, probe_reg, pca_dim):
    rows = []
    ytr = state["ytr"]
    yte = state["yte"]
    point = state["point"]
    for idx, (xtr, xte, v1, v2) in enumerate(
        zip(
            state["pathnorm_train"],
            state["pathnorm_test"],
            state["pathnorm_view1_train"],
            state["pathnorm_view2_train"],
        )
    ):
        row = {
            "model": "backprop_supervised_best",
            "seed": int(point["seed"]),
            "input_dim": int(point["input_dim"]),
            "width": int(point["width"]),
            "depth": int(point["depth"]),
            "layer": idx + 1,
            "setup": "layer_hidden_512",
            "supervised_mapping": "single_linear_classifier",
        }
        row.update(linear_classifier_readout(xtr, xte, ytr, yte, probe_reg))
        row.update(covariance_spectrum(xtr))
        row.update(view_alignment(v1, v2, int(point["seed"]) + idx + 7070))
        rows.append(row)
    return rows


def setup_readouts(state, probe_reg, pca_dim):
    ytr = state["ytr"]
    yte = state["yte"]
    point = state["point"]
    last_tr = state["pathnorm_train"][-1]
    last_te = state["pathnorm_test"][-1]
    all_tr = np.concatenate(state["pathnorm_train"], axis=1)
    all_te = np.concatenate(state["pathnorm_test"], axis=1)
    rows = []
    last = {
        "model": "backprop_supervised_best",
        "seed": int(point["seed"]),
        "input_dim": int(point["input_dim"]),
        "width": int(point["width"]),
        "depth": int(point["depth"]),
        "setup": "last_layer_512",
        "representation": "backprop_last_hidden_512",
        "supervised_mapping": "single_linear_classifier",
    }
    last.update(linear_classifier_readout(last_tr, last_te, ytr, yte, probe_reg))
    all_pca = {
        "model": "backprop_supervised_best",
        "seed": int(point["seed"]),
        "input_dim": int(point["input_dim"]),
        "width": int(point["width"]),
        "depth": int(point["depth"]),
        "setup": "all_layers_pca512",
        "representation": "backprop_all_hidden_concat_to_pca512",
        "supervised_mapping": "single_linear_classifier",
    }
    all_pca.update(pca512_readout(all_tr, all_te, ytr, yte, probe_reg, int(point["seed"]), pca_dim))
    return rows + [last, all_pca]


def selected_content_rows(model, state, probe_reg, recon_reg, pca_dim, max_retrieval, knn_k):
    ytr = state["ytr"]
    yte = state["yte"]
    point = state["point"]
    labels_onehot = class_one_hot(ytr, int(point["num_classes"]))
    first = state["pathnorm_train"][0]
    selected = sorted(set([0, max(0, len(state["pathnorm_train"]) // 2 - 1), len(state["pathnorm_train"]) - 1]))
    features = []
    for idx in selected:
        features.append(
            (
                f"layer{idx + 1}_512",
                state["pathnorm_train"][idx],
                state["pathnorm_test"][idx],
                state["pathnorm_view1_train"][idx],
                state["pathnorm_view2_train"][idx],
                1.0,
            )
        )
    all_tr = np.concatenate(state["pathnorm_train"], axis=1)
    all_te = np.concatenate(state["pathnorm_test"], axis=1)
    all_v1 = np.concatenate(state["pathnorm_view1_train"], axis=1)
    all_v2 = np.concatenate(state["pathnorm_view2_train"], axis=1)
    pca = PCA(n_components=min(pca_dim, all_tr.shape[1]), svd_solver="randomized", iterated_power=3, random_state=int(point["seed"]) + 414)
    all_tr_pca = pca.fit_transform(all_tr).astype(np.float32)
    all_te_pca = pca.transform(all_te).astype(np.float32)
    all_v1_pca = pca.transform(all_v1).astype(np.float32)
    all_v2_pca = pca.transform(all_v2).astype(np.float32)
    features.append(("all_layers_pca512", all_tr_pca, all_te_pca, all_v1_pca, all_v2_pca, float(np.sum(pca.explained_variance_ratio_))))
    rows = []
    for setup, ftr, fte, v1, v2, explained in features:
        readout = linear_classifier_readout(ftr, fte, ytr, yte, probe_reg)
        row = {
            "model": model,
            "seed": int(point["seed"]),
            "input_dim": int(point["input_dim"]),
            "width": int(point["width"]),
            "depth": int(point["depth"]),
            "setup": setup,
            "class_linear_accuracy": readout["test_accuracy"],
            "class_linear_train_accuracy": readout["train_accuracy"],
            "pca_explained_variance": explained,
        }
        row.update(covariance_spectrum(ftr))
        row.update(view_alignment(v1, v2, int(point["seed"]) + 5151))
        row.update(raw_reconstruction_metrics(ftr, fte, state["xtr"], state["xte"], recon_reg))
        row.update(view_retrieval_metrics(v1, v2, max_retrieval))
        row.update(class_knn_metrics(fte, yte, max_retrieval, knn_k))
        row["cka_to_raw_input"] = linear_cka(ftr, state["xtr"])
        row["cka_to_labels"] = linear_cka(ftr, labels_onehot)
        row["cka_to_first_layer"] = linear_cka(ftr, first)
        rows.append(row)
    return rows


def sample_raw_images(point):
    xtr_all, _, xte_all, _ = load_cifar100_full()
    rng = np.random.default_rng(point.seed)
    idx_tr = rng.choice(xtr_all.shape[0], size=point.n_train, replace=False)
    idx_te = rng.choice(xte_all.shape[0], size=point.n_test, replace=False)
    return xtr_all[idx_tr].astype(np.float32), xte_all[idx_te].astype(np.float32)


def attribute_families(images):
    rgb_mean = images.mean(axis=(2, 3))
    rgb_std = images.std(axis=(2, 3))
    gray = images.mean(axis=1)
    brightness = np.stack([gray.mean(axis=(1, 2)), gray.std(axis=(1, 2))], axis=1)
    h = gray.shape[1]
    w = gray.shape[2]
    quadrants = np.stack(
        [
            gray[:, : h // 2, : w // 2].mean(axis=(1, 2)),
            gray[:, : h // 2, w // 2 :].mean(axis=(1, 2)),
            gray[:, h // 2 :, : w // 2].mean(axis=(1, 2)),
            gray[:, h // 2 :, w // 2 :].mean(axis=(1, 2)),
        ],
        axis=1,
    )
    sx = ndimage.sobel(gray, axis=2)
    sy = ndimage.sobel(gray, axis=1)
    edge = np.sqrt(sx * sx + sy * sy)
    edge_stats = np.stack([edge.mean(axis=(1, 2)), edge.std(axis=(1, 2))], axis=1)
    fft = np.fft.fftshift(np.fft.fft2(gray, axes=(1, 2)), axes=(1, 2))
    power = np.log1p(np.abs(fft) ** 2)
    yy, xx = np.mgrid[:h, :w]
    rr = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    bands = []
    for lo, hi in [(0, 4), (4, 10), (10, 100)]:
        mask = (rr >= lo) & (rr < hi)
        bands.append(power[:, mask].mean(axis=1))
    frequency = np.stack(bands, axis=1)
    rg = images[:, 0] - images[:, 1]
    yb = 0.5 * (images[:, 0] + images[:, 1]) - images[:, 2]
    color_opponent = np.stack(
        [
            rg.mean(axis=(1, 2)),
            rg.std(axis=(1, 2)),
            yb.mean(axis=(1, 2)),
            yb.std(axis=(1, 2)),
        ],
        axis=1,
    )
    return {
        "rgb_mean": rgb_mean.astype(np.float32),
        "rgb_std": rgb_std.astype(np.float32),
        "brightness": brightness.astype(np.float32),
        "quadrants": quadrants.astype(np.float32),
        "edge": edge_stats.astype(np.float32),
        "frequency": frequency.astype(np.float32),
        "color_opponent": color_opponent.astype(np.float32),
    }


def zscore_with_train(train, test):
    mean = train.mean(axis=0, keepdims=True)
    std = np.maximum(train.std(axis=0, keepdims=True), 1e-6)
    return (train - mean) / std, (test - mean) / std


def mean_r2(pred, target):
    mse = np.mean((pred - target) ** 2, axis=0)
    var = np.var(target, axis=0)
    return float(np.mean(1.0 - mse / np.maximum(var, 1e-12)))


def ridge_r2(xtr, xte, ytr, yte, reg):
    xtr_z, xte_z = standardize_pair(xtr.astype(np.float32), xte.astype(np.float32))
    ytr_z, yte_z = zscore_with_train(ytr.astype(np.float32), yte.astype(np.float32))
    weight = ridge_map_np(xtr_z, ytr_z, reg=reg, fit_bias=True)
    pred = apply_map_np(xte_z, weight, fit_bias=True)
    return mean_r2(pred, yte_z)


def normalize_rows(x):
    x0 = x.astype(np.float32)
    x0 = x0 - x0.mean(axis=1, keepdims=True)
    return x0 / np.maximum(np.linalg.norm(x0, axis=1, keepdims=True), 1e-12)


def nearest_attribute_ratio(rep, attrs, max_items, seed):
    n = min(max_items, rep.shape[0], attrs.shape[0])
    z = normalize_rows(rep[:n])
    atr, _ = zscore_with_train(attrs[:n], attrs[:n])
    sims = z @ z.T
    np.fill_diagonal(sims, -2.0)
    nn = np.argmax(sims, axis=1)
    same = np.linalg.norm(atr - atr[nn], axis=1).mean()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    random = np.linalg.norm(atr - atr[perm], axis=1).mean()
    return float(same / max(random, 1e-12))


def feature_diagnostic_rows(representations, attr_train, attr_test, reg, max_items, seed):
    rows = []
    for rep_name, xtr, xte in representations:
        for family in sorted(attr_train):
            row = {
                "representation": rep_name,
                "family": family,
                "attr_from_rep_r2": ridge_r2(xtr, xte, attr_train[family], attr_test[family], reg),
                "rep_from_attr_r2": ridge_r2(attr_train[family], attr_test[family], xtr, xte, reg),
                "nn_attr_distance_ratio": nearest_attribute_ratio(xte, attr_test[family], max_items, seed + len(rows)),
            }
            rows.append(row)
    return rows


def make_feature_representations(cf_state, bp_state, pca_dim, seed):
    reps = [("raw_input", cf_state["xtr"], cf_state["xte"])]
    for idx in [0, 11, 23]:
        if idx < len(cf_state["pathnorm_train"]):
            reps.append((f"cf_layer{idx + 1}", cf_state["pathnorm_train"][idx], cf_state["pathnorm_test"][idx]))
    cf_all_tr = np.concatenate(cf_state["pathnorm_train"], axis=1)
    cf_all_te = np.concatenate(cf_state["pathnorm_test"], axis=1)
    pca = PCA(n_components=min(pca_dim, cf_all_tr.shape[1]), svd_solver="randomized", iterated_power=3, random_state=seed + 616)
    reps.append(("cf_all_layers_pca512", pca.fit_transform(cf_all_tr).astype(np.float32), pca.transform(cf_all_te).astype(np.float32)))
    if bp_state is not None:
        for idx in [0, 11, 23]:
            if idx < len(bp_state["pathnorm_train"]):
                reps.append((f"bp_layer{idx + 1}", bp_state["pathnorm_train"][idx], bp_state["pathnorm_test"][idx]))
        bp_all_tr = np.concatenate(bp_state["pathnorm_train"], axis=1)
        bp_all_te = np.concatenate(bp_state["pathnorm_test"], axis=1)
        pca = PCA(n_components=min(pca_dim, bp_all_tr.shape[1]), svd_solver="randomized", iterated_power=3, random_state=seed + 717)
        reps.append(("bp_all_layers_pca512", pca.fit_transform(bp_all_tr).astype(np.float32), pca.transform(bp_all_te).astype(np.float32)))
    return reps


def summarize_backprop(point, train_state, layer_rows, setup_rows, importance_rows):
    best_layer = max(layer_rows, key=lambda row: row["test_accuracy"])
    last_setup = next(row for row in setup_rows if row["setup"] == "last_layer_512")
    all_setup = next(row for row in setup_rows if row["setup"] == "all_layers_pca512")
    ablations = [row for row in importance_rows if row["description"] == "layer_shrunk" and row["alpha"] == 0.0]
    late = [row for row in ablations if row["layer"] > point.depth // 2]
    top = sorted(ablations, key=lambda row: row["corrected_drop"], reverse=True)[:5]
    return {
        "model": "backprop_supervised_best",
        "seed": point.seed,
        "input_dim": point.input_dim,
        "width": point.width,
        "depth": point.depth,
        "epochs": float(train_state["epochs"]),
        "best_epoch": float(train_state["best_epoch"]),
        "best_supervised_accuracy": float(train_state["best_metrics"]["accuracy"]),
        "final_supervised_accuracy": float(train_state["final_metrics"]["accuracy"]),
        "last_layer_readout_accuracy": last_setup["test_accuracy"],
        "all_pca_readout_accuracy": all_setup["test_accuracy"],
        "best_layer_readout_accuracy": best_layer["test_accuracy"],
        "best_layer": int(best_layer["layer"]),
        "max_late_corrected_drop": float(max([row["corrected_drop"] for row in late], default=0.0)),
        "mean_late_corrected_drop": float(np.mean([row["corrected_drop"] for row in late])) if late else 0.0,
        "positive_late_corrected_drop_sum": float(sum(max(row["corrected_drop"], 0.0) for row in late)),
        "top_importance_layers": ",".join(str(row["layer"]) for row in top),
        "top_importance_drops": ",".join(f"{row['corrected_drop']:.4f}" for row in top),
        "fit_time_sec": float(train_state["fit_time_sec"]),
    }


def build_report(out_dir, summaries, content_rows, feature_rows):
    lines = [
        "# Backprop Depth Scaling And Feature Content",
        "",
        "Backprop rows use best evaluated supervised residual-MLP checkpoints. Frozen representation readouts are single linear classifiers.",
        "",
        "| Depth | Best epoch | Supervised acc | Last readout | All PCA512 | Best layer | Max late drop | Late positive sum | Top importance layers |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['depth']} | {row['best_epoch']:.1f} | {row['best_supervised_accuracy']:.4f} | "
            f"{row['last_layer_readout_accuracy']:.4f} | {row['all_pca_readout_accuracy']:.4f} | "
            f"{row['best_layer_readout_accuracy']:.4f} @ L{row['best_layer']} | "
            f"{row['max_late_corrected_drop']:+.4f} | {row['positive_late_corrected_drop_sum']:+.4f} | "
            f"{row['top_importance_layers']} ({row['top_importance_drops']}) |"
        )
    lines.extend(
        [
            "",
            "## Selected Representation Content",
            "",
            "| Model | Depth | Setup | Class acc | Raw R2 | CKA raw | CKA labels | View top1 | kNN class | Rank | View ratio |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in content_rows:
        lines.append(
            f"| {row['model']} | {row['depth']} | {row['setup']} | {row['class_linear_accuracy']:.4f} | "
            f"{row['raw_reconstruction_r2']:.3f} | {row['cka_to_raw_input']:.3f} | "
            f"{row['cka_to_labels']:.3f} | {row['view_retrieval_top1']:.3f} | "
            f"{row['class_knn_purity']:.3f} | {row['effective_rank']:.1f} | "
            f"{row['same_over_shuffled_mse']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Low-Level Attribute Diagnostics",
            "",
            "`attr R2` predicts handcrafted image attributes from the representation. `NN ratio` is nearest-neighbor attribute distance divided by random-pair distance; lower means the representation clusters that attribute.",
            "",
            "| Representation | Family | Attr R2 | NN ratio |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in feature_rows:
        lines.append(
            f"| {row['representation']} | {row['family']} | {row['attr_from_rep_r2']:.3f} | "
            f"{row['nn_attr_distance_ratio']:.3f} |"
        )
    report = "\n".join(lines) + "\n"
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
    summaries = []
    all_layer_rows = []
    all_setup_rows = []
    all_importance_rows = []
    all_content_rows = []
    bp_feature_state = None
    feature_point = None
    for depth in args.depths:
        point = SweepPoint(
            dataset="cifar100_fullres_width",
            axis="backprop_depth_scaled",
            scale_value=args.width,
            seed=args.seed,
            n_train=args.n_train,
            n_test=args.n_test,
            input_dim=args.input_dim,
            width=args.width,
            depth=depth,
            num_classes=100,
        )
        arrays = load_point_data(point)
        train_point = replace(point, lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size)
        print(f"training backprop depth={depth} epochs={args.epochs} device={device_name}", flush=True)
        train_state = train_backprop_best(train_point, arrays, device, args.epochs, args.eval_every_epochs)
        torch.save(
            {**train_state, "point": asdict(train_point), "model_type": "backprop_supervised_depth_scaled_best"},
            args.out_dir / f"backprop_depth{depth}_best.pt",
        )
        bp_state = collect_backprop_representations(point, arrays, train_state, device)
        layer_rows = backprop_readout_rows(bp_state, args.probe_reg, args.pca_dim)
        setup_rows = setup_readouts(bp_state, args.probe_reg, args.pca_dim)
        importance_rows = supervised_layer_importance_rows(bp_state)
        content = selected_content_rows("backprop_supervised_best", bp_state, args.probe_reg, args.recon_reg, args.pca_dim, args.max_retrieval, args.knn_k)
        summary = summarize_backprop(point, train_state, layer_rows, setup_rows, importance_rows)
        summaries.append(summary)
        all_layer_rows.extend(layer_rows)
        all_setup_rows.extend(setup_rows)
        all_importance_rows.extend({"depth": depth, **row} for row in importance_rows)
        all_content_rows.extend(content)
        write_jsonl(args.out_dir / "backprop_summary.partial.jsonl", summaries)
        write_jsonl(args.out_dir / "backprop_layer_readouts.partial.jsonl", all_layer_rows)
        write_jsonl(args.out_dir / "backprop_setup_readouts.partial.jsonl", all_setup_rows)
        write_jsonl(args.out_dir / "backprop_importance.partial.jsonl", all_importance_rows)
        write_jsonl(args.out_dir / "content_rows.partial.jsonl", all_content_rows)
        if depth == args.feature_depth:
            bp_feature_state = bp_state
            feature_point = point
        else:
            del bp_state
        del train_state, arrays
        gc.collect()
        torch.cuda.empty_cache()

    if feature_point is None:
        feature_point = SweepPoint(
            dataset="cifar100_fullres_width",
            axis="feature_content",
            scale_value=args.width,
            seed=args.seed,
            n_train=args.n_train,
            n_test=args.n_test,
            input_dim=args.input_dim,
            width=args.width,
            depth=args.feature_depth,
            num_classes=100,
        )
    transform_kind, schedule_name, activation_name = args.cf_variant.split(":")
    print(f"collecting CF feature state depth={args.feature_depth} variant={args.cf_variant}", flush=True)
    cf_state = collect_depth_representations(
        feature_point,
        device,
        device_name,
        transform_kind=transform_kind,
        schedule_name=schedule_name,
        activation_name=activation_name,
    )
    cf_state["point"]["num_classes"] = feature_point.num_classes
    cf_content = selected_content_rows(args.cf_variant, cf_state, args.probe_reg, args.recon_reg, args.pca_dim, args.max_retrieval, args.knn_k)
    all_content_rows.extend(cf_content)

    xtr_img, xte_img = sample_raw_images(feature_point)
    attr_train = attribute_families(xtr_img)
    attr_test = attribute_families(xte_img)
    reps = make_feature_representations(cf_state, bp_feature_state, args.pca_dim, args.seed)
    feature_rows = feature_diagnostic_rows(reps, attr_train, attr_test, args.feature_reg, args.max_retrieval, args.seed)

    write_jsonl(args.out_dir / "backprop_summary.jsonl", summaries)
    write_jsonl(args.out_dir / "backprop_layer_readouts.jsonl", all_layer_rows)
    write_jsonl(args.out_dir / "backprop_setup_readouts.jsonl", all_setup_rows)
    write_jsonl(args.out_dir / "backprop_importance.jsonl", all_importance_rows)
    write_jsonl(args.out_dir / "content_rows.jsonl", all_content_rows)
    write_jsonl(args.out_dir / "feature_diagnostics.jsonl", feature_rows)
    print(build_report(args.out_dir, summaries, all_content_rows, feature_rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Depth-scaled backprop comparison and feature-content diagnostics.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_backprop_depth_scaled"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.40)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depths", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--feature-depth", type=int, default=24)
    parser.add_argument("--epochs", type=float, default=30.0)
    parser.add_argument("--eval-every-epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pca-dim", type=int, default=512)
    parser.add_argument("--probe-reg", type=float, default=100.0)
    parser.add_argument("--recon-reg", type=float, default=100.0)
    parser.add_argument("--feature-reg", type=float, default=100.0)
    parser.add_argument("--max-retrieval", type=int, default=2000)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--cf-variant", default="cf:relax4:leaky0.2")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
