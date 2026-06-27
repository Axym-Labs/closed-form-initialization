import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cf_mlp_representation import (
    apply_map_np,
    collect_cf_state,
    head_svd_rows,
    one_hot_np,
    ridge_map_np,
    softmax_ce_np,
    standardize_pair,
    supervised_layer_importance_rows,
    tensors_from_arrays,
)
from cf_mlp_scalability import (
    SweepPoint,
    accuracy_from_logits,
    estimate_backprop_step_flops,
    estimate_cf_flops,
    load_point_data,
    write_jsonl,
)
from cf_mlp_scalability_gpu import (
    init_backprop_params,
    normalize_hidden_with_stats_torch,
)


def forward_with_hiddens(x, weights, heads):
    h = x
    hiddens = []
    contributions = []
    logits = []
    cumulative = torch.zeros((x.shape[0], heads[0].shape[1]), dtype=x.dtype, device=x.device)
    for weight, head in zip(weights, heads):
        h = torch.relu(h @ weight)
        contribution = h @ head
        cumulative = cumulative + contribution
        hiddens.append(h)
        contributions.append(contribution)
        logits.append(cumulative)
    return hiddens, contributions, logits


def standardize_many(train, *arrays, eps=1e-6):
    mean = train.mean(axis=0, keepdims=True)
    std = np.maximum(train.std(axis=0, keepdims=True), eps)
    return tuple(((arr - mean) / std).astype(np.float32) for arr in (train, *arrays))


def collect_backprop_state(point, device, device_name, step_multiplier=1.0):
    arrays = load_point_data(point)
    xtr_np, ytr_np, xte_np, yte_np, *_ = arrays
    tensors = tensors_from_arrays(arrays, device)
    xtr, ytr, xte, yte, view1_tr, view2_tr, *_ = tensors
    _, _, norm_mean, norm_scale = normalize_hidden_with_stats_torch(
        [xtr, view1_tr, view2_tr],
        [xte],
    )
    xtr_norm = (xtr - norm_mean) / norm_scale
    xte_norm = (xte - norm_mean) / norm_scale
    view1_norm = (view1_tr - norm_mean) / norm_scale
    view2_norm = (view2_tr - norm_mean) / norm_scale

    weights, heads = init_backprop_params(point, device)
    params = list(weights.parameters()) + list(heads.parameters())
    optimizer = torch.optim.AdamW(params, lr=point.lr, weight_decay=point.weight_decay)
    cf_budget = estimate_cf_flops(point)
    step_flops = estimate_backprop_step_flops(point)
    equal_steps = max(1, int(math.floor(cf_budget / max(step_flops, 1.0))))
    max_steps = max(1, int(math.ceil(equal_steps * step_multiplier)))
    steps_per_epoch = max(1, int(math.ceil(point.n_train / point.batch_size)))
    permutation = torch.randperm(point.n_train, device=device)
    cursor = 0
    losses = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(max_steps):
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
    torch.cuda.synchronize()
    fit_time = time.perf_counter() - start

    with torch.no_grad():
        train_h, train_c, train_logits = forward_with_hiddens(xtr_norm, weights, heads)
        test_h, test_c, test_logits = forward_with_hiddens(xte_norm, weights, heads)
        view1_h, _, _ = forward_with_hiddens(view1_norm, weights, heads)
        view2_h, _, _ = forward_with_hiddens(view2_norm, weights, heads)

    raw_train = [h.detach().cpu().numpy().astype(np.float32) for h in train_h]
    raw_test = [h.detach().cpu().numpy().astype(np.float32) for h in test_h]
    raw_view1_train = [h.detach().cpu().numpy().astype(np.float32) for h in view1_h]
    raw_view2_train = [h.detach().cpu().numpy().astype(np.float32) for h in view2_h]
    pathnorm_train = []
    pathnorm_test = []
    pathnorm_view1_train = []
    pathnorm_view2_train = []
    for htr, hte, hv1, hv2 in zip(raw_train, raw_test, raw_view1_train, raw_view2_train):
        htr_s, hte_s, hv1_s, hv2_s = standardize_many(htr, hte, hv1, hv2)
        pathnorm_train.append(htr_s)
        pathnorm_test.append(hte_s)
        pathnorm_view1_train.append(hv1_s)
        pathnorm_view2_train.append(hv2_s)

    contrib_train = [c.detach().cpu().numpy().astype(np.float32) for c in train_c]
    contrib_test = [c.detach().cpu().numpy().astype(np.float32) for c in test_c]
    head_np = [head.detach().cpu().numpy().astype(np.float32) for head in heads]
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

    final_logits = test_logits[-1].detach().cpu().numpy()
    del tensors
    torch.cuda.empty_cache()
    return {
        "point": asdict(point),
        "device": device_name,
        "fit_time_sec": fit_time,
        "steps": max_steps,
        "equal_flop_steps": equal_steps,
        "effective_epochs": max_steps / steps_per_epoch,
        "mean_train_loss": float(np.mean(losses[-min(len(losses), steps_per_epoch) :])),
        "final_accuracy": accuracy_from_logits(final_logits, yte_np),
        "final_cross_entropy": softmax_ce_np(final_logits, yte_np),
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
        "heads": head_np,
        "contrib_train": contrib_train,
        "contrib_test": contrib_test,
        "layer_rows": layer_rows,
    }


def covariance_spectrum(x):
    x64 = x.astype(np.float64)
    x64 = x64 - x64.mean(axis=0, keepdims=True)
    cov = (x64.T @ x64) / max(1, x64.shape[0] - 1)
    vals = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    vals = np.maximum(vals, 0.0)[::-1]
    total = float(vals.sum())
    if total <= 1e-12:
        return {
            "effective_rank": 0.0,
            "participation_rank": 0.0,
            "top1_var": 0.0,
            "top10_var": 0.0,
            "top50_var": 0.0,
        }
    probs = vals / total
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
    return {
        "effective_rank": float(np.exp(entropy)),
        "participation_rank": float((total * total) / max(float(np.sum(vals * vals)), 1e-12)),
        "top1_var": float(probs[:1].sum()),
        "top10_var": float(probs[:10].sum()),
        "top50_var": float(probs[:50].sum()),
    }


def class_separation_ratio(x, y):
    x64 = x.astype(np.float64)
    total = float(np.sum((x64 - x64.mean(axis=0, keepdims=True)) ** 2) / x64.shape[0])
    within = 0.0
    for cls in np.unique(y):
        part = x64[y == cls]
        within += float(np.sum((part - part.mean(axis=0, keepdims=True)) ** 2))
    within /= x64.shape[0]
    between = max(total - within, 0.0)
    return {
        "within_trace": within,
        "between_trace": between,
        "between_within_ratio": float(between / max(within, 1e-12)),
    }


def view_alignment(x1, x2, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(x2.shape[0])
    same_mse = float(np.mean((x1 - x2) ** 2))
    shuffled_mse = float(np.mean((x1 - x2[idx]) ** 2))
    num = np.sum(x1 * x2, axis=1)
    den = np.linalg.norm(x1, axis=1) * np.linalg.norm(x2, axis=1)
    cosine = float(np.mean(num / np.maximum(den, 1e-12)))
    return {
        "same_view_mse": same_mse,
        "shuffled_view_mse": shuffled_mse,
        "same_over_shuffled_mse": float(same_mse / max(shuffled_mse, 1e-12)),
        "same_view_cosine": cosine,
    }


def probe_metrics(xtr, xte, ytr, yte, reg):
    y_onehot = one_hot_np(ytr, int(np.max(ytr)) + 1)
    weight = ridge_map_np(xtr, y_onehot, reg=reg, fit_bias=True)
    train_logits = apply_map_np(xtr, weight, fit_bias=True)
    test_logits = apply_map_np(xte, weight, fit_bias=True)
    return {
        "probe_train_accuracy": accuracy_from_logits(train_logits, ytr),
        "probe_test_accuracy": accuracy_from_logits(test_logits, yte),
        "probe_train_ce": softmax_ce_np(train_logits, ytr),
        "probe_test_ce": softmax_ce_np(test_logits, yte),
    }


def layer_diagnostic_rows(model_name, state, seed, probe_reg):
    rows = []
    ytr = state["ytr"]
    yte = state["yte"]
    for idx, (xtr, xte, v1, v2, raw) in enumerate(
        zip(
            state["pathnorm_train"],
            state["pathnorm_test"],
            state["pathnorm_view1_train"],
            state["pathnorm_view2_train"],
            state["raw_train"],
        )
    ):
        row = {
            "model": model_name,
            "seed": seed,
            "layer": idx + 1,
            "zero_fraction_raw": float(np.mean(np.abs(raw) < 1e-8)),
            "activation_rms_raw": float(np.sqrt(np.mean(raw * raw))),
        }
        row.update(probe_metrics(xtr, xte, ytr, yte, probe_reg))
        row.update(covariance_spectrum(xtr))
        row.update(class_separation_ratio(xtr, ytr))
        row.update(view_alignment(v1, v2, seed + 1234 + idx))
        rows.append(row)
    return rows


def add_model_prefix(rows, model):
    return [{"model": model, **row} for row in rows]


def aggregate(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    out = []
    numeric_keys = sorted(k for row in rows for k, v in row.items() if isinstance(v, (int, float)) and k not in keys)
    for key, items in sorted(grouped.items()):
        rec = {k: v for k, v in zip(keys, key)}
        rec["runs"] = len(items)
        for nk in numeric_keys:
            vals = [item[nk] for item in items if nk in item]
            if vals:
                rec[f"mean_{nk}"] = float(np.mean(vals))
                rec[f"std_{nk}"] = float(np.std(vals))
        out.append(rec)
    return out


def report_text(point, cf_layers, bp_layers, cf_importance, bp_importance):
    cf_imp = {r["layer"]: r["corrected_drop"] for r in cf_importance if r.get("description") == "layer_shrunk" and r.get("alpha") == 0.0}
    bp_imp = {r["layer"]: r["corrected_drop"] for r in bp_importance if r.get("description") == "layer_shrunk" and r.get("alpha") == 0.0}
    lines = [
        "# CF vs Backprop Layer Mechanistic Diagnostic",
        "",
        f"`input_dim={point.input_dim}`, `width={point.width}`, `depth={point.depth}`, `n_train={point.n_train}`, `n_test={point.n_test}`.",
        "",
        "## Per-Layer Representation Quality",
        "",
        "| Model | Layer | Probe train | Probe test | Class sep | Eff rank | View MSE ratio | View cosine | Corrected drop |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_key = defaultdict(list)
    for row in cf_layers + bp_layers:
        by_key[(row["model"], row["layer"])].append(row)
    for (model, layer), items in sorted(by_key.items()):
        imp = cf_imp.get(layer, 0.0) if model == "cf" else bp_imp.get(layer, 0.0)
        lines.append(
            f"| {model} | {layer} | {np.mean([r['probe_train_accuracy'] for r in items]):.4f} | "
            f"{np.mean([r['probe_test_accuracy'] for r in items]):.4f} | "
            f"{np.mean([r['between_within_ratio'] for r in items]):.4f} | "
            f"{np.mean([r['effective_rank'] for r in items]):.1f} | "
            f"{np.mean([r['same_over_shuffled_mse'] for r in items]):.3f} | "
            f"{np.mean([r['same_view_cosine'] for r in items]):.3f} | {imp:+.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


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

    all_cf_layers = []
    all_bp_layers = []
    all_cf_stream = []
    all_bp_stream = []
    all_cf_svd = []
    all_bp_svd = []
    all_cf_importance = []
    all_bp_importance = []
    point = None
    for seed in args.seeds:
        point = SweepPoint(
            dataset="cifar100_fullres_width",
            axis="mechanistic_layers",
            scale_value=args.width,
            seed=seed,
            n_train=args.n_train,
            n_test=args.n_test,
            input_dim=args.input_dim,
            width=args.width,
            depth=args.depth,
            num_classes=100,
        )
        print(f"CF seed={seed} input_dim={args.input_dim} width={args.width} depth={args.depth}", flush=True)
        cf_state = collect_cf_state(point, device, device_name)
        cf_layers = layer_diagnostic_rows("cf", cf_state, seed, args.probe_reg)
        cf_importance = supervised_layer_importance_rows(cf_state)
        all_cf_layers.extend(cf_layers)
        all_cf_stream.extend({"model": "cf", "seed": seed, **row} for row in cf_state["layer_rows"])
        all_cf_svd.extend({"model": "cf", "seed": seed, **row} for row in head_svd_rows(cf_state))
        all_cf_importance.extend({"model": "cf", "seed": seed, **row} for row in cf_importance)

        print(f"BP seed={seed} step_multiplier={args.bp_step_multiplier}", flush=True)
        bp_state = collect_backprop_state(point, device, device_name, args.bp_step_multiplier)
        bp_layers = layer_diagnostic_rows("backprop", bp_state, seed, args.probe_reg)
        bp_importance = supervised_layer_importance_rows(bp_state)
        all_bp_layers.extend(bp_layers)
        all_bp_stream.extend({"model": "backprop", "seed": seed, **row} for row in bp_state["layer_rows"])
        all_bp_svd.extend({"model": "backprop", "seed": seed, **row} for row in head_svd_rows(bp_state))
        all_bp_importance.extend({"model": "backprop", "seed": seed, **row} for row in bp_importance)

        write_jsonl(args.out_dir / "cf_layer_diagnostics.partial.jsonl", all_cf_layers)
        write_jsonl(args.out_dir / "backprop_layer_diagnostics.partial.jsonl", all_bp_layers)
        write_jsonl(args.out_dir / "cf_stream_rows.partial.jsonl", all_cf_stream)
        write_jsonl(args.out_dir / "backprop_stream_rows.partial.jsonl", all_bp_stream)

    write_jsonl(args.out_dir / "cf_layer_diagnostics.jsonl", all_cf_layers)
    write_jsonl(args.out_dir / "backprop_layer_diagnostics.jsonl", all_bp_layers)
    write_jsonl(args.out_dir / "cf_stream_rows.jsonl", all_cf_stream)
    write_jsonl(args.out_dir / "backprop_stream_rows.jsonl", all_bp_stream)
    write_jsonl(args.out_dir / "cf_head_svd_rows.jsonl", all_cf_svd)
    write_jsonl(args.out_dir / "backprop_head_svd_rows.jsonl", all_bp_svd)
    write_jsonl(args.out_dir / "cf_importance_rows.jsonl", all_cf_importance)
    write_jsonl(args.out_dir / "backprop_importance_rows.jsonl", all_bp_importance)
    write_jsonl(args.out_dir / "layer_diagnostics_aggregate.jsonl", aggregate(all_cf_layers + all_bp_layers, ["model", "layer"]))
    write_jsonl(args.out_dir / "stream_aggregate.jsonl", aggregate(all_cf_stream + all_bp_stream, ["model", "layer"]))
    report = report_text(point, all_cf_layers, all_bp_layers, all_cf_importance, all_bp_importance)
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


def main():
    parser = argparse.ArgumentParser(description="Mechanistic CF/backprop residual stream and per-layer representation diagnostics.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_mechanistic_layers"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.40)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--probe-reg", type=float, default=100.0)
    parser.add_argument("--bp-step-multiplier", type=float, default=1.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
