import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cf_mlp_scalability import (
    aggregate,
    architecture_dims,
    estimate_backprop_step_flops,
    estimate_cf_flops,
    load_point_data,
    make_fullres_diagnostic_sweep,
    make_harder_sweep,
    make_sweep,
    markdown_report,
    paired_summary,
    trend_summary,
    write_jsonl,
)


REG_EPS = 1e-4


def accuracy_from_logits_torch(logits, labels):
    return float((logits.argmax(dim=1) == labels).float().mean().detach().cpu().item())


def normalize_hidden_with_stats_torch(train_arrays, test_arrays):
    mean = sum(arr.mean(dim=0, keepdim=True) for arr in train_arrays) / len(train_arrays)
    centered_train = [arr - mean for arr in train_arrays]
    centered_test = [arr - mean for arr in test_arrays]
    avg_var = sum((arr * arr).mean(dim=0, keepdim=True) for arr in centered_train) / len(centered_train)
    scale = torch.sqrt(torch.clamp(avg_var, min=1e-6))
    return [arr / scale for arr in centered_train], [arr / scale for arr in centered_test], mean, scale


def one_hot_torch(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes).to(torch.float32)


def lambda_from_invariance_strength(invariance_strength):
    return 1.0 / max(float(invariance_strength), 1e-12)


def fit_cf_transform_torch(view1, view2, width, lambda_reg=None, invariance_strength=None, gain_floor=0.0):
    if invariance_strength is not None:
        if lambda_reg is not None:
            raise ValueError("Pass either lambda_reg or invariance_strength, not both")
        lambda_reg = lambda_from_invariance_strength(invariance_strength)
    if lambda_reg is None:
        raise ValueError("lambda_reg or invariance_strength is required")
    dim = view1.shape[1]
    out_dim = min(width, dim)
    mean = 0.5 * (view1.mean(dim=0, keepdim=True) + view2.mean(dim=0, keepdim=True))
    h1 = view1 - mean
    h2 = view2 - mean
    n = float(view1.shape[0])
    sigma1 = (h1.T @ h1) / n
    sigma2 = (h2.T @ h2) / n
    sigma_bar = 0.5 * (sigma1 + sigma2)
    sigma_bar = 0.5 * (sigma_bar + sigma_bar.T)
    delta_h = h1 - h2
    delta = (delta_h.T @ delta_h) / n
    delta = 0.5 * (delta + delta.T)

    evals_sigma, evecs_sigma = torch.linalg.eigh(sigma_bar)
    evals_sigma = torch.clamp(evals_sigma, min=REG_EPS)
    sigma_inv_sqrt = (evecs_sigma / torch.sqrt(evals_sigma).unsqueeze(0)) @ evecs_sigma.T
    m_matrix = sigma_inv_sqrt @ delta @ sigma_inv_sqrt
    m_matrix = 0.5 * (m_matrix + m_matrix.T)
    eigvals, eigvecs = torch.linalg.eigh(m_matrix)
    gains = lambda_reg / (torch.clamp(eigvals, min=0.0) + lambda_reg)
    if float(gain_floor) > 0.0:
        floor = torch.as_tensor(float(gain_floor), dtype=gains.dtype, device=gains.device)
        gains = floor + (1.0 - floor) * gains

    if width >= dim:
        sigma_sqrt = (evecs_sigma * torch.sqrt(evals_sigma).unsqueeze(0)) @ evecs_sigma.T
        g_matrix = (eigvecs * gains.unsqueeze(0)) @ eigvecs.T
        transform = sigma_sqrt @ g_matrix @ sigma_inv_sqrt
        kept_gains = gains
    else:
        order = torch.argsort(gains, descending=True)[:out_dim]
        modes = eigvecs[:, order]
        kept_gains = gains[order]
        transform = sigma_inv_sqrt @ (modes * kept_gains.unsqueeze(0))

    return {
        "transform": transform,
        "lambda_reg": float(lambda_reg),
        "invariance_strength": float(1.0 / max(float(lambda_reg), 1e-12)),
        "gain_floor": float(gain_floor),
        "max_whitened_delta": float(eigvals.max().detach().cpu().item()),
        "min_whitened_delta": float(eigvals.min().detach().cpu().item()),
        "mean_gain": float(kept_gains.mean().detach().cpu().item()),
        "min_gain": float(kept_gains.min().detach().cpu().item()),
    }


def fit_whitening_transform_torch(view1, view2, width):
    dim = view1.shape[1]
    out_dim = min(width, dim)
    mean = 0.5 * (view1.mean(dim=0, keepdim=True) + view2.mean(dim=0, keepdim=True))
    h1 = view1 - mean
    h2 = view2 - mean
    n = float(view1.shape[0])
    sigma1 = (h1.T @ h1) / n
    sigma2 = (h2.T @ h2) / n
    sigma_bar = 0.5 * (sigma1 + sigma2)
    sigma_bar = 0.5 * (sigma_bar + sigma_bar.T)

    evals_sigma, evecs_sigma = torch.linalg.eigh(sigma_bar)
    evals_sigma = torch.clamp(evals_sigma, min=REG_EPS)
    if width >= dim:
        transform = (evecs_sigma / torch.sqrt(evals_sigma).unsqueeze(0)) @ evecs_sigma.T
        kept_evals = evals_sigma
    else:
        order = torch.argsort(evals_sigma, descending=True)[:out_dim]
        kept_evals = evals_sigma[order]
        transform = evecs_sigma[:, order] / torch.sqrt(kept_evals).unsqueeze(0)

    return {
        "transform": transform,
        "lambda_reg": float("nan"),
        "invariance_strength": float("nan"),
        "max_whitened_delta": float("nan"),
        "min_whitened_delta": float("nan"),
        "mean_gain": 1.0,
        "min_gain": 1.0,
        "max_sigma_eval": float(kept_evals.max().detach().cpu().item()),
        "min_sigma_eval": float(kept_evals.min().detach().cpu().item()),
    }


def ridge_regression_torch(x, y, reg):
    gram = x.T @ x
    rhs = x.T @ y
    eye = torch.eye(gram.shape[0], dtype=x.dtype, device=x.device)
    return torch.linalg.solve(gram + reg * eye, rhs)


def tensors_for_point(point, device):
    arrays = load_point_data(point)
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


def fit_cf_mlp_gpu(point, tensors, device_name):
    xtr, ytr, xte, yte, view1_tr, view2_tr, view1_te, view2_te = tensors
    train_arrays, test_arrays, initial_mean, initial_scale = normalize_hidden_with_stats_torch(
        [xtr, view1_tr, view2_tr],
        [xte, view1_te, view2_te],
    )
    base_tr, view1_tr, view2_tr = train_arrays
    base_te, view1_te, view2_te = test_arrays
    y_onehot = one_hot_torch(ytr, point.num_classes)
    yhat_tr = torch.zeros_like(y_onehot)
    yhat_te = torch.zeros((yte.shape[0], point.num_classes), dtype=torch.float32, device=yte.device)
    layers = []
    torch.cuda.synchronize()
    start = time.perf_counter()

    for layer_idx in range(point.depth):
        fitted = fit_cf_transform_torch(view1_tr, view2_tr, point.width, point.lambda_reg)
        transform = fitted["transform"]
        base_tr = torch.relu(base_tr @ transform)
        base_te = torch.relu(base_te @ transform)
        view1_tr = torch.relu(view1_tr @ transform)
        view2_tr = torch.relu(view2_tr @ transform)
        view1_te = torch.relu(view1_te @ transform)
        view2_te = torch.relu(view2_te @ transform)

        out_map = ridge_regression_torch(base_tr, y_onehot - yhat_tr, point.head_reg)
        yhat_tr = yhat_tr + base_tr @ out_map
        yhat_te = yhat_te + base_te @ out_map
        layers.append(
            {
                "depth": layer_idx + 1,
                "accuracy": accuracy_from_logits_torch(yhat_te, yte),
                "max_whitened_delta": fitted["max_whitened_delta"],
                "min_whitened_delta": fitted["min_whitened_delta"],
                "mean_gain": fitted["mean_gain"],
                "min_gain": fitted["min_gain"],
            }
        )

        train_arrays, test_arrays, _, _ = normalize_hidden_with_stats_torch(
            [base_tr, view1_tr, view2_tr],
            [base_te, view1_te, view2_te],
        )
        base_tr, view1_tr, view2_tr = train_arrays
        base_te, view1_te, view2_te = test_arrays

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "model": "cf-mlp",
        "accuracy": accuracy_from_logits_torch(yhat_te, yte),
        "cross_entropy": float(F.cross_entropy(yhat_te, yte).detach().cpu().item()),
        "fit_time_sec": float(elapsed),
        "layers": layers,
        "initial_mean_norm": float(torch.linalg.vector_norm(initial_mean).detach().cpu().item()),
        "initial_scale_mean": float(initial_scale.mean().detach().cpu().item()),
        "backend": "torch-cuda",
        "device": device_name,
    }


def init_backprop_params(point, device):
    torch.manual_seed(point.seed + 404)
    dims = architecture_dims(point.input_dim, point.width, point.depth)
    weights = torch.nn.ParameterList()
    heads = torch.nn.ParameterList()
    for idx in range(point.depth):
        fan_in = dims[idx]
        fan_out = dims[idx + 1]
        weights.append(torch.nn.Parameter(torch.randn((fan_in, fan_out), device=device) * math.sqrt(2.0 / fan_in)))
        heads.append(torch.nn.Parameter(torch.randn((fan_out, point.num_classes), device=device) * math.sqrt(1.0 / fan_out)))
    return weights, heads


def forward_residual_mlp_torch(x, weights, heads):
    h = x
    cumulative = torch.zeros((x.shape[0], heads[0].shape[1]), dtype=x.dtype, device=x.device)
    logits = []
    for weight, head in zip(weights, heads):
        h = torch.relu(h @ weight)
        cumulative = cumulative + h @ head
        logits.append(cumulative)
    return logits


def fit_backprop_residual_mlp_gpu(point, tensors, norm_mean, norm_scale, cf_budget, device_name):
    xtr, ytr, xte, yte, *_ = tensors
    xtr_norm = (xtr - norm_mean) / norm_scale
    xte_norm = (xte - norm_mean) / norm_scale
    weights, heads = init_backprop_params(point, xtr.device)
    params = list(weights.parameters()) + list(heads.parameters())
    optimizer = torch.optim.AdamW(params, lr=point.lr, weight_decay=point.weight_decay)
    step_flops = estimate_backprop_step_flops(point)
    max_steps = max(1, int(math.floor(cf_budget / max(step_flops, 1.0))))
    steps_per_epoch = max(1, int(math.ceil(point.n_train / point.batch_size)))
    cursor = 0
    permutation = torch.randperm(point.n_train, device=xtr.device)
    losses = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(max_steps):
        if cursor + point.batch_size > point.n_train:
            permutation = torch.randperm(point.n_train, device=xtr.device)
            cursor = 0
        batch_idx = permutation[cursor : cursor + point.batch_size]
        cursor += point.batch_size
        logits = forward_residual_mlp_torch(xtr_norm[batch_idx], weights, heads)
        loss = sum(F.cross_entropy(out, ytr[batch_idx]) for out in logits) / len(logits)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    with torch.no_grad():
        depth_logits = forward_residual_mlp_torch(xte_norm, weights, heads)
        final_logits = depth_logits[-1]
        layer_acc = [accuracy_from_logits_torch(out, yte) for out in depth_logits]
    return {
        "model": "backprop-residual-mlp",
        "accuracy": accuracy_from_logits_torch(final_logits, yte),
        "cross_entropy": float(F.cross_entropy(final_logits, yte).detach().cpu().item()),
        "fit_time_sec": float(elapsed),
        "steps": int(max_steps),
        "effective_epochs": float(max_steps / steps_per_epoch),
        "mean_train_loss": float(np.mean(losses[-min(len(losses), steps_per_epoch) :])),
        "step_flops_proxy": float(step_flops),
        "used_flops_proxy": float(max_steps * step_flops),
        "layer_accuracy": layer_acc,
        "backend": "torch-cuda",
        "device": device_name,
    }


def run_point_gpu(point, device, device_name):
    tensors = tensors_for_point(point, device)
    _, _, norm_mean, norm_scale = normalize_hidden_with_stats_torch(
        [tensors[0], tensors[4], tensors[5]],
        [tensors[2], tensors[6], tensors[7]],
    )
    cf_budget = estimate_cf_flops(point)
    start = time.perf_counter()
    cf = fit_cf_mlp_gpu(point, tensors, device_name)
    bp = fit_backprop_residual_mlp_gpu(point, tensors, norm_mean, norm_scale, cf_budget, device_name)
    elapsed = time.perf_counter() - start
    base = {
        **asdict(point),
        "architecture_dims": architecture_dims(point.input_dim, point.width, point.depth),
        "cf_flops_proxy": float(cf_budget),
        "backprop_step_flops_proxy": float(estimate_backprop_step_flops(point)),
        "total_pair_elapsed_sec": float(elapsed),
    }
    rows = []
    for result in (cf, bp):
        rows.append(
            {
                **base,
                **result,
                "error_rate": float(1.0 - result["accuracy"]),
            }
        )
    del tensors
    torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description="GPU CIFAR100 CF-MLP scalability sweep.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_scalability_tests/artifacts"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--harder", action="store_true")
    parser.add_argument("--fullres-diagnostic", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.55)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(args.torch_threads)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)

    if args.fullres_diagnostic:
        points = make_fullres_diagnostic_sweep(quick=args.quick)
    elif args.harder:
        points = make_harder_sweep(quick=args.quick)
    else:
        points = make_sweep(quick=args.quick)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, point in enumerate(points, start=1):
        print(
            f"[{idx}/{len(points)}] {point.dataset} axis={point.axis} scale={point.scale_value} seed={point.seed} "
            f"n={point.n_train} width={point.width} depth={point.depth} device={device_name}",
            flush=True,
        )
        point_rows = run_point_gpu(point, device, device_name)
        rows.extend(point_rows)
        write_jsonl(args.out_dir / "cf_mlp_scalability_rows.partial.jsonl", rows)

    agg_rows = aggregate(rows)
    paired_rows = paired_summary(rows)
    trends = trend_summary(paired_rows)
    write_jsonl(args.out_dir / "cf_mlp_scalability_rows.jsonl", rows)
    write_jsonl(args.out_dir / "cf_mlp_scalability_aggregate.jsonl", agg_rows)
    write_jsonl(args.out_dir / "cf_mlp_scalability_paired.jsonl", paired_rows)
    write_jsonl(args.out_dir / "cf_mlp_scalability_trends.jsonl", trends)
    report = markdown_report(agg_rows, paired_rows, trends)
    (args.out_dir / "cf_mlp_scalability_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
