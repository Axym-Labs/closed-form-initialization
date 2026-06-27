import argparse
import gc
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from cf_mlp_layer_mechanistic import covariance_spectrum, view_alignment
from cf_mlp_representation import (
    apply_map_np,
    one_hot_np,
    ridge_map_np,
    softmax_ce_np,
    standardize_pair,
    tensors_from_arrays,
)
from cf_mlp_scalability import SweepPoint, accuracy_from_logits, load_point_data, write_jsonl
from cf_mlp_scalability_gpu import (
    REG_EPS,
    fit_cf_transform_torch,
    lambda_from_invariance_strength,
    normalize_hidden_with_stats_torch,
)
from cf_mlp_residual_barlow import BP_BT_ACTIVATION, BP_BT_ACTIVATION_ALPHA


def geometric(base, depth, factor):
    return [float(base * (factor**idx)) for idx in range(depth)]


def linear_schedule(start, end, depth):
    if depth <= 1:
        return [float(end)]
    return [float(start + (end - start) * idx / (depth - 1)) for idx in range(depth)]


def scheduled_spec_value(spec, key, layer_idx):
    value = spec[key]
    if isinstance(value, (list, tuple)):
        return value[layer_idx]
    return value


def normalize_hidden_zca_torch(train_arrays, test_arrays, eps=1e-4):
    mean = sum(arr.mean(dim=0, keepdim=True) for arr in train_arrays) / len(train_arrays)
    centered_train = [arr - mean for arr in train_arrays]
    centered_test = [arr - mean for arr in test_arrays]
    cov = None
    for arr in centered_train:
        arr_cov = (arr.T @ arr) / float(arr.shape[0])
        cov = arr_cov if cov is None else cov + arr_cov
    cov = 0.5 * ((cov / len(centered_train)) + (cov / len(centered_train)).T)
    evals, evecs = torch.linalg.eigh(cov)
    evals = torch.clamp(evals, min=float(eps))
    inv_sqrt = (evecs / torch.sqrt(evals).unsqueeze(0)) @ evecs.T
    return [arr @ inv_sqrt for arr in centered_train], [arr @ inv_sqrt for arr in centered_test], mean, inv_sqrt


def normalize_hidden_for_spec_torch(spec, train_arrays, test_arrays):
    norm_kind = spec.get("norm_kind", "feature")
    if norm_kind == "feature":
        return normalize_hidden_with_stats_torch(train_arrays, test_arrays)
    if norm_kind == "fullwhiten":
        return normalize_hidden_zca_torch(train_arrays, test_arrays)
    raise ValueError(f"Unknown hidden normalization kind: {norm_kind}")


def default_schedule(depth):
    return geometric(1.0, depth, 0.25)


def apply_activation_torch(x, activation, alpha):
    if activation == "relu":
        return torch.relu(x)
    if activation == "leaky_gelu":
        return F.gelu(x) + float(alpha) * torch.minimum(x, torch.zeros((), dtype=x.dtype, device=x.device))
    if activation == "identity":
        return x
    raise ValueError(f"Unsupported activation: {activation}")


def leaky_gelu_inverse_newton(y, alpha, steps=8):
    x = torch.where(y < 0, y / max(float(alpha), 1e-4), y)
    sqrt_2pi = float(np.sqrt(2.0 * np.pi))
    inv_sqrt2 = float(1.0 / np.sqrt(2.0))
    for _ in range(steps):
        gelu = F.gelu(x)
        neg = torch.minimum(x, torch.zeros((), dtype=x.dtype, device=x.device))
        fx = gelu + float(alpha) * neg - y
        cdf = 0.5 * (1.0 + torch.erf(x * inv_sqrt2))
        pdf = torch.exp(-0.5 * x * x) / sqrt_2pi
        deriv = cdf + x * pdf + float(alpha) * (x < 0).to(x.dtype)
        x = x - fx / torch.clamp(deriv, min=1e-3)
    return torch.clamp(x, min=-20.0, max=20.0)


def ridge_solve_torch(x, y, reg):
    gram = x.T @ x
    rhs = x.T @ y
    eye = torch.eye(gram.shape[0], dtype=x.dtype, device=x.device)
    return torch.linalg.solve(gram + float(reg) * eye, rhs)


def shared_cov_geometry(view1, view2):
    dim = view1.shape[1]
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
    sigma_sqrt = (evecs_sigma * torch.sqrt(evals_sigma).unsqueeze(0)) @ evecs_sigma.T
    sigma_inv_sqrt = (evecs_sigma / torch.sqrt(evals_sigma).unsqueeze(0)) @ evecs_sigma.T
    m_matrix = sigma_inv_sqrt @ delta @ sigma_inv_sqrt
    m_matrix = 0.5 * (m_matrix + m_matrix.T)
    return sigma_sqrt, sigma_inv_sqrt, m_matrix, dim


def fit_cf_transform_with_prior_torch(view1, view2, width, invariance_strength, prior_transform):
    lambda_reg = lambda_from_invariance_strength(invariance_strength)
    sigma_sqrt, sigma_inv_sqrt, m_matrix, dim = shared_cov_geometry(view1, view2)
    if width < dim:
        raise ValueError("Prior-centered CF transform is only implemented for square/full-width layers.")
    eigvals, eigvecs = torch.linalg.eigh(m_matrix)
    gains = lambda_reg / (torch.clamp(eigvals, min=0.0) + lambda_reg)
    shrink = (eigvecs * gains.unsqueeze(0)) @ eigvecs.T
    prior_whitened = sigma_inv_sqrt @ prior_transform @ sigma_sqrt
    g_matrix = shrink @ prior_whitened
    transform = sigma_sqrt @ g_matrix @ sigma_inv_sqrt
    return {
        "transform": transform,
        "lambda_reg": float(lambda_reg),
        "invariance_strength": float(invariance_strength),
        "max_whitened_delta": float(eigvals.max().detach().cpu().item()),
        "min_whitened_delta": float(eigvals.min().detach().cpu().item()),
        "mean_gain": float(gains.mean().detach().cpu().item()),
        "min_gain": float(gains.min().detach().cpu().item()),
    }


def fit_cf_transform_row_correct_torch(view1, view2, width, invariance_strength):
    lambda_reg = lambda_from_invariance_strength(invariance_strength)
    sigma_sqrt, sigma_inv_sqrt, m_matrix, dim = shared_cov_geometry(view1, view2)
    if width < dim:
        raise ValueError("Row-correct CF transform is only implemented for square/full-width layers.")
    eigvals, eigvecs = torch.linalg.eigh(m_matrix)
    gains = lambda_reg / (torch.clamp(eigvals, min=0.0) + lambda_reg)
    g_matrix = (eigvecs * gains.unsqueeze(0)) @ eigvecs.T
    transform = sigma_inv_sqrt @ g_matrix @ sigma_sqrt
    return {
        "transform": transform,
        "lambda_reg": float(lambda_reg),
        "invariance_strength": float(invariance_strength),
        "max_whitened_delta": float(eigvals.max().detach().cpu().item()),
        "min_whitened_delta": float(eigvals.min().detach().cpu().item()),
        "mean_gain": float(gains.mean().detach().cpu().item()),
        "min_gain": float(gains.min().detach().cpu().item()),
    }


def fit_cf_transform_shared_metric_torch(view1, view2, width, invariance_strength):
    lambda_reg = lambda_from_invariance_strength(invariance_strength)
    dim = view1.shape[1]
    if width < dim:
        raise ValueError("Shared-metric CF transform is only implemented for square/full-width layers.")
    mean = 0.5 * (view1.mean(dim=0, keepdim=True) + view2.mean(dim=0, keepdim=True))
    h1 = view1 - mean
    h2 = view2 - mean
    shared = 0.5 * (h1 + h2)
    diff = 0.5 * (h1 - h2)
    n = float(view1.shape[0])
    sigma_shared = (shared.T @ shared) / n
    sigma_shared = 0.5 * (sigma_shared + sigma_shared.T)
    sigma_diff = (diff.T @ diff) / n
    sigma_diff = 0.5 * (sigma_diff + sigma_diff.T)

    evals_shared, evecs_shared = torch.linalg.eigh(sigma_shared)
    evals_shared = torch.clamp(evals_shared, min=REG_EPS)
    shared_sqrt = (evecs_shared * torch.sqrt(evals_shared).unsqueeze(0)) @ evecs_shared.T
    shared_inv_sqrt = (evecs_shared / torch.sqrt(evals_shared).unsqueeze(0)) @ evecs_shared.T
    m_matrix = shared_inv_sqrt @ sigma_diff @ shared_inv_sqrt
    m_matrix = 0.5 * (m_matrix + m_matrix.T)
    eigvals, eigvecs = torch.linalg.eigh(m_matrix)
    gains = lambda_reg / (torch.clamp(eigvals, min=0.0) + lambda_reg)
    shrink = (eigvecs * gains.unsqueeze(0)) @ eigvecs.T
    transform = shared_sqrt @ shrink @ shared_inv_sqrt
    return {
        "transform": transform,
        "lambda_reg": float(lambda_reg),
        "invariance_strength": float(invariance_strength),
        "max_shared_metric_delta": float(eigvals.max().detach().cpu().item()),
        "min_shared_metric_delta": float(eigvals.min().detach().cpu().item()),
        "mean_gain": float(gains.mean().detach().cpu().item()),
        "min_gain": float(gains.min().detach().cpu().item()),
        "shared_metric": True,
    }


def fit_agreement_mode_transform_torch(view1, view2, width, invariance_strength, use_gain, gate_beta):
    lambda_reg = lambda_from_invariance_strength(invariance_strength)
    _, sigma_inv_sqrt, m_matrix, dim = shared_cov_geometry(view1, view2)
    if width < dim:
        raise ValueError("Agreement-mode transform is only implemented for square/full-width layers.")
    eigvals, eigvecs = torch.linalg.eigh(m_matrix)
    eigvals_clamped = torch.clamp(eigvals, min=0.0)
    gains = lambda_reg / (eigvals_clamped + lambda_reg)
    suppression = 1.0 - gains
    scales = gains if use_gain else torch.ones_like(gains)
    transform = sigma_inv_sqrt @ (eigvecs * scales.unsqueeze(0))
    bias = -float(gate_beta) * suppression
    return {
        "transform": transform,
        "bias": bias,
        "lambda_reg": float(lambda_reg),
        "invariance_strength": float(invariance_strength),
        "max_whitened_delta": float(eigvals.max().detach().cpu().item()),
        "min_whitened_delta": float(eigvals.min().detach().cpu().item()),
        "mean_gain": float(gains.mean().detach().cpu().item()),
        "min_gain": float(gains.min().detach().cpu().item()),
        "mean_suppression": float(suppression.mean().detach().cpu().item()),
        "max_suppression": float(suppression.max().detach().cpu().item()),
        "gate_beta": float(gate_beta),
        "bias_mean": float(bias.mean().detach().cpu().item()),
        "bias_min": float(bias.min().detach().cpu().item()),
        "bias_max": float(bias.max().detach().cpu().item()),
        "mode_aligned": True,
        "mode_gain_used": bool(use_gain),
        "gains": gains.detach(),
        "eigvals": eigvals.detach(),
    }


def deterministic_subspace_mixer_torch(rows, cols, dtype, device):
    row = torch.arange(1, rows + 1, dtype=dtype, device=device).unsqueeze(1)
    col = torch.arange(1, cols + 1, dtype=dtype, device=device).unsqueeze(0)
    mixer = torch.sin(row * col * 12.9898) + torch.cos((row + 0.37) * (col + 0.19) * 78.233)
    mixer = mixer - mixer.mean(dim=0, keepdim=True)
    mixer = mixer / torch.clamp(torch.linalg.vector_norm(mixer, dim=0, keepdim=True), min=1e-6)
    return mixer


def fit_agreement_subspace_expand_transform_torch(view1, view2, width, keep_dim):
    _, sigma_inv_sqrt, m_matrix, dim = shared_cov_geometry(view1, view2)
    if width < dim:
        raise ValueError("Agreement subspace expansion is only implemented for square/full-width layers.")
    keep = int(max(1, min(int(keep_dim), dim)))
    eigvals, eigvecs = torch.linalg.eigh(m_matrix)
    order = torch.argsort(eigvals, descending=False)
    kept = order[:keep]
    kept_modes = eigvecs[:, kept]
    mixer = deterministic_subspace_mixer_torch(keep, width, view1.dtype, view1.device)
    transform = sigma_inv_sqrt @ kept_modes @ mixer
    kept_eigvals = eigvals[kept]
    dropped_eigvals = eigvals[order[keep:]] if keep < dim else eigvals.new_empty((0,))
    dropped_mean = (
        float(dropped_eigvals.mean().detach().cpu().item())
        if dropped_eigvals.numel()
        else float("nan")
    )
    return {
        "transform": transform,
        "lambda_reg": float("nan"),
        "invariance_strength": float("nan"),
        "max_whitened_delta": float(eigvals.max().detach().cpu().item()),
        "min_whitened_delta": float(eigvals.min().detach().cpu().item()),
        "mean_gain": float("nan"),
        "min_gain": float("nan"),
        "agreement_expand_keep_dim": keep,
        "agreement_expand_kept_delta_mean": float(kept_eigvals.mean().detach().cpu().item()),
        "agreement_expand_kept_delta_max": float(kept_eigvals.max().detach().cpu().item()),
        "agreement_expand_dropped_delta_mean": dropped_mean,
        "agreement_expand_rank_fraction": float(keep / dim),
        "agreement_expand": True,
    }


def fit_activation_inverse_prior(base, view1, view2, activation_alpha, reg):
    x = torch.cat([base, view1, view2], dim=0)
    target = leaky_gelu_inverse_newton(x, activation_alpha)
    prior = ridge_solve_torch(x, target, reg)
    with torch.no_grad():
        recon = apply_activation_torch(base @ prior, "leaky_gelu", activation_alpha)
        mse = torch.mean((recon - base) ** 2)
        denom = torch.linalg.vector_norm(recon, dim=1) * torch.linalg.vector_norm(base, dim=1)
        cos = torch.mean(torch.sum(recon * base, dim=1) / torch.clamp(denom, min=1e-12))
    return prior, {
        "prior_recon_mse": float(mse.detach().cpu().item()),
        "prior_recon_cosine": float(cos.detach().cpu().item()),
    }


def offdiag(x):
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_loss_torch(z1, z2, lambd):
    z1 = (z1 - z1.mean(dim=0)) / torch.clamp(z1.std(dim=0), min=1e-4)
    z2 = (z2 - z2.mean(dim=0)) / torch.clamp(z2.std(dim=0), min=1e-4)
    corr = (z1.T @ z2) / z1.shape[0]
    on_diag = torch.diagonal(corr).add(-1.0).pow(2).sum()
    off_diag = offdiag(corr).pow(2).sum()
    return on_diag + float(lambd) * off_diag, on_diag, off_diag


def corr_diag_mean_torch(z1, z2):
    z1 = (z1 - z1.mean(dim=0, keepdim=True)) / torch.clamp(z1.std(dim=0, keepdim=True), min=1e-4)
    z2 = (z2 - z2.mean(dim=0, keepdim=True)) / torch.clamp(z2.std(dim=0, keepdim=True), min=1e-4)
    corr = (z1.T @ z2) / z1.shape[0]
    return torch.diagonal(corr).mean()


def bt_total_per_dim_torch(z1, z2, lambd):
    total, _, _ = barlow_loss_torch(z1, z2, lambd)
    return total / float(z1.shape[1])


def corr1d_torch(a, b):
    a = a.detach().flatten()
    b = b.detach().flatten()
    ac = a - a.mean()
    bc = b - b.mean()
    denom = torch.sqrt(torch.sum(ac * ac) * torch.sum(bc * bc))
    if float(denom.detach().cpu().item()) <= 1e-12:
        return float("nan")
    return float((torch.sum(ac * bc) / denom).detach().cpu().item())


def paired_preactivation_stats(pre1, pre2):
    mean = 0.5 * (pre1.mean(dim=0) + pre2.mean(dim=0))
    centered1 = pre1 - mean
    centered2 = pre2 - mean
    var = 0.5 * (centered1.pow(2).mean(dim=0) + centered2.pow(2).mean(dim=0))
    std = torch.sqrt(torch.clamp(var, min=1e-8))
    rho = (centered1 * centered2).mean(dim=0) / torch.clamp(std * std, min=1e-8)
    rho = torch.clamp(rho, min=-1.0, max=1.0)
    delta = (pre1 - pre2).pow(2).mean(dim=0)
    return mean, std, rho, delta


def fit_postrelu_affine_torch(pre1, pre2, args, optimize_scale=True, objective="bt"):
    sample_count = min(int(args.postrelu_fit_samples), pre1.shape[0])
    if sample_count <= 0 or sample_count >= pre1.shape[0]:
        fit1 = pre1.detach()
        fit2 = pre2.detach()
    else:
        idx = torch.linspace(0, pre1.shape[0] - 1, sample_count, device=pre1.device).long()
        fit1 = pre1.index_select(0, idx).detach()
        fit2 = pre2.index_select(0, idx).detach()

    if objective not in {"bt", "on_diag"}:
        raise ValueError(f"Unknown post-ReLU affine objective: {objective}")
    if optimize_scale:
        log_scale = torch.nn.Parameter(torch.zeros(pre1.shape[1], dtype=pre1.dtype, device=pre1.device))
        params = [log_scale]
    else:
        log_scale = None
        params = []
    bias = torch.nn.Parameter(torch.zeros(pre1.shape[1], dtype=pre1.dtype, device=pre1.device))
    params.append(bias)
    optimizer = torch.optim.Adam(params, lr=float(args.postrelu_lr))

    last_loss = float("nan")
    last_bt = float("nan")
    last_on = float("nan")
    last_off = float("nan")
    for _ in range(int(args.postrelu_steps)):
        optimizer.zero_grad(set_to_none=True)
        if optimize_scale:
            scale = torch.exp(torch.clamp(log_scale, min=-4.0, max=4.0))
        else:
            scale = torch.ones(pre1.shape[1], dtype=pre1.dtype, device=pre1.device)
        z1 = torch.relu(fit1 * scale + bias)
        z2 = torch.relu(fit2 * scale + bias)
        bt, on_diag, off_diag = barlow_loss_torch(z1, z2, args.bt_offdiag_lambda)
        scale_penalty = torch.mean(log_scale * log_scale) if optimize_scale else torch.zeros((), dtype=pre1.dtype, device=pre1.device)
        bias_penalty = torch.mean(bias * bias)
        objective_value = bt if objective == "bt" else on_diag
        loss = (
            objective_value
            + float(args.postrelu_scale_ridge) * scale_penalty
            + float(args.postrelu_bias_ridge) * bias_penalty
        )
        loss.backward()
        if args.postrelu_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, float(args.postrelu_grad_clip))
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())
        last_bt = float(bt.detach().cpu().item())
        last_on = float(on_diag.detach().cpu().item())
        last_off = float(off_diag.detach().cpu().item())

    with torch.no_grad():
        if optimize_scale:
            scale = torch.exp(torch.clamp(log_scale, min=-4.0, max=4.0)).detach()
        else:
            scale = torch.ones(pre1.shape[1], dtype=pre1.dtype, device=pre1.device)
        bias_out = bias.detach()
        aff1 = fit1 * scale + bias_out
        aff2 = fit2 * scale + bias_out
        active = 0.5 * ((aff1 > 0).to(fit1.dtype).mean(dim=0) + (aff2 > 0).to(fit2.dtype).mean(dim=0))
        threshold = -bias_out / torch.clamp(scale, min=1e-8)
        _, pre_std, pre_corr, pre_delta = paired_preactivation_stats(fit1, fit2)
    return scale, bias_out, {
        "postrelu_opt_loss": last_loss,
        "postrelu_opt_bt": last_bt,
        "postrelu_opt_on_diag": last_on,
        "postrelu_opt_off_diag": last_off,
        "postrelu_fit_samples": int(sample_count),
        "postrelu_steps": int(args.postrelu_steps),
        "postrelu_lr": float(args.postrelu_lr),
        "postrelu_optimize_scale": bool(optimize_scale),
        "postrelu_objective": objective,
        "postrelu_scale_mean": float(scale.mean().detach().cpu().item()),
        "postrelu_scale_min": float(scale.min().detach().cpu().item()),
        "postrelu_scale_max": float(scale.max().detach().cpu().item()),
        "postrelu_bias_mean": float(bias_out.mean().detach().cpu().item()),
        "postrelu_bias_min": float(bias_out.min().detach().cpu().item()),
        "postrelu_bias_max": float(bias_out.max().detach().cpu().item()),
        "postrelu_active_mean": float(active.mean().detach().cpu().item()),
        "postrelu_active_min": float(active.min().detach().cpu().item()),
        "postrelu_active_max": float(active.max().detach().cpu().item()),
        "postrelu_threshold_mean": float(threshold.mean().detach().cpu().item()),
        "postrelu_threshold_min": float(threshold.min().detach().cpu().item()),
        "postrelu_threshold_max": float(threshold.max().detach().cpu().item()),
        "postrelu_scale_corr_pre_corr": corr1d_torch(scale, pre_corr),
        "postrelu_bias_corr_pre_corr": corr1d_torch(bias_out, pre_corr),
        "postrelu_threshold_corr_pre_corr": corr1d_torch(threshold, pre_corr),
        "postrelu_scale_corr_pre_std": corr1d_torch(scale, pre_std),
        "postrelu_bias_corr_pre_std": corr1d_torch(bias_out, pre_std),
        "postrelu_threshold_corr_pre_std": corr1d_torch(threshold, pre_std),
        "postrelu_scale_corr_pre_delta": corr1d_torch(scale, pre_delta),
        "postrelu_bias_corr_pre_delta": corr1d_torch(bias_out, pre_delta),
        "postrelu_threshold_corr_pre_delta": corr1d_torch(threshold, pre_delta),
    }


def sample_postrelu_fit_pair(pre1, pre2, args):
    sample_count = min(int(args.postrelu_fit_samples), pre1.shape[0])
    if sample_count <= 0 or sample_count >= pre1.shape[0]:
        return pre1.detach(), pre2.detach(), int(pre1.shape[0])
    idx = torch.linspace(0, pre1.shape[0] - 1, sample_count, device=pre1.device).long()
    return pre1.index_select(0, idx).detach(), pre2.index_select(0, idx).detach(), int(sample_count)


def active_rate_for_bias(pre1, pre2, bias):
    return 0.5 * (
        (pre1 + bias > 0).to(pre1.dtype).mean(dim=0)
        + (pre2 + bias > 0).to(pre2.dtype).mean(dim=0)
    )


def fit_postrelu_active_bias_torch(pre1, pre2, active_target, args):
    fit1, fit2, sample_count = sample_postrelu_fit_pair(pre1, pre2, args)
    target = float(active_target)
    if not 0.0 < target < 1.0:
        raise ValueError(f"active_target must be in (0, 1), got {target}")
    pooled = torch.cat([fit1, fit2], dim=0)
    threshold = torch.quantile(pooled, 1.0 - target, dim=0)
    bias = -threshold.detach()
    with torch.no_grad():
        active = active_rate_for_bias(fit1, fit2, bias)
    return bias, {
        "postrelu_heuristic": "active_target",
        "postrelu_fit_samples": sample_count,
        "postrelu_active_target": target,
        "postrelu_active_mean": float(active.mean().detach().cpu().item()),
        "postrelu_active_min": float(active.min().detach().cpu().item()),
        "postrelu_active_max": float(active.max().detach().cpu().item()),
        "postrelu_bias_mean": float(bias.mean().detach().cpu().item()),
        "postrelu_bias_min": float(bias.min().detach().cpu().item()),
        "postrelu_bias_max": float(bias.max().detach().cpu().item()),
    }


def columnwise_quantile(values, q):
    if values.shape[1] != q.numel():
        raise ValueError("Quantile target count must match feature dimension.")
    q = torch.clamp(q.to(dtype=values.dtype, device=values.device), min=0.0, max=1.0)
    sorted_values, _ = torch.sort(values, dim=0)
    pos = q * float(values.shape[0] - 1)
    lo = torch.floor(pos).long()
    hi = torch.ceil(pos).long()
    mix = (pos - lo.to(values.dtype)).to(values.dtype)
    cols = torch.arange(values.shape[1], device=values.device)
    low_values = sorted_values[lo, cols]
    high_values = sorted_values[hi, cols]
    return low_values * (1.0 - mix) + high_values * mix


def fit_postrelu_active_bias_targets_torch(pre1, pre2, active_targets, args):
    fit1, fit2, sample_count = sample_postrelu_fit_pair(pre1, pre2, args)
    targets = torch.clamp(active_targets.detach().to(dtype=fit1.dtype, device=fit1.device), min=1e-4, max=1.0 - 1e-4)
    pooled = torch.cat([fit1, fit2], dim=0)
    threshold = columnwise_quantile(pooled, 1.0 - targets)
    bias = -threshold.detach()
    with torch.no_grad():
        active = active_rate_for_bias(fit1, fit2, bias)
    return bias, {
        "postrelu_heuristic": "active_targets",
        "postrelu_fit_samples": sample_count,
        "postrelu_active_target_mean": float(targets.mean().detach().cpu().item()),
        "postrelu_active_target_min": float(targets.min().detach().cpu().item()),
        "postrelu_active_target_max": float(targets.max().detach().cpu().item()),
        "postrelu_active_mean": float(active.mean().detach().cpu().item()),
        "postrelu_active_min": float(active.min().detach().cpu().item()),
        "postrelu_active_max": float(active.max().detach().cpu().item()),
        "postrelu_active_target_corr_active": corr1d_torch(targets, active),
        "postrelu_bias_mean": float(bias.mean().detach().cpu().item()),
        "postrelu_bias_min": float(bias.min().detach().cpu().item()),
        "postrelu_bias_max": float(bias.max().detach().cpu().item()),
    }


def fit_postrelu_corrbias_torch(pre1, pre2, beta, args):
    fit1, fit2, sample_count = sample_postrelu_fit_pair(pre1, pre2, args)
    mean, std, rho, _ = paired_preactivation_stats(fit1, fit2)
    bias = (float(beta) * (1.0 - rho) * std - mean).detach()
    with torch.no_grad():
        active = active_rate_for_bias(fit1, fit2, bias)
    return bias, {
        "postrelu_heuristic": "corrbias",
        "postrelu_fit_samples": sample_count,
        "postrelu_corrbias_beta": float(beta),
        "postrelu_pre_corr_mean": float(rho.mean().detach().cpu().item()),
        "postrelu_pre_corr_min": float(rho.min().detach().cpu().item()),
        "postrelu_pre_corr_max": float(rho.max().detach().cpu().item()),
        "postrelu_active_mean": float(active.mean().detach().cpu().item()),
        "postrelu_active_min": float(active.min().detach().cpu().item()),
        "postrelu_active_max": float(active.max().detach().cpu().item()),
        "postrelu_bias_mean": float(bias.mean().detach().cpu().item()),
        "postrelu_bias_min": float(bias.min().detach().cpu().item()),
        "postrelu_bias_max": float(bias.max().detach().cpu().item()),
    }


def fit_postnorm_linear_bt_torch(view1, view2, args):
    sample_count = min(int(args.postnorm_linear_fit_samples), view1.shape[0])
    if sample_count <= 0 or sample_count >= view1.shape[0]:
        fit1 = view1.detach()
        fit2 = view2.detach()
        sample_count = int(view1.shape[0])
    else:
        idx = torch.linspace(0, view1.shape[0] - 1, sample_count, device=view1.device).long()
        fit1 = view1.index_select(0, idx).detach()
        fit2 = view2.index_select(0, idx).detach()
    dim = fit1.shape[1]
    eye = torch.eye(dim, dtype=fit1.dtype, device=fit1.device)
    delta = torch.nn.Parameter(torch.zeros((dim, dim), dtype=fit1.dtype, device=fit1.device))
    optimizer = torch.optim.Adam([delta], lr=float(args.postnorm_linear_lr))
    last_loss = float("nan")
    last_bt = float("nan")
    last_on = float("nan")
    last_off = float("nan")
    for _ in range(int(args.postnorm_linear_steps)):
        optimizer.zero_grad(set_to_none=True)
        w = eye + delta
        z1 = fit1 @ w
        z2 = fit2 @ w
        bt, on_diag, off_diag = barlow_loss_torch(z1, z2, args.bt_offdiag_lambda)
        ridge = torch.mean(delta * delta)
        loss = bt + float(args.postnorm_linear_ridge) * ridge
        loss.backward()
        if args.postnorm_linear_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([delta], float(args.postnorm_linear_grad_clip))
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())
        last_bt = float(bt.detach().cpu().item())
        last_on = float(on_diag.detach().cpu().item())
        last_off = float(off_diag.detach().cpu().item())
    w = (eye + delta).detach()
    return w, {
        "postnorm_linear_opt": True,
        "postnorm_linear_kind": "adam_bt",
        "postnorm_linear_fit_samples": int(sample_count),
        "postnorm_linear_steps": int(args.postnorm_linear_steps),
        "postnorm_linear_lr": float(args.postnorm_linear_lr),
        "postnorm_linear_ridge": float(args.postnorm_linear_ridge),
        "postnorm_linear_loss": last_loss,
        "postnorm_linear_bt": last_bt,
        "postnorm_linear_on_diag": last_on,
        "postnorm_linear_off_diag": last_off,
        "postnorm_linear_fro_delta": float(torch.linalg.matrix_norm(w - eye).detach().cpu().item()),
        "postnorm_linear_fro_w": float(torch.linalg.matrix_norm(w).detach().cpu().item()),
    }


def fit_postnorm_shared_cca_torch(view1, view2, args, blend=1.0, ridge=0.0):
    sample_count = min(int(args.postnorm_linear_fit_samples), view1.shape[0])
    if sample_count <= 0 or sample_count >= view1.shape[0]:
        fit1 = view1.detach()
        fit2 = view2.detach()
        sample_count = int(view1.shape[0])
    else:
        idx = torch.linspace(0, view1.shape[0] - 1, sample_count, device=view1.device).long()
        fit1 = view1.index_select(0, idx).detach()
        fit2 = view2.index_select(0, idx).detach()
    mean = 0.5 * (fit1.mean(dim=0, keepdim=True) + fit2.mean(dim=0, keepdim=True))
    x1 = fit1 - mean
    x2 = fit2 - mean
    n = float(fit1.shape[0])
    cov1 = (x1.T @ x1) / n
    cov2 = (x2.T @ x2) / n
    cov = 0.5 * (cov1 + cov2)
    cov = 0.5 * (cov + cov.T)
    cross = 0.5 * ((x1.T @ x2) / n + (x2.T @ x1) / n)
    cross = 0.5 * (cross + cross.T)
    evals_cov, evecs_cov = torch.linalg.eigh(cov)
    evals_cov = torch.clamp(
        evals_cov + float(ridge),
        min=float(args.postnorm_linear_cca_eps),
    )
    inv_sqrt = (evecs_cov / torch.sqrt(evals_cov).unsqueeze(0)) @ evecs_cov.T
    sym_corr = inv_sqrt @ cross @ inv_sqrt
    sym_corr = 0.5 * (sym_corr + sym_corr.T)
    evals_corr, evecs_corr = torch.linalg.eigh(sym_corr)
    order = torch.argsort(evals_corr, descending=True)
    evals_corr = evals_corr[order]
    evecs_corr = evecs_corr[:, order]
    w_cca = inv_sqrt @ evecs_corr
    if float(blend) < 1.0:
        eye = torch.eye(w_cca.shape[0], dtype=w_cca.dtype, device=w_cca.device)
        w = ((1.0 - float(blend)) * eye + float(blend) * w_cca).detach()
    else:
        w = w_cca.detach()
    with torch.no_grad():
        z1 = fit1 @ w
        z2 = fit2 @ w
        bt, on_diag, off_diag = barlow_loss_torch(z1, z2, args.bt_offdiag_lambda)
    return w, {
        "postnorm_linear_opt": True,
        "postnorm_linear_kind": "shared_cca",
        "postnorm_linear_cca_blend": float(blend),
        "postnorm_linear_cca_ridge": float(ridge),
        "postnorm_linear_fit_samples": int(sample_count),
        "postnorm_linear_cca_eps": float(args.postnorm_linear_cca_eps),
        "postnorm_linear_bt": float(bt.detach().cpu().item()),
        "postnorm_linear_on_diag": float(on_diag.detach().cpu().item()),
        "postnorm_linear_off_diag": float(off_diag.detach().cpu().item()),
        "postnorm_linear_corr_eval_mean": float(evals_corr.mean().detach().cpu().item()),
        "postnorm_linear_corr_eval_min": float(evals_corr.min().detach().cpu().item()),
        "postnorm_linear_corr_eval_max": float(evals_corr.max().detach().cpu().item()),
        "postnorm_linear_fro_w": float(torch.linalg.matrix_norm(w).detach().cpu().item()),
    }


def fit_postnorm_spectral_rotate_torch(view1, view2, args, method):
    sample_count = min(int(args.postnorm_linear_fit_samples), view1.shape[0])
    if sample_count <= 0 or sample_count >= view1.shape[0]:
        fit1 = view1.detach()
        fit2 = view2.detach()
        sample_count = int(view1.shape[0])
    else:
        idx = torch.linspace(0, view1.shape[0] - 1, sample_count, device=view1.device).long()
        fit1 = view1.index_select(0, idx).detach()
        fit2 = view2.index_select(0, idx).detach()
    mean = 0.5 * (fit1.mean(dim=0, keepdim=True) + fit2.mean(dim=0, keepdim=True))
    x1 = fit1 - mean
    x2 = fit2 - mean
    n = float(fit1.shape[0])
    cross = 0.5 * ((x1.T @ x2) / n + (x2.T @ x1) / n)
    cross = 0.5 * (cross + cross.T)
    if method == "cross_eig":
        matrix = cross
    elif method == "cca_rotate":
        cov1 = (x1.T @ x1) / n
        cov2 = (x2.T @ x2) / n
        cov = 0.5 * (cov1 + cov2)
        cov = 0.5 * (cov + cov.T)
        evals_cov, evecs_cov = torch.linalg.eigh(cov)
        evals_cov = torch.clamp(evals_cov, min=float(args.postnorm_linear_cca_eps))
        inv_sqrt = (evecs_cov / torch.sqrt(evals_cov).unsqueeze(0)) @ evecs_cov.T
        matrix = inv_sqrt @ cross @ inv_sqrt
        matrix = 0.5 * (matrix + matrix.T)
    else:
        raise ValueError(f"Unknown spectral rotation method: {method}")
    evals, evecs = torch.linalg.eigh(matrix)
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    w = evecs[:, order].detach()
    with torch.no_grad():
        z1 = fit1 @ w
        z2 = fit2 @ w
        bt, on_diag, off_diag = barlow_loss_torch(z1, z2, args.bt_offdiag_lambda)
    return w, {
        "postnorm_linear_opt": True,
        "postnorm_linear_kind": method,
        "postnorm_linear_fit_samples": int(sample_count),
        "postnorm_linear_bt": float(bt.detach().cpu().item()),
        "postnorm_linear_on_diag": float(on_diag.detach().cpu().item()),
        "postnorm_linear_off_diag": float(off_diag.detach().cpu().item()),
        "postnorm_linear_corr_eval_mean": float(evals.mean().detach().cpu().item()),
        "postnorm_linear_corr_eval_min": float(evals.min().detach().cpu().item()),
        "postnorm_linear_corr_eval_max": float(evals.max().detach().cpu().item()),
        "postnorm_linear_fro_w": float(torch.linalg.matrix_norm(w).detach().cpu().item()),
    }


def fit_postnorm_cca_power_torch(view1, view2, args, power):
    sample_count = min(int(args.postnorm_linear_fit_samples), view1.shape[0])
    if sample_count <= 0 or sample_count >= view1.shape[0]:
        fit1 = view1.detach()
        fit2 = view2.detach()
        sample_count = int(view1.shape[0])
    else:
        idx = torch.linspace(0, view1.shape[0] - 1, sample_count, device=view1.device).long()
        fit1 = view1.index_select(0, idx).detach()
        fit2 = view2.index_select(0, idx).detach()
    mean = 0.5 * (fit1.mean(dim=0, keepdim=True) + fit2.mean(dim=0, keepdim=True))
    x1 = fit1 - mean
    x2 = fit2 - mean
    n = float(fit1.shape[0])
    cov1 = (x1.T @ x1) / n
    cov2 = (x2.T @ x2) / n
    cov = 0.5 * (cov1 + cov2)
    cov = 0.5 * (cov + cov.T)
    cross = 0.5 * ((x1.T @ x2) / n + (x2.T @ x1) / n)
    cross = 0.5 * (cross + cross.T)
    evals_cov, evecs_cov = torch.linalg.eigh(cov)
    evals_cov = torch.clamp(evals_cov, min=float(args.postnorm_linear_cca_eps))
    cov_power = (evecs_cov / torch.pow(evals_cov, float(power)).unsqueeze(0)) @ evecs_cov.T
    matrix = cov_power @ cross @ cov_power
    matrix = 0.5 * (matrix + matrix.T)
    evals_corr, evecs_corr = torch.linalg.eigh(matrix)
    order = torch.argsort(evals_corr, descending=True)
    evals_corr = evals_corr[order]
    evecs_corr = evecs_corr[:, order]
    w = (cov_power @ evecs_corr).detach()
    with torch.no_grad():
        z1 = fit1 @ w
        z2 = fit2 @ w
        bt, on_diag, off_diag = barlow_loss_torch(z1, z2, args.bt_offdiag_lambda)
    return w, {
        "postnorm_linear_opt": True,
        "postnorm_linear_kind": "cca_power",
        "postnorm_linear_cca_power": float(power),
        "postnorm_linear_fit_samples": int(sample_count),
        "postnorm_linear_cca_eps": float(args.postnorm_linear_cca_eps),
        "postnorm_linear_bt": float(bt.detach().cpu().item()),
        "postnorm_linear_on_diag": float(on_diag.detach().cpu().item()),
        "postnorm_linear_off_diag": float(off_diag.detach().cpu().item()),
        "postnorm_linear_corr_eval_mean": float(evals_corr.mean().detach().cpu().item()),
        "postnorm_linear_corr_eval_min": float(evals_corr.min().detach().cpu().item()),
        "postnorm_linear_corr_eval_max": float(evals_corr.max().detach().cpu().item()),
        "postnorm_linear_fro_w": float(torch.linalg.matrix_norm(w).detach().cpu().item()),
    }


def fit_postnorm_align_ls_torch(view1, view2, args, ridge):
    sample_count = min(int(args.postnorm_linear_fit_samples), view1.shape[0])
    if sample_count <= 0 or sample_count >= view1.shape[0]:
        fit1 = view1.detach()
        fit2 = view2.detach()
        sample_count = int(view1.shape[0])
    else:
        idx = torch.linspace(0, view1.shape[0] - 1, sample_count, device=view1.device).long()
        fit1 = view1.index_select(0, idx).detach()
        fit2 = view2.index_select(0, idx).detach()
    midpoint = 0.5 * (fit1 + fit2)
    dim = fit1.shape[1]
    eye = torch.eye(dim, dtype=fit1.dtype, device=fit1.device)
    reg = float(ridge) * float(fit1.shape[0])
    lhs = fit1.T @ fit1 + fit2.T @ fit2 + reg * eye
    rhs = fit1.T @ midpoint + fit2.T @ midpoint + reg * eye
    w = torch.linalg.solve(lhs, rhs).detach()
    with torch.no_grad():
        z1 = fit1 @ w
        z2 = fit2 @ w
        bt, on_diag, off_diag = barlow_loss_torch(z1, z2, args.bt_offdiag_lambda)
        align_mse = torch.mean((z1 - midpoint).pow(2) + (z2 - midpoint).pow(2))
    return w, {
        "postnorm_linear_opt": True,
        "postnorm_linear_kind": "align_ls",
        "postnorm_linear_fit_samples": int(sample_count),
        "postnorm_linear_ridge": float(ridge),
        "postnorm_linear_align_mse": float(align_mse.detach().cpu().item()),
        "postnorm_linear_bt": float(bt.detach().cpu().item()),
        "postnorm_linear_on_diag": float(on_diag.detach().cpu().item()),
        "postnorm_linear_off_diag": float(off_diag.detach().cpu().item()),
        "postnorm_linear_fro_delta": float(torch.linalg.matrix_norm(w - eye).detach().cpu().item()),
        "postnorm_linear_fro_w": float(torch.linalg.matrix_norm(w).detach().cpu().item()),
    }


def apply_postnorm_linear_if_needed(spec, args, train_arrays, test_arrays, layer_idx=None):
    if not spec.get("postnorm_linear_opt", False):
        return train_arrays, test_arrays, {}
    kind = spec.get("postnorm_linear_kind", "adam_bt")
    if kind == "adam_bt":
        w, stats = fit_postnorm_linear_bt_torch(train_arrays[1], train_arrays[2], args)
    elif kind == "shared_cca":
        w, stats = fit_postnorm_shared_cca_torch(
            train_arrays[1],
            train_arrays[2],
            args,
            blend=spec.get("postnorm_linear_cca_blend", 1.0),
            ridge=spec.get("postnorm_linear_cca_ridge", 0.0),
        )
    elif kind in {"cross_eig", "cca_rotate"}:
        w, stats = fit_postnorm_spectral_rotate_torch(train_arrays[1], train_arrays[2], args, kind)
    elif kind == "cca_power":
        if layer_idx is None:
            power = spec.get("postnorm_linear_cca_power", 0.5)
        else:
            power = scheduled_spec_value(spec, "postnorm_linear_cca_power", layer_idx)
        w, stats = fit_postnorm_cca_power_torch(
            train_arrays[1],
            train_arrays[2],
            args,
            power,
        )
    elif kind == "align_ls":
        w, stats = fit_postnorm_align_ls_torch(
            train_arrays[1],
            train_arrays[2],
            args,
            spec.get("postnorm_linear_ridge", 1.0),
        )
    elif kind == "adaptive_fullwhiten":
        corr_diag = corr_diag_mean_torch(train_arrays[1], train_arrays[2])
        threshold = float(spec.get("postnorm_adaptive_corr_threshold", 0.9))
        apply_whiten = bool(float(corr_diag.detach().cpu().item()) >= threshold)
        stats = {
            "postnorm_linear_opt": True,
            "postnorm_linear_kind": "adaptive_fullwhiten",
            "postnorm_adaptive_corr_threshold": threshold,
            "postnorm_adaptive_corr_diag_mean": float(corr_diag.detach().cpu().item()),
            "postnorm_adaptive_fullwhiten_applied": apply_whiten,
        }
        if apply_whiten:
            train_arrays, test_arrays, _, _ = normalize_hidden_zca_torch(
                train_arrays,
                test_arrays,
                eps=float(args.postnorm_linear_cca_eps),
            )
        return train_arrays, test_arrays, stats
    else:
        raise ValueError(f"Unknown postnorm linear kind: {kind}")
    train_arrays = [arr @ w for arr in train_arrays]
    test_arrays = [arr @ w for arr in test_arrays]
    train_arrays, test_arrays, _, _ = normalize_hidden_with_stats_torch(train_arrays, test_arrays)
    return train_arrays, test_arrays, stats


def normalized_hybrid_transform(view1, view2, left_transform, right_transform, mix):
    left1 = view1 @ left_transform
    left2 = view2 @ left_transform
    right1 = view1 @ right_transform
    right2 = view2 @ right_transform
    left_mean, left_std, _, _ = paired_preactivation_stats(left1, left2)
    right_mean, right_std, _, _ = paired_preactivation_stats(right1, right2)
    m = float(mix)
    transform = (1.0 - m) * (left_transform / torch.clamp(left_std, min=1e-4).unsqueeze(0))
    transform = transform + m * (right_transform / torch.clamp(right_std, min=1e-4).unsqueeze(0))
    base_bias = -(
        (1.0 - m) * left_mean / torch.clamp(left_std, min=1e-4)
        + m * right_mean / torch.clamp(right_std, min=1e-4)
    )
    return transform, base_bias.detach(), {
        "hybrid_mix": m,
        "hybrid_left_std_mean": float(left_std.mean().detach().cpu().item()),
        "hybrid_right_std_mean": float(right_std.mean().detach().cpu().item()),
    }


def fit_linearized_bt_residual(view1, view2, args):
    dtype = view1.dtype
    device = view1.device
    n = float(view1.shape[0])
    dim = view1.shape[1]
    x1 = view1
    x2 = view2
    d = view1 - view2
    e = x1 - x2

    eye = torch.eye(dim, dtype=dtype, device=device)
    see = (e.T @ e) / n
    sed = (e.T @ d) / n
    align_init = -torch.linalg.solve(see + float(args.linearized_ridge) * eye, sed)
    b = torch.nn.Parameter(align_init.detach())

    c0 = (view1.T @ view2) / n
    a = (x1.T @ view2) / n
    q = (view1.T @ x2) / n
    optimizer = torch.optim.Adam([b], lr=args.linearized_lr)
    last_loss = float("nan")
    last_align = float("nan")
    last_bt = float("nan")
    for _ in range(args.linearized_steps):
        optimizer.zero_grad(set_to_none=True)
        align = torch.trace(b.T @ see @ b) + 2.0 * torch.sum(b * sed)
        c_lin = c0 + b.T @ a + q @ b
        on = torch.diagonal(c_lin).add(-1.0).pow(2).sum()
        off = offdiag(c_lin).pow(2).sum()
        bt = on + float(args.bt_offdiag_lambda) * off
        ridge = b.pow(2).mean()
        loss = float(args.align_weight) * align + float(args.bt_weight) * bt + float(args.linearized_ridge) * ridge
        loss.backward()
        if args.linearized_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([b], args.linearized_grad_clip)
        optimizer.step()
        with torch.no_grad():
            if args.linearized_max_norm > 0:
                norm = torch.linalg.matrix_norm(b)
                if norm > args.linearized_max_norm:
                    b.mul_(args.linearized_max_norm / norm)
        last_loss = float(loss.detach().cpu().item())
        last_align = float(align.detach().cpu().item())
        last_bt = float(bt.detach().cpu().item())
    return b.detach(), {
        "linearized_loss": last_loss,
        "linearized_align_term": last_align,
        "linearized_bt_term": last_bt,
        "residual_transform_fro": float(torch.linalg.matrix_norm(b.detach()).detach().cpu().item()),
    }


def metadata_without_transform(fitted):
    return {key: value for key, value in fitted.items() if key not in {"transform", "bias", "scale", "gains", "eigvals"}}


def linear_classifier_readout(xtr, xte, ytr, yte, reg):
    y_onehot = one_hot_np(ytr, int(np.max(ytr)) + 1)
    ztr, zte = standardize_pair(xtr, xte)
    weight = ridge_map_np(ztr, y_onehot, reg=reg, fit_bias=True)
    train_logits = apply_map_np(ztr, weight, fit_bias=True)
    test_logits = apply_map_np(zte, weight, fit_bias=True)
    return {
        "train_accuracy": accuracy_from_logits(train_logits, ytr),
        "test_accuracy": accuracy_from_logits(test_logits, yte),
        "train_ce": softmax_ce_np(train_logits, ytr),
        "test_ce": softmax_ce_np(test_logits, yte),
    }


def pca512_readout(xtr, xte, ytr, yte, reg, seed, pca_dim):
    max_dim = min(pca_dim, xtr.shape[1], xtr.shape[0] - 1)
    pca = PCA(n_components=max_dim, svd_solver="randomized", iterated_power=3, random_state=seed)
    start = time.perf_counter()
    ztr = pca.fit_transform(xtr)
    zte = pca.transform(xte)
    pca_time = time.perf_counter() - start
    metrics = linear_classifier_readout(ztr.astype(np.float32), zte.astype(np.float32), ytr, yte, reg)
    metrics.update(
        {
            "readout_dim": int(max_dim),
            "source_dim": int(xtr.shape[1]),
            "explained_variance": float(np.sum(pca.explained_variance_ratio_[:max_dim])),
            "pca_time_sec": float(pca_time),
        }
    )
    return metrics


def variant_spec(name, depth):
    bpbt_activation = {"activation": BP_BT_ACTIVATION, "alpha": BP_BT_ACTIVATION_ALPHA}
    relu_activation = {"activation": "relu", "alpha": 0.0}
    if name.startswith("plain_cf_relu_constinv"):
        value = float(name.rsplit("constinv", 1)[1])
        return {"kind": "plain_cf", **relu_activation, "schedule": [value] * depth}
    if name.startswith("plain_cf_relu_relax2"):
        return {"kind": "plain_cf", **relu_activation, "schedule": geometric(1.0, depth, 0.5)}
    if name == "plain_cf_relu":
        return {"kind": "plain_cf", **relu_activation, "schedule": default_schedule(depth)}
    if name == "plain_cf_relu_fullwhiten":
        return {"kind": "plain_cf", **relu_activation, "schedule": default_schedule(depth), "norm_kind": "fullwhiten"}
    if name == "plain_cf_rowcorrect_relu":
        return {"kind": "plain_cf_rowcorrect", **relu_activation, "schedule": default_schedule(depth)}
    if name == "plain_cf_sharedmetric_relu":
        return {"kind": "plain_cf_sharedmetric", **relu_activation, "schedule": default_schedule(depth)}
    if name == "plain_cf_postrelu_affineopt_relu":
        return {
            "kind": "plain_cf_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "postrelu_optimize_scale": True,
            "postrelu_objective": "bt",
        }
    if name == "plain_cf_postrelu_biasopt_relu":
        return {
            "kind": "plain_cf_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "postrelu_optimize_scale": False,
            "postrelu_objective": "bt",
        }
    if name == "plain_cf_postrelu_biasopt_linearopt_relu":
        return {
            "kind": "plain_cf_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "postrelu_optimize_scale": False,
            "postrelu_objective": "bt",
            "postnorm_linear_opt": True,
            "postnorm_linear_kind": "adam_bt",
        }
    if name == "plain_cf_postrelu_biasopt_ccalinear_relu":
        return {
            "kind": "plain_cf_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "postrelu_optimize_scale": False,
            "postrelu_objective": "bt",
            "postnorm_linear_opt": True,
            "postnorm_linear_kind": "shared_cca",
        }
    if name.startswith("plain_cf_postrelu_biasopt_ccaridge_relu_r"):
        return {
            "kind": "plain_cf_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "postrelu_optimize_scale": False,
            "postrelu_objective": "bt",
            "postnorm_linear_opt": True,
            "postnorm_linear_kind": "shared_cca",
            "postnorm_linear_cca_ridge": float(name.rsplit("_r", 1)[1]),
        }
    if name.startswith("plain_cf_postrelu_biasopt_ccablend_relu_m"):
        return {
            "kind": "plain_cf_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "postrelu_optimize_scale": False,
            "postrelu_objective": "bt",
            "postnorm_linear_opt": True,
            "postnorm_linear_kind": "shared_cca",
            "postnorm_linear_cca_blend": float(name.rsplit("_m", 1)[1]),
        }
    if name == "plain_cf_postrelu_biasopt_relu_fullwhiten":
        return {
            "kind": "plain_cf_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "postrelu_optimize_scale": False,
            "postrelu_objective": "bt",
            "norm_kind": "fullwhiten",
        }
    if name == "plain_cf_postrelu_diagopt_relu":
        return {
            "kind": "plain_cf_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "postrelu_optimize_scale": True,
            "postrelu_objective": "on_diag",
        }
    if name == "plain_cf_postrelu_biasdiagopt_relu":
        return {
            "kind": "plain_cf_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "postrelu_optimize_scale": False,
            "postrelu_objective": "on_diag",
        }
    if name.startswith("plain_cf_postrelu_active_relu_a"):
        return {
            "kind": "plain_cf_postrelu_active",
            **relu_activation,
            "schedule": default_schedule(depth),
            "active_target": float(name.rsplit("_a", 1)[1]),
        }
    if name == "plain_cf_postrelu_active_ramp_relu":
        return {
            "kind": "plain_cf_postrelu_active",
            **relu_activation,
            "schedule": default_schedule(depth),
            "active_target": linear_schedule(0.25, 0.9, depth),
        }
    if name.startswith("plain_cf_postrelu_active_ramp_relu_lo"):
        lo_text, hi_text = name.rsplit("_hi", 1)
        return {
            "kind": "plain_cf_postrelu_active",
            **relu_activation,
            "schedule": default_schedule(depth),
            "active_target": linear_schedule(float(lo_text.rsplit("_lo", 1)[1]), float(hi_text), depth),
        }
    if name.startswith("plain_cf_postrelu_corrbias_relu_b"):
        return {
            "kind": "plain_cf_postrelu_corrbias",
            **relu_activation,
            "schedule": default_schedule(depth),
            "corrbias_beta": float(name.rsplit("_b", 1)[1]),
        }
    if name == "plain_cf_agreement_biasopt_relu":
        return {
            "kind": "plain_cf_agreement_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": False,
            "postrelu_optimize_scale": False,
            "postrelu_objective": "bt",
        }
    if name.startswith("plain_cf_agreement_corrbias_ccalinear_relu_b"):
        return {
            "kind": "plain_cf_agreement_corrbias",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": False,
            "corrbias_beta": float(name.rsplit("_b", 1)[1]),
            "postnorm_linear_opt": True,
            "postnorm_linear_kind": "shared_cca",
        }
    if name.startswith("plain_cf_agreement_corrbias_relu_b"):
        return {
            "kind": "plain_cf_agreement_corrbias",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": False,
            "corrbias_beta": float(name.rsplit("_b", 1)[1]),
        }
    if name.startswith("plain_cf_agreement_activegain_relu_lo"):
        lo_text, hi_text = name.rsplit("_lo", 1)[1].split("_hi", 1)
        return {
            "kind": "plain_cf_agreement_activegain",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": False,
            "postrelu_active_lo": float(lo_text),
            "postrelu_active_hi": float(hi_text),
        }
    if name.startswith("plain_cf_agreement_activerank_relu_lo"):
        lo_text, hi_text = name.rsplit("_lo", 1)[1].split("_hi", 1)
        return {
            "kind": "plain_cf_agreement_activerank",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": False,
            "postrelu_active_lo": float(lo_text),
            "postrelu_active_hi": float(hi_text),
        }
    if name.startswith("plain_cf_agreement_activerank_ccalinear_relu_lo"):
        lo_text, hi_text = name.rsplit("_lo", 1)[1].split("_hi", 1)
        return {
            "kind": "plain_cf_agreement_activerank",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": False,
            "postrelu_active_lo": float(lo_text),
            "postrelu_active_hi": float(hi_text),
            "postnorm_linear_opt": True,
            "postnorm_linear_kind": "shared_cca",
        }
    if name == "plain_cf_agreement_biasopt_linearopt_relu":
        return {
            "kind": "plain_cf_agreement_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": False,
            "postrelu_optimize_scale": False,
            "postrelu_objective": "bt",
            "postnorm_linear_opt": True,
            "postnorm_linear_kind": "adam_bt",
        }
    if name == "plain_cf_agreement_biasopt_ccalinear_relu":
        return {
            "kind": "plain_cf_agreement_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": False,
            "postrelu_optimize_scale": False,
            "postrelu_objective": "bt",
            "postnorm_linear_opt": True,
            "postnorm_linear_kind": "shared_cca",
        }
    if name == "plain_cf_agreement_biasopt_crosseig_relu":
        return {
            "kind": "plain_cf_agreement_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": False,
            "postrelu_optimize_scale": False,
            "postrelu_objective": "bt",
            "postnorm_linear_opt": True,
            "postnorm_linear_kind": "cross_eig",
        }
    if name == "plain_cf_agreement_biasopt_ccarotate_relu":
        return {
            "kind": "plain_cf_agreement_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": False,
            "postrelu_optimize_scale": False,
            "postrelu_objective": "bt",
            "postnorm_linear_opt": True,
            "postnorm_linear_kind": "cca_rotate",
        }
    if name.startswith("plain_cf_agreement_biasopt_ccapower_relu_p"):
        return {
            "kind": "plain_cf_agreement_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": False,
            "postrelu_optimize_scale": False,
            "postrelu_objective": "bt",
            "postnorm_linear_opt": True,
            "postnorm_linear_kind": "cca_power",
            "postnorm_linear_cca_power": float(name.rsplit("_p", 1)[1]),
        }
    if name.startswith("plain_cf_agreement_biasopt_ccapowersched_relu_p"):
        start_text, end_text = name.rsplit("_p", 1)[1].split("_to", 1)
        return {
            "kind": "plain_cf_agreement_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": False,
            "postrelu_optimize_scale": False,
            "postrelu_objective": "bt",
            "postnorm_linear_opt": True,
            "postnorm_linear_kind": "cca_power",
            "postnorm_linear_cca_power": linear_schedule(float(start_text), float(end_text), depth),
        }
    if name.startswith("plain_cf_agreement_biasopt_alignls_relu_r"):
        return {
            "kind": "plain_cf_agreement_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": False,
            "postrelu_optimize_scale": False,
            "postrelu_objective": "bt",
            "postnorm_linear_opt": True,
            "postnorm_linear_kind": "align_ls",
            "postnorm_linear_ridge": float(name.rsplit("_r", 1)[1]),
        }
    if name.startswith("plain_cf_agreement_biasopt_ccaridge_relu_r"):
        return {
            "kind": "plain_cf_agreement_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": False,
            "postrelu_optimize_scale": False,
            "postrelu_objective": "bt",
            "postnorm_linear_opt": True,
            "postnorm_linear_kind": "shared_cca",
            "postnorm_linear_cca_ridge": float(name.rsplit("_r", 1)[1]),
        }
    if name.startswith("plain_cf_agreement_biasopt_ccablend_relu_m"):
        return {
            "kind": "plain_cf_agreement_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": False,
            "postrelu_optimize_scale": False,
            "postrelu_objective": "bt",
            "postnorm_linear_opt": True,
            "postnorm_linear_kind": "shared_cca",
            "postnorm_linear_cca_blend": float(name.rsplit("_m", 1)[1]),
        }
    if name == "plain_cf_agreement_biasopt_relu_fullwhiten":
        return {
            "kind": "plain_cf_agreement_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": False,
            "postrelu_optimize_scale": False,
            "postrelu_objective": "bt",
            "norm_kind": "fullwhiten",
        }
    if name == "plain_cf_agreement_diagopt_relu":
        return {
            "kind": "plain_cf_agreement_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": False,
            "postrelu_optimize_scale": False,
            "postrelu_objective": "on_diag",
        }
    if name == "plain_cf_eigenshrink_biasopt_relu":
        return {
            "kind": "plain_cf_agreement_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": True,
            "postrelu_optimize_scale": False,
            "postrelu_objective": "bt",
        }
    if name == "plain_cf_eigenshrink_diagopt_relu":
        return {
            "kind": "plain_cf_agreement_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": True,
            "postrelu_optimize_scale": False,
            "postrelu_objective": "on_diag",
        }
    if name.startswith("plain_cf_hybrid_agreement_biasopt_relu_m"):
        return {
            "kind": "plain_cf_hybrid_agreement_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "hybrid_mix": float(name.rsplit("_m", 1)[1]),
            "mode_gain_used": False,
            "postrelu_optimize_scale": False,
            "postrelu_objective": "bt",
        }
    if name.startswith("plain_cf_hybrid_agreement_diagopt_relu_m"):
        return {
            "kind": "plain_cf_hybrid_agreement_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "hybrid_mix": float(name.rsplit("_m", 1)[1]),
            "mode_gain_used": False,
            "postrelu_optimize_scale": False,
            "postrelu_objective": "on_diag",
        }
    if name.startswith("plain_cf_bias_then_agreement_biasopt_relu_s"):
        return {
            "kind": "plain_cf_switch_bias_agreement",
            **relu_activation,
            "schedule": default_schedule(depth),
            "switch_layer": int(name.rsplit("_s", 1)[1]),
            "postrelu_optimize_scale": False,
            "postrelu_objective": "bt",
        }
    if name == "plain_cf_eigen_shrink_relu":
        return {
            "kind": "plain_cf_agreement_mode",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": True,
            "gate_beta": 0.0,
        }
    if name.startswith("plain_cf_agreement_gate_relu_b") and "_constinv" in name:
        beta_text, inv_text = name.rsplit("_constinv", 1)
        return {
            "kind": "plain_cf_agreement_mode",
            **relu_activation,
            "schedule": [float(inv_text)] * depth,
            "mode_gain_used": False,
            "gate_beta": float(beta_text.rsplit("_b", 1)[1]),
        }
    if name.startswith("plain_cf_agreement_gate_relu_b"):
        return {
            "kind": "plain_cf_agreement_mode",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": False,
            "gate_beta": float(name.rsplit("_b", 1)[1]),
        }
    if name.startswith("plain_cf_agreement_expand_adaptwhiten_relu_k"):
        prefix, threshold_text = name.rsplit("_t", 1)
        return {
            "kind": "plain_cf_agreement_expand",
            **relu_activation,
            "schedule": default_schedule(depth),
            "agreement_expand_keep_dim": int(prefix.rsplit("_k", 1)[1]),
            "postnorm_linear_opt": True,
            "postnorm_linear_kind": "adaptive_fullwhiten",
            "postnorm_adaptive_corr_threshold": float(threshold_text),
        }
    if name.startswith("plain_cf_agreement_expand_fullwhiten_relu_k"):
        return {
            "kind": "plain_cf_agreement_expand",
            **relu_activation,
            "schedule": default_schedule(depth),
            "agreement_expand_keep_dim": int(name.rsplit("_k", 1)[1]),
            "norm_kind": "fullwhiten",
        }
    if name.startswith("plain_cf_agreement_expand_relu_k"):
        return {
            "kind": "plain_cf_agreement_expand",
            **relu_activation,
            "schedule": default_schedule(depth),
            "agreement_expand_keep_dim": int(name.rsplit("_k", 1)[1]),
        }
    if name in {"plain_cf_leakygelu0.5", "plain_cf_bpbt_nonlinearity"}:
        return {"kind": "plain_cf", **bpbt_activation, "schedule": default_schedule(depth)}
    if name in {"inverse_prior_ols_leakygelu0.5", "inverse_prior_ols_bpbt_nonlinearity"}:
        return {"kind": "inverse_prior", **bpbt_activation, "schedule": default_schedule(depth)}
    if name == "residual_cf_branch_relu":
        return {"kind": "residual_cf_branch", **relu_activation, "schedule": default_schedule(depth)}
    if name == "residual_cf_branch_rowcorrect_relu":
        return {"kind": "residual_cf_branch_rowcorrect", **relu_activation, "schedule": default_schedule(depth)}
    if name == "residual_cf_branch_postrelu_affineopt_relu":
        return {
            "kind": "residual_cf_branch_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "postrelu_optimize_scale": True,
            "postrelu_objective": "bt",
        }
    if name == "residual_cf_branch_postrelu_biasopt_relu":
        return {
            "kind": "residual_cf_branch_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "postrelu_optimize_scale": False,
            "postrelu_objective": "bt",
        }
    if name == "residual_cf_branch_postrelu_diagopt_relu":
        return {
            "kind": "residual_cf_branch_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "postrelu_optimize_scale": True,
            "postrelu_objective": "on_diag",
        }
    if name == "residual_cf_branch_postrelu_biasdiagopt_relu":
        return {
            "kind": "residual_cf_branch_postrelu_affineopt",
            **relu_activation,
            "schedule": default_schedule(depth),
            "postrelu_optimize_scale": False,
            "postrelu_objective": "on_diag",
        }
    if name.startswith("residual_cf_branch_postrelu_active_relu_a"):
        return {
            "kind": "residual_cf_branch_postrelu_active",
            **relu_activation,
            "schedule": default_schedule(depth),
            "active_target": float(name.rsplit("_a", 1)[1]),
        }
    if name == "residual_cf_branch_postrelu_active_ramp_relu":
        return {
            "kind": "residual_cf_branch_postrelu_active",
            **relu_activation,
            "schedule": default_schedule(depth),
            "active_target": linear_schedule(0.25, 0.9, depth),
        }
    if name.startswith("residual_cf_branch_postrelu_active_ramp_relu_lo"):
        lo_text, hi_text = name.rsplit("_hi", 1)
        return {
            "kind": "residual_cf_branch_postrelu_active",
            **relu_activation,
            "schedule": default_schedule(depth),
            "active_target": linear_schedule(float(lo_text.rsplit("_lo", 1)[1]), float(hi_text), depth),
        }
    if name.startswith("residual_cf_branch_postrelu_corrbias_relu_b"):
        return {
            "kind": "residual_cf_branch_postrelu_corrbias",
            **relu_activation,
            "schedule": default_schedule(depth),
            "corrbias_beta": float(name.rsplit("_b", 1)[1]),
        }
    if name == "residual_cf_branch_eigen_shrink_relu":
        return {
            "kind": "residual_cf_branch_agreement_mode",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": True,
            "gate_beta": 0.0,
        }
    if name.startswith("residual_cf_branch_agreement_gate_relu_b") and "_constinv" in name:
        beta_text, inv_text = name.rsplit("_constinv", 1)
        return {
            "kind": "residual_cf_branch_agreement_mode",
            **relu_activation,
            "schedule": [float(inv_text)] * depth,
            "mode_gain_used": False,
            "gate_beta": float(beta_text.rsplit("_b", 1)[1]),
        }
    if name.startswith("residual_cf_branch_agreement_gate_relu_b"):
        return {
            "kind": "residual_cf_branch_agreement_mode",
            **relu_activation,
            "schedule": default_schedule(depth),
            "mode_gain_used": False,
            "gate_beta": float(name.rsplit("_b", 1)[1]),
        }
    if name in {"residual_cf_branch_leakygelu0.5", "residual_cf_branch_bpbt_nonlinearity"}:
        return {"kind": "residual_cf_branch", **bpbt_activation, "schedule": default_schedule(depth)}
    if name == "linearized_bt_residual":
        return {"kind": "linearized_bt_residual", "activation": "identity", "alpha": 0.0, "schedule": [None] * depth}
    raise ValueError(f"Unknown variant: {name}")


def update_path(
    variant,
    spec,
    layer_idx,
    base_tr,
    base_te,
    view1_tr,
    view2_tr,
    view1_te,
    view2_te,
    args,
):
    stats = {"layer": layer_idx + 1}
    kind = spec["kind"]
    if kind == "plain_cf":
        fitted = fit_cf_transform_torch(view1_tr, view2_tr, args.width, invariance_strength=spec["schedule"][layer_idx])
        transform = fitted["transform"]
        fn = lambda x: apply_activation_torch(x @ transform, spec["activation"], spec["alpha"])
        stats.update(metadata_without_transform(fitted))
    elif kind == "plain_cf_postrelu_affineopt":
        fitted = fit_cf_transform_torch(view1_tr, view2_tr, args.width, invariance_strength=spec["schedule"][layer_idx])
        transform = fitted["transform"]
        pre1 = view1_tr @ transform
        pre2 = view2_tr @ transform
        scale, bias, opt_stats = fit_postrelu_affine_torch(
            pre1,
            pre2,
            args,
            optimize_scale=spec.get("postrelu_optimize_scale", True),
            objective=spec.get("postrelu_objective", "bt"),
        )
        fn = lambda x: apply_activation_torch((x @ transform) * scale + bias, spec["activation"], spec["alpha"])
        stats.update(metadata_without_transform(fitted))
        stats.update(opt_stats)
    elif kind == "plain_cf_postrelu_active":
        fitted = fit_cf_transform_torch(view1_tr, view2_tr, args.width, invariance_strength=spec["schedule"][layer_idx])
        transform = fitted["transform"]
        pre1 = view1_tr @ transform
        pre2 = view2_tr @ transform
        bias, heuristic_stats = fit_postrelu_active_bias_torch(
            pre1,
            pre2,
            scheduled_spec_value(spec, "active_target", layer_idx),
            args,
        )
        fn = lambda x: apply_activation_torch(x @ transform + bias, spec["activation"], spec["alpha"])
        stats.update(metadata_without_transform(fitted))
        stats.update(heuristic_stats)
    elif kind == "plain_cf_postrelu_corrbias":
        fitted = fit_cf_transform_torch(view1_tr, view2_tr, args.width, invariance_strength=spec["schedule"][layer_idx])
        transform = fitted["transform"]
        pre1 = view1_tr @ transform
        pre2 = view2_tr @ transform
        bias, heuristic_stats = fit_postrelu_corrbias_torch(pre1, pre2, spec["corrbias_beta"], args)
        fn = lambda x: apply_activation_torch(x @ transform + bias, spec["activation"], spec["alpha"])
        stats.update(metadata_without_transform(fitted))
        stats.update(heuristic_stats)
    elif kind == "plain_cf_rowcorrect":
        fitted = fit_cf_transform_row_correct_torch(view1_tr, view2_tr, args.width, spec["schedule"][layer_idx])
        transform = fitted["transform"]
        fn = lambda x: apply_activation_torch(x @ transform, spec["activation"], spec["alpha"])
        stats.update(metadata_without_transform(fitted))
        stats["row_corrected"] = True
    elif kind == "plain_cf_sharedmetric":
        fitted = fit_cf_transform_shared_metric_torch(view1_tr, view2_tr, args.width, spec["schedule"][layer_idx])
        transform = fitted["transform"]
        fn = lambda x: apply_activation_torch(x @ transform, spec["activation"], spec["alpha"])
        stats.update(metadata_without_transform(fitted))
    elif kind == "plain_cf_agreement_mode":
        fitted = fit_agreement_mode_transform_torch(
            view1_tr,
            view2_tr,
            args.width,
            spec["schedule"][layer_idx],
            spec["mode_gain_used"],
            spec["gate_beta"],
        )
        transform = fitted["transform"]
        bias = fitted["bias"]
        fn = lambda x: apply_activation_torch(x @ transform + bias, spec["activation"], spec["alpha"])
        stats.update(metadata_without_transform(fitted))
    elif kind == "plain_cf_agreement_expand":
        fitted = fit_agreement_subspace_expand_transform_torch(
            view1_tr,
            view2_tr,
            args.width,
            spec["agreement_expand_keep_dim"],
        )
        transform = fitted["transform"]
        fn = lambda x: apply_activation_torch(x @ transform, spec["activation"], spec["alpha"])
        stats.update(metadata_without_transform(fitted))
    elif kind == "plain_cf_agreement_postrelu_affineopt":
        fitted = fit_agreement_mode_transform_torch(
            view1_tr,
            view2_tr,
            args.width,
            spec["schedule"][layer_idx],
            spec["mode_gain_used"],
            gate_beta=0.0,
        )
        transform = fitted["transform"]
        pre1 = view1_tr @ transform
        pre2 = view2_tr @ transform
        scale, bias, opt_stats = fit_postrelu_affine_torch(
            pre1,
            pre2,
            args,
            optimize_scale=spec.get("postrelu_optimize_scale", False),
            objective=spec.get("postrelu_objective", "bt"),
        )
        fn = lambda x: apply_activation_torch((x @ transform) * scale + bias, spec["activation"], spec["alpha"])
        stats.update(metadata_without_transform(fitted))
        stats.update(opt_stats)
        stats["agreement_postrelu_fit"] = True
    elif kind == "plain_cf_agreement_corrbias":
        fitted = fit_agreement_mode_transform_torch(
            view1_tr,
            view2_tr,
            args.width,
            spec["schedule"][layer_idx],
            spec["mode_gain_used"],
            gate_beta=0.0,
        )
        transform = fitted["transform"]
        pre1 = view1_tr @ transform
        pre2 = view2_tr @ transform
        bias, heuristic_stats = fit_postrelu_corrbias_torch(pre1, pre2, spec["corrbias_beta"], args)
        fn = lambda x: apply_activation_torch(x @ transform + bias, spec["activation"], spec["alpha"])
        stats.update(metadata_without_transform(fitted))
        stats.update(heuristic_stats)
        stats["agreement_corrbias"] = True
    elif kind == "plain_cf_agreement_activegain":
        fitted = fit_agreement_mode_transform_torch(
            view1_tr,
            view2_tr,
            args.width,
            spec["schedule"][layer_idx],
            spec["mode_gain_used"],
            gate_beta=0.0,
        )
        transform = fitted["transform"]
        gains = fitted["gains"]
        lo = float(spec["postrelu_active_lo"])
        hi = float(spec["postrelu_active_hi"])
        active_targets = lo + (hi - lo) * gains
        pre1 = view1_tr @ transform
        pre2 = view2_tr @ transform
        bias, active_stats = fit_postrelu_active_bias_targets_torch(pre1, pre2, active_targets, args)
        fn = lambda x: apply_activation_torch(x @ transform + bias, spec["activation"], spec["alpha"])
        stats.update(metadata_without_transform(fitted))
        stats.update(active_stats)
        stats["postrelu_active_lo"] = lo
        stats["postrelu_active_hi"] = hi
        stats["agreement_activegain"] = True
    elif kind == "plain_cf_agreement_activerank":
        fitted = fit_agreement_mode_transform_torch(
            view1_tr,
            view2_tr,
            args.width,
            spec["schedule"][layer_idx],
            spec["mode_gain_used"],
            gate_beta=0.0,
        )
        transform = fitted["transform"]
        lo = float(spec["postrelu_active_lo"])
        hi = float(spec["postrelu_active_hi"])
        dim = transform.shape[1]
        rank_agreement = torch.linspace(1.0, 0.0, dim, dtype=transform.dtype, device=transform.device)
        active_targets = lo + (hi - lo) * rank_agreement
        pre1 = view1_tr @ transform
        pre2 = view2_tr @ transform
        bias, active_stats = fit_postrelu_active_bias_targets_torch(pre1, pre2, active_targets, args)
        fn = lambda x: apply_activation_torch(x @ transform + bias, spec["activation"], spec["alpha"])
        stats.update(metadata_without_transform(fitted))
        stats.update(active_stats)
        stats["postrelu_active_lo"] = lo
        stats["postrelu_active_hi"] = hi
        stats["agreement_activerank"] = True
    elif kind == "plain_cf_hybrid_agreement_postrelu_affineopt":
        cf_fitted = fit_cf_transform_torch(view1_tr, view2_tr, args.width, invariance_strength=spec["schedule"][layer_idx])
        ag_fitted = fit_agreement_mode_transform_torch(
            view1_tr,
            view2_tr,
            args.width,
            spec["schedule"][layer_idx],
            spec["mode_gain_used"],
            gate_beta=0.0,
        )
        transform, base_bias, hybrid_stats = normalized_hybrid_transform(
            view1_tr,
            view2_tr,
            cf_fitted["transform"],
            ag_fitted["transform"],
            spec["hybrid_mix"],
        )
        pre1 = view1_tr @ transform + base_bias
        pre2 = view2_tr @ transform + base_bias
        scale, bias, opt_stats = fit_postrelu_affine_torch(
            pre1,
            pre2,
            args,
            optimize_scale=spec.get("postrelu_optimize_scale", False),
            objective=spec.get("postrelu_objective", "bt"),
        )
        full_bias = base_bias + bias
        fn = lambda x: apply_activation_torch((x @ transform) * scale + full_bias, spec["activation"], spec["alpha"])
        stats.update(metadata_without_transform(cf_fitted))
        stats.update(hybrid_stats)
        stats.update(opt_stats)
        stats["hybrid_agreement_postrelu_fit"] = True
    elif kind == "plain_cf_switch_bias_agreement":
        if layer_idx < int(spec["switch_layer"]):
            fitted = fit_cf_transform_torch(view1_tr, view2_tr, args.width, invariance_strength=spec["schedule"][layer_idx])
            basis = "ordinary_cf"
        else:
            fitted = fit_agreement_mode_transform_torch(
                view1_tr,
                view2_tr,
                args.width,
                spec["schedule"][layer_idx],
                use_gain=False,
                gate_beta=0.0,
            )
            basis = "agreement_mode"
        transform = fitted["transform"]
        pre1 = view1_tr @ transform
        pre2 = view2_tr @ transform
        scale, bias, opt_stats = fit_postrelu_affine_torch(
            pre1,
            pre2,
            args,
            optimize_scale=spec.get("postrelu_optimize_scale", False),
            objective=spec.get("postrelu_objective", "bt"),
        )
        fn = lambda x: apply_activation_torch((x @ transform) * scale + bias, spec["activation"], spec["alpha"])
        stats.update(metadata_without_transform(fitted))
        stats.update(opt_stats)
        stats["switch_layer"] = int(spec["switch_layer"])
        stats["switch_basis"] = basis
    elif kind == "inverse_prior":
        prior, prior_stats = fit_activation_inverse_prior(base_tr, view1_tr, view2_tr, spec["alpha"], args.inverse_reg)
        fitted = fit_cf_transform_with_prior_torch(view1_tr, view2_tr, args.width, spec["schedule"][layer_idx], prior)
        transform = fitted["transform"]
        fn = lambda x: apply_activation_torch(x @ transform, spec["activation"], spec["alpha"])
        stats.update(metadata_without_transform(fitted))
        stats.update(prior_stats)
    elif kind == "residual_cf_branch":
        fitted = fit_cf_transform_torch(view1_tr, view2_tr, args.width, invariance_strength=spec["schedule"][layer_idx])
        transform = fitted["transform"]
        scale = float(args.residual_scale)
        fn = lambda x: x + scale * apply_activation_torch(x @ transform, spec["activation"], spec["alpha"])
        stats.update(metadata_without_transform(fitted))
        stats["residual_scale"] = scale
    elif kind == "residual_cf_branch_postrelu_affineopt":
        fitted = fit_cf_transform_torch(view1_tr, view2_tr, args.width, invariance_strength=spec["schedule"][layer_idx])
        transform = fitted["transform"]
        pre1 = view1_tr @ transform
        pre2 = view2_tr @ transform
        affine_scale, bias, opt_stats = fit_postrelu_affine_torch(
            pre1,
            pre2,
            args,
            optimize_scale=spec.get("postrelu_optimize_scale", True),
            objective=spec.get("postrelu_objective", "bt"),
        )
        scale = float(args.residual_scale)
        fn = lambda x: x + scale * apply_activation_torch((x @ transform) * affine_scale + bias, spec["activation"], spec["alpha"])
        stats.update(metadata_without_transform(fitted))
        stats.update(opt_stats)
        stats["residual_scale"] = scale
    elif kind == "residual_cf_branch_postrelu_active":
        fitted = fit_cf_transform_torch(view1_tr, view2_tr, args.width, invariance_strength=spec["schedule"][layer_idx])
        transform = fitted["transform"]
        pre1 = view1_tr @ transform
        pre2 = view2_tr @ transform
        bias, heuristic_stats = fit_postrelu_active_bias_torch(
            pre1,
            pre2,
            scheduled_spec_value(spec, "active_target", layer_idx),
            args,
        )
        scale = float(args.residual_scale)
        fn = lambda x: x + scale * apply_activation_torch(x @ transform + bias, spec["activation"], spec["alpha"])
        stats.update(metadata_without_transform(fitted))
        stats.update(heuristic_stats)
        stats["residual_scale"] = scale
    elif kind == "residual_cf_branch_postrelu_corrbias":
        fitted = fit_cf_transform_torch(view1_tr, view2_tr, args.width, invariance_strength=spec["schedule"][layer_idx])
        transform = fitted["transform"]
        pre1 = view1_tr @ transform
        pre2 = view2_tr @ transform
        bias, heuristic_stats = fit_postrelu_corrbias_torch(pre1, pre2, spec["corrbias_beta"], args)
        scale = float(args.residual_scale)
        fn = lambda x: x + scale * apply_activation_torch(x @ transform + bias, spec["activation"], spec["alpha"])
        stats.update(metadata_without_transform(fitted))
        stats.update(heuristic_stats)
        stats["residual_scale"] = scale
    elif kind == "residual_cf_branch_rowcorrect":
        fitted = fit_cf_transform_row_correct_torch(view1_tr, view2_tr, args.width, spec["schedule"][layer_idx])
        transform = fitted["transform"]
        scale = float(args.residual_scale)
        fn = lambda x: x + scale * apply_activation_torch(x @ transform, spec["activation"], spec["alpha"])
        stats.update(metadata_without_transform(fitted))
        stats["residual_scale"] = scale
        stats["row_corrected"] = True
    elif kind == "residual_cf_branch_agreement_mode":
        fitted = fit_agreement_mode_transform_torch(
            view1_tr,
            view2_tr,
            args.width,
            spec["schedule"][layer_idx],
            spec["mode_gain_used"],
            spec["gate_beta"],
        )
        transform = fitted["transform"]
        bias = fitted["bias"]
        scale = float(args.residual_scale)
        fn = lambda x: x + scale * apply_activation_torch(x @ transform + bias, spec["activation"], spec["alpha"])
        stats.update(metadata_without_transform(fitted))
        stats["residual_scale"] = scale
    elif kind == "linearized_bt_residual":
        transform, lin_stats = fit_linearized_bt_residual(view1_tr, view2_tr, args)
        scale = float(args.linearized_residual_scale)
        fn = lambda x: x + scale * (x @ transform)
        stats.update(
            {
                "lambda_reg": float("nan"),
                "invariance_strength": float("nan"),
                "max_whitened_delta": float("nan"),
                "min_whitened_delta": float("nan"),
                "mean_gain": float("nan"),
                "min_gain": float("nan"),
                "residual_scale": scale,
            }
        )
        stats.update(lin_stats)
        stats["linearized_residual_scale"] = scale
    else:
        raise ValueError(f"Unknown variant kind: {kind}")

    return (
        fn(base_tr),
        fn(base_te),
        fn(view1_tr),
        fn(view2_tr),
        fn(view1_te),
        fn(view2_te),
        stats,
    )


def collect_variant_state(point, variant, args, device, device_name):
    spec = variant_spec(variant, point.depth)
    arrays = load_point_data(point)
    xtr_np, ytr_np, xte_np, yte_np, *_ = arrays
    tensors = tensors_from_arrays(arrays, device)
    xtr, _, xte, _, view1_tr, view2_tr, view1_te, view2_te = tensors
    train_arrays, test_arrays, _, _ = normalize_hidden_for_spec_torch(
        spec,
        [xtr, view1_tr, view2_tr],
        [xte, view1_te, view2_te],
    )
    base_tr, view1_tr, view2_tr = train_arrays
    base_te, view1_te, view2_te = test_arrays
    pathnorm_train = []
    pathnorm_test = []
    pathnorm_view1_train = []
    pathnorm_view2_train = []
    pathnorm_view1_test = []
    pathnorm_view2_test = []
    transform_rows = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for layer_idx in range(point.depth):
        base_tr, base_te, view1_tr, view2_tr, view1_te, view2_te, row = update_path(
            variant,
            spec,
            layer_idx,
            base_tr,
            base_te,
            view1_tr,
            view2_tr,
            view1_te,
            view2_te,
            args,
        )
        train_arrays, test_arrays, _, _ = normalize_hidden_for_spec_torch(
            spec,
            [base_tr, view1_tr, view2_tr],
            [base_te, view1_te, view2_te],
        )
        train_arrays, test_arrays, postnorm_stats = apply_postnorm_linear_if_needed(
            spec,
            args,
            train_arrays,
            test_arrays,
            layer_idx,
        )
        base_tr, view1_tr, view2_tr = train_arrays
        base_te, view1_te, view2_te = test_arrays
        pathnorm_train.append(base_tr.detach().cpu().numpy().astype(np.float32))
        pathnorm_test.append(base_te.detach().cpu().numpy().astype(np.float32))
        pathnorm_view1_train.append(view1_tr.detach().cpu().numpy().astype(np.float32))
        pathnorm_view2_train.append(view2_tr.detach().cpu().numpy().astype(np.float32))
        pathnorm_view1_test.append(view1_te.detach().cpu().numpy().astype(np.float32))
        pathnorm_view2_test.append(view2_te.detach().cpu().numpy().astype(np.float32))
        transform_rows.append(
            {
                "variant": variant,
                "seed": point.seed,
                "depth": point.depth,
                "kind": spec["kind"],
                "activation": spec["activation"],
                "activation_alpha": float(spec["alpha"]),
                **row,
                **postnorm_stats,
            }
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    fit_time = time.perf_counter() - start
    del tensors
    torch.cuda.empty_cache()
    return {
        "point": asdict(point),
        "device": device_name,
        "variant": variant,
        "kind": spec["kind"],
        "activation": spec["activation"],
        "activation_alpha": float(spec["alpha"]),
        "schedule_json": json.dumps(spec["schedule"]),
        "fit_time_sec": float(fit_time),
        "xtr": xtr_np.astype(np.float32),
        "xte": xte_np.astype(np.float32),
        "ytr": ytr_np.astype(np.int64),
        "yte": yte_np.astype(np.int64),
        "pathnorm_train": pathnorm_train,
        "pathnorm_test": pathnorm_test,
        "pathnorm_view1_train": pathnorm_view1_train,
        "pathnorm_view2_train": pathnorm_view2_train,
        "pathnorm_view1_test": pathnorm_view1_test,
        "pathnorm_view2_test": pathnorm_view2_test,
        "transform_rows": transform_rows,
    }


def evaluate_state(state, probe_reg, pca_dim):
    ytr = state["ytr"]
    yte = state["yte"]
    layer_rows = []
    for idx, (xtr, xte, v1, v2) in enumerate(
        zip(
            state["pathnorm_train"],
            state["pathnorm_test"],
            state["pathnorm_view1_train"],
            state["pathnorm_view2_train"],
        )
    ):
        row = {
            "variant": state["variant"],
            "kind": state["kind"],
            "seed": int(state["point"]["seed"]),
            "dataset": state["point"]["dataset"],
            "input_dim": int(state["point"]["input_dim"]),
            "width": int(state["point"]["width"]),
            "depth": int(state["point"]["depth"]),
            "layer": idx + 1,
        }
        row.update(linear_classifier_readout(xtr, xte, ytr, yte, probe_reg))
        row.update(covariance_spectrum(xtr))
        row.update(view_alignment(v1, v2, int(state["point"]["seed"]) + 8100 + idx))
        layer_rows.append(row)
    last_tr = state["pathnorm_train"][-1]
    last_te = state["pathnorm_test"][-1]
    all_tr = np.concatenate(state["pathnorm_train"], axis=1)
    all_te = np.concatenate(state["pathnorm_test"], axis=1)
    setup_rows = []
    last = {
        "variant": state["variant"],
        "kind": state["kind"],
        "seed": int(state["point"]["seed"]),
        "dataset": state["point"]["dataset"],
        "input_dim": int(state["point"]["input_dim"]),
        "width": int(state["point"]["width"]),
        "depth": int(state["point"]["depth"]),
        "setup": "last_layer_512",
        "source_dim": int(last_tr.shape[1]),
    }
    last.update(linear_classifier_readout(last_tr, last_te, ytr, yte, probe_reg))
    setup_rows.append(last)
    all_pca = {
        "variant": state["variant"],
        "kind": state["kind"],
        "seed": int(state["point"]["seed"]),
        "dataset": state["point"]["dataset"],
        "input_dim": int(state["point"]["input_dim"]),
        "width": int(state["point"]["width"]),
        "depth": int(state["point"]["depth"]),
        "setup": "all_layers_pca512",
    }
    all_pca.update(pca512_readout(all_tr, all_te, ytr, yte, probe_reg, int(state["point"]["seed"]), pca_dim))
    setup_rows.append(all_pca)
    best_layer = max(layer_rows, key=lambda row: row["test_accuracy"])
    last_layer = layer_rows[-1]
    summary = {
        "variant": state["variant"],
        "kind": state["kind"],
        "seed": int(state["point"]["seed"]),
        "dataset": state["point"]["dataset"],
        "input_dim": int(state["point"]["input_dim"]),
        "width": int(state["point"]["width"]),
        "depth": int(state["point"]["depth"]),
        "activation": state["activation"],
        "activation_alpha": float(state["activation_alpha"]),
        "last_layer_accuracy": last["test_accuracy"],
        "all_pca_accuracy": all_pca["test_accuracy"],
        "all_pca_explained_variance": all_pca["explained_variance"],
        "best_layer_accuracy": best_layer["test_accuracy"],
        "best_layer": int(best_layer["layer"]),
        "last_minus_first_accuracy": last_layer["test_accuracy"] - layer_rows[0]["test_accuracy"],
        "last_layer_effective_rank": last_layer["effective_rank"],
        "last_layer_view_mse_ratio": last_layer["same_over_shuffled_mse"],
        "last_layer_view_cosine": last_layer["same_view_cosine"],
        "fit_time_sec": float(state["fit_time_sec"]),
    }
    return layer_rows, setup_rows, summary


def aggregate(rows, key_fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in key_fields)].append(row)
    numeric = sorted(
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float, np.integer, np.floating)) and key not in key_fields
    )
    out = []
    for key, items in sorted(grouped.items()):
        rec = {field: value for field, value in zip(key_fields, key)}
        rec["runs"] = len(items)
        for name in numeric:
            vals = [float(item[name]) for item in items if name in item and np.isfinite(float(item[name]))]
            if vals:
                rec[f"mean_{name}"] = float(np.mean(vals))
                rec[f"std_{name}"] = float(np.std(vals, ddof=0))
        out.append(rec)
    return out


def fmt(row, key):
    return f"{row.get(f'mean_{key}', float('nan')):.4f}"


def build_report(out_dir, summaries, aggregates):
    lines = [
        "# Residual BT CF-MLP Variant Sweep",
        "",
        "Readouts are clean frozen-representation probes: final 512D hidden state, all hidden layers PCA-compressed to 512D, and best individual layer.",
        "",
        "## Aggregate",
        "",
        "| Variant | Depth | Runs | Last layer | All PCA512 | Best layer | Best layer idx | Last-first | Last rank | View ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(aggregates, key=lambda rec: (rec["depth"], rec["variant"])):
        lines.append(
            f"| {row['variant']} | {row['depth']} | {row['runs']} | "
            f"{fmt(row, 'last_layer_accuracy')} | {fmt(row, 'all_pca_accuracy')} | "
            f"{fmt(row, 'best_layer_accuracy')} | {row.get('mean_best_layer', float('nan')):.1f} | "
            f"{fmt(row, 'last_minus_first_accuracy')} | "
            f"{row.get('mean_last_layer_effective_rank', float('nan')):.1f} | "
            f"{row.get('mean_last_layer_view_mse_ratio', float('nan')):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Single Runs",
            "",
            "| Variant | Depth | Last layer | All PCA512 | Best layer | Best layer idx | Last rank | View ratio | Fit sec |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(summaries, key=lambda rec: (rec["depth"], rec["variant"], rec["seed"])):
        lines.append(
            f"| {row['variant']} | {row['depth']} | {row['last_layer_accuracy']:.4f} | "
            f"{row['all_pca_accuracy']:.4f} | {row['best_layer_accuracy']:.4f} | "
            f"{row['best_layer']} | {row['last_layer_effective_rank']:.1f} | "
            f"{row['last_layer_view_mse_ratio']:.3f} | {row['fit_time_sec']:.1f} |"
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

    all_layers = []
    all_setups = []
    all_transforms = []
    summaries = []
    for depth in args.depths:
        for seed in args.seeds:
            point = SweepPoint(
                dataset=args.dataset,
                axis="residual_bt_variants",
                scale_value=args.width,
                seed=seed,
                n_train=args.n_train,
                n_test=args.n_test,
                input_dim=args.input_dim,
                width=args.width,
                depth=depth,
                num_classes=args.num_classes,
            )
            for variant in args.variants:
                print(
                    f"seed={seed} dataset={args.dataset} depth={depth} width={args.width} variant={variant} device={device_name}",
                    flush=True,
                )
                state = collect_variant_state(point, variant, args, device, device_name)
                layer_rows, setup_rows, summary = evaluate_state(state, args.probe_reg, args.pca_dim)
                all_layers.extend(layer_rows)
                all_setups.extend(setup_rows)
                all_transforms.extend(state["transform_rows"])
                summaries.append(summary)
                write_jsonl(args.out_dir / "layer_readouts.partial.jsonl", all_layers)
                write_jsonl(args.out_dir / "setup_readouts.partial.jsonl", all_setups)
                write_jsonl(args.out_dir / "transform_rows.partial.jsonl", all_transforms)
                write_jsonl(args.out_dir / "summary.partial.jsonl", summaries)
                del state, layer_rows, setup_rows, summary
                torch.cuda.empty_cache()
                gc.collect()

    aggregates = aggregate(summaries, ["variant", "kind", "dataset", "input_dim", "width", "depth"])
    write_jsonl(args.out_dir / "layer_readouts.jsonl", all_layers)
    write_jsonl(args.out_dir / "setup_readouts.jsonl", all_setups)
    write_jsonl(args.out_dir / "transform_rows.jsonl", all_transforms)
    write_jsonl(args.out_dir / "summary.jsonl", summaries)
    write_jsonl(args.out_dir / "summary_aggregate.jsonl", aggregates)
    print(build_report(args.out_dir, summaries, aggregates), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Residual and activation-aware CF-MLP representation variants.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_residual_bt_variants_seed7"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.38)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--dataset", default="cifar100_simclr")
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depths", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--pca-dim", type=int, default=512)
    parser.add_argument("--probe-reg", type=float, default=100.0)
    parser.add_argument("--inverse-reg", type=float, default=1e-2)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--linearized-residual-scale", type=float, default=0.25)
    parser.add_argument("--align-weight", type=float, default=1.0)
    parser.add_argument("--bt-weight", type=float, default=0.05)
    parser.add_argument("--bt-offdiag-lambda", type=float, default=0.005)
    parser.add_argument("--linearized-ridge", type=float, default=1.0)
    parser.add_argument("--linearized-lr", type=float, default=0.05)
    parser.add_argument("--linearized-steps", type=int, default=30)
    parser.add_argument("--linearized-grad-clip", type=float, default=10.0)
    parser.add_argument("--linearized-max-norm", type=float, default=16.0)
    parser.add_argument("--postrelu-fit-samples", type=int, default=1024)
    parser.add_argument("--postrelu-steps", type=int, default=40)
    parser.add_argument("--postrelu-lr", type=float, default=0.05)
    parser.add_argument("--postrelu-scale-ridge", type=float, default=1e-4)
    parser.add_argument("--postrelu-bias-ridge", type=float, default=1e-4)
    parser.add_argument("--postrelu-grad-clip", type=float, default=10.0)
    parser.add_argument("--postnorm-linear-fit-samples", type=int, default=2048)
    parser.add_argument("--postnorm-linear-steps", type=int, default=60)
    parser.add_argument("--postnorm-linear-lr", type=float, default=0.03)
    parser.add_argument("--postnorm-linear-ridge", type=float, default=1e-4)
    parser.add_argument("--postnorm-linear-grad-clip", type=float, default=10.0)
    parser.add_argument("--postnorm-linear-cca-eps", type=float, default=1e-4)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "plain_cf_bpbt_nonlinearity",
            "inverse_prior_ols_bpbt_nonlinearity",
            "residual_cf_branch_bpbt_nonlinearity",
            "linearized_bt_residual",
        ],
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
