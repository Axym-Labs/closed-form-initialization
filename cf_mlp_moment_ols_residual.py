import argparse
import csv
import gc
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cf_mlp_barlow_clean import load_tensors
from cf_mlp_bpbt_spectral_diagnostic import agreement_spectrum_metrics, transition_metrics
from cf_mlp_bt_objective_by_layer import bt_hidden_metrics
from cf_mlp_clean_readouts import linear_classifier_readout, pca512_readout
from cf_mlp_layer_mechanistic import covariance_spectrum, standardize_many
from cf_mlp_residual_barlow import leaky_gelu
from cf_mlp_residual_bt_variants import fit_cf_transform_with_prior_torch
from cf_mlp_scalability import SweepPoint, write_jsonl
from cf_mlp_scalability_gpu import fit_cf_transform_torch, fit_whitening_transform_torch, normalize_hidden_with_stats_torch


def point_for(args, depth, seed):
    return SweepPoint(
        dataset=args.dataset,
        axis="moment_ols_residual",
        scale_value=args.width,
        seed=seed,
        n_train=args.n_train,
        n_test=args.n_test,
        input_dim=args.input_dim,
        width=args.width,
        depth=depth,
        num_classes=args.num_classes,
    )


def standardize_with_stats_torch(x, eps=1e-4):
    x = x.to(dtype=torch.float32)
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    scale = torch.sqrt(torch.mean(centered * centered, dim=0, keepdim=True))
    scale = torch.clamp(scale, min=eps)
    return centered / scale, mean, scale


def standardize_torch(x, eps=1e-4):
    standardized, _, _ = standardize_with_stats_torch(x, eps)
    return standardized


def center_torch(x):
    return x.to(dtype=torch.float32) - x.to(dtype=torch.float32).mean(dim=0, keepdim=True)


def rms_torch(x):
    return torch.sqrt(torch.mean(x * x))


def row_cosine_torch(a, b):
    num = torch.sum(a * b, dim=1)
    den = torch.linalg.vector_norm(a, dim=1) * torch.linalg.vector_norm(b, dim=1)
    return float(torch.mean(num / torch.clamp(den, min=1e-12)).detach().cpu().item())


def bt_corr_and_gradient(view1, view2, bt_lambda):
    z1, _, scale1 = standardize_with_stats_torch(view1)
    z2, _, scale2 = standardize_with_stats_torch(view2)
    corr = (z1.T @ z2) / float(z1.shape[0])
    grad = 2.0 * float(bt_lambda) * corr
    diag = torch.arange(corr.shape[0], device=corr.device)
    grad[diag, diag] = 2.0 * (corr[diag, diag] - 1.0)
    return z1, z2, corr, grad, scale1.squeeze(0), scale2.squeeze(0)


def precondition_bt_gradient(grad, diag_multiplier):
    if float(diag_multiplier) == 1.0:
        return grad
    conditioned = grad.clone()
    diag = torch.arange(conditioned.shape[0], device=conditioned.device)
    conditioned[diag, diag] = conditioned[diag, diag] * float(diag_multiplier)
    return conditioned


def polar_delta_target(corr, eta_layer, weight):
    u, _, vh = torch.linalg.svd(corr, full_matrices=False)
    polar = u @ vh
    return float(eta_layer) * float(weight) * (polar - corr)


def moment_delta_target(args, corr, bt_grad, eta_layer):
    bt_target = -float(eta_layer) * precondition_bt_gradient(bt_grad, args.diag_gradient_multiplier)
    if args.moment_target_kind == "bt":
        return bt_target
    polar_target = polar_delta_target(corr, eta_layer, args.polar_target_weight)
    if args.moment_target_kind == "polar":
        return polar_target
    if args.moment_target_kind == "bt_plus_polar":
        return bt_target + polar_target
    raise ValueError(f"Unknown moment target kind: {args.moment_target_kind}")


def standardization_tangent_project(delta_z, z):
    return delta_z - delta_z.mean(dim=0, keepdim=True) - z * torch.mean(z * delta_z, dim=0, keepdim=True)


def standardization_tangent_context(reference, eps=1e-4):
    z, _, scale = standardize_with_stats_torch(reference, eps)
    inv_scale = 1.0 / torch.clamp(scale.squeeze(0), min=1e-12)
    return z, inv_scale


def apply_standardized_tangent(delta, z, inv_scale):
    return standardization_tangent_project(delta * inv_scale.unsqueeze(0), z)


def row_layernorm_torch(x, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), eps=float(eps))


def row_layernorm_tangent(delta, reference, eps=1e-5):
    ref = reference.to(dtype=delta.dtype)
    centered_ref = ref - ref.mean(dim=1, keepdim=True)
    scale = torch.sqrt(torch.mean(centered_ref * centered_ref, dim=1, keepdim=True) + float(eps))
    normalized_ref = centered_ref / torch.clamp(scale, min=1e-12)
    centered_delta = delta - delta.mean(dim=1, keepdim=True)
    radial = normalized_ref * torch.mean(normalized_ref * centered_delta, dim=1, keepdim=True)
    return (centered_delta - radial) / torch.clamp(scale, min=1e-12)


def mode_balance_delta(delta_z, z, power, eps, min_gain, max_gain):
    if float(power) == 0.0:
        return delta_z
    operator = mode_balance_operator_from_cov(
        (z.T @ z) / float(z.shape[0]),
        power,
        eps,
        min_gain,
        max_gain,
    )
    return delta_z @ operator


def mode_balance_operator_from_cov(cov, power, eps, min_gain, max_gain):
    cov = 0.5 * (cov + cov.T)
    evals, evecs = torch.linalg.eigh(cov)
    evals = torch.clamp(evals, min=float(eps))
    gains = evals.pow(-float(power))
    gains = gains / torch.clamp(torch.sqrt(torch.mean(gains * gains)), min=1e-12)
    if float(min_gain) > 0.0 or float(max_gain) > 0.0:
        lo = float(min_gain) if float(min_gain) > 0.0 else 0.0
        hi = float(max_gain) if float(max_gain) > 0.0 else float("inf")
        gains = torch.clamp(gains, min=lo, max=hi)
        gains = gains / torch.clamp(torch.sqrt(torch.mean(gains * gains)), min=1e-12)
    return (evecs * gains.unsqueeze(0)) @ evecs.T


def pair_covariance_operator(view1, view2, power, eps, min_gain, max_gain):
    mean = 0.5 * (view1.mean(dim=0, keepdim=True) + view2.mean(dim=0, keepdim=True))
    h1 = view1 - mean
    h2 = view2 - mean
    cov = 0.5 * ((h1.T @ h1) / float(h1.shape[0]) + (h2.T @ h2) / float(h2.shape[0]))
    return mode_balance_operator_from_cov(cov, power, eps, min_gain, max_gain)


def activation_gradient_targets(
    z1,
    z2,
    target_grad,
    scale1,
    scale2,
    eta_layer,
    sample_scale,
    mode_balance_power=0.0,
    mode_balance_eps=1e-3,
    mode_balance_min_gain=0.0,
    mode_balance_max_gain=0.0,
):
    delta_z1 = -float(eta_layer) * float(sample_scale) * (z2 @ target_grad.T)
    delta_z2 = -float(eta_layer) * float(sample_scale) * (z1 @ target_grad)
    delta_z1 = mode_balance_delta(
        delta_z1,
        z1,
        mode_balance_power,
        mode_balance_eps,
        mode_balance_min_gain,
        mode_balance_max_gain,
    )
    delta_z2 = mode_balance_delta(
        delta_z2,
        z2,
        mode_balance_power,
        mode_balance_eps,
        mode_balance_min_gain,
        mode_balance_max_gain,
    )
    delta_z1 = standardization_tangent_project(delta_z1, z1)
    delta_z2 = standardization_tangent_project(delta_z2, z2)
    return delta_z1 * scale1.unsqueeze(0), delta_z2 * scale2.unsqueeze(0)


def residualize_sample_target(reference, target, ridge, rescale_max=0.0):
    ref = center_torch(reference)
    centered_target = center_torch(target)
    n = float(ref.shape[0])
    eye = torch.eye(ref.shape[1], dtype=ref.dtype, device=ref.device)
    gram = (ref.T @ ref) / n + float(ridge) * eye
    rhs = (ref.T @ centered_target) / n
    coef = torch.linalg.solve(gram, rhs)
    predicted = ref @ coef
    residual = centered_target - predicted
    target_energy = torch.sum(centered_target * centered_target)
    residual_energy = torch.sum(residual * residual)
    energy_fraction = residual_energy / torch.clamp(target_energy, min=1e-12)
    gain = torch.ones((), dtype=residual.dtype, device=residual.device)
    if float(rescale_max) > 0.0:
        gain = torch.rsqrt(torch.clamp(energy_fraction, min=1e-12))
        gain = torch.clamp(gain, max=float(rescale_max))
        residual = residual * gain
    return residual, {
        "sample_target_residual_energy_fraction": float(energy_fraction.detach().cpu().item()),
        "sample_target_projection_r2": float((1.0 - energy_fraction).detach().cpu().item()),
        "sample_target_residual_rescale_gain": float(gain.detach().cpu().item()),
    }


def project_features_onto_reference(reference, features, ridge):
    ref = center_torch(reference)
    feat = center_torch(features)
    n = float(ref.shape[0])
    eye = torch.eye(ref.shape[1], dtype=ref.dtype, device=ref.device)
    gram = (ref.T @ ref) / n + float(ridge) * eye
    rhs = (ref.T @ feat) / n
    coef = torch.linalg.solve(gram, rhs)
    projected = ref @ coef
    projected_energy = torch.sum(projected * projected)
    feature_energy = torch.sum(feat * feat)
    energy_fraction = projected_energy / torch.clamp(feature_energy, min=1e-12)
    return projected, {
        "old_span_feature_projection_energy_fraction": float(energy_fraction.detach().cpu().item()),
    }


def stable_mode_basis(view1, view2, count, ridge, max_delta, kind):
    if int(count) <= 0:
        return None, {
            "stable_mode_count": 0,
            "stable_mode_mean_delta": float("nan"),
            "stable_mode_max_delta": float("nan"),
        }
    out_dtype = view1.dtype
    h1 = center_torch(view1).to(dtype=torch.float64)
    h2 = center_torch(view2).to(dtype=torch.float64)
    n = float(h1.shape[0])
    dim = h1.shape[1]
    sigma = 0.5 * ((h1.T @ h1) / n + (h2.T @ h2) / n)
    diff = h1 - h2
    delta = (diff.T @ diff) / n
    sigma = 0.5 * (sigma + sigma.T)
    delta = 0.5 * (delta + delta.T)
    evals_sigma, evecs_sigma = torch.linalg.eigh(sigma)
    if kind == "pca":
        order = torch.argsort(evals_sigma, descending=True)[: min(int(count), dim)]
        basis = evecs_sigma[:, order]
        denom = torch.sum(basis * (sigma @ basis), dim=0)
        selected = torch.sum(basis * (delta @ basis), dim=0) / torch.clamp(denom, min=1e-12)
        selected = torch.clamp(selected, min=0.0)
        return basis.to(dtype=out_dtype), {
            "stable_mode_count": int(order.numel()),
            "stable_mode_mean_delta": float(torch.mean(selected).detach().cpu().item()),
            "stable_mode_max_delta": float(torch.max(selected).detach().cpu().item()),
        }
    if kind != "agreement":
        raise ValueError(f"Unknown stable-mode kind: {kind}")
    evals_sigma = torch.clamp(evals_sigma, min=float(ridge))
    sigma_inv_sqrt = (evecs_sigma / torch.sqrt(evals_sigma).unsqueeze(0)) @ evecs_sigma.T
    whitened_delta = sigma_inv_sqrt @ delta @ sigma_inv_sqrt
    whitened_delta = 0.5 * (whitened_delta + whitened_delta.T)
    delta_evals, delta_evecs = torch.linalg.eigh(whitened_delta)
    delta_evals = torch.clamp(delta_evals, min=0.0)
    order = torch.argsort(delta_evals, descending=False)
    if float(max_delta) > 0.0:
        order = order[delta_evals[order] <= float(max_delta)]
    order = order[: min(int(count), int(order.numel()))]
    if int(order.numel()) == 0:
        return None, {
            "stable_mode_count": 0,
            "stable_mode_mean_delta": float("nan"),
            "stable_mode_max_delta": float("nan"),
        }
    basis = sigma_inv_sqrt @ delta_evecs[:, order]
    basis = basis / torch.clamp(torch.linalg.vector_norm(basis, dim=0, keepdim=True), min=1e-12)
    selected = delta_evals[order]
    return basis.to(dtype=out_dtype), {
        "stable_mode_count": int(order.numel()),
        "stable_mode_mean_delta": float(torch.mean(selected).detach().cpu().item()),
        "stable_mode_max_delta": float(torch.max(selected).detach().cpu().item()),
    }


def bt_total_per_dim_torch(view1, view2, bt_lambda):
    _, _, corr, _, _, _ = bt_corr_and_gradient(view1, view2, bt_lambda)
    diag = torch.diagonal(corr)
    on_diag = torch.sum((diag - 1.0) ** 2)
    off_diag = torch.sum(corr * corr) - torch.sum(diag * diag)
    return (on_diag + float(bt_lambda) * off_diag) / corr.shape[0]


def bt_quadratic_energy_per_dim(delta_corr, bt_lambda):
    diag = torch.diagonal(delta_corr)
    diag_energy = torch.sum(diag * diag)
    off_energy = torch.sum(delta_corr * delta_corr) - diag_energy
    return (diag_energy + float(bt_lambda) * off_energy) / delta_corr.shape[0]


def normalized_candidate_arrays(args, base, view1, view2, delta_base, delta_v1, delta_v2, scale):
    cand_base = base + float(scale) * delta_base
    cand_v1 = view1 + float(scale) * delta_v1
    cand_v2 = view2 + float(scale) * delta_v2
    if args.residual_normalization == "layernorm":
        return [
            row_layernorm_torch(cand_base, args.layernorm_eps),
            row_layernorm_torch(cand_v1, args.layernorm_eps),
            row_layernorm_torch(cand_v2, args.layernorm_eps),
        ]
    cand_train, _, _, _ = normalize_hidden_with_stats_torch([cand_base, cand_v1, cand_v2], [])
    return cand_train


def bt_quadratic_scale_from_realized_corr(args, base, view1, view2, delta_base, delta_v1, delta_v2):
    _, _, corr, grad, _, _ = bt_corr_and_gradient(view1, view2, args.bt_lambda)
    cand = normalized_candidate_arrays(args, base, view1, view2, delta_base, delta_v1, delta_v2, 1.0)
    _, _, corr_after, _, _, _ = bt_corr_and_gradient(cand[1], cand[2], args.bt_lambda)
    delta_corr = corr_after - corr
    first_order = torch.sum(grad * delta_corr) / corr.shape[0]
    quadratic = bt_quadratic_energy_per_dim(delta_corr, args.bt_lambda)
    if first_order >= 0.0 or quadratic <= 0.0:
        scale = torch.zeros((), dtype=view1.dtype, device=view1.device)
    else:
        scale = -first_order / torch.clamp(2.0 * quadratic, min=1e-12)
        scale = torch.clamp(scale, min=0.0, max=float(args.bt_quadratic_scale_max))
    predicted_delta = scale * first_order + scale * scale * quadratic
    return float(scale.detach().cpu().item()), {
        "bt_quadratic_full_first_order_delta": float(first_order.detach().cpu().item()),
        "bt_quadratic_full_second_order_delta": float(quadratic.detach().cpu().item()),
        "bt_quadratic_predicted_delta": float(predicted_delta.detach().cpu().item()),
        "bt_quadratic_full_scale_unclipped": float(
            (-first_order / torch.clamp(2.0 * quadratic, min=1e-12)).detach().cpu().item()
        )
        if float(quadratic.detach().cpu().item()) > 0.0
        else float("nan"),
    }


def bt_score_stats_torch(view1, view2, bt_lambda):
    _, _, corr, _, _, _ = bt_corr_and_gradient(view1, view2, bt_lambda)
    diag = torch.diagonal(corr)
    on_diag = torch.sum((diag - 1.0) ** 2)
    off_diag = torch.sum(corr * corr) - torch.sum(diag * diag)
    bt = (on_diag + float(bt_lambda) * off_diag) / corr.shape[0]
    nuclear = torch.sum(torch.linalg.svdvals(corr)) / corr.shape[0]
    return {
        "bt": float(bt.detach().cpu().item()),
        "nuclear": float(nuclear.detach().cpu().item()),
    }


def old_span_adaptive_score(stats, args):
    if args.old_span_adaptive_metric == "bt":
        return float(stats["bt"])
    if args.old_span_adaptive_metric == "bt_plus_nuclear":
        return float(stats["bt"]) - float(args.old_span_adaptive_nuclear_weight) * float(stats["nuclear"])
    raise ValueError(f"Unknown old-span adaptive metric: {args.old_span_adaptive_metric}")


def self_corr_offdiag_per_dim_torch(x):
    z = standardize_torch(x)
    corr = (z.T @ z) / float(z.shape[0])
    diag = torch.diagonal(corr)
    off_diag = torch.sum(corr * corr) - torch.sum(diag * diag)
    return off_diag / corr.shape[0]


def combined_train_score_torch(view1, view2, bt_lambda, self_cov_weight):
    score = bt_total_per_dim_torch(view1, view2, bt_lambda)
    if float(self_cov_weight) > 0.0:
        score = score + 0.5 * float(self_cov_weight) * (
            self_corr_offdiag_per_dim_torch(view1) + self_corr_offdiag_per_dim_torch(view2)
        )
    return score


def paired_self_corr_offdiag_per_dim_torch(view1, view2):
    return 0.5 * (self_corr_offdiag_per_dim_torch(view1) + self_corr_offdiag_per_dim_torch(view2))


def covariance_effective_rank_torch(x):
    x0 = center_torch(x)
    cov = (x0.T @ x0) / max(1.0, float(x0.shape[0] - 1))
    cov = 0.5 * (cov + cov.T)
    vals = torch.clamp(torch.linalg.eigvalsh(cov), min=0.0)
    total = torch.sum(vals)
    if float(total.detach().cpu().item()) <= 1e-12:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    probs = vals / total
    entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-12)))
    return torch.exp(entropy)


def offdiag_matrix(matrix):
    return matrix - torch.diag(torch.diagonal(matrix))


def deterministic_orthogonal_matrix(rows, cols, seed, dtype, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    mat = torch.randn((rows, cols), generator=gen, dtype=dtype, device=device)
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q[:, :cols]


def branch_matrix(args, variant, layer_idx, view1, view2, seed):
    dim = view1.shape[1]
    branch_dim = int(args.branch_dim)
    if variant == "random_orth":
        return deterministic_orthogonal_matrix(dim, branch_dim, seed + 1009 * (layer_idx + 1), view1.dtype, view1.device)
    if variant == "cf_shrink":
        prior = torch.eye(dim, dtype=view1.dtype, device=view1.device)
        fitted = fit_cf_transform_with_prior_torch(
            view1,
            view2,
            dim,
            invariance_strength=args.cf_invariance,
            prior_transform=prior,
        )
        transform = fitted["transform"]
        if float(args.branch_random_blend) > 0.0:
            random = deterministic_orthogonal_matrix(
                dim,
                dim,
                seed + 3571 * (layer_idx + 1),
                view1.dtype,
                view1.device,
            )
            random = random * (rms_torch(transform) / torch.clamp(rms_torch(random), min=1e-12))
            blend = float(args.branch_random_blend)
            transform = (1.0 - blend) * transform + blend * random
        if float(args.branch_mode_balance_power) != 0.0:
            balance = pair_covariance_operator(
                view1,
                view2,
                args.branch_mode_balance_power,
                args.branch_mode_balance_eps,
                args.branch_mode_balance_min_gain,
                args.branch_mode_balance_max_gain,
            )
            if args.branch_mode_balance_side == "input":
                transform = balance @ transform
            elif args.branch_mode_balance_side == "output":
                transform = transform @ balance
            elif args.branch_mode_balance_side == "both":
                transform = balance @ transform @ balance
            else:
                raise ValueError(f"Unknown branch mode-balance side: {args.branch_mode_balance_side}")
        if branch_dim == dim:
            return transform
        q, _ = torch.linalg.qr(transform, mode="reduced")
        return q[:, :branch_dim]
    if variant == "shared_cross":
        mean = 0.5 * (view1.mean(dim=0, keepdim=True) + view2.mean(dim=0, keepdim=True))
        h1 = view1 - mean
        h2 = view2 - mean
        n = float(view1.shape[0])
        cross = 0.5 * ((h1.T @ h2) + (h2.T @ h1)) / n
        cross = 0.5 * (cross + cross.T)
        evals, evecs = torch.linalg.eigh(cross)
        order = torch.argsort(evals, descending=True)
        if branch_dim < dim:
            order = order[:branch_dim]
        modes = evecs[:, order]
        if float(args.branch_shared_power) > 0.0:
            gains = torch.clamp(evals[order], min=0.0) ** float(args.branch_shared_power)
            gains = gains / torch.clamp(torch.sqrt(torch.mean(gains * gains)), min=1e-12)
            modes = modes * gains.unsqueeze(0)
        return modes
    raise ValueError(f"Unknown branch variant: {variant}")


def apply_activation(x, alpha):
    return leaky_gelu(x, alpha)


def branch_features(args, arrays, branch):
    raw = [apply_activation(arr @ branch, args.activation_alpha) for arr in arrays]
    train_normed, test_normed, mean, scale = normalize_hidden_with_stats_torch(raw[:3], raw[3:])
    return train_normed, test_normed, mean, scale


def branch_covariance_moments(view1, view2, ridge):
    out_dtype = view1.dtype
    h1 = center_torch(view1).to(dtype=torch.float64)
    h2 = center_torch(view2).to(dtype=torch.float64)
    n = float(h1.shape[0])
    sigma = 0.5 * ((h1.T @ h1) / n + (h2.T @ h2) / n)
    diff = h1 - h2
    delta = (diff.T @ diff) / n
    sigma = 0.5 * (sigma + sigma.T)
    delta = 0.5 * (delta + delta.T)
    evals, evecs = torch.linalg.eigh(sigma)
    evals = torch.clamp(evals, min=float(ridge))
    sigma_inv_sqrt = (evecs / torch.sqrt(evals).unsqueeze(0)) @ evecs.T
    return sigma.to(dtype=out_dtype), delta.to(dtype=out_dtype), sigma_inv_sqrt.to(dtype=out_dtype)


def branch_reach_adjoint(args, current_train, branch_train):
    _, view1, view2 = current_train
    _, phi_v1, phi_v2 = branch_train
    z1, z2, corr, grad, scale1, scale2 = bt_corr_and_gradient(view1, view2, args.bt_lambda)
    target = -precondition_bt_gradient(grad, args.diag_gradient_multiplier)
    inv_scale1 = 1.0 / torch.clamp(scale1, min=1e-12)
    inv_scale2 = 1.0 / torch.clamp(scale2, min=1e-12)
    if args.residual_normalization == "layernorm":
        term = {
            "type": "moment_layernorm",
            "phi1": center_torch(phi_v1),
            "phi2": center_torch(phi_v2),
            "reference1": view1,
            "reference2": view2,
            "z1": z1,
            "z2": z2,
            "inv_scale1": inv_scale1,
            "inv_scale2": inv_scale2,
            "layernorm_eps": float(args.layernorm_eps),
        }
    else:
        phi1 = center_torch(phi_v1)
        phi2 = center_torch(phi_v2)
        n = float(z1.shape[0])
        term = {
            "type": "moment",
            "m1": (phi1.T @ z2) / n,
            "m2": (z1.T @ phi2) / n,
            "inv_scale1": inv_scale1,
            "inv_scale2": inv_scale2,
            "context": operator_context(args, phi1, phi2, z1, z2, corr),
        }
    return adjoint_term_operator(target, term)


def grad_reach_branch_transform(args, current_train, branch_train):
    sigma, delta, sigma_inv_sqrt = branch_covariance_moments(
        branch_train[1],
        branch_train[2],
        args.branch_post_cov_ridge,
    )
    adjoint = branch_reach_adjoint(args, current_train, branch_train)
    reach = (adjoint @ adjoint.T) / float(adjoint.shape[1])
    reach = 0.5 * (reach + reach.T)
    reach_w = sigma_inv_sqrt @ reach @ sigma_inv_sqrt
    delta_w = sigma_inv_sqrt @ delta @ sigma_inv_sqrt
    reach_w = 0.5 * (reach_w + reach_w.T)
    delta_w = 0.5 * (delta_w + delta_w.T)
    invariance_ridge = 1.0 / max(float(args.branch_post_invariance), 1e-12)
    denom = delta_w + invariance_ridge * torch.eye(
        delta_w.shape[0],
        dtype=delta_w.dtype,
        device=delta_w.device,
    )
    denom_evals, denom_evecs = torch.linalg.eigh(denom.to(dtype=torch.float64))
    denom_evals = torch.clamp(denom_evals, min=float(args.branch_post_cov_ridge))
    denom_inv_sqrt = (denom_evecs / torch.sqrt(denom_evals).unsqueeze(0)) @ denom_evecs.T
    score = denom_inv_sqrt @ reach_w.to(dtype=torch.float64) @ denom_inv_sqrt
    score = 0.5 * (score + score.T)
    score_evals, score_evecs = torch.linalg.eigh(score)
    order = torch.argsort(score_evals, descending=True)
    count = min(branch_train[1].shape[1], int(args.branch_post_dim) if int(args.branch_post_dim) > 0 else branch_train[1].shape[1])
    order = order[:count]
    basis_w = denom_inv_sqrt @ score_evecs[:, order]
    transform = sigma_inv_sqrt.to(dtype=torch.float64) @ basis_w
    transform = transform.to(dtype=branch_train[1].dtype)
    selected = basis_w
    selected_reach = torch.sum(selected * (reach_w.to(dtype=torch.float64) @ selected), dim=0)
    selected_delta = torch.sum(selected * (delta_w.to(dtype=torch.float64) @ selected), dim=0)
    return transform, {
        "branch_post_reach_score_mean": float(torch.mean(score_evals[order]).detach().cpu().item()),
        "branch_post_reach_score_max": float(torch.max(score_evals[order]).detach().cpu().item()),
        "branch_post_reach_mean": float(torch.mean(selected_reach).detach().cpu().item()),
        "branch_post_delta_mean": float(torch.mean(selected_delta).detach().cpu().item()),
    }


def branch_post_transform_features(args, current_train, branch_train, branch_test):
    metrics = {
        "branch_post_transform": args.branch_post_transform,
        "branch_post_invariance": float(args.branch_post_invariance),
        "branch_post_mean_gain": float("nan"),
        "branch_post_min_gain": float("nan"),
        "branch_post_max_delta": float("nan"),
        "branch_post_reach_score_mean": float("nan"),
        "branch_post_reach_score_max": float("nan"),
        "branch_post_reach_mean": float("nan"),
        "branch_post_delta_mean": float("nan"),
    }
    if args.branch_post_transform == "none":
        return branch_train, branch_test, metrics
    if args.branch_post_transform == "cf_shrink":
        fitted = fit_cf_transform_torch(
            branch_train[1],
            branch_train[2],
            branch_train[1].shape[1],
            invariance_strength=args.branch_post_invariance,
        )
        transform = fitted["transform"]
        transformed_train = [arr @ transform for arr in branch_train]
        transformed_test = [arr @ transform for arr in branch_test]
        transformed_train, transformed_test, _, _ = normalize_hidden_with_stats_torch(
            transformed_train,
            transformed_test,
        )
        metrics.update(
            {
                "branch_post_mean_gain": fitted["mean_gain"],
                "branch_post_min_gain": fitted["min_gain"],
                "branch_post_max_delta": fitted["max_whitened_delta"],
            }
        )
        return transformed_train, transformed_test, metrics
    if args.branch_post_transform == "whiten":
        fitted = fit_whitening_transform_torch(
            branch_train[1],
            branch_train[2],
            branch_train[1].shape[1],
        )
        transform = fitted["transform"]
        transformed_train = [arr @ transform for arr in branch_train]
        transformed_test = [arr @ transform for arr in branch_test]
        transformed_train, transformed_test, _, _ = normalize_hidden_with_stats_torch(
            transformed_train,
            transformed_test,
        )
        metrics.update(
            {
                "branch_post_mean_gain": fitted["mean_gain"],
                "branch_post_min_gain": fitted["min_gain"],
                "branch_post_max_delta": fitted["max_whitened_delta"],
            }
        )
        return transformed_train, transformed_test, metrics
    if args.branch_post_transform == "grad_reach":
        transform, reach_metrics = grad_reach_branch_transform(args, current_train, branch_train)
        transformed_train = [arr @ transform for arr in branch_train]
        transformed_test = [arr @ transform for arr in branch_test]
        transformed_train, transformed_test, _, _ = normalize_hidden_with_stats_torch(
            transformed_train,
            transformed_test,
        )
        metrics.update(reach_metrics)
        return transformed_train, transformed_test, metrics
    raise ValueError(f"Unknown branch post-transform: {args.branch_post_transform}")


def residualized_branch_features(current_train, current_test, branch_train, branch_test, ridge):
    ref_train = torch.cat(current_train, dim=0)
    phi_train = torch.cat(branch_train, dim=0)
    ref_mean = ref_train.mean(dim=0, keepdim=True)
    phi_mean = phi_train.mean(dim=0, keepdim=True)
    ref_centered = ref_train - ref_mean
    phi_centered = phi_train - phi_mean
    n = float(ref_centered.shape[0])
    eye = torch.eye(ref_centered.shape[1], dtype=ref_centered.dtype, device=ref_centered.device)
    gram = (ref_centered.T @ ref_centered) / n + float(ridge) * eye
    rhs = (ref_centered.T @ phi_centered) / n
    coef = torch.linalg.solve(gram, rhs)

    def residualize(ref, phi):
        return (phi - phi_mean) - (ref - ref_mean) @ coef

    residual_train = [residualize(ref, phi) for ref, phi in zip(current_train, branch_train)]
    residual_test = [residualize(ref, phi) for ref, phi in zip(current_test, branch_test)]
    numerator = sum(torch.sum(res * res) for res in residual_train)
    denominator = sum(torch.sum((phi - phi_mean) * (phi - phi_mean)) for phi in branch_train)
    energy_fraction = float((numerator / torch.clamp(denominator, min=1e-12)).detach().cpu().item())
    return residual_train, residual_test, {
        "branch_residual_energy_fraction": energy_fraction,
        "branch_projection_r2": 1.0 - energy_fraction,
    }


def project_onto_basis(features, basis, ridge):
    gram = basis.T @ basis
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    coef = torch.linalg.solve(gram + float(ridge) * eye, basis.T)
    return features @ basis @ coef


def filter_residual_branch_features(args, residual_train, residual_test, metrics):
    metrics = dict(metrics)
    metrics.update(
        {
            "branch_novelty_filter_count": 0,
            "branch_novelty_filter_mean_delta": float("nan"),
            "branch_novelty_filter_max_delta": float("nan"),
            "branch_novelty_filter_energy_fraction": float("nan"),
            "branch_novelty_filter_mean_gain": float("nan"),
            "branch_novelty_filter_min_gain": float("nan"),
        }
    )
    if args.branch_novelty_filter == "none":
        return residual_train, residual_test, metrics
    if args.branch_novelty_filter == "agreement_shrink":
        fitted = fit_cf_transform_torch(
            residual_train[1],
            residual_train[2],
            residual_train[1].shape[1],
            invariance_strength=args.branch_novelty_filter_invariance,
        )
        transform = fitted["transform"]
        filtered_train = [arr @ transform for arr in residual_train]
        filtered_test = [arr @ transform for arr in residual_test]
        numerator = sum(torch.sum(arr * arr) for arr in filtered_train)
        denominator = sum(torch.sum(arr * arr) for arr in residual_train)
        metrics.update(
            {
                "branch_novelty_filter_count": int(transform.shape[1]),
                "branch_novelty_filter_mean_delta": fitted["max_whitened_delta"],
                "branch_novelty_filter_max_delta": fitted["max_whitened_delta"],
                "branch_novelty_filter_energy_fraction": float(
                    (numerator / torch.clamp(denominator, min=1e-12)).detach().cpu().item()
                ),
                "branch_novelty_filter_mean_delta": float("nan"),
                "branch_novelty_filter_mean_gain": fitted["mean_gain"],
                "branch_novelty_filter_min_gain": fitted["min_gain"],
            }
        )
        return filtered_train, filtered_test, metrics
    basis, basis_metrics = stable_mode_basis(
        residual_train[1],
        residual_train[2],
        args.branch_novelty_filter_count,
        args.branch_novelty_filter_ridge,
        args.branch_novelty_filter_max_delta,
        args.branch_novelty_filter,
    )
    metrics.update(
        {
            "branch_novelty_filter_count": basis_metrics["stable_mode_count"],
            "branch_novelty_filter_mean_delta": basis_metrics["stable_mode_mean_delta"],
            "branch_novelty_filter_max_delta": basis_metrics["stable_mode_max_delta"],
        }
    )
    if basis is None:
        filtered_train = [torch.zeros_like(arr) for arr in residual_train]
        filtered_test = [torch.zeros_like(arr) for arr in residual_test]
        metrics["branch_novelty_filter_energy_fraction"] = 0.0
        return filtered_train, filtered_test, metrics
    filtered_train = [
        project_onto_basis(arr, basis, args.branch_novelty_filter_projection_ridge) for arr in residual_train
    ]
    filtered_test = [
        project_onto_basis(arr, basis, args.branch_novelty_filter_projection_ridge) for arr in residual_test
    ]
    numerator = sum(torch.sum(arr * arr) for arr in filtered_train)
    denominator = sum(torch.sum(arr * arr) for arr in residual_train)
    metrics["branch_novelty_filter_energy_fraction"] = float(
        (numerator / torch.clamp(denominator, min=1e-12)).detach().cpu().item()
    )
    return filtered_train, filtered_test, metrics


def build_branch_dictionary(args, current_train, current_test, branch_train, branch_test):
    if args.branch_novelty_mode == "mix" and float(args.branch_novelty_mix) <= 0.0:
        return branch_train, branch_test, {
            "branch_residual_energy_fraction": float("nan"),
            "branch_projection_r2": float("nan"),
            "branch_novelty_filter_count": 0,
            "branch_novelty_filter_mean_delta": float("nan"),
            "branch_novelty_filter_max_delta": float("nan"),
            "branch_novelty_filter_energy_fraction": float("nan"),
            "branch_novelty_filter_mean_gain": float("nan"),
            "branch_novelty_filter_min_gain": float("nan"),
        }
    residual_train, residual_test, metrics = residualized_branch_features(
        current_train,
        current_test,
        branch_train,
        branch_test,
        args.branch_residual_ridge,
    )
    residual_train, residual_test, metrics = filter_residual_branch_features(args, residual_train, residual_test, metrics)
    if args.branch_novelty_mode == "mix":
        mix = float(args.branch_novelty_mix)
        mixed_train = [(1.0 - mix) * old + mix * new for old, new in zip(branch_train, residual_train)]
        mixed_test = [(1.0 - mix) * old + mix * new for old, new in zip(branch_test, residual_test)]
        train_normed, test_normed, _, _ = normalize_hidden_with_stats_torch(mixed_train, mixed_test)
        return train_normed, test_normed, metrics
    if args.branch_novelty_mode == "concat":
        residual_train_normed, residual_test_normed, _, _ = normalize_hidden_with_stats_torch(residual_train, residual_test)
        scale = float(args.branch_novelty_scale)
        concat_train = [
            torch.cat([old, scale * new], dim=1) for old, new in zip(branch_train, residual_train_normed)
        ]
        concat_test = [
            torch.cat([old, scale * new], dim=1) for old, new in zip(branch_test, residual_test_normed)
        ]
        return concat_train, concat_test, metrics
    raise ValueError(f"Unknown branch novelty mode: {args.branch_novelty_mode}")


def fit_indices(args, layer_idx, seed, n, device):
    if args.moment_batch_size <= 0 or args.moment_batch_size >= n:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + 7919 * (int(layer_idx) + 1))
    return torch.randperm(n, generator=gen, device=device)[: int(args.moment_batch_size)]


def fit_index_list(args, layer_idx, seed, n, device):
    if args.moment_batch_size <= 0 or args.moment_batch_size >= n:
        return [None]
    indices = []
    for ensemble_idx in range(max(1, int(args.moment_ensembles))):
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) + 7919 * (int(layer_idx) + 1) + 104729 * ensemble_idx)
        indices.append(torch.randperm(n, generator=gen, device=device)[: int(args.moment_batch_size)])
    return indices


def line_search_eval_indices(args, layer_idx, seed, n, device):
    if int(args.line_search_eval_size) <= 0 or int(args.line_search_eval_size) >= n:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + 65537 * (int(layer_idx) + 1))
    return torch.randperm(n, generator=gen, device=device)[: int(args.line_search_eval_size)]


def bt_quadratic_eval_indices(args, layer_idx, seed, n, device):
    if int(args.bt_quadratic_eval_size) <= 0 or int(args.bt_quadratic_eval_size) >= n:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + 433494437 * (int(layer_idx) + 1))
    return torch.randperm(n, generator=gen, device=device)[: int(args.bt_quadratic_eval_size)]


def old_span_adaptive_eval_indices(args, layer_idx, seed, n, device):
    if int(args.old_span_adaptive_eval_size) <= 0 or int(args.old_span_adaptive_eval_size) >= n:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + 99991 * (int(layer_idx) + 1))
    return torch.randperm(n, generator=gen, device=device)[: int(args.old_span_adaptive_eval_size)]


def operator_context(args, phi1, phi2, z1, z2, corr):
    context = {
        "mode": args.standardization_jacobian,
        "corr": corr,
    }
    if args.standardization_jacobian == "projected":
        n = float(z1.shape[0])
        context["n1"] = (phi1.T @ z1) / n
        context["n2"] = (phi2.T @ z2) / n
    return context


def apply_moment_operator(b, m1, m2, inv_scale1, inv_scale2, context):
    b1 = b * inv_scale1.unsqueeze(0)
    b2 = b * inv_scale2.unsqueeze(0)
    out = b1.T @ m1 + m2 @ b2
    if context["mode"] == "projected":
        q1 = torch.diagonal(b1.T @ context["n1"])
        q2 = torch.diagonal(b2.T @ context["n2"])
        out = out - q1.unsqueeze(1) * context["corr"] - context["corr"] * q2.unsqueeze(0)
    return out


def apply_layernorm_moment_operator(b, term):
    delta1 = row_layernorm_tangent(term["phi1"] @ b, term["reference1"], term["layernorm_eps"])
    delta2 = row_layernorm_tangent(term["phi2"] @ b, term["reference2"], term["layernorm_eps"])
    dz1 = apply_standardized_tangent(delta1, term["z1"], term["inv_scale1"])
    dz2 = apply_standardized_tangent(delta2, term["z2"], term["inv_scale2"])
    n = float(term["z1"].shape[0])
    return (dz1.T @ term["z2"] + term["z1"].T @ dz2) / n


def apply_layernorm_sample_operator(b, term):
    out = row_layernorm_tangent(term["features"] @ b, term["reference"], term["layernorm_eps"])
    if "right" in term:
        out = out @ term["right"]
    return out


def adjoint_moment_operator(e, m1, m2, inv_scale1, inv_scale2, context):
    grad = (m1 @ e.T) * inv_scale1.unsqueeze(0) + (m2.T @ e) * inv_scale2.unsqueeze(0)
    if context["mode"] == "projected":
        weighted = e * context["corr"]
        row_weights = torch.sum(weighted, dim=1)
        col_weights = torch.sum(weighted, dim=0)
        grad = grad - (context["n1"] * row_weights.unsqueeze(0)) * inv_scale1.unsqueeze(0)
        grad = grad - (context["n2"] * col_weights.unsqueeze(0)) * inv_scale2.unsqueeze(0)
    return grad


def adjoint_layernorm_moment_operator(e, term):
    n = float(term["z1"].shape[0])
    grad_dz1 = (term["z2"] @ e.T) / n
    grad_dz2 = (term["z1"] @ e) / n
    grad_y1 = apply_standardized_tangent(grad_dz1, term["z1"], term["inv_scale1"])
    grad_y2 = apply_standardized_tangent(grad_dz2, term["z2"], term["inv_scale2"])
    grad_u1 = row_layernorm_tangent(grad_y1, term["reference1"], term["layernorm_eps"])
    grad_u2 = row_layernorm_tangent(grad_y2, term["reference2"], term["layernorm_eps"])
    return term["phi1"].T @ grad_u1 + term["phi2"].T @ grad_u2


def adjoint_layernorm_sample_operator(e, term):
    back = e
    if "right" in term:
        back = back @ term["right"].T
    back = row_layernorm_tangent(back, term["reference"], term["layernorm_eps"])
    return (term["features"].T @ back) / float(term["features"].shape[0])


def normal_operator(b, m1, m2, inv_scale1, inv_scale2, context, rho):
    e = apply_moment_operator(b, m1, m2, inv_scale1, inv_scale2, context)
    return adjoint_moment_operator(e, m1, m2, inv_scale1, inv_scale2, context) + float(rho) * b


def apply_term_operator(b, term):
    if term.get("type") == "moment_layernorm":
        return apply_layernorm_moment_operator(b, term)
    if term.get("type") == "layernorm_sample":
        return apply_layernorm_sample_operator(b, term)
    if term.get("type") in {"sample", "sample_tangent"}:
        out = term["features"] @ b
        if term.get("type") == "sample_tangent":
            out = apply_standardized_tangent(out, term["reference_z"], term["reference_inv_scale"])
        if "right" in term:
            out = out @ term["right"]
        return out
    return apply_moment_operator(
        b,
        term["m1"],
        term["m2"],
        term["inv_scale1"],
        term["inv_scale2"],
        term["context"],
    )


def term_probe_energy(term, probe):
    achieved = apply_term_operator(probe, term)
    return torch.mean(achieved * achieved)


def effective_old_span_weight(args, penalty, operator_scale):
    weight = float(penalty) * float(operator_scale)
    if float(args.old_span_update_weight_min) > 0.0 or float(args.old_span_update_weight_max) > 0.0:
        lo = float(args.old_span_update_weight_min) if float(args.old_span_update_weight_min) > 0.0 else 0.0
        hi = float(args.old_span_update_weight_max) if float(args.old_span_update_weight_max) > 0.0 else float("inf")
        weight = min(max(weight, lo), hi)
    return weight


def effective_stable_mode_weight(args, operator_scale):
    weight = float(args.stable_mode_penalty) * float(operator_scale)
    if float(args.stable_mode_weight_min) > 0.0 or float(args.stable_mode_weight_max) > 0.0:
        lo = float(args.stable_mode_weight_min) if float(args.stable_mode_weight_min) > 0.0 else 0.0
        hi = float(args.stable_mode_weight_max) if float(args.stable_mode_weight_max) > 0.0 else float("inf")
        weight = min(max(weight, lo), hi)
    return weight


def configure_old_span_terms(terms, args, penalty):
    weights = []
    for term in terms:
        if not term.get("old_span_term", False):
            continue
        weight = effective_old_span_weight(args, penalty, term.get("old_span_operator_scale", 1.0))
        term["weight"] = float(term["old_span_fit_weight"]) * weight
        weights.append(weight)
    return weights


def old_span_update_rms_from_terms(terms, b):
    vals = []
    for term in terms:
        if term.get("old_span_term", False):
            achieved = apply_term_operator(b, term)
            vals.append(torch.mean(achieved * achieved))
    if not vals:
        return float("nan")
    return float(torch.sqrt(torch.mean(torch.stack(vals))).detach().cpu().item())


def select_old_span_candidate(candidates, before_score, args):
    best_score = min(item["score"] for item in candidates)
    best_gain = float(before_score) - best_score
    for item in candidates:
        item["gain"] = float(before_score) - float(item["score"])
    if args.old_span_adaptive_rule == "fraction":
        if best_gain > 0.0:
            min_gain = float(args.old_span_adaptive_bt_fraction) * best_gain
            eligible = [item for item in candidates if item["gain"] >= min_gain]
        else:
            eligible = candidates
        if not eligible:
            eligible = candidates
        selected = min(eligible, key=lambda item: (item["old_rms"], item["score"]))
        selected["selection_score"] = selected["gain"] / max(best_gain, 1e-12) if best_gain > 0.0 else 0.0
        return selected, best_score, best_gain
    if args.old_span_adaptive_rule == "density":
        positive = [item for item in candidates if item["gain"] > 0.0]
        if not positive:
            selected = min(candidates, key=lambda item: (item["score"], item["old_rms"]))
            selected["selection_score"] = 0.0
            return selected, best_score, best_gain
        for item in positive:
            item["selection_score"] = item["gain"] / max(item["old_rms"] ** 2, 1e-12)
        selected = max(positive, key=lambda item: (item["selection_score"], item["gain"], -item["old_rms"]))
        return selected, best_score, best_gain
    if args.old_span_adaptive_rule == "knee":
        positive = [item for item in candidates if item["gain"] > 0.0]
        if not positive:
            selected = min(candidates, key=lambda item: (item["score"], item["old_rms"]))
            selected["selection_score"] = 0.0
            return selected, best_score, best_gain
        min_cost = min(item["old_rms"] for item in positive)
        max_cost = max(item["old_rms"] for item in positive)
        min_gain = min(item["gain"] for item in positive)
        max_gain = max(item["gain"] for item in positive)
        cost_span = max(max_cost - min_cost, 1e-12)
        gain_span = max(max_gain - min_gain, 1e-12)
        for item in positive:
            cost_norm = (item["old_rms"] - min_cost) / cost_span
            gain_norm = (item["gain"] - min_gain) / gain_span if max_gain > min_gain else 1.0
            item["selection_score"] = gain_norm - cost_norm
        selected = max(positive, key=lambda item: (item["selection_score"], item["gain"], -item["old_rms"]))
        return selected, best_score, best_gain
    raise ValueError(f"Unknown old-span adaptive rule: {args.old_span_adaptive_rule}")


def adjoint_term_operator(e, term):
    if term.get("type") == "moment_layernorm":
        return adjoint_layernorm_moment_operator(e, term)
    if term.get("type") == "layernorm_sample":
        return adjoint_layernorm_sample_operator(e, term)
    if term.get("type") == "sample":
        if "right" in term:
            return (term["features"].T @ e @ term["right"].T) / float(term["features"].shape[0])
        return (term["features"].T @ e) / float(term["features"].shape[0])
    if term.get("type") == "sample_tangent":
        back = e
        if "right" in term:
            back = back @ term["right"].T
        back = apply_standardized_tangent(back, term["reference_z"], term["reference_inv_scale"])
        return (term["features"].T @ back) / float(term["features"].shape[0])
    return adjoint_moment_operator(
        e,
        term["m1"],
        term["m2"],
        term["inv_scale1"],
        term["inv_scale2"],
        term["context"],
    )


def first_term_named(terms, name):
    for term in terms:
        if term["name"] == name:
            return term
    return None


def normal_operator_terms(b, terms, rho):
    out = float(rho) * b
    for term in terms:
        achieved = apply_term_operator(b, term)
        out = out + float(term["weight"]) * adjoint_term_operator(achieved, term)
    return out


def cg_solve_moment_ols_terms(terms, shape, dtype, device, rho, max_iter, tol):
    rhs = torch.zeros(shape, dtype=dtype, device=device)
    for term in terms:
        rhs = rhs + float(term["weight"]) * adjoint_term_operator(term["target"], term)
    x = torch.zeros_like(rhs)
    r = rhs - normal_operator_terms(x, terms, rho)
    p = r.clone()
    rs_old = torch.sum(r * r)
    rhs_norm = torch.sqrt(torch.sum(rhs * rhs))
    iters = 0
    for idx in range(int(max_iter)):
        ap = normal_operator_terms(p, terms, rho)
        denom = torch.sum(p * ap)
        alpha = rs_old / torch.clamp(denom, min=1e-20)
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = torch.sum(r * r)
        iters = idx + 1
        if torch.sqrt(rs_new) <= float(tol) * torch.clamp(rhs_norm, min=1e-20):
            rs_old = rs_new
            break
        beta = rs_new / torch.clamp(rs_old, min=1e-20)
        p = r + beta * p
        rs_old = rs_new
    residual = torch.sqrt(rs_old) / torch.clamp(rhs_norm, min=1e-20)
    return x, {
        "cg_iters": iters,
        "cg_relative_residual": float(residual.detach().cpu().item()),
        "cg_rhs_norm": float(rhs_norm.detach().cpu().item()),
    }


def cg_solve_moment_ols(m1, m2, target, inv_scale1, inv_scale2, context, rho, max_iter, tol):
    rhs = adjoint_moment_operator(target, m1, m2, inv_scale1, inv_scale2, context)
    x = torch.zeros_like(rhs)
    r = rhs - normal_operator(x, m1, m2, inv_scale1, inv_scale2, context, rho)
    p = r.clone()
    rs_old = torch.sum(r * r)
    rhs_norm = torch.sqrt(torch.sum(rhs * rhs))
    iters = 0
    for idx in range(int(max_iter)):
        ap = normal_operator(p, m1, m2, inv_scale1, inv_scale2, context, rho)
        denom = torch.sum(p * ap)
        alpha = rs_old / torch.clamp(denom, min=1e-20)
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = torch.sum(r * r)
        iters = idx + 1
        if torch.sqrt(rs_new) <= float(tol) * torch.clamp(rhs_norm, min=1e-20):
            rs_old = rs_new
            break
        beta = rs_new / torch.clamp(rs_old, min=1e-20)
        p = r + beta * p
        rs_old = rs_new
    residual = torch.sqrt(rs_old) / torch.clamp(rhs_norm, min=1e-20)
    return x, {
        "cg_iters": iters,
        "cg_relative_residual": float(residual.detach().cpu().item()),
        "cg_rhs_norm": float(rhs_norm.detach().cpu().item()),
    }


def finite_difference_corr_delta(view1, view2, delta1, delta2, corr, scale, residual_normalization, layernorm_eps):
    cand1 = view1 + float(scale) * delta1
    cand2 = view2 + float(scale) * delta2
    if residual_normalization == "layernorm":
        cand1 = row_layernorm_torch(cand1, layernorm_eps)
        cand2 = row_layernorm_torch(cand2, layernorm_eps)
    _, _, corr_after, _, _, _ = bt_corr_and_gradient(
        cand1,
        cand2,
        1.0,
    )
    return (corr_after - corr) / float(scale)


def cosine_matrix(a, b):
    af = a.reshape(-1)
    bf = b.reshape(-1)
    return float((torch.dot(af, bf) / torch.clamp(torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf), min=1e-12)).detach().cpu().item())


def relative_r2(pred, target):
    err = torch.sum((pred - target) ** 2)
    base = torch.sum(target * target)
    return float((1.0 - err / torch.clamp(base, min=1e-12)).detach().cpu().item())


def rms_scalar(x):
    return float(torch.sqrt(torch.mean(x * x)).detach().cpu().item())


def layernorm_post_update_ratio(reference, delta, scale, eps):
    updated = row_layernorm_torch(reference + float(scale) * delta, eps)
    return rms_torch(updated - reference) / torch.clamp(rms_torch(reference), min=1e-12)


def layernorm_post_update_cap_scale(reference, delta, cap, eps):
    if float(cap) <= 0.0:
        return 1.0
    full_ratio = layernorm_post_update_ratio(reference, delta, 1.0, eps)
    if full_ratio <= float(cap):
        return 1.0
    lo = 0.0
    hi = 1.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        ratio = layernorm_post_update_ratio(reference, delta, mid, eps)
        if ratio <= float(cap):
            lo = mid
        else:
            hi = mid
    return lo


def stable_mode_update_diagnostics(
    view1_before,
    view2_before,
    view1_after,
    view2_after,
    delta_v1,
    delta_v2,
    args,
):
    basis, metrics = stable_mode_basis(
        view1_before,
        view2_before,
        args.stable_mode_count,
        args.stable_mode_ridge,
        args.stable_mode_max_delta,
        args.stable_mode_kind,
    )
    out = {
        "stable_mode_diag_count": metrics["stable_mode_count"],
        "stable_mode_raw_update_rms": float("nan"),
        "stable_mode_tangent_update_rms": float("nan"),
        "stable_mode_actual_delta_rms": float("nan"),
        "stable_mode_raw_actual_cosine": float("nan"),
        "stable_mode_tangent_actual_cosine": float("nan"),
    }
    if basis is None:
        return out
    z1_before, inv_scale1 = standardization_tangent_context(view1_before)
    z2_before, inv_scale2 = standardization_tangent_context(view2_before)
    raw1 = delta_v1 @ basis
    raw2 = delta_v2 @ basis
    tangent1 = apply_standardized_tangent(delta_v1, z1_before, inv_scale1) @ basis
    tangent2 = apply_standardized_tangent(delta_v2, z2_before, inv_scale2) @ basis
    actual1 = (standardize_torch(view1_after) - standardize_torch(view1_before)) @ basis
    actual2 = (standardize_torch(view2_after) - standardize_torch(view2_before)) @ basis
    raw = torch.cat([raw1, raw2], dim=0)
    tangent = torch.cat([tangent1, tangent2], dim=0)
    actual = torch.cat([actual1, actual2], dim=0)
    out.update(
        {
            "stable_mode_raw_update_rms": rms_scalar(raw),
            "stable_mode_tangent_update_rms": rms_scalar(tangent),
            "stable_mode_actual_delta_rms": rms_scalar(actual),
            "stable_mode_raw_actual_cosine": cosine_matrix(raw, actual),
            "stable_mode_tangent_actual_cosine": cosine_matrix(tangent, actual),
        }
    )
    return out


def make_layernorm_kinetic_terms(args, fit_weight, layer_idx, fit_idx, seed, bt_term, streams, solve_shape, dtype, device):
    raw_terms = []
    for name, features, reference in streams:
        raw_terms.append(
            {
                "type": "layernorm_sample",
                "name": f"layernorm_kinetic_{name}",
                "weight": 0.0,
                "features": features,
                "reference": reference,
                "layernorm_eps": float(args.layernorm_eps),
                "target": torch.zeros_like(reference),
            }
        )
    operator_scale = 1.0
    if args.layernorm_kinetic_normalization == "operator":
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) + 67867979 * (int(layer_idx) + 1) + 86028121 * (int(fit_idx) + 1))
        probe = torch.randn(solve_shape, generator=gen, dtype=dtype, device=device)
        moment_energy = term_probe_energy(bt_term, probe)
        kinetic_energy = torch.mean(torch.stack([term_probe_energy(term, probe) for term in raw_terms]))
        operator_scale = float((moment_energy / torch.clamp(kinetic_energy, min=1e-12)).detach().cpu().item())
    elif args.layernorm_kinetic_normalization != "none":
        raise ValueError(f"Unknown LayerNorm kinetic normalization: {args.layernorm_kinetic_normalization}")
    effective_weight = float(args.layernorm_kinetic_weight) * operator_scale
    per_term_weight = fit_weight * effective_weight / max(1, len(raw_terms))
    for term in raw_terms:
        term["weight"] = per_term_weight
    return raw_terms, {
        "layernorm_kinetic_stream_count": len(raw_terms),
        "layernorm_kinetic_operator_scale": operator_scale,
        "layernorm_kinetic_effective_weight": effective_weight,
    }


def shared_difference_metrics(view1, view2):
    x1 = np.asarray(view1, dtype=np.float64)
    x2 = np.asarray(view2, dtype=np.float64)
    shared = 0.5 * (x1 + x2)
    diff = 0.5 * (x1 - x2)
    shared = shared - shared.mean(axis=0, keepdims=True)
    diff = diff - diff.mean(axis=0, keepdims=True)
    shared_trace = float(np.mean(shared * shared))
    diff_trace = float(np.mean(diff * diff))
    return {
        "shared_trace_per_dim": shared_trace,
        "diff_trace_per_dim": diff_trace,
        "shared_diff_ratio": shared_trace / max(diff_trace, 1e-12),
        "diff_fraction": diff_trace / max(shared_trace + diff_trace, 1e-12),
    }


def numpy_path(train, test, view1, view2):
    htr, hte, hv1, hv2 = standardize_many(
        train.detach().cpu().numpy().astype(np.float32),
        test.detach().cpu().numpy().astype(np.float32),
        view1.detach().cpu().numpy().astype(np.float32),
        view2.detach().cpu().numpy().astype(np.float32),
    )
    return htr, hte, hv1, hv2


def run_variant(args, point, tensors, variant, device):
    if args.residual_normalization != "layernorm" and float(args.layernorm_kinetic_weight) > 0.0:
        raise ValueError("--layernorm-kinetic-weight requires --residual-normalization layernorm")
    if args.residual_normalization == "layernorm":
        unsupported = []
        if float(args.sample_gradient_weight) > 0.0:
            unsupported.append("--sample-gradient-weight")
        if float(args.self_cov_weight) > 0.0:
            unsupported.append("--self-cov-weight")
        if float(args.old_span_update_penalty) > 0.0 or args.old_span_adaptive_path:
            unsupported.append("--old-span-update-penalty/--old-span-adaptive-path")
        if float(args.stable_mode_penalty) > 0.0:
            unsupported.append("--stable-mode-penalty")
        if unsupported:
            joined = ", ".join(unsupported)
            raise ValueError(f"LayerNorm residual mode currently supports the clean BT moment objective only; got {joined}")
    xtr = tensors["xtr"]
    xte = tensors["xte"]
    view1_tr = tensors["view1_tr"]
    view2_tr = tensors["view2_tr"]
    view1_te = tensors["view1_te"]
    view2_te = tensors["view2_te"]
    train_arrays, test_arrays, _, _ = normalize_hidden_with_stats_torch(
        [xtr, view1_tr, view2_tr],
        [xte, view1_te, view2_te],
    )
    base_tr, view1_tr, view2_tr = train_arrays
    base_te, view1_te, view2_te = test_arrays
    if args.residual_normalization == "layernorm":
        base_tr = row_layernorm_torch(base_tr, args.layernorm_eps)
        view1_tr = row_layernorm_torch(view1_tr, args.layernorm_eps)
        view2_tr = row_layernorm_torch(view2_tr, args.layernorm_eps)
        base_te = row_layernorm_torch(base_te, args.layernorm_eps)
        view1_te = row_layernorm_torch(view1_te, args.layernorm_eps)
        view2_te = row_layernorm_torch(view2_te, args.layernorm_eps)

    path_train = []
    path_test = []
    path_view1 = []
    path_view2 = []
    path_view1_test = []
    path_view2_test = []
    rows = []
    prev_train_np = None
    eta_layer = float(args.eta_total) / float(point.depth)

    for layer_idx in range(point.depth):
        before_train_np, before_test_np, before_v1_np, before_v2_np = numpy_path(base_tr, base_te, view1_tr, view2_tr)
        before_bt = bt_hidden_metrics(before_v1_np, before_v2_np, args.bt_lambda)
        before_sd = shared_difference_metrics(before_v1_np, before_v2_np)
        branch = branch_matrix(args, variant, layer_idx, view1_tr, view2_tr, point.seed)
        branch_train, branch_test, _, _ = branch_features(
            args,
            [base_tr, view1_tr, view2_tr, base_te, view1_te, view2_te],
            branch,
        )
        branch_train, branch_test, branch_post_metrics = branch_post_transform_features(
            args,
            [base_tr, view1_tr, view2_tr],
            branch_train,
            branch_test,
        )
        branch_train, branch_test, branch_projection_metrics = build_branch_dictionary(
            args,
            [base_tr, view1_tr, view2_tr],
            [base_te, view1_te, view2_te],
            branch_train,
            branch_test,
        )
        phi_base, phi_v1, phi_v2 = branch_train
        phi_base_te, phi_v1_te, phi_v2_te = branch_test
        idx_fits = fit_index_list(args, layer_idx, point.seed, view1_tr.shape[0], view1_tr.device)
        terms = []
        fit_weight = 1.0 / float(len(idx_fits))
        diagnostic = {}
        sample_target_metrics = []
        old_span_metrics = []
        old_span_effective_weights = []
        stable_mode_metrics = []
        stable_mode_effective_weights = []
        layernorm_kinetic_metrics = []
        for fit_idx, idx_fit in enumerate(idx_fits):
            if idx_fit is None:
                base_fit = base_tr
                view1_fit = view1_tr
                view2_fit = view2_tr
                phi_base_fit = phi_base
                phi_v1_fit = phi_v1
                phi_v2_fit = phi_v2
            else:
                base_fit = base_tr[idx_fit]
                view1_fit = view1_tr[idx_fit]
                view2_fit = view2_tr[idx_fit]
                phi_base_fit = phi_base[idx_fit]
                phi_v1_fit = phi_v1[idx_fit]
                phi_v2_fit = phi_v2[idx_fit]
            z1_i, z2_i, corr_i, grad_i, scale1_i, scale2_i = bt_corr_and_gradient(view1_fit, view2_fit, args.bt_lambda)
            target_grad_i = precondition_bt_gradient(grad_i, args.diag_gradient_multiplier)
            target_i = moment_delta_target(args, corr_i, grad_i, eta_layer)
            phi1_i = center_torch(phi_v1_fit)
            phi2_i = center_torch(phi_v2_fit)
            m1_i = (phi1_i.T @ z2_i) / float(z1_i.shape[0])
            m2_i = (z1_i.T @ phi2_i) / float(z1_i.shape[0])
            inv_scale1_i = 1.0 / torch.clamp(scale1_i, min=1e-12)
            inv_scale2_i = 1.0 / torch.clamp(scale2_i, min=1e-12)
            context_i = operator_context(args, phi1_i, phi2_i, z1_i, z2_i, corr_i)
            if args.residual_normalization == "layernorm":
                bt_term_i = {
                    "type": "moment_layernorm",
                    "name": "bt_cross",
                    "weight": fit_weight * float(args.moment_target_weight),
                    "phi1": phi1_i,
                    "phi2": phi2_i,
                    "reference1": view1_fit,
                    "reference2": view2_fit,
                    "z1": z1_i,
                    "z2": z2_i,
                    "inv_scale1": inv_scale1_i,
                    "inv_scale2": inv_scale2_i,
                    "layernorm_eps": float(args.layernorm_eps),
                    "target": target_i,
                }
            else:
                bt_term_i = {
                    "type": "moment",
                    "name": "bt_cross",
                    "weight": fit_weight * float(args.moment_target_weight),
                    "m1": m1_i,
                    "m2": m2_i,
                    "inv_scale1": inv_scale1_i,
                    "inv_scale2": inv_scale2_i,
                    "context": context_i,
                    "target": target_i,
                }
            terms.append(bt_term_i)
            if float(args.layernorm_kinetic_weight) > 0.0:
                kinetic_streams = [
                    ("view1", phi_v1_fit, view1_fit),
                    ("view2", phi_v2_fit, view2_fit),
                ]
                if args.layernorm_kinetic_include_base:
                    kinetic_streams.insert(0, ("base", phi_base_fit, base_fit))
                kinetic_terms, kinetic_metrics_i = make_layernorm_kinetic_terms(
                    args,
                    fit_weight,
                    layer_idx,
                    fit_idx,
                    point.seed,
                    bt_term_i,
                    kinetic_streams,
                    (phi_v1.shape[1], view1_tr.shape[1]),
                    view1_tr.dtype,
                    view1_tr.device,
                )
                layernorm_kinetic_metrics.append(kinetic_metrics_i)
                terms.extend(kinetic_terms)
            if float(args.sample_gradient_weight) > 0.0:
                sample_target1_i, sample_target2_i = activation_gradient_targets(
                    z1_i,
                    z2_i,
                    target_grad_i,
                    scale1_i,
                    scale2_i,
                    eta_layer,
                    args.sample_gradient_scale,
                    args.sample_mode_balance_power,
                    args.sample_mode_balance_eps,
                    args.sample_mode_balance_min_gain,
                    args.sample_mode_balance_max_gain,
                )
                if args.sample_target_projection == "residual":
                    sample_target1_i, sample_metrics1_i = residualize_sample_target(
                        view1_fit,
                        sample_target1_i,
                        args.sample_target_residual_ridge,
                        args.sample_target_rescale_max,
                    )
                    sample_target2_i, sample_metrics2_i = residualize_sample_target(
                        view2_fit,
                        sample_target2_i,
                        args.sample_target_residual_ridge,
                        args.sample_target_rescale_max,
                    )
                    sample_target_metrics.append(
                        {
                            "sample_target_residual_energy_fraction": 0.5
                            * (
                                sample_metrics1_i["sample_target_residual_energy_fraction"]
                                + sample_metrics2_i["sample_target_residual_energy_fraction"]
                            ),
                            "sample_target_projection_r2": 0.5
                            * (
                                sample_metrics1_i["sample_target_projection_r2"]
                                + sample_metrics2_i["sample_target_projection_r2"]
                            ),
                            "sample_target_residual_rescale_gain": 0.5
                            * (
                                sample_metrics1_i["sample_target_residual_rescale_gain"]
                                + sample_metrics2_i["sample_target_residual_rescale_gain"]
                            ),
                        }
                    )
                elif args.sample_target_projection != "raw":
                    raise ValueError(f"Unknown sample target projection: {args.sample_target_projection}")
                terms.extend(
                    [
                        {
                            "type": "sample",
                            "name": "sample_grad_view1",
                            "weight": fit_weight * float(args.sample_gradient_weight),
                            "features": phi_v1_fit,
                            "target": sample_target1_i,
                        },
                        {
                            "type": "sample",
                            "name": "sample_grad_view2",
                            "weight": fit_weight * float(args.sample_gradient_weight),
                            "features": phi_v2_fit,
                            "target": sample_target2_i,
                        },
                    ]
                )
            if float(args.old_span_update_penalty) > 0.0:
                old_phi_v1_i, old_metrics1_i = project_features_onto_reference(
                    view1_fit,
                    phi_v1_fit,
                    args.old_span_update_ridge,
                )
                old_phi_v2_i, old_metrics2_i = project_features_onto_reference(
                    view2_fit,
                    phi_v2_fit,
                    args.old_span_update_ridge,
                )
                old_span_metrics.append(
                    {
                        "old_span_feature_projection_energy_fraction": 0.5
                        * (
                            old_metrics1_i["old_span_feature_projection_energy_fraction"]
                            + old_metrics2_i["old_span_feature_projection_energy_fraction"]
                        )
                    }
                )
                old_span_term_type = (
                    "sample_tangent" if args.old_span_update_tangent == "view_standardized" else "sample"
                )
                old_z1_i, old_inv_scale1_i = standardization_tangent_context(view1_fit)
                old_z2_i, old_inv_scale2_i = standardization_tangent_context(view2_fit)
                old_term1_i = {
                    "type": old_span_term_type,
                    "name": "old_span_update_view1",
                    "weight": fit_weight * float(args.old_span_update_penalty),
                    "features": old_phi_v1_i,
                    "target": torch.zeros_like(view1_fit),
                    "old_span_term": True,
                    "old_span_fit_weight": fit_weight,
                    "old_span_operator_scale": 1.0,
                    "reference_z": old_z1_i,
                    "reference_inv_scale": old_inv_scale1_i,
                }
                old_term2_i = {
                    "type": old_span_term_type,
                    "name": "old_span_update_view2",
                    "weight": fit_weight * float(args.old_span_update_penalty),
                    "features": old_phi_v2_i,
                    "target": torch.zeros_like(view2_fit),
                    "old_span_term": True,
                    "old_span_fit_weight": fit_weight,
                    "old_span_operator_scale": 1.0,
                    "reference_z": old_z2_i,
                    "reference_inv_scale": old_inv_scale2_i,
                }
                old_span_operator_scale = 1.0
                if args.old_span_update_normalization == "operator":
                    gen = torch.Generator(device=view1_tr.device)
                    gen.manual_seed(
                        int(point.seed)
                        + 15485863 * (int(layer_idx) + 1)
                        + 32452843 * (int(fit_idx) + 1)
                    )
                    probe = torch.randn(
                        (phi_v1_fit.shape[1], z1_i.shape[1]),
                        generator=gen,
                        dtype=z1_i.dtype,
                        device=z1_i.device,
                    )
                    moment_energy = term_probe_energy(bt_term_i, probe)
                    old_energy = 0.5 * (term_probe_energy(old_term1_i, probe) + term_probe_energy(old_term2_i, probe))
                    old_span_operator_scale = float(
                        (moment_energy / torch.clamp(old_energy, min=1e-12)).detach().cpu().item()
                    )
                elif args.old_span_update_normalization != "none":
                    raise ValueError(f"Unknown old-span update normalization: {args.old_span_update_normalization}")
                old_term1_i["old_span_operator_scale"] = old_span_operator_scale
                old_term2_i["old_span_operator_scale"] = old_span_operator_scale
                old_span_weight = effective_old_span_weight(args, args.old_span_update_penalty, old_span_operator_scale)
                old_term1_i["weight"] = fit_weight * old_span_weight
                old_term2_i["weight"] = fit_weight * old_span_weight
                old_span_effective_weights.append(old_span_weight)
                terms.extend(
                    [
                        old_term1_i,
                        old_term2_i,
                    ]
                )
            if float(args.stable_mode_penalty) > 0.0 and int(args.stable_mode_count) > 0:
                stable_basis_i, stable_metrics_i = stable_mode_basis(
                    view1_fit,
                    view2_fit,
                    args.stable_mode_count,
                    args.stable_mode_ridge,
                    args.stable_mode_max_delta,
                    args.stable_mode_kind,
                )
                stable_mode_metrics.append(stable_metrics_i)
                if stable_basis_i is not None:
                    stable_term_type = (
                        "sample_tangent" if args.stable_mode_tangent == "view_standardized" else "sample"
                    )
                    stable_z1_i, stable_inv_scale1_i = standardization_tangent_context(view1_fit)
                    stable_z2_i, stable_inv_scale2_i = standardization_tangent_context(view2_fit)
                    stable_term1_i = {
                        "type": stable_term_type,
                        "name": "stable_mode_view1",
                        "weight": fit_weight * float(args.stable_mode_penalty),
                        "features": phi_v1_fit,
                        "right": stable_basis_i,
                        "target": torch.zeros(
                            (view1_fit.shape[0], stable_basis_i.shape[1]),
                            dtype=view1_fit.dtype,
                            device=view1_fit.device,
                        ),
                        "stable_mode_term": True,
                        "stable_mode_fit_weight": fit_weight,
                        "stable_mode_operator_scale": 1.0,
                        "reference_z": stable_z1_i,
                        "reference_inv_scale": stable_inv_scale1_i,
                    }
                    stable_term2_i = {
                        "type": stable_term_type,
                        "name": "stable_mode_view2",
                        "weight": fit_weight * float(args.stable_mode_penalty),
                        "features": phi_v2_fit,
                        "right": stable_basis_i,
                        "target": torch.zeros(
                            (view2_fit.shape[0], stable_basis_i.shape[1]),
                            dtype=view2_fit.dtype,
                            device=view2_fit.device,
                        ),
                        "stable_mode_term": True,
                        "stable_mode_fit_weight": fit_weight,
                        "stable_mode_operator_scale": 1.0,
                        "reference_z": stable_z2_i,
                        "reference_inv_scale": stable_inv_scale2_i,
                    }
                    stable_operator_scale = 1.0
                    if args.stable_mode_normalization == "operator":
                        gen = torch.Generator(device=view1_tr.device)
                        gen.manual_seed(
                            int(point.seed)
                            + 49979687 * (int(layer_idx) + 1)
                            + 67867967 * (int(fit_idx) + 1)
                        )
                        probe = torch.randn(
                            (phi_v1_fit.shape[1], z1_i.shape[1]),
                            generator=gen,
                            dtype=z1_i.dtype,
                            device=z1_i.device,
                        )
                        moment_energy = term_probe_energy(bt_term_i, probe)
                        stable_energy = 0.5 * (
                            term_probe_energy(stable_term1_i, probe)
                            + term_probe_energy(stable_term2_i, probe)
                        )
                        stable_operator_scale = float(
                            (moment_energy / torch.clamp(stable_energy, min=1e-12)).detach().cpu().item()
                        )
                    elif args.stable_mode_normalization != "none":
                        raise ValueError(f"Unknown stable-mode normalization: {args.stable_mode_normalization}")
                    stable_weight = effective_stable_mode_weight(args, stable_operator_scale)
                    stable_term1_i["stable_mode_operator_scale"] = stable_operator_scale
                    stable_term2_i["stable_mode_operator_scale"] = stable_operator_scale
                    stable_term1_i["weight"] = fit_weight * stable_weight
                    stable_term2_i["weight"] = fit_weight * stable_weight
                    stable_mode_effective_weights.append(stable_weight)
                    terms.extend([stable_term1_i, stable_term2_i])
            self_corr1_i = (z1_i.T @ z1_i) / float(z1_i.shape[0])
            self_corr2_i = (z2_i.T @ z2_i) / float(z2_i.shape[0])
            if float(args.self_cov_weight) > 0.0:
                self_target1_i = -eta_layer * offdiag_matrix(self_corr1_i)
                self_context1_i = operator_context(args, phi1_i, phi1_i, z1_i, z1_i, self_corr1_i)
                terms.append(
                    {
                        "name": "self_cov_view1",
                        "weight": fit_weight * float(args.self_cov_weight),
                        "m1": (phi1_i.T @ z1_i) / float(z1_i.shape[0]),
                        "m2": (z1_i.T @ phi1_i) / float(z1_i.shape[0]),
                        "inv_scale1": inv_scale1_i,
                        "inv_scale2": inv_scale1_i,
                        "context": self_context1_i,
                        "target": self_target1_i,
                    }
                )
                self_target2_i = -eta_layer * offdiag_matrix(self_corr2_i)
                self_context2_i = operator_context(args, phi2_i, phi2_i, z2_i, z2_i, self_corr2_i)
                terms.append(
                    {
                        "name": "self_cov_view2",
                        "weight": fit_weight * float(args.self_cov_weight),
                        "m1": (phi2_i.T @ z2_i) / float(z2_i.shape[0]),
                        "m2": (z2_i.T @ phi2_i) / float(z2_i.shape[0]),
                        "inv_scale1": inv_scale2_i,
                        "inv_scale2": inv_scale2_i,
                        "context": self_context2_i,
                        "target": self_target2_i,
                    }
                )
            if fit_idx == 0:
                diagnostic = {
                    "idx_fit": idx_fit,
                    "view1_fit": view1_fit,
                    "view2_fit": view2_fit,
                    "z1": z1_i,
                    "z2": z2_i,
                    "corr": corr_i,
                    "grad": grad_i,
                    "target_grad": target_grad_i,
                    "target": target_i,
                    "self_corr1": self_corr1_i,
                    "self_corr2": self_corr2_i,
                }
        if sample_target_metrics:
            sample_target_summary = {
                key: float(np.mean([item[key] for item in sample_target_metrics]))
                for key in sample_target_metrics[0]
            }
        else:
            sample_target_summary = {
                "sample_target_residual_energy_fraction": float("nan"),
                "sample_target_projection_r2": float("nan"),
                "sample_target_residual_rescale_gain": float("nan"),
            }
        if old_span_metrics:
            old_span_summary = {
                key: float(np.mean([item[key] for item in old_span_metrics]))
                for key in old_span_metrics[0]
            }
        else:
            old_span_summary = {
                "old_span_feature_projection_energy_fraction": float("nan"),
            }
        if stable_mode_metrics:
            stable_mode_summary = {
                key: float(np.mean([item[key] for item in stable_mode_metrics]))
                for key in stable_mode_metrics[0]
            }
        else:
            stable_mode_summary = {
                "stable_mode_count": 0,
                "stable_mode_mean_delta": float("nan"),
                "stable_mode_max_delta": float("nan"),
            }
        if layernorm_kinetic_metrics:
            layernorm_kinetic_summary = {
                key: float(np.mean([item[key] for item in layernorm_kinetic_metrics]))
                for key in layernorm_kinetic_metrics[0]
            }
        else:
            layernorm_kinetic_summary = {
                "layernorm_kinetic_stream_count": 0,
                "layernorm_kinetic_operator_scale": float("nan"),
                "layernorm_kinetic_effective_weight": float("nan"),
            }
        old_span_effective_weight = (
            float(np.mean(old_span_effective_weights)) if old_span_effective_weights else float("nan")
        )
        stable_mode_effective_weight = (
            float(np.mean(stable_mode_effective_weights)) if stable_mode_effective_weights else float("nan")
        )
        old_span_selected_penalty = float(args.old_span_update_penalty)
        old_span_adaptive_before_bt = float("nan")
        old_span_adaptive_before_nuclear = float("nan")
        old_span_adaptive_best_bt = float("nan")
        old_span_adaptive_best_gain = float("nan")
        old_span_adaptive_selected_bt = float("nan")
        old_span_adaptive_selected_nuclear = float("nan")
        old_span_adaptive_selected_gain = float("nan")
        old_span_adaptive_selected_score = float("nan")
        old_span_adaptive_selected_old_rms = float("nan")
        old_span_adaptive_candidate_count = 0
        if args.old_span_adaptive_path and any(term.get("old_span_term", False) for term in terms):
            idx_old_eval = old_span_adaptive_eval_indices(
                args, layer_idx, point.seed, view1_tr.shape[0], view1_tr.device
            )
            if idx_old_eval is None:
                old_eval_v1 = view1_tr
                old_eval_v2 = view2_tr
            else:
                old_eval_v1 = view1_tr[idx_old_eval]
                old_eval_v2 = view2_tr[idx_old_eval]
            old_span_before_stats = bt_score_stats_torch(old_eval_v1, old_eval_v2, args.bt_lambda)
            old_span_adaptive_before_bt = old_span_before_stats["bt"]
            old_span_adaptive_before_nuclear = old_span_before_stats["nuclear"]
            old_span_before_score = old_span_adaptive_score(old_span_before_stats, args)
            path_candidates = []
            for path_penalty in args.old_span_adaptive_path:
                path_weights = configure_old_span_terms(terms, args, path_penalty)
                b_path, _ = cg_solve_moment_ols_terms(
                    terms,
                    (phi_v1.shape[1], view1_tr.shape[1]),
                    view1_tr.dtype,
                    view1_tr.device,
                    args.ols_ridge,
                    args.cg_iters,
                    args.cg_tol,
                )
                path_scale = 1.0
                delta_base_path = phi_base @ b_path
                update_ratio_path = rms_torch(delta_base_path) / torch.clamp(rms_torch(base_tr), min=1e-12)
                if args.max_update_ratio > 0 and update_ratio_path > float(args.max_update_ratio):
                    path_scale = float((float(args.max_update_ratio) / update_ratio_path).detach().cpu().item())
                    b_path = b_path * path_scale
                    delta_base_path = delta_base_path * path_scale
                delta_v1_path = phi_v1 @ b_path
                delta_v2_path = phi_v2 @ b_path
                cand_train, _, _, _ = normalize_hidden_with_stats_torch(
                    [base_tr + delta_base_path, view1_tr + delta_v1_path, view2_tr + delta_v2_path],
                    [],
                )
                if idx_old_eval is None:
                    cand_eval = cand_train
                else:
                    cand_eval = [arr[idx_old_eval] for arr in cand_train]
                cand_stats = bt_score_stats_torch(cand_eval[1], cand_eval[2], args.bt_lambda)
                cand_score = old_span_adaptive_score(cand_stats, args)
                old_rms = old_span_update_rms_from_terms(terms, b_path)
                path_candidates.append(
                    {
                        "penalty": float(path_penalty),
                        "bt": cand_stats["bt"],
                        "nuclear": cand_stats["nuclear"],
                        "score": cand_score,
                        "old_rms": old_rms,
                        "mean_effective_weight": float(np.mean(path_weights)) if path_weights else float("nan"),
                    }
                )
            old_span_adaptive_candidate_count = len(path_candidates)
            selected, best_score, best_gain = select_old_span_candidate(path_candidates, old_span_before_score, args)
            old_span_adaptive_best_bt = min(item["bt"] for item in path_candidates)
            old_span_adaptive_best_gain = best_gain
            old_span_selected_penalty = selected["penalty"]
            old_span_adaptive_selected_bt = selected["bt"]
            old_span_adaptive_selected_nuclear = selected["nuclear"]
            old_span_adaptive_selected_gain = selected["gain"]
            old_span_adaptive_selected_score = selected["selection_score"]
            old_span_adaptive_selected_old_rms = selected["old_rms"]
            old_span_effective_weights = configure_old_span_terms(terms, args, old_span_selected_penalty)
            old_span_effective_weight = (
                float(np.mean(old_span_effective_weights)) if old_span_effective_weights else float("nan")
            )
        idx_fit = diagnostic["idx_fit"]
        view1_fit = diagnostic["view1_fit"]
        view2_fit = diagnostic["view2_fit"]
        z1 = diagnostic["z1"]
        z2 = diagnostic["z2"]
        corr = diagnostic["corr"]
        grad = diagnostic["grad"]
        target = diagnostic["target"]
        self_corr1 = diagnostic["self_corr1"]
        self_corr2 = diagnostic["self_corr2"]
        b, solve = cg_solve_moment_ols_terms(
            terms,
            (phi_v1.shape[1], z1.shape[1]),
            z1.dtype,
            z1.device,
            args.ols_ridge,
            args.cg_iters,
            args.cg_tol,
        )
        bt_term = first_term_named(terms, "bt_cross")
        self1_term = first_term_named(terms, "self_cov_view1")
        self2_term = first_term_named(terms, "self_cov_view2")
        achieved = apply_term_operator(b, bt_term)
        self_achieved1 = apply_term_operator(b, self1_term) if self1_term is not None else torch.zeros_like(corr)
        self_achieved2 = apply_term_operator(b, self2_term) if self2_term is not None else torch.zeros_like(corr)

        base_before = base_tr
        view1_before = view1_tr
        view2_before = view2_tr
        base_rms_before = rms_torch(base_before)
        view1_rms_before = rms_torch(view1_before)
        delta_base = phi_base @ b
        update_ratio = rms_torch(delta_base) / torch.clamp(base_rms_before, min=1e-12)
        applied_scale = 1.0
        if args.max_update_ratio > 0 and update_ratio > float(args.max_update_ratio):
            applied_scale = float((float(args.max_update_ratio) / update_ratio).detach().cpu().item())
            b = b * applied_scale
            achieved = achieved * applied_scale
            self_achieved1 = self_achieved1 * applied_scale
            self_achieved2 = self_achieved2 * applied_scale
            delta_base = delta_base * applied_scale
        if args.residual_normalization == "layernorm" and float(args.max_postnorm_update_ratio) > 0.0:
            post_scale = layernorm_post_update_cap_scale(
                base_before,
                delta_base,
                args.max_postnorm_update_ratio,
                args.layernorm_eps,
            )
            if post_scale < 1.0:
                b = b * post_scale
                achieved = achieved * post_scale
                self_achieved1 = self_achieved1 * post_scale
                self_achieved2 = self_achieved2 * post_scale
                delta_base = delta_base * post_scale
                applied_scale = applied_scale * post_scale

        delta_v1 = phi_v1 @ b
        delta_v2 = phi_v2 @ b
        delta_base_te = phi_base_te @ b
        delta_v1_te = phi_v1_te @ b
        delta_v2_te = phi_v2_te @ b

        bt_quadratic_scale = 1.0
        bt_quadratic_metrics = {
            "bt_quadratic_full_first_order_delta": float("nan"),
            "bt_quadratic_full_second_order_delta": float("nan"),
            "bt_quadratic_predicted_delta": float("nan"),
            "bt_quadratic_full_scale_unclipped": float("nan"),
        }
        if args.bt_quadratic_scale:
            idx_quad_eval = bt_quadratic_eval_indices(
                args, layer_idx, point.seed, view1_tr.shape[0], view1_tr.device
            )
            if idx_quad_eval is None:
                quad_base = base_tr
                quad_v1 = view1_tr
                quad_v2 = view2_tr
                quad_delta_base = delta_base
                quad_delta_v1 = delta_v1
                quad_delta_v2 = delta_v2
            else:
                quad_base = base_tr[idx_quad_eval]
                quad_v1 = view1_tr[idx_quad_eval]
                quad_v2 = view2_tr[idx_quad_eval]
                quad_delta_base = delta_base[idx_quad_eval]
                quad_delta_v1 = delta_v1[idx_quad_eval]
                quad_delta_v2 = delta_v2[idx_quad_eval]
            bt_quadratic_scale, bt_quadratic_metrics = bt_quadratic_scale_from_realized_corr(
                args,
                quad_base,
                quad_v1,
                quad_v2,
                quad_delta_base,
                quad_delta_v1,
                quad_delta_v2,
            )
            b = b * bt_quadratic_scale
            achieved = achieved * bt_quadratic_scale
            self_achieved1 = self_achieved1 * bt_quadratic_scale
            self_achieved2 = self_achieved2 * bt_quadratic_scale
            delta_base = delta_base * bt_quadratic_scale
            delta_v1 = delta_v1 * bt_quadratic_scale
            delta_v2 = delta_v2 * bt_quadratic_scale
            delta_base_te = delta_base_te * bt_quadratic_scale
            delta_v1_te = delta_v1_te * bt_quadratic_scale
            delta_v2_te = delta_v2_te * bt_quadratic_scale
            applied_scale = applied_scale * bt_quadratic_scale

        line_search_scale = 1.0
        line_search_bt = float("nan")
        line_search_self_cov = float("nan")
        line_search_effective_rank = float("nan")
        line_search_feasible_count = -1
        if len(args.line_search_scales) > 1:
            candidate_scales = list(args.line_search_scales)
            if int(args.line_search_cap_after_layer) > 0 and (layer_idx + 1) > int(args.line_search_cap_after_layer):
                candidate_scales = [
                    scale for scale in candidate_scales if float(scale) <= float(args.line_search_max_scale_after_cap)
                ]
            if args.line_search_include_zero and 0.0 not in candidate_scales:
                candidate_scales.append(0.0)
            if not candidate_scales:
                candidate_scales = [0.0]
            idx_line_eval = line_search_eval_indices(args, layer_idx, point.seed, view1_tr.shape[0], view1_tr.device)
            if idx_line_eval is None:
                eval_base = base_tr
                eval_v1 = view1_tr
                eval_v2 = view2_tr
            else:
                eval_base = base_tr[idx_line_eval]
                eval_v1 = view1_tr[idx_line_eval]
                eval_v2 = view2_tr[idx_line_eval]
            before_bt_score = float(bt_total_per_dim_torch(eval_v1, eval_v2, args.bt_lambda).detach().cpu().item())
            before_self_score = float(paired_self_corr_offdiag_per_dim_torch(eval_v1, eval_v2).detach().cpu().item())
            before_rank_score = float(covariance_effective_rank_torch(eval_base).detach().cpu().item())
            self_limit = before_self_score * (1.0 + float(args.line_search_self_cov_rel_tol)) + float(
                args.line_search_self_cov_abs_tol
            )
            rank_floor = before_rank_score * (1.0 - float(args.line_search_rank_rel_tol)) - float(
                args.line_search_rank_abs_tol
            )
            candidates = []
            for candidate_scale in candidate_scales:
                cand_base = base_tr + float(candidate_scale) * delta_base
                cand_v1 = view1_tr + float(candidate_scale) * delta_v1
                cand_v2 = view2_tr + float(candidate_scale) * delta_v2
                if args.residual_normalization == "layernorm":
                    cand_train = [
                        row_layernorm_torch(cand_base, args.layernorm_eps),
                        row_layernorm_torch(cand_v1, args.layernorm_eps),
                        row_layernorm_torch(cand_v2, args.layernorm_eps),
                    ]
                else:
                    cand_train, _, _, _ = normalize_hidden_with_stats_torch([cand_base, cand_v1, cand_v2], [])
                if idx_line_eval is None:
                    cand_eval = cand_train
                else:
                    cand_eval = [arr[idx_line_eval] for arr in cand_train]
                cand_bt = float(bt_total_per_dim_torch(cand_eval[1], cand_eval[2], args.bt_lambda).detach().cpu().item())
                cand_self = float(paired_self_corr_offdiag_per_dim_torch(cand_eval[1], cand_eval[2]).detach().cpu().item())
                cand_rank = float(covariance_effective_rank_torch(cand_eval[0]).detach().cpu().item())
                score = cand_bt + (
                    args.line_search_self_cov_weight if args.line_search_self_cov_weight >= 0.0 else args.self_cov_weight
                ) * cand_self
                improves_bt = cand_bt <= before_bt_score - float(args.line_search_min_bt_gain)
                if args.line_search_mode == "bt_self_cap":
                    feasible = improves_bt and cand_self <= self_limit
                elif args.line_search_mode == "bt_rank_floor":
                    feasible = improves_bt and cand_rank >= rank_floor
                else:
                    feasible = False
                candidates.append(
                    {
                        "scale": float(candidate_scale),
                        "bt": cand_bt,
                        "self": cand_self,
                        "rank": cand_rank,
                        "score": float(score),
                        "feasible": bool(feasible),
                        "bt_violation": max(0.0, cand_bt - (before_bt_score - float(args.line_search_min_bt_gain))),
                        "self_violation": max(0.0, cand_self - self_limit),
                        "rank_violation": max(0.0, rank_floor - cand_rank),
                    }
                )
            feasible = [item for item in candidates if item["feasible"]]
            line_search_feasible_count = len(feasible)
            if args.line_search_mode == "score":
                best = min(candidates, key=lambda item: item["score"])
            elif args.line_search_mode == "bt_self_cap":
                if feasible:
                    best = min(feasible, key=lambda item: item["bt"])
                elif args.line_search_include_zero:
                    best = min(candidates, key=lambda item: (item["bt_violation"] > 0.0, item["self_violation"], item["bt"]))
                else:
                    best = min(candidates, key=lambda item: (item["self_violation"], item["bt_violation"], item["bt"]))
            elif args.line_search_mode == "bt_rank_floor":
                if feasible:
                    best = min(feasible, key=lambda item: item["bt"])
                elif args.line_search_include_zero:
                    best = min(candidates, key=lambda item: (item["bt_violation"] > 0.0, item["rank_violation"], item["bt"]))
                else:
                    best = min(candidates, key=lambda item: (item["rank_violation"], item["bt_violation"], item["bt"]))
            else:
                raise ValueError(f"Unknown line search mode: {args.line_search_mode}")
            best_scale = best["scale"]
            line_search_scale = best_scale
            line_search_bt = best["bt"]
            line_search_self_cov = best["self"]
            line_search_effective_rank = best["rank"]
            b = b * line_search_scale
            achieved = achieved * line_search_scale
            self_achieved1 = self_achieved1 * line_search_scale
            self_achieved2 = self_achieved2 * line_search_scale
            delta_base = delta_base * line_search_scale
            delta_v1 = delta_v1 * line_search_scale
            delta_v2 = delta_v2 * line_search_scale
            delta_base_te = delta_base_te * line_search_scale
            delta_v1_te = delta_v1_te * line_search_scale
            delta_v2_te = delta_v2_te * line_search_scale
            applied_scale = applied_scale * line_search_scale

        if idx_fit is None:
            delta_v1_fit = delta_v1
            delta_v2_fit = delta_v2
        else:
            delta_v1_fit = delta_v1[idx_fit]
            delta_v2_fit = delta_v2[idx_fit]
        fd_corr_delta = finite_difference_corr_delta(
            view1_fit,
            view2_fit,
            delta_v1_fit,
            delta_v2_fit,
            corr,
            args.fd_scale,
            args.residual_normalization,
            args.layernorm_eps,
        )

        base_next = base_tr + delta_base
        view1_next = view1_tr + delta_v1
        view2_next = view2_tr + delta_v2
        base_next_te = base_te + delta_base_te
        view1_next_te = view1_te + delta_v1_te
        view2_next_te = view2_te + delta_v2_te

        if args.residual_normalization == "layernorm":
            base_tr = row_layernorm_torch(base_next, args.layernorm_eps)
            view1_tr = row_layernorm_torch(view1_next, args.layernorm_eps)
            view2_tr = row_layernorm_torch(view2_next, args.layernorm_eps)
            base_te = row_layernorm_torch(base_next_te, args.layernorm_eps)
            view1_te = row_layernorm_torch(view1_next_te, args.layernorm_eps)
            view2_te = row_layernorm_torch(view2_next_te, args.layernorm_eps)
        else:
            train_arrays, test_arrays, _, _ = normalize_hidden_with_stats_torch(
                [base_next, view1_next, view2_next],
                [base_next_te, view1_next_te, view2_next_te],
            )
            base_tr, view1_tr, view2_tr = train_arrays
            base_te, view1_te, view2_te = test_arrays
        if idx_fit is None:
            view1_fit_after = view1_tr
            view2_fit_after = view2_tr
        else:
            view1_fit_after = view1_tr[idx_fit]
            view2_fit_after = view2_tr[idx_fit]
        _, _, corr_after_torch, _, _, _ = bt_corr_and_gradient(view1_fit_after, view2_fit_after, args.bt_lambda)
        actual_corr_delta = corr_after_torch - corr
        z1_after = standardize_torch(view1_tr)
        z2_after = standardize_torch(view2_tr)
        self_corr1_after = (z1_after.T @ z1_after) / float(z1_after.shape[0])
        self_corr2_after = (z2_after.T @ z2_after) / float(z2_after.shape[0])
        z1_fit_after = standardize_torch(view1_fit_after)
        z2_fit_after = standardize_torch(view2_fit_after)
        self_corr1_fit_after = (z1_fit_after.T @ z1_fit_after) / float(z1_fit_after.shape[0])
        self_corr2_fit_after = (z2_fit_after.T @ z2_fit_after) / float(z2_fit_after.shape[0])
        actual_self_delta1 = self_corr1_fit_after - self_corr1
        actual_self_delta2 = self_corr2_fit_after - self_corr2
        train_np, test_np, v1_np, v2_np = numpy_path(base_tr, base_te, view1_tr, view2_tr)
        _, _, v1_test_np, v2_test_np = numpy_path(base_tr, base_te, view1_te, view2_te)
        path_train.append(train_np)
        path_test.append(test_np)
        path_view1.append(v1_np)
        path_view2.append(v2_np)
        path_view1_test.append(v1_test_np)
        path_view2_test.append(v2_test_np)

        after_bt = bt_hidden_metrics(v1_np, v2_np, args.bt_lambda)
        after_test_bt = bt_hidden_metrics(v1_test_np, v2_test_np, args.bt_lambda)
        after_sd = shared_difference_metrics(v1_np, v2_np)
        actual_base_update = base_tr - base_before
        actual_view1_update = view1_tr - view1_before
        if float(args.stable_mode_penalty) > 0.0 or args.stable_mode_diagnostic:
            stable_update_diag = stable_mode_update_diagnostics(
                view1_before,
                view2_before,
                view1_tr,
                view2_tr,
                delta_v1,
                delta_v2,
                args,
            )
        else:
            stable_update_diag = {
                "stable_mode_diag_count": 0,
                "stable_mode_raw_update_rms": float("nan"),
                "stable_mode_tangent_update_rms": float("nan"),
                "stable_mode_actual_delta_rms": float("nan"),
                "stable_mode_raw_actual_cosine": float("nan"),
                "stable_mode_tangent_actual_cosine": float("nan"),
            }
        row = {
            "variant": variant,
            "seed": point.seed,
            "dataset": point.dataset,
            "depth": point.depth,
            "width": point.width,
            "residual_normalization": args.residual_normalization,
            "layernorm_eps": float(args.layernorm_eps),
            "layernorm_kinetic_weight": float(args.layernorm_kinetic_weight),
            "layernorm_kinetic_normalization": args.layernorm_kinetic_normalization,
            "layernorm_kinetic_include_base": bool(args.layernorm_kinetic_include_base),
            "layernorm_kinetic_stream_count": layernorm_kinetic_summary["layernorm_kinetic_stream_count"],
            "layernorm_kinetic_operator_scale": layernorm_kinetic_summary["layernorm_kinetic_operator_scale"],
            "layernorm_kinetic_effective_weight": layernorm_kinetic_summary["layernorm_kinetic_effective_weight"],
            "branch_dim": int(args.branch_dim),
            "branch_feature_dim": int(phi_v1.shape[1]),
            "branch_random_blend": float(args.branch_random_blend),
            "branch_shared_power": float(args.branch_shared_power),
            "branch_post_transform": branch_post_metrics["branch_post_transform"],
            "branch_post_invariance": branch_post_metrics["branch_post_invariance"],
            "branch_post_mean_gain": branch_post_metrics["branch_post_mean_gain"],
            "branch_post_min_gain": branch_post_metrics["branch_post_min_gain"],
            "branch_post_max_delta": branch_post_metrics["branch_post_max_delta"],
            "branch_post_reach_score_mean": branch_post_metrics["branch_post_reach_score_mean"],
            "branch_post_reach_score_max": branch_post_metrics["branch_post_reach_score_max"],
            "branch_post_reach_mean": branch_post_metrics["branch_post_reach_mean"],
            "branch_post_delta_mean": branch_post_metrics["branch_post_delta_mean"],
            "branch_mode_balance_power": float(args.branch_mode_balance_power),
            "branch_mode_balance_side": args.branch_mode_balance_side,
            "branch_mode_balance_eps": float(args.branch_mode_balance_eps),
            "branch_mode_balance_min_gain": float(args.branch_mode_balance_min_gain),
            "branch_mode_balance_max_gain": float(args.branch_mode_balance_max_gain),
            "branch_novelty_mode": args.branch_novelty_mode,
            "branch_novelty_mix": float(args.branch_novelty_mix),
            "branch_novelty_scale": float(args.branch_novelty_scale),
            "branch_residual_ridge": float(args.branch_residual_ridge),
            "branch_residual_energy_fraction": branch_projection_metrics["branch_residual_energy_fraction"],
            "branch_projection_r2": branch_projection_metrics["branch_projection_r2"],
            "branch_novelty_filter": args.branch_novelty_filter,
            "branch_novelty_filter_count_requested": int(args.branch_novelty_filter_count),
            "branch_novelty_filter_count": branch_projection_metrics["branch_novelty_filter_count"],
            "branch_novelty_filter_ridge": float(args.branch_novelty_filter_ridge),
            "branch_novelty_filter_max_delta_threshold": float(args.branch_novelty_filter_max_delta),
            "branch_novelty_filter_mean_delta": branch_projection_metrics["branch_novelty_filter_mean_delta"],
            "branch_novelty_filter_max_delta": branch_projection_metrics["branch_novelty_filter_max_delta"],
            "branch_novelty_filter_energy_fraction": branch_projection_metrics[
                "branch_novelty_filter_energy_fraction"
            ],
            "branch_novelty_filter_mean_gain": branch_projection_metrics["branch_novelty_filter_mean_gain"],
            "branch_novelty_filter_min_gain": branch_projection_metrics["branch_novelty_filter_min_gain"],
            "branch_novelty_filter_projection_ridge": float(args.branch_novelty_filter_projection_ridge),
            "moment_batch_size": int(args.moment_batch_size),
            "moment_ensembles": int(args.moment_ensembles),
            "layer": layer_idx + 1,
            "eta_total": float(args.eta_total),
            "eta_layer": eta_layer,
            "ols_ridge": float(args.ols_ridge),
            "moment_target_kind": args.moment_target_kind,
            "moment_target_weight": float(args.moment_target_weight),
            "polar_target_weight": float(args.polar_target_weight),
            "sample_gradient_weight": float(args.sample_gradient_weight),
            "sample_gradient_scale": float(args.sample_gradient_scale),
            "sample_target_projection": args.sample_target_projection,
            "sample_target_residual_ridge": float(args.sample_target_residual_ridge),
            "sample_target_rescale_max": float(args.sample_target_rescale_max),
            "sample_target_residual_energy_fraction": sample_target_summary[
                "sample_target_residual_energy_fraction"
            ],
            "sample_target_projection_r2": sample_target_summary["sample_target_projection_r2"],
            "sample_target_residual_rescale_gain": sample_target_summary[
                "sample_target_residual_rescale_gain"
            ],
            "sample_mode_balance_power": float(args.sample_mode_balance_power),
            "sample_mode_balance_eps": float(args.sample_mode_balance_eps),
            "sample_mode_balance_min_gain": float(args.sample_mode_balance_min_gain),
            "sample_mode_balance_max_gain": float(args.sample_mode_balance_max_gain),
            "stable_mode_penalty": float(args.stable_mode_penalty),
            "stable_mode_count_requested": int(args.stable_mode_count),
            "stable_mode_kind": args.stable_mode_kind,
            "stable_mode_tangent": args.stable_mode_tangent,
            "stable_mode_count": stable_mode_summary["stable_mode_count"],
            "stable_mode_ridge": float(args.stable_mode_ridge),
            "stable_mode_max_delta_threshold": float(args.stable_mode_max_delta),
            "stable_mode_mean_delta": stable_mode_summary["stable_mode_mean_delta"],
            "stable_mode_max_delta": stable_mode_summary["stable_mode_max_delta"],
            "stable_mode_normalization": args.stable_mode_normalization,
            "stable_mode_effective_weight": stable_mode_effective_weight,
            "stable_mode_weight_min": float(args.stable_mode_weight_min),
            "stable_mode_weight_max": float(args.stable_mode_weight_max),
            "stable_mode_diag_count": stable_update_diag["stable_mode_diag_count"],
            "stable_mode_raw_update_rms": stable_update_diag["stable_mode_raw_update_rms"],
            "stable_mode_tangent_update_rms": stable_update_diag["stable_mode_tangent_update_rms"],
            "stable_mode_actual_delta_rms": stable_update_diag["stable_mode_actual_delta_rms"],
            "stable_mode_raw_actual_cosine": stable_update_diag["stable_mode_raw_actual_cosine"],
            "stable_mode_tangent_actual_cosine": stable_update_diag["stable_mode_tangent_actual_cosine"],
            "old_span_update_penalty": float(args.old_span_update_penalty),
            "old_span_update_ridge": float(args.old_span_update_ridge),
            "old_span_update_normalization": args.old_span_update_normalization,
            "old_span_update_tangent": args.old_span_update_tangent,
            "old_span_update_weight_min": float(args.old_span_update_weight_min),
            "old_span_update_weight_max": float(args.old_span_update_weight_max),
            "old_span_effective_weight": old_span_effective_weight,
            "old_span_adaptive_path": " ".join(str(value) for value in args.old_span_adaptive_path),
            "old_span_adaptive_rule": args.old_span_adaptive_rule,
            "old_span_adaptive_metric": args.old_span_adaptive_metric,
            "old_span_adaptive_eval_size": int(args.old_span_adaptive_eval_size),
            "old_span_adaptive_bt_fraction": float(args.old_span_adaptive_bt_fraction),
            "old_span_adaptive_nuclear_weight": float(args.old_span_adaptive_nuclear_weight),
            "old_span_selected_penalty": old_span_selected_penalty,
            "old_span_adaptive_before_bt": old_span_adaptive_before_bt,
            "old_span_adaptive_before_nuclear": old_span_adaptive_before_nuclear,
            "old_span_adaptive_best_bt": old_span_adaptive_best_bt,
            "old_span_adaptive_best_gain": old_span_adaptive_best_gain,
            "old_span_adaptive_selected_bt": old_span_adaptive_selected_bt,
            "old_span_adaptive_selected_nuclear": old_span_adaptive_selected_nuclear,
            "old_span_adaptive_selected_gain": old_span_adaptive_selected_gain,
            "old_span_adaptive_selected_score": old_span_adaptive_selected_score,
            "old_span_adaptive_selected_old_rms": old_span_adaptive_selected_old_rms,
            "old_span_adaptive_candidate_count": old_span_adaptive_candidate_count,
            "old_span_feature_projection_energy_fraction": old_span_summary[
                "old_span_feature_projection_energy_fraction"
            ],
            "diag_gradient_multiplier": float(args.diag_gradient_multiplier),
            "self_cov_weight": float(args.self_cov_weight),
            "max_update_ratio": float(args.max_update_ratio),
            "max_postnorm_update_ratio": float(args.max_postnorm_update_ratio),
            "applied_scale": applied_scale,
            "bt_quadratic_scale_enabled": bool(args.bt_quadratic_scale),
            "bt_quadratic_scale": bt_quadratic_scale,
            "bt_quadratic_eval_size": int(args.bt_quadratic_eval_size),
            "bt_quadratic_scale_max": float(args.bt_quadratic_scale_max),
            "bt_quadratic_full_first_order_delta": bt_quadratic_metrics[
                "bt_quadratic_full_first_order_delta"
            ],
            "bt_quadratic_full_second_order_delta": bt_quadratic_metrics[
                "bt_quadratic_full_second_order_delta"
            ],
            "bt_quadratic_predicted_delta": bt_quadratic_metrics["bt_quadratic_predicted_delta"],
            "bt_quadratic_full_scale_unclipped": bt_quadratic_metrics["bt_quadratic_full_scale_unclipped"],
            "line_search_scale": line_search_scale,
            "line_search_mode": args.line_search_mode,
            "line_search_eval_size": int(args.line_search_eval_size),
            "line_search_cap_after_layer": int(args.line_search_cap_after_layer),
            "line_search_max_scale_after_cap": float(args.line_search_max_scale_after_cap),
            "line_search_self_cov_weight": float(
                args.line_search_self_cov_weight if args.line_search_self_cov_weight >= 0.0 else args.self_cov_weight
            ),
            "line_search_bt_total_per_dim": line_search_bt,
            "line_search_self_cov_offdiag_per_dim": line_search_self_cov,
            "line_search_effective_rank": line_search_effective_rank,
            "line_search_feasible_count": line_search_feasible_count,
            "line_search_self_cov_rel_tol": float(args.line_search_self_cov_rel_tol),
            "line_search_self_cov_abs_tol": float(args.line_search_self_cov_abs_tol),
            "line_search_rank_rel_tol": float(args.line_search_rank_rel_tol),
            "line_search_rank_abs_tol": float(args.line_search_rank_abs_tol),
            "line_search_min_bt_gain": float(args.line_search_min_bt_gain),
            "before_bt_total_per_dim": before_bt["bt_total_per_dim"],
            "after_bt_total_per_dim": after_bt["bt_total_per_dim"],
            "delta_bt_total_per_dim": after_bt["bt_total_per_dim"] - before_bt["bt_total_per_dim"],
            "after_test_bt_total_per_dim": after_test_bt["bt_total_per_dim"],
            "before_corr_diag_mean": before_bt["corr_diag_mean"],
            "after_corr_diag_mean": after_bt["corr_diag_mean"],
            "after_test_corr_diag_mean": after_test_bt["corr_diag_mean"],
            "after_corr_nuclear_per_dim": after_bt["corr_nuclear_per_dim"],
            "after_corr_singular_max": after_bt["corr_singular_max"],
            "after_corr_singular_effective_rank": after_bt["corr_singular_effective_rank"],
            "after_corr_trace_to_nuclear": after_bt["corr_trace_to_nuclear"],
            "after_test_corr_nuclear_per_dim": after_test_bt["corr_nuclear_per_dim"],
            "after_test_corr_trace_to_nuclear": after_test_bt["corr_trace_to_nuclear"],
            "before_shared_diff_ratio": before_sd["shared_diff_ratio"],
            "after_shared_diff_ratio": after_sd["shared_diff_ratio"],
            "target_norm": float(torch.linalg.vector_norm(target).detach().cpu().item()),
            "achieved_norm": float(torch.linalg.vector_norm(achieved).detach().cpu().item()),
            "actual_corr_delta_norm": float(torch.linalg.vector_norm(actual_corr_delta).detach().cpu().item()),
            "fd_corr_delta_norm": float(torch.linalg.vector_norm(fd_corr_delta).detach().cpu().item()),
            "target_diag_delta_mean": float(torch.diagonal(target).mean().detach().cpu().item()),
            "achieved_diag_delta_mean": float(torch.diagonal(achieved).mean().detach().cpu().item()),
            "actual_corr_diag_delta_mean": float(torch.diagonal(actual_corr_delta).mean().detach().cpu().item()),
            "fd_corr_diag_delta_mean": float(torch.diagonal(fd_corr_delta).mean().detach().cpu().item()),
            "linearized_target_cosine": cosine_matrix(achieved, target),
            "linearized_target_r2": relative_r2(achieved, target),
            "fd_delta_target_cosine": cosine_matrix(fd_corr_delta, target),
            "fd_delta_achieved_cosine": cosine_matrix(fd_corr_delta, achieved),
            "fd_delta_vs_achieved_r2": relative_r2(fd_corr_delta, achieved),
            "actual_delta_target_cosine": cosine_matrix(actual_corr_delta, target),
            "actual_delta_achieved_cosine": cosine_matrix(actual_corr_delta, achieved),
            "actual_delta_vs_achieved_r2": relative_r2(actual_corr_delta, achieved),
            "before_self_corr_offdiag_per_dim": float(
                (
                    0.5 * (self_corr_offdiag_per_dim_torch(view1_before) + self_corr_offdiag_per_dim_torch(view2_before))
                )
                .detach()
                .cpu()
                .item()
            ),
            "after_self_corr_offdiag_per_dim": float(
                (
                    0.5 * (self_corr_offdiag_per_dim_torch(view1_tr) + self_corr_offdiag_per_dim_torch(view2_tr))
                )
                .detach()
                .cpu()
                .item()
            ),
            "self1_target_cosine": cosine_matrix(self_achieved1, -eta_layer * offdiag_matrix(self_corr1)),
            "self2_target_cosine": cosine_matrix(self_achieved2, -eta_layer * offdiag_matrix(self_corr2)),
            "actual_self1_achieved_cosine": cosine_matrix(actual_self_delta1, self_achieved1),
            "actual_self2_achieved_cosine": cosine_matrix(actual_self_delta2, self_achieved2),
            "first_order_predicted_loss_delta_per_dim": float(torch.sum(grad * achieved).detach().cpu().item() / grad.shape[0]),
            "first_order_actual_loss_delta_per_dim": float(torch.sum(grad * actual_corr_delta).detach().cpu().item() / grad.shape[0]),
            "corr_grad_norm": float(torch.linalg.vector_norm(grad).detach().cpu().item()),
            "residual_update_over_input_rms": float((rms_torch(delta_base) / torch.clamp(base_rms_before, min=1e-12)).detach().cpu().item()),
            "view1_update_over_input_rms": float((rms_torch(delta_v1) / torch.clamp(view1_rms_before, min=1e-12)).detach().cpu().item()),
            "actual_postnorm_update_over_input_rms": float(
                (rms_torch(actual_base_update) / torch.clamp(base_rms_before, min=1e-12)).detach().cpu().item()
            ),
            "actual_view1_postnorm_update_over_input_rms": float(
                (rms_torch(actual_view1_update) / torch.clamp(view1_rms_before, min=1e-12)).detach().cpu().item()
            ),
            "update_input_row_cosine": row_cosine_torch(delta_base, base_before),
            "actual_postnorm_update_input_row_cosine": row_cosine_torch(actual_base_update, base_before),
        }
        row.update(solve)
        row.update({f"after_{key}": value for key, value in covariance_spectrum(train_np).items()})
        row.update({f"after_{key}": value for key, value in agreement_spectrum_metrics(v1_np, v2_np, args, device).items()})
        if prev_train_np is None:
            row.update(
                {
                    "prev_to_cur_cka": float("nan"),
                    "prev_to_cur_forward_r2": float("nan"),
                    "prev_to_cur_reverse_r2": float("nan"),
                    "prev_to_cur_sym_r2": float("nan"),
                    "prev_to_cur_linear_novelty": float("nan"),
                }
            )
        else:
            row.update(transition_metrics(prev_train_np, train_np, args, device))
        prev_train_np = train_np
        rows.append(row)

    return {
        "pathnorm_train": path_train,
        "pathnorm_test": path_test,
        "pathnorm_view1_train": path_view1,
        "pathnorm_view2_train": path_view2,
        "pathnorm_view1_test": path_view1_test,
        "pathnorm_view2_test": path_view2_test,
        "rows": rows,
    }


def readout_rows(args, point, variant, state, ytr, yte):
    layer_rows = []
    for idx, (xtr, xte) in enumerate(zip(state["pathnorm_train"], state["pathnorm_test"])):
        row = {
            "variant": variant,
            "seed": point.seed,
            "dataset": point.dataset,
            "depth": point.depth,
            "layer": idx + 1,
            "setup": "layer_hidden_512",
        }
        row.update(linear_classifier_readout(xtr, xte, ytr, yte, args.probe_reg))
        row.update(covariance_spectrum(xtr))
        layer_rows.append(row)
    all_tr = np.concatenate(state["pathnorm_train"], axis=1)
    all_te = np.concatenate(state["pathnorm_test"], axis=1)
    all_pca = {
        "variant": variant,
        "seed": point.seed,
        "dataset": point.dataset,
        "depth": point.depth,
        "setup": "all_layers_pca512",
    }
    all_pca.update(pca512_readout(all_tr, all_te, ytr, yte, args.probe_reg, point.seed, args.pca_dim))
    last = layer_rows[-1]
    best = max(layer_rows, key=lambda row: row["test_accuracy"])
    summary = {
        "variant": variant,
        "seed": point.seed,
        "dataset": point.dataset,
        "depth": point.depth,
        "last_layer_accuracy": last["test_accuracy"],
        "all_pca_accuracy": all_pca["test_accuracy"],
        "best_layer_accuracy": best["test_accuracy"],
        "best_layer": int(best["layer"]),
        "last_effective_rank": last["effective_rank"],
    }
    return layer_rows, [all_pca], summary


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def safe_nanmean(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def summarize(mech_rows, readout_summaries):
    summaries = []
    for variant, depth, seed in sorted({(row["variant"], row["depth"], row["seed"]) for row in mech_rows}):
        rows = sorted(
            [row for row in mech_rows if row["variant"] == variant and row["depth"] == depth and row["seed"] == seed],
            key=lambda row: row["layer"],
        )
        readout = next((row for row in readout_summaries if row["variant"] == variant and row["depth"] == depth and row["seed"] == seed), {})
        final = rows[-1]
        summaries.append(
            {
                "variant": variant,
                "depth": depth,
                "seed": seed,
                "final_train_bt_total_per_dim": final["after_bt_total_per_dim"],
                "final_test_bt_total_per_dim": final["after_test_bt_total_per_dim"],
                "bt_improving_step_fraction": float(np.mean([row["delta_bt_total_per_dim"] < 0.0 for row in rows])),
                "final_corr_diag_mean": final["after_corr_diag_mean"],
                "final_test_corr_diag_mean": final["after_test_corr_diag_mean"],
                "final_shared_diff_ratio": final["after_shared_diff_ratio"],
                "mean_linearized_target_cosine": float(np.mean([row["linearized_target_cosine"] for row in rows])),
                "mean_linearized_target_r2": float(np.mean([row["linearized_target_r2"] for row in rows])),
                "mean_fd_delta_target_cosine": float(np.mean([row["fd_delta_target_cosine"] for row in rows])),
                "mean_fd_delta_achieved_cosine": float(np.mean([row["fd_delta_achieved_cosine"] for row in rows])),
                "mean_fd_delta_vs_achieved_r2": float(np.mean([row["fd_delta_vs_achieved_r2"] for row in rows])),
                "mean_actual_delta_target_cosine": float(np.mean([row["actual_delta_target_cosine"] for row in rows])),
                "mean_actual_delta_achieved_cosine": float(np.mean([row["actual_delta_achieved_cosine"] for row in rows])),
                "mean_actual_delta_vs_achieved_r2": float(np.mean([row["actual_delta_vs_achieved_r2"] for row in rows])),
                "final_self_corr_offdiag_per_dim": final["after_self_corr_offdiag_per_dim"],
                "mean_self1_target_cosine": float(np.mean([row["self1_target_cosine"] for row in rows])),
                "mean_self2_target_cosine": float(np.mean([row["self2_target_cosine"] for row in rows])),
                "mean_actual_self1_achieved_cosine": float(np.mean([row["actual_self1_achieved_cosine"] for row in rows])),
                "mean_actual_self2_achieved_cosine": float(np.mean([row["actual_self2_achieved_cosine"] for row in rows])),
                "mean_first_order_predicted_loss_delta_per_dim": float(
                    np.mean([row["first_order_predicted_loss_delta_per_dim"] for row in rows])
                ),
                "mean_first_order_actual_loss_delta_per_dim": float(
                    np.mean([row["first_order_actual_loss_delta_per_dim"] for row in rows])
                ),
                "mean_update_over_input_rms": float(np.mean([row["residual_update_over_input_rms"] for row in rows])),
                "mean_actual_postnorm_update_over_input_rms": float(
                    np.mean([row["actual_postnorm_update_over_input_rms"] for row in rows])
                ),
                "mean_actual_postnorm_update_input_row_cosine": float(
                    np.mean([row["actual_postnorm_update_input_row_cosine"] for row in rows])
                ),
                "mean_layernorm_kinetic_operator_scale": safe_nanmean(
                    [row["layernorm_kinetic_operator_scale"] for row in rows]
                ),
                "mean_layernorm_kinetic_effective_weight": safe_nanmean(
                    [row["layernorm_kinetic_effective_weight"] for row in rows]
                ),
                "mean_applied_scale": float(np.mean([row["applied_scale"] for row in rows])),
                "mean_bt_quadratic_scale": safe_nanmean([row["bt_quadratic_scale"] for row in rows]),
                "mean_bt_quadratic_full_first_order_delta": safe_nanmean(
                    [row["bt_quadratic_full_first_order_delta"] for row in rows]
                ),
                "mean_bt_quadratic_full_second_order_delta": safe_nanmean(
                    [row["bt_quadratic_full_second_order_delta"] for row in rows]
                ),
                "mean_bt_quadratic_predicted_delta": safe_nanmean(
                    [row["bt_quadratic_predicted_delta"] for row in rows]
                ),
                "mean_bt_quadratic_full_scale_unclipped": safe_nanmean(
                    [row["bt_quadratic_full_scale_unclipped"] for row in rows]
                ),
                "mean_line_search_scale": float(np.mean([row["line_search_scale"] for row in rows])),
                "mean_sample_target_residual_energy_fraction": safe_nanmean(
                    [row["sample_target_residual_energy_fraction"] for row in rows]
                ),
                "mean_sample_target_projection_r2": safe_nanmean(
                    [row["sample_target_projection_r2"] for row in rows]
                ),
                "mean_sample_target_residual_rescale_gain": safe_nanmean(
                    [row["sample_target_residual_rescale_gain"] for row in rows]
                ),
                "mean_branch_post_mean_gain": safe_nanmean([row["branch_post_mean_gain"] for row in rows]),
                "mean_branch_post_min_gain": safe_nanmean([row["branch_post_min_gain"] for row in rows]),
                "mean_branch_post_max_delta": safe_nanmean([row["branch_post_max_delta"] for row in rows]),
                "mean_branch_post_reach_score_mean": safe_nanmean(
                    [row["branch_post_reach_score_mean"] for row in rows]
                ),
                "mean_branch_post_reach_score_max": safe_nanmean(
                    [row["branch_post_reach_score_max"] for row in rows]
                ),
                "mean_branch_post_reach_mean": safe_nanmean([row["branch_post_reach_mean"] for row in rows]),
                "mean_branch_post_delta_mean": safe_nanmean([row["branch_post_delta_mean"] for row in rows]),
                "mean_old_span_feature_projection_energy_fraction": safe_nanmean(
                    [row["old_span_feature_projection_energy_fraction"] for row in rows]
                ),
                "mean_branch_novelty_filter_count": safe_nanmean(
                    [row["branch_novelty_filter_count"] for row in rows]
                ),
                "mean_branch_novelty_filter_mean_delta": safe_nanmean(
                    [row["branch_novelty_filter_mean_delta"] for row in rows]
                ),
                "mean_branch_novelty_filter_energy_fraction": safe_nanmean(
                    [row["branch_novelty_filter_energy_fraction"] for row in rows]
                ),
                "mean_branch_novelty_filter_mean_gain": safe_nanmean(
                    [row["branch_novelty_filter_mean_gain"] for row in rows]
                ),
                "mean_branch_novelty_filter_min_gain": safe_nanmean(
                    [row["branch_novelty_filter_min_gain"] for row in rows]
                ),
                "mean_old_span_effective_weight": safe_nanmean(
                    [row["old_span_effective_weight"] for row in rows]
                ),
                "mean_old_span_selected_penalty": safe_nanmean(
                    [row["old_span_selected_penalty"] for row in rows]
                ),
                "mean_old_span_adaptive_selected_old_rms": safe_nanmean(
                    [row["old_span_adaptive_selected_old_rms"] for row in rows]
                ),
                "mean_old_span_adaptive_best_gain": safe_nanmean(
                    [row["old_span_adaptive_best_gain"] for row in rows]
                ),
                "mean_old_span_adaptive_selected_gain": safe_nanmean(
                    [row["old_span_adaptive_selected_gain"] for row in rows]
                ),
                "mean_old_span_adaptive_selected_score": safe_nanmean(
                    [row["old_span_adaptive_selected_score"] for row in rows]
                ),
                "mean_stable_mode_count": safe_nanmean(
                    [row["stable_mode_count"] for row in rows]
                ),
                "mean_stable_mode_mean_delta": safe_nanmean(
                    [row["stable_mode_mean_delta"] for row in rows]
                ),
                "mean_stable_mode_effective_weight": safe_nanmean(
                    [row["stable_mode_effective_weight"] for row in rows]
                ),
                "mean_stable_mode_raw_update_rms": safe_nanmean(
                    [row["stable_mode_raw_update_rms"] for row in rows]
                ),
                "mean_stable_mode_tangent_update_rms": safe_nanmean(
                    [row["stable_mode_tangent_update_rms"] for row in rows]
                ),
                "mean_stable_mode_actual_delta_rms": safe_nanmean(
                    [row["stable_mode_actual_delta_rms"] for row in rows]
                ),
                "mean_stable_mode_raw_actual_cosine": safe_nanmean(
                    [row["stable_mode_raw_actual_cosine"] for row in rows]
                ),
                "mean_stable_mode_tangent_actual_cosine": safe_nanmean(
                    [row["stable_mode_tangent_actual_cosine"] for row in rows]
                ),
                "final_effective_rank": final["after_effective_rank"],
                "mean_linear_novelty": float(np.nanmean([row["prev_to_cur_linear_novelty"] for row in rows])),
                "last_layer_accuracy": readout.get("last_layer_accuracy", float("nan")),
                "all_pca_accuracy": readout.get("all_pca_accuracy", float("nan")),
                "best_layer_accuracy": readout.get("best_layer_accuracy", float("nan")),
                "best_layer": readout.get("best_layer", -1),
            }
        )
    return summaries


def fmt(value):
    if isinstance(value, float) and np.isnan(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_report(path, summaries):
    lines = [
        "# Moment-Space OLS Residual CF-BT",
        "",
        "Nonlinear residual layers solve a moment-space least-squares problem for the BT correlation step.",
        "Primary judgment is mechanistic: trajectory shape, train/test BT behavior, rank preservation, update size, and target fit.",
        "",
        "| Variant | Depth | Train BT | Test BT | Improve frac | Corr train/test | Shared/diff | Self-cov off | Target cos | Actual-pred cos | Update/input | Rank | Novelty | Last acc | All PCA | Best layer |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(summaries, key=lambda rec: (rec["depth"], rec["variant"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    str(row["depth"]),
                    fmt(row["final_train_bt_total_per_dim"]),
                    fmt(row["final_test_bt_total_per_dim"]),
                    fmt(row["bt_improving_step_fraction"]),
                    f"{fmt(row['final_corr_diag_mean'])}/{fmt(row['final_test_corr_diag_mean'])}",
                    fmt(row["final_shared_diff_ratio"]),
                    fmt(row["final_self_corr_offdiag_per_dim"]),
                    fmt(row["mean_linearized_target_cosine"]),
                    fmt(row["mean_actual_delta_achieved_cosine"]),
                    fmt(row["mean_update_over_input_rms"]),
                    fmt(row["final_effective_rank"]),
                    fmt(row["mean_linear_novelty"]),
                    fmt(row["last_layer_accuracy"]),
                    fmt(row["all_pca_accuracy"]),
                    f"{fmt(row['best_layer_accuracy'])} @ {row['best_layer']}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Files: `mech_rows.jsonl/csv`, `layer_readouts.jsonl/csv`, `setup_readouts.jsonl`, `summary.jsonl/csv`.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args):
    torch.set_num_threads(args.torch_threads)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
        torch.backends.cudnn.allow_tf32 = not args.no_tf32
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mech_rows = []
    layer_readouts = []
    setup_readouts = []
    readout_summaries = []
    for depth in args.depths:
        for seed in args.seeds:
            point = point_for(args, depth, seed)
            tensors = load_tensors(point, device)
            for variant in args.variants:
                print(f"moment OLS residual variant={variant} depth={depth} seed={seed}", flush=True)
                state = run_variant(args, point, tensors, variant, device)
                mech_rows.extend(state["rows"])
                lr, sr, summary = readout_rows(args, point, variant, state, tensors["ytr_np"], tensors["yte_np"])
                layer_readouts.extend(lr)
                setup_readouts.extend(sr)
                readout_summaries.append(summary)
                write_jsonl(args.out_dir / "mech_rows.partial.jsonl", mech_rows)
                write_jsonl(args.out_dir / "layer_readouts.partial.jsonl", layer_readouts)
                write_jsonl(args.out_dir / "readout_summary.partial.jsonl", readout_summaries)
                del state, lr, sr, summary
                torch.cuda.empty_cache()
                gc.collect()
            del tensors
            torch.cuda.empty_cache()
            gc.collect()

    summaries = summarize(mech_rows, readout_summaries)
    write_jsonl(args.out_dir / "mech_rows.jsonl", mech_rows)
    write_jsonl(args.out_dir / "layer_readouts.jsonl", layer_readouts)
    write_jsonl(args.out_dir / "setup_readouts.jsonl", setup_readouts)
    write_jsonl(args.out_dir / "readout_summary.jsonl", readout_summaries)
    write_jsonl(args.out_dir / "summary.jsonl", summaries)
    write_csv(args.out_dir / "mech_rows.csv", mech_rows)
    write_csv(args.out_dir / "layer_readouts.csv", layer_readouts)
    write_csv(args.out_dir / "summary.csv", summaries)
    write_report(args.out_dir / "report.md", summaries)
    print((args.out_dir / "report.md").read_text(encoding="utf-8"), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Nonlinear residual CF-BT with moment-space OLS BT-correlation updates.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_moment_ols_residual_seed7"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.7)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--variants", nargs="+", default=["random_orth", "cf_shrink"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--dataset", default="cifar100_simclr")
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depths", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--bt-lambda", type=float, default=0.005)
    parser.add_argument("--eta-total", type=float, default=2.0)
    parser.add_argument("--moment-target-kind", choices=["bt", "polar", "bt_plus_polar"], default="bt")
    parser.add_argument("--moment-target-weight", type=float, default=1.0)
    parser.add_argument("--polar-target-weight", type=float, default=1.0)
    parser.add_argument("--sample-gradient-weight", type=float, default=0.0)
    parser.add_argument("--sample-gradient-scale", type=float, default=1.0)
    parser.add_argument("--sample-target-projection", choices=["raw", "residual"], default="raw")
    parser.add_argument("--sample-target-residual-ridge", type=float, default=1e-3)
    parser.add_argument("--sample-target-rescale-max", type=float, default=0.0)
    parser.add_argument("--sample-mode-balance-power", type=float, default=0.0)
    parser.add_argument("--sample-mode-balance-eps", type=float, default=1e-3)
    parser.add_argument("--sample-mode-balance-min-gain", type=float, default=0.0)
    parser.add_argument("--sample-mode-balance-max-gain", type=float, default=0.0)
    parser.add_argument("--stable-mode-penalty", type=float, default=0.0)
    parser.add_argument("--stable-mode-kind", choices=["agreement", "pca"], default="agreement")
    parser.add_argument("--stable-mode-tangent", choices=["raw", "view_standardized"], default="raw")
    parser.add_argument("--stable-mode-diagnostic", action="store_true")
    parser.add_argument("--stable-mode-count", type=int, default=128)
    parser.add_argument("--stable-mode-ridge", type=float, default=1e-3)
    parser.add_argument("--stable-mode-max-delta", type=float, default=0.0)
    parser.add_argument("--stable-mode-normalization", choices=["none", "operator"], default="operator")
    parser.add_argument("--stable-mode-weight-min", type=float, default=0.0)
    parser.add_argument("--stable-mode-weight-max", type=float, default=0.0)
    parser.add_argument("--old-span-update-penalty", type=float, default=0.0)
    parser.add_argument("--old-span-update-ridge", type=float, default=1e-3)
    parser.add_argument("--old-span-update-normalization", choices=["none", "operator"], default="none")
    parser.add_argument("--old-span-update-tangent", choices=["raw", "view_standardized"], default="raw")
    parser.add_argument("--old-span-update-weight-min", type=float, default=0.0)
    parser.add_argument("--old-span-update-weight-max", type=float, default=0.0)
    parser.add_argument("--old-span-adaptive-path", type=float, nargs="+", default=[])
    parser.add_argument("--old-span-adaptive-rule", choices=["fraction", "knee", "density"], default="fraction")
    parser.add_argument("--old-span-adaptive-metric", choices=["bt", "bt_plus_nuclear"], default="bt")
    parser.add_argument("--old-span-adaptive-eval-size", type=int, default=0)
    parser.add_argument("--old-span-adaptive-bt-fraction", type=float, default=0.95)
    parser.add_argument("--old-span-adaptive-nuclear-weight", type=float, default=1.0)
    parser.add_argument("--diag-gradient-multiplier", type=float, default=1.0)
    parser.add_argument("--ols-ridge", type=float, default=1e-2)
    parser.add_argument("--cg-iters", type=int, default=40)
    parser.add_argument("--cg-tol", type=float, default=1e-4)
    parser.add_argument("--standardization-jacobian", choices=["frozen", "projected"], default="projected")
    parser.add_argument("--fd-scale", type=float, default=1e-3)
    parser.add_argument("--line-search-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--line-search-include-zero", action="store_true")
    parser.add_argument("--line-search-mode", choices=["score", "bt_self_cap", "bt_rank_floor"], default="score")
    parser.add_argument("--line-search-eval-size", type=int, default=0)
    parser.add_argument("--line-search-cap-after-layer", type=int, default=0)
    parser.add_argument("--line-search-max-scale-after-cap", type=float, default=1.0)
    parser.add_argument("--line-search-self-cov-rel-tol", type=float, default=0.0)
    parser.add_argument("--line-search-self-cov-abs-tol", type=float, default=0.0)
    parser.add_argument("--line-search-rank-rel-tol", type=float, default=0.0)
    parser.add_argument("--line-search-rank-abs-tol", type=float, default=0.0)
    parser.add_argument("--line-search-min-bt-gain", type=float, default=0.0)
    parser.add_argument("--self-cov-weight", type=float, default=0.0)
    parser.add_argument("--line-search-self-cov-weight", type=float, default=-1.0)
    parser.add_argument("--bt-quadratic-scale", action="store_true")
    parser.add_argument("--bt-quadratic-eval-size", type=int, default=0)
    parser.add_argument("--bt-quadratic-scale-max", type=float, default=1.0)
    parser.add_argument("--moment-batch-size", type=int, default=0)
    parser.add_argument("--moment-ensembles", type=int, default=1)
    parser.add_argument("--residual-normalization", choices=["feature", "layernorm"], default="feature")
    parser.add_argument("--layernorm-eps", type=float, default=1e-5)
    parser.add_argument("--layernorm-kinetic-weight", type=float, default=0.0)
    parser.add_argument("--layernorm-kinetic-normalization", choices=["none", "operator"], default="operator")
    parser.add_argument("--layernorm-kinetic-include-base", action="store_true")
    parser.add_argument("--max-postnorm-update-ratio", type=float, default=0.0)
    parser.add_argument("--branch-dim", type=int, default=128)
    parser.add_argument("--branch-random-blend", type=float, default=0.0)
    parser.add_argument("--branch-shared-power", type=float, default=0.0)
    parser.add_argument("--branch-post-transform", choices=["none", "cf_shrink", "whiten", "grad_reach"], default="none")
    parser.add_argument("--branch-post-invariance", type=float, default=1.0)
    parser.add_argument("--branch-post-dim", type=int, default=0)
    parser.add_argument("--branch-post-cov-ridge", type=float, default=1e-4)
    parser.add_argument("--branch-mode-balance-power", type=float, default=0.0)
    parser.add_argument("--branch-mode-balance-side", choices=["input", "output", "both"], default="input")
    parser.add_argument("--branch-mode-balance-eps", type=float, default=1e-3)
    parser.add_argument("--branch-mode-balance-min-gain", type=float, default=0.0)
    parser.add_argument("--branch-mode-balance-max-gain", type=float, default=0.0)
    parser.add_argument("--branch-novelty-mode", choices=["mix", "concat"], default="mix")
    parser.add_argument("--branch-novelty-mix", type=float, default=0.0)
    parser.add_argument("--branch-novelty-scale", type=float, default=1.0)
    parser.add_argument(
        "--branch-novelty-filter",
        choices=["none", "agreement", "pca", "agreement_shrink"],
        default="none",
    )
    parser.add_argument("--branch-novelty-filter-invariance", type=float, default=1.0)
    parser.add_argument("--branch-novelty-filter-count", type=int, default=128)
    parser.add_argument("--branch-novelty-filter-ridge", type=float, default=1e-3)
    parser.add_argument("--branch-novelty-filter-max-delta", type=float, default=0.0)
    parser.add_argument("--branch-novelty-filter-projection-ridge", type=float, default=1e-5)
    parser.add_argument("--branch-residual-ridge", type=float, default=1e-3)
    parser.add_argument("--activation-alpha", type=float, default=0.5)
    parser.add_argument("--cf-invariance", type=float, default=1.0)
    parser.add_argument("--max-update-ratio", type=float, default=0.35)
    parser.add_argument("--probe-reg", type=float, default=100.0)
    parser.add_argument("--pca-dim", type=int, default=512)
    parser.add_argument("--max-spectrum-samples", type=int, default=50000)
    parser.add_argument("--max-transition-samples", type=int, default=12000)
    parser.add_argument("--ridge-reg", type=float, default=1e-3)
    parser.add_argument("--spectrum-eps", type=float, default=1e-6)
    parser.add_argument("--cut-thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--soft-lambdas", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.25, 1.0])
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
