"""Active transformer utilities.

The previous multi-variant attention parametrization driver was archived to
`archive/legacy/transformer_cifar_compare_multi_attention.py`.

The active path only supports `spectral-self`.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

import cifar_shared
import closed_form_attention as cfatt


ACTIVE_ATTENTION_KIND = "spectral-self"


def patchify_numpy(images, patch_size):
    n, c, h, w = images.shape
    assert h % patch_size == 0 and w % patch_size == 0
    gh = h // patch_size
    gw = w // patch_size
    patches = images.reshape(n, c, gh, patch_size, gw, patch_size)
    patches = patches.transpose(0, 2, 4, 1, 3, 5)
    return patches.reshape(n, gh * gw, c * patch_size * patch_size)


def patchify_torch(images, patch_size):
    n, c, h, w = images.shape
    gh = h // patch_size
    gw = w // patch_size
    patches = images.reshape(n, c, gh, patch_size, gw, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    return patches.reshape(n, gh * gw, c * patch_size * patch_size)


def token_layer_norm(tokens, eps=1e-5):
    mean = tokens.mean(axis=-1, keepdims=True)
    var = np.mean((tokens - mean) ** 2, axis=-1, keepdims=True)
    return (tokens - mean) / np.sqrt(var + eps)


def make_2d_sincos_pos_embed(embed_dim, grid_size):
    assert embed_dim % 4 == 0
    grid_h = np.arange(grid_size, dtype=np.float64)
    grid_w = np.arange(grid_size, dtype=np.float64)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, -1)

    half = embed_dim // 2
    emb_h = _sincos_1d(half, grid[0])
    emb_w = _sincos_1d(half, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _sincos_1d(embed_dim, positions):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    out = np.einsum("m,d->md", positions, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def random_affine_torch(images):
    n, _, _, _ = images.shape
    device = images.device
    dtype = images.dtype
    angles = (torch.rand(n, device=device, dtype=dtype) * 2 - 1) * math.radians(cifar_shared.RANDOM_AFFINE_MAX_DEG)
    scales = torch.empty(n, device=device, dtype=dtype).uniform_(
        cifar_shared.RANDOM_AFFINE_MIN_SCALE,
        cifar_shared.RANDOM_AFFINE_MAX_SCALE,
    )
    tx = torch.empty(n, device=device, dtype=dtype).uniform_(
        -cifar_shared.RANDOM_AFFINE_MAX_SHIFT,
        cifar_shared.RANDOM_AFFINE_MAX_SHIFT,
    )
    ty = torch.empty(n, device=device, dtype=dtype).uniform_(
        -cifar_shared.RANDOM_AFFINE_MAX_SHIFT,
        cifar_shared.RANDOM_AFFINE_MAX_SHIFT,
    )
    theta = torch.zeros((n, 2, 3), device=device, dtype=dtype)
    cos = torch.cos(angles) * scales
    sin = torch.sin(angles) * scales
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def random_crop_torch(images):
    n, _, h, w = images.shape
    device = images.device
    pad = 4
    padded = F.pad(images, (pad, pad, pad, pad), mode="reflect")
    max_offset = pad * 2
    xs = torch.randint(0, max_offset + 1, (n,), device=device)
    ys = torch.randint(0, max_offset + 1, (n,), device=device)
    out = torch.empty_like(images)
    for idx in range(n):
        out[idx] = padded[idx, :, ys[idx] : ys[idx] + h, xs[idx] : xs[idx] + w]
    return out


def augment_torch(images, suite_name):
    if suite_name == "random-affine":
        return random_affine_torch(images)
    if suite_name == "random-crop":
        return random_crop_torch(images)
    raise ValueError(f"Unsupported suite for transformer compare: {suite_name}")


def fit_attention_block(config, view1_tr, view2_tr):
    if config.attention_kind != ACTIVE_ATTENTION_KIND:
        raise ValueError(
            f"Unsupported attention kind: {config.attention_kind}. "
            "Legacy attention parametrizations were archived to "
            "`archive/legacy/transformer_cifar_compare_multi_attention.py`."
        )
    return cfatt.fit_spectral_self_attention_from_token_pairs(
        view1_tr,
        view2_tr,
        lambda_reg=config.lambda_reg,
        total_rank=config.analytic_attention_rank,
        num_heads=config.resolved_analytic_heads,
        target_mode=config.attention_target,
    )


def apply_attention_block(tokens, config, att_model):
    if config.attention_kind != ACTIVE_ATTENTION_KIND:
        raise ValueError(
            f"Unsupported attention kind: {config.attention_kind}. "
            "Legacy attention parametrizations were archived to "
            "`archive/legacy/transformer_cifar_compare_multi_attention.py`."
        )
    return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))

