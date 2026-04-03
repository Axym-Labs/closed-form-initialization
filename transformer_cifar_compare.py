import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import cifar_shared
import closed_form_attention as cfatt
import closed_form_barlow_twins as cfbt
from project_paths import default_json_path, resolve_json_path


SEED = 7
DATASET = "cifar100"
SUITE = "random-affine"
PATCH_SIZE = 8
DEPTH = 3
N_TRAIN = 10000
N_TEST = 2000

HEAD_REG = 100.0
LAMBDA_REG = 1.0
NUM_LANDMARKS = 8

NUM_HEADS = 4
MLP_RATIO = 2.0
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-2

ATTENTION_KINDS = [
    "landmark",
    "spectral-self",
    "spectral-self-centered",
    "spectral-self-token-stats",
    "spectral-self-token-stats-gain",
    "spectral-self-token-centered",
    "spectral-self-interleaved",
    "spectral-self-whitened",
    "score-operator-self",
    "score-operator-self-gain",
    "score-operator-self-bagged-gain",
    "score-kernel-dictionary",
    "score-metric-self-bagged-gain",
    "score-operator-projector-basis-gain",
    "score-self-block-gain",
    "score-self-block-bagged-gain",
    "score-self-power-aligned-bagged-gain",
    "score-self-power",
    "score-self-power-gain",
    "score-self-power-headgain",
    "score-self-power-deflated-gain",
    "score-self-cosine-gain",
    "score-self-power-multistart-gain",
    "score-self-power-holdout-gain",
    "score-self-power-bagged-gain",
    "score-self-power-bagged-shrink-gain",
    "score-self-power-bagged-consensus-gain",
    "score-self-power-raw",
    "mixed-self-objective",
    "mixed-self-objective-gain",
    "token-self-maxent",
    "mixed-token-random",
    "head-pool-gain",
    "random-self-ridge",
    "random-self-untrained",
    "cca-self",
    "cca-self-centered",
    "spectral-landmark",
    "spectral-landmark-bt",
    "spectral-bt-context",
    "spectral-bt-context-centered",
    "spectral-bt-context-weighted",
    "local-spectral",
    "hybrid-spectral",
    "hybrid-spectral-bt",
]
ATTENTION_TARGET = "mean"
LOCAL_SIGMA = 1.5


@dataclass(frozen=True)
class TransformerConfig:
    dataset: str = DATASET
    suite: str = SUITE
    patch_size: int = PATCH_SIZE
    depth: int = DEPTH
    n_train: int = N_TRAIN
    n_test: int = N_TEST
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    lr: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    head_reg: float = HEAD_REG
    lambda_reg: float = LAMBDA_REG
    num_landmarks: int = NUM_LANDMARKS
    num_heads: int = NUM_HEADS
    analytic_num_heads: int = 0
    mlp_ratio: float = MLP_RATIO
    attention_kind: str = "landmark"
    attention_target: str = ATTENTION_TARGET
    attention_rank: int = 0
    local_sigma: float = LOCAL_SIGMA
    attention_power_iters: int = 8
    attention_num_bags: int = 4
    attention_bag_fraction: float = 0.7
    attention_seed: int = -1
    seed: int = SEED
    image_size: int = 32
    in_chans: int = 3

    @property
    def model_dim(self):
        return self.in_chans * self.patch_size * self.patch_size

    @property
    def num_patches(self):
        side = self.image_size // self.patch_size
        return side * side

    @property
    def analytic_attention_rank(self):
        if self.attention_rank > 0:
            return self.attention_rank
        return self.num_landmarks * self.resolved_analytic_heads

    @property
    def resolved_analytic_heads(self):
        return self.analytic_num_heads if self.analytic_num_heads > 0 else self.num_heads

    @property
    def resolved_attention_seed(self):
        return self.attention_seed if self.attention_seed >= 0 else self.seed + 1009


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def ridge_regression(X, Y, reg):
    gram = X.T @ X
    rhs = X.T @ Y
    return np.linalg.solve(gram + reg * np.eye(gram.shape[0], dtype=np.float64), rhs)


def one_hot(y, num_classes):
    eye = np.eye(num_classes, dtype=np.float64)
    return eye[y]


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


def evaluate_logits(logits, y):
    return float((np.argmax(logits, axis=1) == y).mean())


def load_cifar_subset(config):
    dataset = cifar_shared.load_cifar_numpy(
        config.dataset,
        n_train=config.n_train,
        n_test=config.n_test,
        seed=config.seed,
        width=3072,
    )
    xtr_img = dataset["xtr_img"].astype(np.float32)
    xte_img = dataset["xte_img"].astype(np.float32)
    ytr = dataset["ytr"]
    yte = dataset["yte"]
    mean_img = xtr_img.mean(axis=0, keepdims=True)
    return {
        "xtr_img": xtr_img,
        "xte_img": xte_img,
        "ytr": ytr,
        "yte": yte,
        "mean_img": mean_img,
    }


def sample_ssl_views(images, suite_name, mean_img, seed):
    rng = np.random.default_rng(seed)
    view1 = cifar_shared.apply_augmentation(images, suite_name, rng).astype(np.float32)
    view2 = cifar_shared.apply_augmentation(images, suite_name, rng).astype(np.float32)
    return view1 - mean_img, view2 - mean_img


def prepare_token_views(config, data):
    xtr_img = data["xtr_img"]
    xte_img = data["xte_img"]
    mean_img = data["mean_img"]

    base_tr = patchify_numpy(xtr_img - mean_img, config.patch_size)
    base_te = patchify_numpy(xte_img - mean_img, config.patch_size)
    view1_tr, view2_tr = sample_ssl_views(xtr_img, config.suite, mean_img, seed=config.seed + 101)
    view1_te, view2_te = sample_ssl_views(xte_img, config.suite, mean_img, seed=config.seed + 202)
    view1_tr = patchify_numpy(view1_tr, config.patch_size)
    view2_tr = patchify_numpy(view2_tr, config.patch_size)
    view1_te = patchify_numpy(view1_te, config.patch_size)
    view2_te = patchify_numpy(view2_te, config.patch_size)

    grid = int(round(math.sqrt(config.num_patches)))
    pos = make_2d_sincos_pos_embed(config.model_dim, grid).astype(np.float32)

    def add_pos(tokens):
        return tokens + pos[None, :, :]

    return {
        "base_tr": add_pos(base_tr),
        "base_te": add_pos(base_te),
        "view1_tr": add_pos(view1_tr),
        "view2_tr": add_pos(view2_tr),
        "view1_te": add_pos(view1_te),
        "view2_te": add_pos(view2_te),
        "pos_embed": pos.astype(np.float32),
    }


def fit_attention_block(config, view1_tr, view2_tr):
    if config.attention_kind == "landmark":
        if config.attention_target not in {"mean", "residual"}:
            raise ValueError("landmark attention only supports attention_target in {'mean', 'residual'}")
        return cfatt.fit_landmark_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            num_landmarks=config.num_landmarks,
            target_mode=config.attention_target,
        )
    if config.attention_kind == "spectral-self":
        return cfatt.fit_spectral_self_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
        )
    if config.attention_kind == "spectral-self-centered":
        return cfatt.fit_spectral_self_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            center_values=True,
        )
    if config.attention_kind == "spectral-self-token-stats":
        return cfatt.fit_token_stats_spectral_self_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
        )
    if config.attention_kind == "spectral-self-token-stats-gain":
        return cfatt.fit_token_stats_scaled_self_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
        )
    if config.attention_kind == "spectral-self-token-centered":
        return cfatt.fit_token_centered_spectral_self_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
        )
    if config.attention_kind == "spectral-self-interleaved":
        return cfatt.fit_spectral_self_attention_interleaved_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
        )
    if config.attention_kind == "spectral-self-whitened":
        return cfatt.fit_spectral_self_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            whiten_values=True,
        )
    if config.attention_kind == "score-self-power":
        return cfatt.fit_score_power_self_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
        )
    if config.attention_kind == "score-operator-self":
        return cfatt.fit_score_operator_self_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
        )
    if config.attention_kind == "score-operator-self-gain":
        return cfatt.fit_score_operator_self_attention_scaled_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
        )
    if config.attention_kind == "score-operator-self-bagged-gain":
        return cfatt.fit_score_operator_self_attention_bagged_scaled_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
            num_bags=config.attention_num_bags,
            bag_fraction=config.attention_bag_fraction,
        )
    if config.attention_kind == "score-kernel-dictionary":
        return cfatt.fit_score_kernel_dictionary_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
        )
    if config.attention_kind == "score-metric-self-bagged-gain":
        return cfatt.fit_score_metric_bagged_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
        )
    if config.attention_kind == "score-operator-projector-basis-gain":
        return cfatt.fit_score_operator_on_projector_basis_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
        )
    if config.attention_kind == "score-self-block-gain":
        return cfatt.fit_score_blockpower_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
        )
    if config.attention_kind == "score-self-block-bagged-gain":
        return cfatt.fit_score_blockpower_bagged_gain_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
            num_bags=config.attention_num_bags,
            bag_fraction=config.attention_bag_fraction,
        )
    if config.attention_kind == "score-self-power-aligned-bagged-gain":
        return cfatt.fit_score_power_aligned_bagged_gain_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
            num_bags=config.attention_num_bags,
            bag_fraction=config.attention_bag_fraction,
        )
    if config.attention_kind == "score-self-power-gain":
        return cfatt.fit_score_power_scaled_self_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
        )
    if config.attention_kind == "score-self-power-headgain":
        return cfatt.fit_score_power_per_head_scaled_self_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
        )
    if config.attention_kind == "score-self-power-deflated-gain":
        return cfatt.fit_score_power_deflated_scaled_self_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
        )
    if config.attention_kind == "score-self-cosine-gain":
        return cfatt.fit_score_cosine_scaled_self_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
        )
    if config.attention_kind == "score-self-power-multistart-gain":
        return cfatt.fit_score_power_multistart_gain_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
        )
    if config.attention_kind == "score-self-power-holdout-gain":
        return cfatt.fit_score_power_holdout_gain_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
        )
    if config.attention_kind == "score-self-power-bagged-gain":
        return cfatt.fit_score_power_bagged_gain_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
            num_bags=config.attention_num_bags,
            bag_fraction=config.attention_bag_fraction,
        )
    if config.attention_kind == "score-self-power-bagged-shrink-gain":
        return cfatt.fit_score_power_bagged_shrink_gain_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
            num_bags=config.attention_num_bags,
            bag_fraction=config.attention_bag_fraction,
        )
    if config.attention_kind == "score-self-power-bagged-consensus-gain":
        return cfatt.fit_score_power_bagged_consensus_gain_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
            num_bags=config.attention_num_bags,
            bag_fraction=config.attention_bag_fraction,
        )
    if config.attention_kind == "score-self-power-raw":
        return cfatt.fit_score_power_raw_self_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
        )
    if config.attention_kind == "token-self-maxent":
        return cfatt.fit_token_maxent_self_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            seed=config.resolved_attention_seed,
        )
    if config.attention_kind == "mixed-self-objective":
        return cfatt.fit_mixed_self_objective_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
        )
    if config.attention_kind == "mixed-self-objective-gain":
        return cfatt.fit_mixed_self_objective_scaled_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
        )
    if config.attention_kind == "mixed-token-random":
        return cfatt.fit_mixed_token_random_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            seed=config.resolved_attention_seed,
        )
    if config.attention_kind == "head-pool-gain":
        return cfatt.fit_head_pool_gain_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            num_power_iters=config.attention_power_iters,
            seed=config.resolved_attention_seed,
        )
    if config.attention_kind == "random-self-ridge":
        return cfatt.fit_random_self_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            seed=config.resolved_attention_seed,
        )
    if config.attention_kind == "random-self-untrained":
        return cfatt.fit_random_untrained_self_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            seed=config.resolved_attention_seed,
        )
    if config.attention_kind == "cca-self":
        return cfatt.fit_cca_self_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
        )
    if config.attention_kind == "cca-self-centered":
        return cfatt.fit_cca_self_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            center_values=True,
        )
    if config.attention_kind in {"spectral-landmark", "spectral-landmark-bt"}:
        target_mode = "bt-residual" if config.attention_kind == "spectral-landmark-bt" else config.attention_target
        return cfatt.fit_spectral_landmark_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            num_landmarks=config.num_landmarks,
            target_mode=target_mode,
        )
    if config.attention_kind in {
        "spectral-bt-context",
        "spectral-bt-context-centered",
        "spectral-bt-context-weighted",
    }:
        return cfatt.fit_spectral_bt_context_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            center_values=config.attention_kind == "spectral-bt-context-centered",
            head_weight_mode="spectral" if config.attention_kind == "spectral-bt-context-weighted" else "uniform",
        )
    if config.attention_kind == "local-spectral":
        return cfatt.fit_local_spectral_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=config.attention_target,
            local_sigma=config.local_sigma,
        )
    if config.attention_kind in {"hybrid-spectral", "hybrid-spectral-bt"}:
        target_mode = "bt-residual" if config.attention_kind == "hybrid-spectral-bt" else config.attention_target
        return cfatt.fit_hybrid_spectral_attention_from_token_pairs(
            view1_tr,
            view2_tr,
            lambda_reg=config.lambda_reg,
            total_rank=config.analytic_attention_rank,
            num_heads=config.resolved_analytic_heads,
            target_mode=target_mode,
            local_sigma=config.local_sigma,
        )
    raise ValueError(f"Unsupported attention kind: {config.attention_kind}")


def apply_attention_block(tokens, config, att_model):
    if config.attention_kind == "landmark":
        return token_layer_norm(cfatt.apply_token_attention(tokens, att_model))
    if config.attention_kind == "spectral-self":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "spectral-self-centered":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "spectral-self-token-stats":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "spectral-self-token-stats-gain":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "spectral-self-token-centered":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "spectral-self-interleaved":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "spectral-self-whitened":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "score-self-power":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "score-operator-self":
        return token_layer_norm(cfatt.apply_score_operator_attention(tokens, att_model))
    if config.attention_kind == "score-operator-self-gain":
        return token_layer_norm(cfatt.apply_score_operator_attention(tokens, att_model))
    if config.attention_kind == "score-operator-self-bagged-gain":
        return token_layer_norm(cfatt.apply_score_operator_attention(tokens, att_model))
    if config.attention_kind == "score-kernel-dictionary":
        return token_layer_norm(cfatt.apply_score_kernel_dictionary_attention(tokens, att_model))
    if config.attention_kind == "score-metric-self-bagged-gain":
        return token_layer_norm(cfatt.apply_score_operator_attention(tokens, att_model))
    if config.attention_kind == "score-operator-projector-basis-gain":
        return token_layer_norm(cfatt.apply_score_operator_attention(tokens, att_model))
    if config.attention_kind == "score-self-block-gain":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "score-self-block-bagged-gain":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "score-self-power-aligned-bagged-gain":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "score-self-power-gain":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "score-self-power-headgain":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "score-self-power-deflated-gain":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "score-self-cosine-gain":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "score-self-power-multistart-gain":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "score-self-power-holdout-gain":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "score-self-power-bagged-gain":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "score-self-power-bagged-shrink-gain":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "score-self-power-bagged-consensus-gain":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "score-self-power-raw":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "mixed-self-objective":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "mixed-self-objective-gain":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "token-self-maxent":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "mixed-token-random":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind == "head-pool-gain":
        return token_layer_norm(cfatt.apply_spectral_self_attention(tokens, att_model))
    if config.attention_kind in {"random-self-ridge", "random-self-untrained"}:
        return token_layer_norm(cfatt.apply_random_self_attention(tokens, att_model))
    if config.attention_kind in {"cca-self", "cca-self-centered"}:
        return token_layer_norm(cfatt.apply_cca_self_attention(tokens, att_model))
    if config.attention_kind in {"spectral-landmark", "spectral-landmark-bt"}:
        return token_layer_norm(cfatt.apply_spectral_landmark_attention(tokens, att_model))
    if config.attention_kind in {
        "spectral-bt-context",
        "spectral-bt-context-centered",
        "spectral-bt-context-weighted",
    }:
        return token_layer_norm(cfatt.apply_spectral_bt_context_attention(tokens, att_model))
    if config.attention_kind == "local-spectral":
        return token_layer_norm(cfatt.apply_local_spectral_attention(tokens, att_model))
    if config.attention_kind in {"hybrid-spectral", "hybrid-spectral-bt"}:
        return token_layer_norm(cfatt.apply_hybrid_spectral_attention(tokens, att_model))
    raise ValueError(f"Unsupported attention kind: {config.attention_kind}")


def run_closed_form_transformer(config):
    data = load_cifar_subset(config)
    views = prepare_token_views(config, data)
    ytr = data["ytr"]
    yte = data["yte"]
    ytr_onehot = one_hot(ytr, int(np.max(ytr) + 1))
    yhat_tr = np.zeros_like(ytr_onehot)
    yhat_te = np.zeros((len(yte), ytr_onehot.shape[1]), dtype=np.float64)

    base_tr = views["base_tr"].astype(np.float64)
    base_te = views["base_te"].astype(np.float64)
    view1_tr = views["view1_tr"].astype(np.float64)
    view2_tr = views["view2_tr"].astype(np.float64)
    view1_te = views["view1_te"].astype(np.float64)
    view2_te = views["view2_te"].astype(np.float64)

    layer_metrics = []
    output_param_count = 0
    hidden_param_count = 0

    start = time.perf_counter()
    for layer_idx in range(config.depth):
        pooled_tr = base_tr.mean(axis=1)
        pooled_te = base_te.mean(axis=1)
        out_map = ridge_regression(pooled_tr, ytr_onehot - yhat_tr, reg=config.head_reg)
        output_param_count += int(out_map.size)
        yhat_tr = yhat_tr + pooled_tr @ out_map
        yhat_te = yhat_te + pooled_te @ out_map

        att_model = fit_attention_block(config, view1_tr, view2_tr)
        hidden_param_count += int(att_model["parameter_count"])
        base_tr = apply_attention_block(base_tr, config, att_model)
        base_te = apply_attention_block(base_te, config, att_model)
        view1_tr = apply_attention_block(view1_tr, config, att_model)
        view2_tr = apply_attention_block(view2_tr, config, att_model)
        view1_te = apply_attention_block(view1_te, config, att_model)
        view2_te = apply_attention_block(view2_te, config, att_model)
        shared_eigs = att_model.get("shared_eigenvalues", [])
        top_shared_eigenvalue = float(shared_eigs[0]) if len(shared_eigs) else float("nan")

        flat1 = view1_tr.reshape(-1, view1_tr.shape[-1])
        flat2 = view2_tr.reshape(-1, view2_tr.shape[-1])
        ffn_model = cfbt.fit_layer(flat1, flat2, lambda_reg=config.lambda_reg)
        hidden_param_count += int(ffn_model["transform_base"].size)

        def apply_ffn(tokens):
            flat = tokens.reshape(-1, tokens.shape[-1])
            ffn = cfbt.apply_activation(flat @ ffn_model["transform_base"], "relu")
            return token_layer_norm(tokens + ffn.reshape(tokens.shape))

        base_tr = apply_ffn(base_tr)
        base_te = apply_ffn(base_te)
        view1_tr = apply_ffn(view1_tr)
        view2_tr = apply_ffn(view2_tr)
        view1_te = apply_ffn(view1_te)
        view2_te = apply_ffn(view2_te)

        layer_metrics.append(
            {
                "depth": layer_idx + 1,
                "classifier_accuracy": evaluate_logits(yhat_te, yte),
                "attention_kind": config.attention_kind,
                "attention_rank": int(
                    att_model.get("landmark_count", att_model.get("projection_rank", config.num_landmarks))
                ),
                "attention_mix_scale": float(att_model.get("mix_scale", 0.0)),
                "selected_score_scales": att_model.get("selected_score_scales"),
                "selected_prior_weight": att_model.get("selected_prior_weight"),
                "train_fit_loss": float(att_model["train_fit_loss"]) if "train_fit_loss" in att_model else None,
                "top_shared_eigenvalue": top_shared_eigenvalue,
            }
        )
    fit_time = time.perf_counter() - start

    return {
        "model": f"closed-form-transformer:{config.attention_kind}",
        "config": asdict(config),
        "classifier_accuracy": evaluate_logits(yhat_te, yte),
        "layers": layer_metrics,
        "output_param_count": output_param_count,
        "hidden_param_count": hidden_param_count,
        "total_parameter_count": output_param_count + hidden_param_count,
        "fit_time_sec": fit_time,
    }


class UnifiedLearnedTransformer(nn.Module):
    def __init__(self, config, num_classes, pos_embed):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.model_dim * config.mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.patch_size = config.patch_size
        self.blocks = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(config.depth)])
        self.output_heads = nn.ModuleList([nn.Linear(config.model_dim, num_classes) for _ in range(config.depth)])
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float(), persistent=False)

    def forward(self, images):
        tokens = patchify_torch(images, self.patch_size)
        tokens = tokens + self.pos_embed.unsqueeze(0)
        cumulative = None
        depth_logits = []
        for block, head in zip(self.blocks, self.output_heads):
            tokens = block(tokens)
            pooled = tokens.mean(dim=1)
            logits = head(pooled)
            cumulative = logits if cumulative is None else cumulative + logits
            depth_logits.append(cumulative)
        return depth_logits


def run_learned_transformer(config):
    set_seed(config.seed)
    data = load_cifar_subset(config)
    pos_embed = prepare_token_views(config, data)["pos_embed"]

    xtr_img = torch.from_numpy(data["xtr_img"])
    xte_img = torch.from_numpy(data["xte_img"])
    ytr = torch.from_numpy(data["ytr"]).long()
    yte = torch.from_numpy(data["yte"]).long()
    mean_img = torch.from_numpy(data["mean_img"]).float()

    train_ds = TensorDataset(xtr_img, ytr)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnifiedLearnedTransformer(
        config=config,
        num_classes=int(torch.max(ytr).item() + 1),
        pos_embed=pos_embed,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    start = time.perf_counter()
    epoch_stats = []
    for _ in range(config.epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = augment_torch(xb.to(device).float(), config.suite)
            xb = xb - mean_img.to(device)
            yb = yb.to(device)
            depth_logits = model(xb)
            loss = sum(criterion(logits, yb) for logits in depth_logits) / len(depth_logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_stats.append({"loss": float(np.mean(losses))})
    fit_time = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        xte = xte_img.to(device).float() - mean_img.to(device)
        depth_logits = model(xte)
        depth_metrics = []
        for depth_idx, logits in enumerate(depth_logits, start=1):
            pred = logits.argmax(dim=1).cpu().numpy()
            depth_metrics.append({"depth": depth_idx, "classifier_accuracy": float((pred == yte.numpy()).mean())})
        final_pred = depth_logits[-1].argmax(dim=1).cpu().numpy()

    output_param_count = int(sum(p.numel() for p in model.output_heads.parameters()))
    hidden_param_count = int(sum(p.numel() for p in model.blocks.parameters()))
    return {
        "model": "learned-transformer",
        "config": asdict(config),
        "classifier_accuracy": float((final_pred == yte.numpy()).mean()),
        "layers": depth_metrics,
        "output_param_count": output_param_count,
        "hidden_param_count": hidden_param_count,
        "total_parameter_count": output_param_count + hidden_param_count,
        "fit_time_sec": fit_time,
        "epoch_stats": epoch_stats,
        "device": str(device),
    }


def expand_attention_kinds(kind_arg):
    if kind_arg == "all":
        return ATTENTION_KINDS
    return [kind_arg]


def main():
    parser = argparse.ArgumentParser(description="Compare a closed-form transformer and a learned transformer on CIFAR.")
    parser.add_argument("--dataset", default=DATASET, choices=["cifar10", "cifar100"])
    parser.add_argument("--suite", default=SUITE, choices=["random-affine", "random-crop"])
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-landmarks", type=int, default=NUM_LANDMARKS)
    parser.add_argument("--analytic-heads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--attention-kind", default="landmark", choices=ATTENTION_KINDS + ["all"])
    parser.add_argument(
        "--attention-target",
        default=ATTENTION_TARGET,
        choices=["mean", "mean-centered", "residual", "residual-centered", "cross", "bt", "bt-residual"],
    )
    parser.add_argument("--attention-rank", type=int, default=0)
    parser.add_argument("--local-sigma", type=float, default=LOCAL_SIGMA)
    parser.add_argument("--attention-power-iters", type=int, default=8)
    parser.add_argument("--attention-num-bags", type=int, default=4)
    parser.add_argument("--attention-bag-fraction", type=float, default=0.7)
    parser.add_argument("--attention-seed", type=int, default=-1)
    parser.add_argument("--skip-learned", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    base_config = TransformerConfig(
        dataset=args.dataset,
        suite=args.suite,
        patch_size=args.patch_size,
        depth=args.depth,
        n_train=args.n_train,
        n_test=args.n_test,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_landmarks=args.num_landmarks,
        analytic_num_heads=args.analytic_heads,
        attention_target=args.attention_target,
        attention_rank=args.attention_rank,
        local_sigma=args.local_sigma,
        attention_power_iters=args.attention_power_iters,
        attention_num_bags=args.attention_num_bags,
        attention_bag_fraction=args.attention_bag_fraction,
        attention_seed=args.attention_seed,
        seed=args.seed,
    )

    attention_kinds = expand_attention_kinds(args.attention_kind)
    results = []
    for attention_kind in attention_kinds:
        config = replace(base_config, attention_kind=attention_kind)
        results.append(run_closed_form_transformer(config))
    if not args.skip_learned:
        results.append(run_learned_transformer(base_config))

    summary = {
        "config": asdict(base_config),
        "attention_kinds": attention_kinds,
        "results": results,
    }
    json_suffix = args.attention_kind.replace("-", "_")
    json_name = f"transformer_compare_{json_suffix}_{base_config.dataset}_{base_config.suite}.json"
    json_path = default_json_path(json_name) if args.json_out is None else resolve_json_path(args.json_out)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved results to {json_path}")
    for row in results:
        print(
            f"{row['model']:24s} acc={row['classifier_accuracy']:.4f} "
            f"params={row['total_parameter_count']} time={row['fit_time_sec']:.2f}s "
            f"depths={[round(x['classifier_accuracy'], 4) for x in row['layers']]}"
        )


if __name__ == "__main__":
    main()
