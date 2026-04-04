import argparse
import copy
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.datasets import fetch_covtype
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets as tv_datasets

import cifar_shared
import closed_form_barlow_twins as cfbt
import dual_path_residual_cifar as dpr
import transformer_cifar_compare as tcc
from project_paths import default_json_path, default_plot_path, resolve_json_path


MAIN_SEEDS = [7, 11, 19]
SCALING_SEEDS = [7, 11]
SCALING_DATA_VALUES = [1000, 2000, 4000, 8000]
SCALING_MLP_WIDTHS = [128, 256, 512, 1024]
SCALING_TRANSFORMER_PATCH_SIZES = [2, 4, 8, 16]
SCALING_MLP_COMPUTE_DEPTHS = [1, 2, 4, 8, 16]
SCALING_TRANSFORMER_COMPUTE_DEPTHS = [1, 2, 4, 8]

MLP_CANDIDATES = [
    "closed-form-barlow",
    "paper-cca-shared",
    "whitened-shared-pca",
]
DEFAULT_MLP_WINNER = "closed-form-barlow"

TRANSFORMER_CANDIDATES = [
    "spectral-self",
    "spectral-self-token-stats",
    "spectral-self-whitened",
    "score-self-power-bagged-gain",
    "score-self-power-bagged-consensus-gain",
]
DEFAULT_TRANSFORMER_WINNER = "spectral-self"

DEFAULT_DATASETS = ["covtype", "svhn", "cifar100", "ag_news"]
DEFAULT_SCALING_DATASET = "svhn"

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
TEXT_PAD = 0
TEXT_UNK = 1


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    suite: str
    n_train: int
    n_test: int
    seed: int
    image_size: int = 32
    text_vocab_size: int = 512
    text_seq_len: int = 32
    text_embed_dim: int = 64
    text_drop_prob: float = 0.2


@dataclass(frozen=True)
class MLPEvalConfig:
    width: int = 512
    depth: int = 3
    activation: str = "relu"
    lambda_reg: float = 1.0
    head_reg: float = 100.0
    epochs: int = 8
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class TransformerEvalConfig:
    depth: int = 3
    patch_size: int = 8
    lambda_reg: float = 1.0
    head_reg: float = 100.0
    epochs: int = 5
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 5e-2
    num_heads: int = 4
    analytic_num_heads: int = 2
    num_landmarks: int = 8
    mlp_ratio: float = 2.0
    attention_target: str = "mean"
    attention_rank: int = 0
    local_sigma: float = 1.5
    attention_power_iters: int = 8
    attention_num_bags: int = 4
    attention_bag_fraction: float = 0.7


@dataclass(frozen=True)
class GenericAttentionConfig:
    attention_kind: str
    token_dim: int
    num_tokens: int
    depth: int
    head_reg: float
    lambda_reg: float
    num_heads: int
    analytic_num_heads: int
    num_landmarks: int
    attention_target: str
    attention_rank: int
    local_sigma: float
    attention_power_iters: int
    attention_num_bags: int
    attention_bag_fraction: float
    seed: int
    attention_seed: int = -1

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


@dataclass
class RawDatasetBundle:
    name: str
    suite: str
    modality: str
    train_raw: np.ndarray
    test_raw: np.ndarray
    ytr: np.ndarray
    yte: np.ndarray
    num_classes: int
    image_size: int = 0
    image_mean: np.ndarray | None = None
    text_embedding: np.ndarray | None = None
    text_pos: np.ndarray | None = None
    text_drop_prob: float = 0.0
    tabular_mean: np.ndarray | None = None
    tabular_std: np.ndarray | None = None
    tabular_token_embedding: np.ndarray | None = None
    tabular_pos: np.ndarray | None = None
    tabular_drop_prob: float = 0.0
    tabular_noise_std: float = 0.0


DATASET_REGISTRY = {
    "covtype": {"suite": "feature-mask", "n_train": 12000, "n_test": 4000},
    "svhn": {"suite": "random-affine", "n_train": 6000, "n_test": 2000},
    "cifar100": {"suite": "random-affine", "n_train": 8000, "n_test": 2000},
    "ag_news": {"suite": "token-mask", "n_train": 8000, "n_test": 2000},
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def one_hot(y, num_classes):
    eye = np.eye(num_classes, dtype=np.float64)
    return eye[y]


def ridge_regression(X, Y, reg):
    gram = X.T @ X
    rhs = X.T @ Y
    return np.linalg.solve(gram + reg * np.eye(gram.shape[0], dtype=np.float64), rhs)


def evaluate_logits(logits, y):
    return float((np.argmax(logits, axis=1) == y).mean())


def mean_std(values):
    vals = np.asarray(values, dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=0))


def confidence_interval(values):
    vals = np.asarray(values, dtype=np.float64)
    if vals.size <= 1:
        return 0.0
    return float(1.96 * vals.std(ddof=1) / np.sqrt(vals.size))


def maybe_import_pyplot():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def choose_num_heads(token_dim, preferred=4):
    for candidate in [preferred, 3, 2, 1]:
        if token_dim % candidate == 0:
            return candidate
    return 1


def make_1d_sincos_pos_embed(embed_dim, seq_len):
    assert embed_dim % 2 == 0
    positions = np.arange(seq_len, dtype=np.float64)
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    out = np.einsum("m,d->md", positions, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)


def resize_images(images, image_size):
    if images.shape[-1] == image_size and images.shape[-2] == image_size:
        return images.astype(np.float32)
    tensor = torch.from_numpy(images).float()
    resized = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return resized.numpy().astype(np.float32)


@lru_cache(maxsize=2)
def load_grayscale_dataset_full(dataset_name):
    dataset_cls = {
        "mnist": tv_datasets.MNIST,
        "fashion_mnist": tv_datasets.FashionMNIST,
    }[dataset_name]
    train_ds = dataset_cls(root="./data", train=True, download=True)
    test_ds = dataset_cls(root="./data", train=False, download=True)
    xtr = train_ds.data.numpy().astype(np.float32)[:, None, :, :] / 255.0
    xte = test_ds.data.numpy().astype(np.float32)[:, None, :, :] / 255.0
    ytr = train_ds.targets.numpy()
    yte = test_ds.targets.numpy()
    return xtr, ytr, xte, yte


@lru_cache(maxsize=1)
def load_svhn_full():
    train_ds = tv_datasets.SVHN(root="./data", split="train", download=True)
    test_ds = tv_datasets.SVHN(root="./data", split="test", download=True)
    xtr = train_ds.data.astype(np.float32) / 255.0
    xte = test_ds.data.astype(np.float32) / 255.0
    ytr = np.asarray(train_ds.labels, dtype=np.int64)
    yte = np.asarray(test_ds.labels, dtype=np.int64)
    return xtr, ytr, xte, yte


@lru_cache(maxsize=1)
def load_covtype_full():
    bunch = fetch_covtype(data_home="./data", download_if_missing=True)
    X = bunch.data.astype(np.float32)
    y = bunch.target.astype(np.int64) - 1
    return X, y


@lru_cache(maxsize=8)
def load_ag_news_tokenized(vocab_size, seq_len):
    dataset = load_dataset("ag_news")
    train_texts = dataset["train"]["text"]
    test_texts = dataset["test"]["text"]
    train_labels = np.asarray(dataset["train"]["label"], dtype=np.int64)
    test_labels = np.asarray(dataset["test"]["label"], dtype=np.int64)

    counter = Counter()
    for text in train_texts:
        counter.update(token.lower() for token in TOKEN_PATTERN.findall(text))

    most_common = [token for token, _ in counter.most_common(max(0, vocab_size - 2))]
    vocab = {token: idx + 2 for idx, token in enumerate(most_common)}

    def encode(texts):
        encoded = np.zeros((len(texts), seq_len), dtype=np.int64)
        for row_idx, text in enumerate(texts):
            tokens = [vocab.get(token.lower(), TEXT_UNK) for token in TOKEN_PATTERN.findall(text)]
            tokens = tokens[:seq_len]
            if tokens:
                encoded[row_idx, : len(tokens)] = tokens
        return encoded

    return {
        "train_ids": encode(train_texts),
        "test_ids": encode(test_texts),
        "train_labels": train_labels,
        "test_labels": test_labels,
    }


def build_text_embedding(vocab_size, embed_dim, seed):
    rng = np.random.default_rng(seed)
    embedding = rng.standard_normal((vocab_size, embed_dim)).astype(np.float32)
    embedding[TEXT_PAD] = 0.0
    norms = np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-8)
    return embedding / norms


def sample_indices(total, count, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(total, size=count, replace=False)


def load_dataset_bundle(spec: DatasetSpec):
    if spec.name == "covtype":
        X_all, y_all = load_covtype_full()
        all_indices = sample_indices(len(X_all), spec.n_train + spec.n_test, spec.seed)
        idx_tr = all_indices[: spec.n_train]
        idx_te = all_indices[spec.n_train :]
        xtr = X_all[idx_tr]
        xte = X_all[idx_te]
        ytr = y_all[idx_tr]
        yte = y_all[idx_te]
        mean = xtr.mean(axis=0, keepdims=True)
        std = np.maximum(xtr.std(axis=0, keepdims=True), 1e-6)
        xtr = ((xtr - mean) / std).astype(np.float32)
        xte = ((xte - mean) / std).astype(np.float32)
        token_embedding = build_text_embedding(xtr.shape[1], spec.text_embed_dim, seed=17)
        pos = make_1d_sincos_pos_embed(spec.text_embed_dim, xtr.shape[1])
        return RawDatasetBundle(
            name=spec.name,
            suite=spec.suite,
            modality="tabular",
            train_raw=xtr,
            test_raw=xte,
            ytr=ytr,
            yte=yte,
            num_classes=int(max(ytr.max(), yte.max()) + 1),
            tabular_mean=mean.astype(np.float32),
            tabular_std=std.astype(np.float32),
            tabular_token_embedding=token_embedding,
            tabular_pos=pos,
            tabular_drop_prob=0.15,
            tabular_noise_std=0.1,
        )

    if spec.name in {"mnist", "fashion_mnist"}:
        xtr_all, ytr_all, xte_all, yte_all = load_grayscale_dataset_full(spec.name)
        idx_tr = sample_indices(len(xtr_all), spec.n_train, spec.seed)
        idx_te = sample_indices(len(xte_all), spec.n_test, spec.seed + 1)
        xtr = resize_images(xtr_all[idx_tr], spec.image_size)
        xte = resize_images(xte_all[idx_te], spec.image_size)
        ytr = ytr_all[idx_tr]
        yte = yte_all[idx_te]
        return RawDatasetBundle(
            name=spec.name,
            suite=spec.suite,
            modality="image",
            train_raw=xtr,
            test_raw=xte,
            ytr=ytr,
            yte=yte,
            num_classes=int(max(ytr.max(), yte.max()) + 1),
            image_size=spec.image_size,
            image_mean=xtr.mean(axis=0, keepdims=True).astype(np.float32),
        )

    if spec.name == "svhn":
        xtr_all, ytr_all, xte_all, yte_all = load_svhn_full()
        idx_tr = sample_indices(len(xtr_all), spec.n_train, spec.seed)
        idx_te = sample_indices(len(xte_all), spec.n_test, spec.seed + 1)
        xtr = xtr_all[idx_tr]
        xte = xte_all[idx_te]
        ytr = ytr_all[idx_tr]
        yte = yte_all[idx_te]
        return RawDatasetBundle(
            name=spec.name,
            suite=spec.suite,
            modality="image",
            train_raw=xtr,
            test_raw=xte,
            ytr=ytr,
            yte=yte,
            num_classes=int(max(ytr.max(), yte.max()) + 1),
            image_size=xtr.shape[-1],
            image_mean=xtr.mean(axis=0, keepdims=True).astype(np.float32),
        )

    if spec.name in {"cifar10", "cifar100"}:
        dataset = cifar_shared.load_cifar_numpy(
            spec.name,
            n_train=spec.n_train,
            n_test=spec.n_test,
            seed=spec.seed,
            width=3072,
        )
        return RawDatasetBundle(
            name=spec.name,
            suite=spec.suite,
            modality="image",
            train_raw=dataset["xtr_img"].astype(np.float32),
            test_raw=dataset["xte_img"].astype(np.float32),
            ytr=dataset["ytr"].astype(np.int64),
            yte=dataset["yte"].astype(np.int64),
            num_classes=int(max(dataset["ytr"].max(), dataset["yte"].max()) + 1),
            image_size=dataset["xtr_img"].shape[-1],
            image_mean=dataset["xtr_img"].mean(axis=0, keepdims=True).astype(np.float32),
        )

    if spec.name == "ag_news":
        tokenized = load_ag_news_tokenized(spec.text_vocab_size, spec.text_seq_len)
        idx_tr = sample_indices(len(tokenized["train_ids"]), spec.n_train, spec.seed)
        idx_te = sample_indices(len(tokenized["test_ids"]), spec.n_test, spec.seed + 1)
        embedding = build_text_embedding(spec.text_vocab_size, spec.text_embed_dim, seed=0)
        pos = make_1d_sincos_pos_embed(spec.text_embed_dim, spec.text_seq_len)
        return RawDatasetBundle(
            name=spec.name,
            suite=spec.suite,
            modality="text",
            train_raw=tokenized["train_ids"][idx_tr],
            test_raw=tokenized["test_ids"][idx_te],
            ytr=tokenized["train_labels"][idx_tr],
            yte=tokenized["test_labels"][idx_te],
            num_classes=int(max(tokenized["train_labels"].max(), tokenized["test_labels"].max()) + 1),
            text_embedding=embedding,
            text_pos=pos,
            text_drop_prob=spec.text_drop_prob,
        )

    raise ValueError(f"Unsupported dataset: {spec.name}")


def flatten_images(images, mean_image):
    centered = images.astype(np.float64) - mean_image.astype(np.float64)
    return centered.reshape(centered.shape[0], -1)


def bow_from_ids(ids, vocab_size):
    bow = np.zeros((ids.shape[0], vocab_size), dtype=np.float64)
    for row_idx, row in enumerate(ids):
        valid = row[row != TEXT_PAD]
        if valid.size == 0:
            continue
        uniq, counts = np.unique(valid, return_counts=True)
        bow[row_idx, uniq] = counts.astype(np.float64)
    bow[:, TEXT_PAD] = 0.0
    norms = np.maximum(np.linalg.norm(bow, axis=1, keepdims=True), 1e-8)
    return bow / norms


def augment_text_ids_numpy(ids, seed, drop_prob):
    rng = np.random.default_rng(seed)
    augmented = ids.copy()
    mask = (augmented != TEXT_PAD) & (rng.random(augmented.shape) < drop_prob)
    augmented[mask] = TEXT_PAD
    return augmented


def augment_text_ids_torch(ids, drop_prob):
    augmented = ids.clone()
    mask = (augmented != TEXT_PAD) & (torch.rand_like(augmented.float()) < drop_prob)
    augmented[mask] = TEXT_PAD
    return augmented


def augment_tabular_numpy(features, seed, drop_prob, noise_std):
    rng = np.random.default_rng(seed)
    augmented = features.copy()
    mask = rng.random(augmented.shape) < drop_prob
    augmented[mask] = 0.0
    augmented = augmented + rng.standard_normal(augmented.shape).astype(np.float32) * noise_std
    return augmented


def augment_tabular_torch(features, drop_prob, noise_std):
    augmented = features.clone()
    mask = torch.rand_like(augmented) < drop_prob
    augmented[mask] = 0.0
    augmented = augmented + torch.randn_like(augmented) * noise_std
    return augmented


def build_mlp_arrays(bundle: RawDatasetBundle, seed: int):
    if bundle.modality == "image":
        base_tr = flatten_images(bundle.train_raw, bundle.image_mean)
        base_te = flatten_images(bundle.test_raw, bundle.image_mean)
        rng_seed = seed + 101
        view1_tr = flatten_images(
            cifar_shared.apply_augmentation(bundle.train_raw, bundle.suite, np.random.default_rng(rng_seed)).astype(np.float32),
            bundle.image_mean,
        )
        view2_tr = flatten_images(
            cifar_shared.apply_augmentation(bundle.train_raw, bundle.suite, np.random.default_rng(rng_seed + 1)).astype(np.float32),
            bundle.image_mean,
        )
        view1_te = flatten_images(
            cifar_shared.apply_augmentation(bundle.test_raw, bundle.suite, np.random.default_rng(seed + 202)).astype(np.float32),
            bundle.image_mean,
        )
        view2_te = flatten_images(
            cifar_shared.apply_augmentation(bundle.test_raw, bundle.suite, np.random.default_rng(seed + 203)).astype(np.float32),
            bundle.image_mean,
        )
        return base_tr, base_te, view1_tr, view2_tr, view1_te, view2_te

    if bundle.modality == "text":
        base_tr = bow_from_ids(bundle.train_raw, bundle.text_embedding.shape[0])
        base_te = bow_from_ids(bundle.test_raw, bundle.text_embedding.shape[0])
        view1_tr = bow_from_ids(
            augment_text_ids_numpy(bundle.train_raw, seed + 101, bundle.text_drop_prob),
            bundle.text_embedding.shape[0],
        )
        view2_tr = bow_from_ids(
            augment_text_ids_numpy(bundle.train_raw, seed + 102, bundle.text_drop_prob),
            bundle.text_embedding.shape[0],
        )
        view1_te = bow_from_ids(
            augment_text_ids_numpy(bundle.test_raw, seed + 202, bundle.text_drop_prob),
            bundle.text_embedding.shape[0],
        )
        view2_te = bow_from_ids(
            augment_text_ids_numpy(bundle.test_raw, seed + 203, bundle.text_drop_prob),
            bundle.text_embedding.shape[0],
        )
        mean = base_tr.mean(axis=0, keepdims=True)
        return (
            base_tr - mean,
            base_te - mean,
            view1_tr - mean,
            view2_tr - mean,
            view1_te - mean,
            view2_te - mean,
        )

    if bundle.modality == "tabular":
        base_tr = bundle.train_raw.astype(np.float64)
        base_te = bundle.test_raw.astype(np.float64)
        view1_tr = augment_tabular_numpy(bundle.train_raw, seed + 101, bundle.tabular_drop_prob, bundle.tabular_noise_std).astype(np.float64)
        view2_tr = augment_tabular_numpy(bundle.train_raw, seed + 102, bundle.tabular_drop_prob, bundle.tabular_noise_std).astype(np.float64)
        view1_te = augment_tabular_numpy(bundle.test_raw, seed + 202, bundle.tabular_drop_prob, bundle.tabular_noise_std).astype(np.float64)
        view2_te = augment_tabular_numpy(bundle.test_raw, seed + 203, bundle.tabular_drop_prob, bundle.tabular_noise_std).astype(np.float64)
        return base_tr, base_te, view1_tr, view2_tr, view1_te, view2_te

    raise ValueError(f"Unsupported modality: {bundle.modality}")


def build_image_tokens(images, mean_image, patch_size):
    tokens = tcc.patchify_numpy(images.astype(np.float32) - mean_image.astype(np.float32), patch_size).astype(np.float64)
    grid = int(round(math.sqrt(tokens.shape[1])))
    pos = tcc.make_2d_sincos_pos_embed(tokens.shape[-1], grid).astype(np.float64)
    return tokens + pos[None, :, :]


def build_text_tokens(ids, embedding, pos):
    return embedding[ids].astype(np.float64) + pos[None, :, :].astype(np.float64)


def build_tabular_tokens(features, embedding, pos):
    return features[:, :, None].astype(np.float64) * embedding[None, :, :].astype(np.float64) + pos[None, :, :].astype(np.float64)


def build_transformer_arrays(bundle: RawDatasetBundle, seed: int, patch_size: int):
    if bundle.modality == "image":
        base_tr = build_image_tokens(bundle.train_raw, bundle.image_mean, patch_size)
        base_te = build_image_tokens(bundle.test_raw, bundle.image_mean, patch_size)
        view1_tr = build_image_tokens(
            cifar_shared.apply_augmentation(bundle.train_raw, bundle.suite, np.random.default_rng(seed + 101)).astype(np.float32),
            bundle.image_mean,
            patch_size,
        )
        view2_tr = build_image_tokens(
            cifar_shared.apply_augmentation(bundle.train_raw, bundle.suite, np.random.default_rng(seed + 102)).astype(np.float32),
            bundle.image_mean,
            patch_size,
        )
        view1_te = build_image_tokens(
            cifar_shared.apply_augmentation(bundle.test_raw, bundle.suite, np.random.default_rng(seed + 202)).astype(np.float32),
            bundle.image_mean,
            patch_size,
        )
        view2_te = build_image_tokens(
            cifar_shared.apply_augmentation(bundle.test_raw, bundle.suite, np.random.default_rng(seed + 203)).astype(np.float32),
            bundle.image_mean,
            patch_size,
        )
        return base_tr, base_te, view1_tr, view2_tr, view1_te, view2_te

    if bundle.modality == "text":
        base_tr = build_text_tokens(bundle.train_raw, bundle.text_embedding, bundle.text_pos)
        base_te = build_text_tokens(bundle.test_raw, bundle.text_embedding, bundle.text_pos)
        view1_tr = build_text_tokens(
            augment_text_ids_numpy(bundle.train_raw, seed + 101, bundle.text_drop_prob),
            bundle.text_embedding,
            bundle.text_pos,
        )
        view2_tr = build_text_tokens(
            augment_text_ids_numpy(bundle.train_raw, seed + 102, bundle.text_drop_prob),
            bundle.text_embedding,
            bundle.text_pos,
        )
        view1_te = build_text_tokens(
            augment_text_ids_numpy(bundle.test_raw, seed + 202, bundle.text_drop_prob),
            bundle.text_embedding,
            bundle.text_pos,
        )
        view2_te = build_text_tokens(
            augment_text_ids_numpy(bundle.test_raw, seed + 203, bundle.text_drop_prob),
            bundle.text_embedding,
            bundle.text_pos,
        )
        return base_tr, base_te, view1_tr, view2_tr, view1_te, view2_te

    if bundle.modality == "tabular":
        base_tr = build_tabular_tokens(bundle.train_raw, bundle.tabular_token_embedding, bundle.tabular_pos)
        base_te = build_tabular_tokens(bundle.test_raw, bundle.tabular_token_embedding, bundle.tabular_pos)
        view1_tr = build_tabular_tokens(
            augment_tabular_numpy(bundle.train_raw, seed + 101, bundle.tabular_drop_prob, bundle.tabular_noise_std),
            bundle.tabular_token_embedding,
            bundle.tabular_pos,
        )
        view2_tr = build_tabular_tokens(
            augment_tabular_numpy(bundle.train_raw, seed + 102, bundle.tabular_drop_prob, bundle.tabular_noise_std),
            bundle.tabular_token_embedding,
            bundle.tabular_pos,
        )
        view1_te = build_tabular_tokens(
            augment_tabular_numpy(bundle.test_raw, seed + 202, bundle.tabular_drop_prob, bundle.tabular_noise_std),
            bundle.tabular_token_embedding,
            bundle.tabular_pos,
        )
        view2_te = build_tabular_tokens(
            augment_tabular_numpy(bundle.test_raw, seed + 203, bundle.tabular_drop_prob, bundle.tabular_noise_std),
            bundle.tabular_token_embedding,
            bundle.tabular_pos,
        )
        return base_tr, base_te, view1_tr, view2_tr, view1_te, view2_te

    raise ValueError(f"Unsupported modality: {bundle.modality}")


def build_base_transformer_tokens(bundle: RawDatasetBundle, raw_inputs, patch_size: int):
    if bundle.modality == "image":
        return build_image_tokens(raw_inputs, bundle.image_mean, patch_size)
    if bundle.modality == "text":
        return build_text_tokens(raw_inputs, bundle.text_embedding, bundle.text_pos)
    if bundle.modality == "tabular":
        return build_tabular_tokens(raw_inputs, bundle.tabular_token_embedding, bundle.tabular_pos)
    raise ValueError(f"Unsupported modality: {bundle.modality}")


def image_tokens_from_torch(images, mean_image, patch_size):
    centered = images - mean_image
    tokens = tcc.patchify_torch(centered, patch_size)
    grid = int(round(math.sqrt(tokens.shape[1])))
    pos = tcc.make_2d_sincos_pos_embed(tokens.shape[-1], grid).astype(np.float32)
    pos_t = torch.from_numpy(pos).to(images.device)
    return tokens + pos_t.unsqueeze(0)


def text_tokens_from_torch(ids, embedding, pos):
    return embedding[ids] + pos.unsqueeze(0)


def tabular_tokens_from_torch(features, embedding, pos):
    return features.unsqueeze(-1) * embedding.unsqueeze(0) + pos.unsqueeze(0)


def normalize_hidden_with_stats(train_arrays, eval_arrays):
    mean = sum(arr.mean(axis=0, keepdims=True) for arr in train_arrays) / len(train_arrays)
    centered_train = [arr - mean for arr in train_arrays]
    centered_eval = [arr - mean for arr in eval_arrays]
    avg_var = sum(np.mean(arr * arr, axis=0, keepdims=True) for arr in centered_train) / len(centered_train)
    scale = np.sqrt(np.maximum(avg_var, 1e-6))
    scaled_train = [arr / scale for arr in centered_train]
    scaled_eval = [arr / scale for arr in centered_eval]
    return scaled_train, scaled_eval, mean, scale


def build_base_mlp_features(bundle: RawDatasetBundle, raw_inputs):
    if bundle.modality == "image":
        return flatten_images(raw_inputs, bundle.image_mean)
    if bundle.modality == "text":
        return bow_from_ids(raw_inputs, bundle.text_embedding.shape[0])
    if bundle.modality == "tabular":
        return raw_inputs.astype(np.float64)
    raise ValueError(f"Unsupported modality: {bundle.modality}")


def make_ood_variants(bundle: RawDatasetBundle, seed: int):
    if bundle.modality == "image":
        rng = np.random.default_rng(seed + 3001)
        noisy = np.clip(bundle.test_raw + 0.15 * rng.standard_normal(bundle.test_raw.shape).astype(np.float32), 0.0, 1.0)
        masked = bundle.test_raw.copy()
        h = masked.shape[-2]
        w = masked.shape[-1]
        top = h // 4
        left = w // 4
        masked[:, :, top : top + h // 2, left : left + w // 2] = 0.0
        return {
            "gaussian-noise": noisy,
            "center-mask": masked,
        }
    if bundle.modality == "text":
        heavy_mask = augment_text_ids_numpy(bundle.test_raw, seed + 3001, min(0.5, bundle.text_drop_prob * 2.0))
        truncated = bundle.test_raw.copy()
        truncated[:, truncated.shape[1] // 2 :] = TEXT_PAD
        return {
            "heavy-mask": heavy_mask,
            "truncate-half": truncated,
        }
    if bundle.modality == "tabular":
        masked = augment_tabular_numpy(bundle.test_raw, seed + 3001, 0.3, 0.0)
        noisy = augment_tabular_numpy(bundle.test_raw, seed + 3002, 0.0, 0.25)
        return {
            "feature-mask": masked,
            "gaussian-noise": noisy,
        }
    raise ValueError(f"Unsupported modality: {bundle.modality}")


def effective_rank(features):
    centered = features - features.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(centered.shape[0], 1)
    eigvals = np.maximum(np.linalg.eigvalsh(cov), 1e-12)
    probs = eigvals / eigvals.sum()
    entropy = -np.sum(probs * np.log(probs))
    return float(np.exp(entropy))


def centroid_margin(train_repr, ytr, test_repr, yte):
    centroids = []
    for cls_idx in range(int(max(ytr.max(), yte.max()) + 1)):
        cls_mask = ytr == cls_idx
        if not np.any(cls_mask):
            centroids.append(np.zeros(train_repr.shape[1], dtype=np.float64))
            continue
        centroids.append(train_repr[cls_mask].mean(axis=0))
    centroids = np.asarray(centroids, dtype=np.float64)
    centroids = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8)
    test_norm = test_repr / np.maximum(np.linalg.norm(test_repr, axis=1, keepdims=True), 1e-8)
    sims = test_norm @ centroids.T
    margins = []
    for idx, label in enumerate(yte):
        own = sims[idx, label]
        others = np.delete(sims[idx], label)
        margins.append(own - np.max(others))
    return float(np.mean(margins))


def masked_batch_for_sample(bundle: RawDatasetBundle, sample, patch_size):
    if bundle.modality == "image":
        channels, height, width = sample.shape
        patch = patch_size
        masked = []
        group_specs = []
        for top in range(0, height, patch):
            for left in range(0, width, patch):
                clone = sample.copy()
                clone[:, top : top + patch, left : left + patch] = 0.0
                masked.append(clone)
                group_specs.append((top, left, patch))
        return np.asarray(masked), group_specs
    if bundle.modality == "text":
        masked = []
        group_specs = []
        for token_idx in range(sample.shape[0]):
            clone = sample.copy()
            clone[token_idx] = TEXT_PAD
            masked.append(clone)
            group_specs.append(token_idx)
        return np.asarray(masked), group_specs
    if bundle.modality == "tabular":
        masked = []
        group_specs = []
        for feat_idx in range(sample.shape[0]):
            clone = sample.copy()
            clone[feat_idx] = 0.0
            masked.append(clone)
            group_specs.append(feat_idx)
        return np.asarray(masked), group_specs
    raise ValueError(f"Unsupported modality: {bundle.modality}")


def apply_group_masks(bundle: RawDatasetBundle, sample, selected_groups, patch_size):
    clone = sample.copy()
    if bundle.modality == "image":
        for top, left, patch in selected_groups:
            clone[:, top : top + patch, left : left + patch] = 0.0
        return clone
    if bundle.modality == "text":
        clone[np.asarray(selected_groups, dtype=np.int64)] = TEXT_PAD
        return clone
    if bundle.modality == "tabular":
        clone[np.asarray(selected_groups, dtype=np.int64)] = 0.0
        return clone
    raise ValueError(f"Unsupported modality: {bundle.modality}")


def occlusion_summary(bundle: RawDatasetBundle, raw_test, predict_logits_fn, patch_size, max_samples=16):
    sample_count = min(max_samples, len(raw_test))
    top1_fracs = []
    top20_fracs = []
    deletion_drops = []
    baseline_logits = predict_logits_fn(raw_test[:sample_count])
    baseline_pred = np.argmax(baseline_logits, axis=1)
    for idx in range(sample_count):
        masked_batch, group_specs = masked_batch_for_sample(bundle, raw_test[idx], patch_size)
        masked_logits = predict_logits_fn(masked_batch)
        pred_cls = baseline_pred[idx]
        drops = np.maximum(baseline_logits[idx, pred_cls] - masked_logits[:, pred_cls], 0.0)
        total = float(np.sum(drops))
        if total <= 1e-8:
            top1_fracs.append(0.0)
            top20_fracs.append(0.0)
            deletion_drops.append(0.0)
            continue
        top1_fracs.append(float(np.max(drops) / total))
        k = max(1, int(math.ceil(0.2 * len(drops))))
        topk_idx = np.argsort(drops)[-k:]
        top20_fracs.append(float(np.sum(drops[topk_idx]) / total))
        selected = [group_specs[group_idx] for group_idx in topk_idx]
        masked_sample = apply_group_masks(bundle, raw_test[idx], selected, patch_size)
        masked_score = predict_logits_fn(masked_sample[None])[0, pred_cls]
        deletion_drops.append(float(baseline_logits[idx, pred_cls] - masked_score))
    return {
        "top1_importance_fraction": float(np.mean(top1_fracs)),
        "top20_importance_fraction": float(np.mean(top20_fracs)),
        "top20_deletion_logit_drop": float(np.mean(deletion_drops)),
    }


def run_closed_form_mlp(bundle: RawDatasetBundle, method_name: str, config: MLPEvalConfig, seed: int, collect_analysis: bool = False):
    set_seed(seed)
    base_tr, base_te, view1_tr, view2_tr, view1_te, view2_te = build_mlp_arrays(bundle, seed)
    train_arrays, test_arrays, initial_mean, initial_scale = normalize_hidden_with_stats(
        [base_tr, view1_tr, view2_tr],
        [base_te, view1_te, view2_te],
    )
    base_tr, view1_tr, view2_tr = train_arrays
    base_te, view1_te, view2_te = test_arrays

    ytr_onehot = one_hot(bundle.ytr, bundle.num_classes)
    layers = []
    activation_param_count = 0
    output_param_count = 0
    layer_states = []
    output_map = None

    start = time.perf_counter()
    for layer_idx in range(config.depth):
        fitted = dpr.fit_activation_transforms(
            method_name=method_name,
            base_tr=base_tr,
            view1_tr=view1_tr,
            view2_tr=view2_tr,
            width=config.width,
            lambda_reg=config.lambda_reg,
            layer_seed=seed + 97 * (layer_idx + 1),
            ytr=bundle.ytr,
        )
        if fitted["apply_kind"] != "linear":
            raise ValueError(f"Unsupported apply kind for MLP suite: {fitted['apply_kind']}")

        activation_param_count += int(fitted["transform_base"].size)
        base_tr = cfbt.apply_layer(base_tr, fitted["transform_base"], activation=config.activation)
        base_te = cfbt.apply_layer(base_te, fitted["transform_base"], activation=config.activation)
        view1_tr = cfbt.apply_layer(view1_tr, fitted["transform_view1"], activation=config.activation)
        view2_tr = cfbt.apply_layer(view2_tr, fitted["transform_view2"], activation=config.activation)
        view1_te = cfbt.apply_layer(view1_te, fitted["transform_view1"], activation=config.activation)
        view2_te = cfbt.apply_layer(view2_te, fitted["transform_view2"], activation=config.activation)

        train_arrays, test_arrays, norm_mean, norm_scale = normalize_hidden_with_stats(
            [base_tr, view1_tr, view2_tr],
            [base_te, view1_te, view2_te],
        )
        base_tr, view1_tr, view2_tr = train_arrays
        base_te, view1_te, view2_te = test_arrays

        output_map = ridge_regression(base_tr, ytr_onehot, reg=config.head_reg)
        output_param_count = int(output_map.size)
        logits_te = base_te @ output_map
        layer_states.append(
            {
                "fitted": fitted,
                "norm_mean": norm_mean,
                "norm_scale": norm_scale,
            }
        )
        layers.append(
            {
                "depth": layer_idx + 1,
                "classifier_accuracy": evaluate_logits(logits_te, bundle.yte),
                **fitted["method_stats"],
            }
        )
    fit_time = time.perf_counter() - start

    analysis = None
    if collect_analysis:
        if bundle.modality == "text":
            text_mean = bow_from_ids(bundle.train_raw, bundle.text_embedding.shape[0]).mean(axis=0, keepdims=True)
        else:
            text_mean = None

        def encode_raw(raw_inputs):
            hidden = build_base_mlp_features(bundle, raw_inputs)
            if bundle.modality == "text":
                hidden = hidden - text_mean
            hidden = (hidden - initial_mean) / initial_scale
            for state in layer_states:
                fitted = state["fitted"]
                hidden = cfbt.apply_layer(hidden, fitted["transform_base"], activation=config.activation)
                hidden = (hidden - state["norm_mean"]) / state["norm_scale"]
            return hidden

        def predict_logits(raw_inputs):
            return encode_raw(raw_inputs) @ output_map

        ood_results = []
        for shift_name, shifted_inputs in make_ood_variants(bundle, seed).items():
            logits = predict_logits(shifted_inputs)
            ood_results.append({"shift": shift_name, "accuracy": evaluate_logits(logits, bundle.yte)})

        train_repr = encode_raw(bundle.train_raw)
        test_repr = encode_raw(bundle.test_raw)
        analysis = {
            "ood_results": ood_results,
            "interpretability": {
                "effective_rank": effective_rank(test_repr),
                "centroid_margin": centroid_margin(train_repr, bundle.ytr, test_repr, bundle.yte),
                **occlusion_summary(bundle, bundle.test_raw, predict_logits, patch_size=8),
            },
        }

    return {
        "architecture": "mlp",
        "model": "closed-form",
        "dataset": bundle.name,
        "suite": bundle.suite,
        "seed": seed,
        "closed_form_method": method_name,
        "classifier_accuracy": layers[-1]["classifier_accuracy"],
        "layers": layers,
        "hidden_param_count": activation_param_count,
        "output_param_count": output_param_count,
        "total_parameter_count": activation_param_count + output_param_count,
        "fit_time_sec": fit_time,
        "config": asdict(config),
        "analysis": analysis,
    }


class SupervisedMLP(nn.Module):
    def __init__(self, input_dim, width, depth, num_classes):
        super().__init__()
        dims = [input_dim] + [width] * depth
        self.hidden = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(depth)])
        self.head = nn.Linear(width, num_classes)

    def encode(self, x):
        for layer in self.hidden:
            x = torch.relu(layer(x))
        return x

    def forward(self, x):
        return self.head(self.encode(x))


def run_backprop_mlp(bundle: RawDatasetBundle, config: MLPEvalConfig, seed: int, collect_analysis: bool = False):
    set_seed(seed)
    base_tr, base_te, _, _, _, _ = build_mlp_arrays(bundle, seed)
    xtr = torch.from_numpy(base_tr).float()
    xte = torch.from_numpy(base_te).float()
    ytr = torch.from_numpy(bundle.ytr).long()
    yte = torch.from_numpy(bundle.yte).long()

    train_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=config.batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SupervisedMLP(base_tr.shape[1], config.width, config.depth, bundle.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    start = time.perf_counter()
    epoch_stats = []
    for _ in range(config.epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_stats.append({"loss": float(np.mean(losses))})
    fit_time = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        pred = model(xte.to(device)).argmax(dim=1).cpu().numpy()

    analysis = None
    if collect_analysis:
        if bundle.modality == "text":
            text_mean = bow_from_ids(bundle.train_raw, bundle.text_embedding.shape[0]).mean(axis=0, keepdims=True)
        else:
            text_mean = None

        def features_from_raw(raw_inputs):
            feats = build_base_mlp_features(bundle, raw_inputs)
            if bundle.modality == "text":
                feats = feats - text_mean
            return feats

        def predict_logits(raw_inputs):
            with torch.no_grad():
                feats = torch.from_numpy(features_from_raw(raw_inputs)).float().to(device)
                return model(feats).cpu().numpy()

        def encode_repr(raw_inputs):
            with torch.no_grad():
                feats = torch.from_numpy(features_from_raw(raw_inputs)).float().to(device)
                return model.encode(feats).cpu().numpy().astype(np.float64)

        ood_results = []
        for shift_name, shifted_inputs in make_ood_variants(bundle, seed).items():
            logits = predict_logits(shifted_inputs)
            ood_results.append({"shift": shift_name, "accuracy": evaluate_logits(logits, bundle.yte)})

        train_repr = encode_repr(bundle.train_raw)
        test_repr = encode_repr(bundle.test_raw)
        analysis = {
            "ood_results": ood_results,
            "interpretability": {
                "effective_rank": effective_rank(test_repr),
                "centroid_margin": centroid_margin(train_repr, bundle.ytr, test_repr, bundle.yte),
                **occlusion_summary(bundle, bundle.test_raw, predict_logits, patch_size=8),
            },
        }

    return {
        "architecture": "mlp",
        "model": "backprop",
        "dataset": bundle.name,
        "suite": bundle.suite,
        "seed": seed,
        "classifier_accuracy": float((pred == bundle.yte).mean()),
        "hidden_param_count": int(sum(p.numel() for p in model.hidden.parameters())),
        "output_param_count": int(sum(p.numel() for p in model.head.parameters())),
        "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "fit_time_sec": fit_time,
        "epoch_stats": epoch_stats,
        "config": asdict(config),
        "analysis": analysis,
    }


def make_attention_config(tokens, attention_kind, config: TransformerEvalConfig, seed: int):
    token_dim = int(tokens.shape[-1])
    num_heads = choose_num_heads(token_dim, preferred=config.num_heads)
    analytic_heads = min(config.analytic_num_heads, num_heads)
    return GenericAttentionConfig(
        attention_kind=attention_kind,
        token_dim=token_dim,
        num_tokens=int(tokens.shape[1]),
        depth=config.depth,
        head_reg=config.head_reg,
        lambda_reg=config.lambda_reg,
        num_heads=num_heads,
        analytic_num_heads=max(1, analytic_heads),
        num_landmarks=config.num_landmarks,
        attention_target=config.attention_target,
        attention_rank=config.attention_rank,
        local_sigma=config.local_sigma,
        attention_power_iters=config.attention_power_iters,
        attention_num_bags=config.attention_num_bags,
        attention_bag_fraction=config.attention_bag_fraction,
        seed=seed,
    )


def run_closed_form_transformer(bundle: RawDatasetBundle, attention_kind: str, config: TransformerEvalConfig, seed: int, collect_analysis: bool = False):
    set_seed(seed)
    base_tr, base_te, view1_tr, view2_tr, view1_te, view2_te = build_transformer_arrays(bundle, seed, config.patch_size)
    attn_cfg = make_attention_config(base_tr, attention_kind, config, seed)
    ytr_onehot = one_hot(bundle.ytr, bundle.num_classes)
    yhat_tr = np.zeros_like(ytr_onehot)
    yhat_te = np.zeros((len(bundle.yte), bundle.num_classes), dtype=np.float64)

    layers = []
    output_param_count = 0
    hidden_param_count = 0
    layer_states = []

    start = time.perf_counter()
    for layer_idx in range(config.depth):
        pooled_tr = base_tr.mean(axis=1)
        pooled_te = base_te.mean(axis=1)
        out_map = ridge_regression(pooled_tr, ytr_onehot - yhat_tr, reg=config.head_reg)
        output_param_count += int(out_map.size)
        yhat_tr = yhat_tr + pooled_tr @ out_map
        yhat_te = yhat_te + pooled_te @ out_map

        att_model = tcc.fit_attention_block(attn_cfg, view1_tr, view2_tr)
        hidden_param_count += int(att_model["parameter_count"])
        base_tr = tcc.apply_attention_block(base_tr, attn_cfg, att_model)
        base_te = tcc.apply_attention_block(base_te, attn_cfg, att_model)
        view1_tr = tcc.apply_attention_block(view1_tr, attn_cfg, att_model)
        view2_tr = tcc.apply_attention_block(view2_tr, attn_cfg, att_model)
        view1_te = tcc.apply_attention_block(view1_te, attn_cfg, att_model)
        view2_te = tcc.apply_attention_block(view2_te, attn_cfg, att_model)

        flat1 = view1_tr.reshape(-1, view1_tr.shape[-1])
        flat2 = view2_tr.reshape(-1, view2_tr.shape[-1])
        ffn_model = cfbt.fit_layer(flat1, flat2, lambda_reg=config.lambda_reg)
        hidden_param_count += int(ffn_model["transform_base"].size)

        def apply_ffn(tokens):
            flat = tokens.reshape(-1, tokens.shape[-1])
            ffn = cfbt.apply_activation(flat @ ffn_model["transform_base"], "relu")
            return tcc.token_layer_norm(tokens + ffn.reshape(tokens.shape))

        base_tr = apply_ffn(base_tr)
        base_te = apply_ffn(base_te)
        view1_tr = apply_ffn(view1_tr)
        view2_tr = apply_ffn(view2_tr)
        view1_te = apply_ffn(view1_te)
        view2_te = apply_ffn(view2_te)
        layer_states.append(
            {
                "out_map": out_map,
                "att_model": att_model,
                "ffn_model": ffn_model,
            }
        )

        layers.append(
            {
                "depth": layer_idx + 1,
                "classifier_accuracy": evaluate_logits(yhat_te, bundle.yte),
                "attention_kind": attention_kind,
                "attention_rank": int(att_model.get("landmark_count", att_model.get("projection_rank", config.num_landmarks))),
                "train_fit_loss": float(att_model["train_fit_loss"]) if "train_fit_loss" in att_model else None,
            }
        )
    fit_time = time.perf_counter() - start

    analysis = None
    if collect_analysis:
        def encode_raw(raw_inputs):
            tokens = build_base_transformer_tokens(bundle, raw_inputs, config.patch_size)
            for state in layer_states:
                tokens = tcc.apply_attention_block(tokens, attn_cfg, state["att_model"])
                flat = tokens.reshape(-1, tokens.shape[-1])
                ffn = cfbt.apply_activation(flat @ state["ffn_model"]["transform_base"], "relu")
                tokens = tcc.token_layer_norm(tokens + ffn.reshape(tokens.shape))
            return tokens

        def predict_logits(raw_inputs):
            tokens = build_base_transformer_tokens(bundle, raw_inputs, config.patch_size)
            logits = np.zeros((tokens.shape[0], bundle.num_classes), dtype=np.float64)
            for state in layer_states:
                pooled = tokens.mean(axis=1)
                logits = logits + pooled @ state["out_map"]
                tokens = tcc.apply_attention_block(tokens, attn_cfg, state["att_model"])
                flat = tokens.reshape(-1, tokens.shape[-1])
                ffn = cfbt.apply_activation(flat @ state["ffn_model"]["transform_base"], "relu")
                tokens = tcc.token_layer_norm(tokens + ffn.reshape(tokens.shape))
            return logits

        ood_results = []
        for shift_name, shifted_inputs in make_ood_variants(bundle, seed).items():
            logits = predict_logits(shifted_inputs)
            ood_results.append({"shift": shift_name, "accuracy": evaluate_logits(logits, bundle.yte)})

        train_repr = encode_raw(bundle.train_raw).mean(axis=1)
        test_repr = encode_raw(bundle.test_raw).mean(axis=1)
        analysis = {
            "ood_results": ood_results,
            "interpretability": {
                "effective_rank": effective_rank(test_repr),
                "centroid_margin": centroid_margin(train_repr, bundle.ytr, test_repr, bundle.yte),
                **occlusion_summary(bundle, bundle.test_raw, predict_logits, patch_size=max(1, config.patch_size)),
            },
        }

    return {
        "architecture": "transformer",
        "model": "closed-form",
        "dataset": bundle.name,
        "suite": bundle.suite,
        "seed": seed,
        "closed_form_method": attention_kind,
        "classifier_accuracy": evaluate_logits(yhat_te, bundle.yte),
        "layers": layers,
        "hidden_param_count": hidden_param_count,
        "output_param_count": output_param_count,
        "total_parameter_count": hidden_param_count + output_param_count,
        "fit_time_sec": fit_time,
        "config": asdict(config),
        "analysis": analysis,
    }


class TokenTransformerClassifier(nn.Module):
    def __init__(self, token_dim, num_heads, depth, num_classes, mlp_ratio):
        super().__init__()
        hidden_ffn = max(int(token_dim * mlp_ratio), token_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=hidden_ffn,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(depth)])
        self.output_heads = nn.ModuleList([nn.Linear(token_dim, num_classes) for _ in range(depth)])

    def encode(self, tokens):
        hidden = tokens
        for block in self.blocks:
            hidden = block(hidden)
        return hidden

    def forward(self, tokens):
        cumulative = None
        depth_logits = []
        hidden = tokens
        for block, head in zip(self.blocks, self.output_heads):
            hidden = block(hidden)
            pooled = hidden.mean(dim=1)
            logits = head(pooled)
            cumulative = logits if cumulative is None else cumulative + logits
            depth_logits.append(cumulative)
        return depth_logits


def run_backprop_transformer(bundle: RawDatasetBundle, config: TransformerEvalConfig, seed: int, collect_analysis: bool = False):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if bundle.modality == "image":
        train_raw = torch.from_numpy(bundle.train_raw).float()
        test_raw = torch.from_numpy(bundle.test_raw).float()
        mean_image = torch.from_numpy(bundle.image_mean).float().to(device)
        token_dim = int(bundle.train_raw.shape[1] * config.patch_size * config.patch_size)
        num_heads = choose_num_heads(token_dim, preferred=config.num_heads)
        train_ds = TensorDataset(train_raw, torch.from_numpy(bundle.ytr).long())

        def to_tokens(xb, augment):
            xb = xb.to(device)
            if augment:
                xb = tcc.augment_torch(xb, bundle.suite)
            return image_tokens_from_torch(xb, mean_image, config.patch_size)

    elif bundle.modality == "text":
        train_raw = torch.from_numpy(bundle.train_raw).long()
        test_raw = torch.from_numpy(bundle.test_raw).long()
        embedding = torch.from_numpy(bundle.text_embedding).float().to(device)
        pos = torch.from_numpy(bundle.text_pos).float().to(device)
        token_dim = int(bundle.text_embedding.shape[1])
        num_heads = choose_num_heads(token_dim, preferred=config.num_heads)
        train_ds = TensorDataset(train_raw, torch.from_numpy(bundle.ytr).long())

        def to_tokens(xb, augment):
            xb = xb.to(device)
            if augment:
                xb = augment_text_ids_torch(xb, bundle.text_drop_prob)
            return text_tokens_from_torch(xb, embedding, pos)

    elif bundle.modality == "tabular":
        train_raw = torch.from_numpy(bundle.train_raw).float()
        test_raw = torch.from_numpy(bundle.test_raw).float()
        embedding = torch.from_numpy(bundle.tabular_token_embedding).float().to(device)
        pos = torch.from_numpy(bundle.tabular_pos).float().to(device)
        token_dim = int(bundle.tabular_token_embedding.shape[1])
        num_heads = choose_num_heads(token_dim, preferred=config.num_heads)
        train_ds = TensorDataset(train_raw, torch.from_numpy(bundle.ytr).long())

        def to_tokens(xb, augment):
            xb = xb.to(device)
            if augment:
                xb = augment_tabular_torch(xb, bundle.tabular_drop_prob, bundle.tabular_noise_std)
            return tabular_tokens_from_torch(xb, embedding, pos)

    else:
        raise ValueError(f"Unsupported modality: {bundle.modality}")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
    model = TokenTransformerClassifier(token_dim, num_heads, config.depth, bundle.num_classes, config.mlp_ratio).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    start = time.perf_counter()
    epoch_stats = []
    for _ in range(config.epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            tokens = to_tokens(xb, augment=True)
            yb = yb.to(device)
            depth_logits = model(tokens)
            loss = sum(criterion(logits, yb) for logits in depth_logits) / len(depth_logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_stats.append({"loss": float(np.mean(losses))})
    fit_time = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        tokens_te = to_tokens(test_raw, augment=False)
        depth_logits = model(tokens_te)
        pred = depth_logits[-1].argmax(dim=1).cpu().numpy()
        layers = []
        for depth_idx, logits in enumerate(depth_logits, start=1):
            layers.append(
                {
                    "depth": depth_idx,
                    "classifier_accuracy": float((logits.argmax(dim=1).cpu().numpy() == bundle.yte).mean()),
                }
            )

    analysis = None
    if collect_analysis:
        if bundle.modality == "image":
            def raw_to_tokens(raw_inputs):
                tensor = torch.from_numpy(raw_inputs).float().to(device)
                return image_tokens_from_torch(tensor, mean_image, config.patch_size)

        elif bundle.modality == "text":
            def raw_to_tokens(raw_inputs):
                tensor = torch.from_numpy(raw_inputs).long().to(device)
                return text_tokens_from_torch(tensor, embedding, pos)

        else:
            def raw_to_tokens(raw_inputs):
                tensor = torch.from_numpy(raw_inputs).float().to(device)
                return tabular_tokens_from_torch(tensor, embedding, pos)

        def predict_logits(raw_inputs):
            with torch.no_grad():
                return model(raw_to_tokens(raw_inputs))[-1].cpu().numpy()

        def encode_repr(raw_inputs):
            with torch.no_grad():
                return model.encode(raw_to_tokens(raw_inputs)).mean(dim=1).cpu().numpy().astype(np.float64)

        ood_results = []
        for shift_name, shifted_inputs in make_ood_variants(bundle, seed).items():
            logits = predict_logits(shifted_inputs)
            ood_results.append({"shift": shift_name, "accuracy": evaluate_logits(logits, bundle.yte)})

        train_repr = encode_repr(bundle.train_raw)
        test_repr = encode_repr(bundle.test_raw)
        analysis = {
            "ood_results": ood_results,
            "interpretability": {
                "effective_rank": effective_rank(test_repr),
                "centroid_margin": centroid_margin(train_repr, bundle.ytr, test_repr, bundle.yte),
                **occlusion_summary(bundle, bundle.test_raw, predict_logits, patch_size=max(1, config.patch_size)),
            },
        }

    return {
        "architecture": "transformer",
        "model": "backprop",
        "dataset": bundle.name,
        "suite": bundle.suite,
        "seed": seed,
        "classifier_accuracy": float((pred == bundle.yte).mean()),
        "layers": layers,
        "hidden_param_count": int(sum(p.numel() for p in model.blocks.parameters())),
        "output_param_count": int(sum(p.numel() for p in model.output_heads.parameters())),
        "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "fit_time_sec": fit_time,
        "epoch_stats": epoch_stats,
        "config": asdict(config),
        "analysis": analysis,
    }


def aggregate_by_method(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["closed_form_method"]].append(row)

    dataset_seed_best = {}
    for row in rows:
        key = (row["dataset"], row["seed"])
        dataset_seed_best[key] = max(dataset_seed_best.get(key, 0.0), row["classifier_accuracy"])

    summary = []
    for method, method_rows in grouped.items():
        per_dataset_mean = {}
        per_dataset_std = {}
        per_dataset_gap = {}
        wins = 0
        datasets = sorted({row["dataset"] for row in method_rows})
        for dataset in datasets:
            vals = [row["classifier_accuracy"] for row in method_rows if row["dataset"] == dataset]
            per_dataset_mean[dataset], per_dataset_std[dataset] = mean_std(vals)
            dataset_best = max(
                np.mean([r["classifier_accuracy"] for r in rows if r["dataset"] == dataset and r["closed_form_method"] == candidate])
                for candidate in {r["closed_form_method"] for r in rows if r["dataset"] == dataset}
            )
            per_dataset_gap[dataset] = float(dataset_best - per_dataset_mean[dataset])
        for row in method_rows:
            if abs(row["classifier_accuracy"] - dataset_seed_best[(row["dataset"], row["seed"])]) <= 1e-9:
                wins += 1
        avg_accuracy = float(np.mean(list(per_dataset_mean.values())))
        avg_seed_std = float(np.mean(list(per_dataset_std.values())))
        avg_gap = float(np.mean(list(per_dataset_gap.values())))
        summary.append(
            {
                "closed_form_method": method,
                "avg_dataset_accuracy": avg_accuracy,
                "avg_seed_std": avg_seed_std,
                "avg_gap_to_dataset_best": avg_gap,
                "win_rate": float(wins / max(len(method_rows), 1)),
                "selection_score": avg_accuracy - avg_seed_std - avg_gap,
                "dataset_mean_accuracy": per_dataset_mean,
                "dataset_seed_std": per_dataset_std,
                "dataset_gap_to_best": per_dataset_gap,
            }
        )
    summary.sort(key=lambda row: row["selection_score"], reverse=True)
    return summary


def infer_winners_from_existing_logs():
    base = default_json_path("placeholder.json").parent
    mlp_scores = defaultdict(list)
    transformer_scores = defaultdict(list)

    for path in base.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        rows = []
        if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
            rows = data["results"]
        elif isinstance(data, dict) and "classifier_accuracy" in data:
            rows = [data]

        for row in rows:
            method = row.get("layer_method")
            acc = row.get("classifier_accuracy")
            if method in MLP_CANDIDATES and acc is not None:
                mlp_scores[method].append(float(acc))

            model_name = row.get("model", "")
            if isinstance(model_name, str) and model_name.startswith("closed-form-transformer:") and acc is not None:
                kind = model_name.split(":", 1)[1]
                if kind in TRANSFORMER_CANDIDATES:
                    transformer_scores[kind].append(float(acc))

    def choose(scores, fallback):
        filtered = {name: vals for name, vals in scores.items() if len(vals) >= 2}
        source = filtered if filtered else scores
        if not source:
            return fallback, []
        summary = [
            {
                "method": name,
                "mean_accuracy": float(np.mean(vals)),
                "std_accuracy": float(np.std(vals, ddof=0)),
                "best_accuracy": float(np.max(vals)),
                "num_logs": len(vals),
                # Favor strong average performance, lightly penalize volatility, and
                # reward methods with broader empirical support from prior runs.
                "selection_score": float(np.mean(vals) - 0.25 * np.std(vals, ddof=0) + 0.005 * np.log1p(len(vals))),
            }
            for name, vals in sorted(source.items())
        ]
        summary.sort(
            key=lambda row: (
                row["selection_score"],
                row["mean_accuracy"],
                row["best_accuracy"],
                row["num_logs"],
            ),
            reverse=True,
        )
        winner = summary[0]["method"]
        return winner, summary

    mlp_winner, mlp_summary = choose(mlp_scores, DEFAULT_MLP_WINNER)
    transformer_winner, transformer_summary = choose(transformer_scores, DEFAULT_TRANSFORMER_WINNER)
    return {
        "mlp_winner": mlp_winner,
        "transformer_winner": transformer_winner,
        "mlp_log_summary": mlp_summary,
        "transformer_log_summary": transformer_summary,
    }


def summarize_main_results(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["architecture"], row["model"], row["dataset"])].append(row["classifier_accuracy"])

    summary = []
    for (architecture, model, dataset), vals in sorted(grouped.items()):
        summary.append(
            {
                "architecture": architecture,
                "model": model,
                "dataset": dataset,
                "mean_accuracy": float(np.mean(vals)),
                "std_accuracy": float(np.std(vals, ddof=0)),
                "ci95_accuracy": confidence_interval(vals),
                "runs": len(vals),
            }
        )
    return summary


def build_analytics_tables(rows):
    run_table = []
    ood_table = []
    for row in rows:
        run_id = f"{row['architecture']}::{row['model']}::{row['dataset']}::{row['seed']}"
        analysis = row.get("analysis") or {}
        interpretability = analysis.get("interpretability") or {}
        run_table.append(
            {
                "run_id": run_id,
                "architecture": row["architecture"],
                "model": row["model"],
                "dataset": row["dataset"],
                "suite": row["suite"],
                "seed": row["seed"],
                "classifier_accuracy": row["classifier_accuracy"],
                "total_parameter_count": row["total_parameter_count"],
                "fit_time_sec": row["fit_time_sec"],
                "effective_rank": interpretability.get("effective_rank"),
                "centroid_margin": interpretability.get("centroid_margin"),
                "top1_importance_fraction": interpretability.get("top1_importance_fraction"),
                "top20_importance_fraction": interpretability.get("top20_importance_fraction"),
                "top20_deletion_logit_drop": interpretability.get("top20_deletion_logit_drop"),
            }
        )
        for item in analysis.get("ood_results") or []:
            ood_table.append(
                {
                    "run_id": run_id,
                    "architecture": row["architecture"],
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "seed": row["seed"],
                    "shift": item["shift"],
                    "accuracy": item["accuracy"],
                }
            )
    return {
        "run_table": run_table,
        "ood_table": ood_table,
    }


def fit_power_law(x_values, y_values):
    xs = np.asarray(x_values, dtype=np.float64)
    ys = np.asarray(y_values, dtype=np.float64)
    mask = (xs > 0.0) & (ys > 0.0)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 3:
        return None
    log_x = np.log(xs)
    log_y = np.log(ys)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    pred = slope * log_x + intercept
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "x_min": float(xs.min()),
        "x_max": float(xs.max()),
        "range_ratio": float(xs.max() / xs.min()),
    }


def summarize_scaling_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["architecture"], row["model"], row["axis"], row["scale_value"])].append(row)

    aggregated = []
    for (architecture, model, axis, scale_value), items in sorted(grouped.items()):
        accs = [item["classifier_accuracy"] for item in items]
        errs = [1.0 - item["classifier_accuracy"] for item in items]
        params = [item["total_parameter_count"] for item in items]
        times = [item["fit_time_sec"] for item in items]
        aggregated.append(
            {
                "architecture": architecture,
                "model": model,
                "axis": axis,
                "scale_value": scale_value,
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs, ddof=0)),
                "mean_error": float(np.mean(errs)),
                "mean_parameter_count": float(np.mean(params)),
                "mean_fit_time_sec": float(np.mean(times)),
            }
        )

    fits = []
    for architecture in sorted({row["architecture"] for row in aggregated}):
        for model in sorted({row["model"] for row in aggregated if row["architecture"] == architecture}):
            for axis in sorted({row["axis"] for row in aggregated if row["architecture"] == architecture and row["model"] == model}):
                rows_axis = [row for row in aggregated if row["architecture"] == architecture and row["model"] == model and row["axis"] == axis]
                rows_axis.sort(key=lambda row: row["scale_value"])
                if axis == "data":
                    x_vals = [row["scale_value"] for row in rows_axis]
                elif axis == "parameters":
                    x_vals = [row["mean_parameter_count"] for row in rows_axis]
                elif axis == "compute":
                    x_vals = [row["mean_fit_time_sec"] for row in rows_axis]
                else:
                    continue
                y_vals = [row["mean_error"] for row in rows_axis]
                fit = fit_power_law(x_vals, y_vals)
                if fit is None:
                    continue
                fits.append(
                    {
                        "architecture": architecture,
                        "model": model,
                        "axis": axis,
                        **fit,
                    }
                )
    return aggregated, fits


def plot_main_results(summary_rows, output_path):
    plt = maybe_import_pyplot()
    if plt is None:
        return False
    datasets = sorted({row["dataset"] for row in summary_rows})
    model_order = [(row["architecture"], row["model"]) for row in summary_rows if row["dataset"] == datasets[0]]
    x = np.arange(len(datasets))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for idx, (architecture, model) in enumerate(model_order):
        means = []
        cis = []
        for dataset in datasets:
            row = next(
                item
                for item in summary_rows
                if item["architecture"] == architecture and item["model"] == model and item["dataset"] == dataset
            )
            means.append(row["mean_accuracy"])
            cis.append(row["ci95_accuracy"])
        ax.bar(x + (idx - 1.5) * width, means, width=width, yerr=cis, label=f"{architecture}-{model}", capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Accuracy")
    ax.set_title("Backprop vs closed-form across datasets")
    ax.set_ylim(0.0, min(1.0, max(row["mean_accuracy"] + row["ci95_accuracy"] for row in summary_rows) + 0.08))
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def plot_scaling_results(aggregated_rows, fit_rows, output_path):
    plt = maybe_import_pyplot()
    if plt is None:
        return False
    axes_order = ["data", "parameters", "compute"]
    architectures = ["mlp", "transformer"]
    fig, axs = plt.subplots(len(architectures), len(axes_order), figsize=(12, 7))

    for row_idx, architecture in enumerate(architectures):
        for col_idx, axis_name in enumerate(axes_order):
            ax = axs[row_idx, col_idx]
            subset = [row for row in aggregated_rows if row["architecture"] == architecture and row["axis"] == axis_name]
            fit_subset = [row for row in fit_rows if row["architecture"] == architecture and row["axis"] == axis_name]
            for model in sorted({row["model"] for row in subset}):
                rows_model = [row for row in subset if row["model"] == model]
                if axis_name == "data":
                    x_vals = [row["scale_value"] for row in rows_model]
                elif axis_name == "parameters":
                    x_vals = [row["mean_parameter_count"] for row in rows_model]
                else:
                    x_vals = [row["mean_fit_time_sec"] for row in rows_model]
                y_vals = [row["mean_error"] for row in rows_model]
                ax.plot(x_vals, y_vals, marker="o", label=model)
            for fit_idx, fit in enumerate(fit_subset):
                ax.text(
                    0.02,
                    0.95 - 0.1 * fit_idx,
                    f"{fit['model']}: slope={fit['slope']:.2f}, R2={fit['r2']:.2f}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{architecture} {axis_name}")
            ax.set_xlabel(axis_name)
            ax.set_ylabel("Error")
            ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def run_closed_form_selection(dataset_specs, mlp_config, transformer_config):
    mlp_rows = []
    transformer_rows = []
    for spec in dataset_specs:
        bundle = load_dataset_bundle(spec)
        for method in MLP_CANDIDATES:
            mlp_rows.append(run_closed_form_mlp(bundle, method, mlp_config, spec.seed))
        for method in TRANSFORMER_CANDIDATES:
            transformer_rows.append(run_closed_form_transformer(bundle, method, transformer_config, spec.seed))
    return mlp_rows, transformer_rows


def run_main_comparison(dataset_specs, mlp_config, transformer_config, mlp_winner, transformer_winner):
    results = []
    for spec in dataset_specs:
        bundle = load_dataset_bundle(spec)
        results.append(run_closed_form_mlp(bundle, mlp_winner, mlp_config, spec.seed, collect_analysis=True))
        results.append(run_backprop_mlp(bundle, mlp_config, spec.seed, collect_analysis=True))
        results.append(run_closed_form_transformer(bundle, transformer_winner, transformer_config, spec.seed, collect_analysis=True))
        results.append(run_backprop_transformer(bundle, transformer_config, spec.seed, collect_analysis=True))
    return results


def scaling_specs_for_axis(dataset_name, axis_name, seeds):
    base = DATASET_REGISTRY[dataset_name]
    specs = []
    if axis_name == "data":
        for seed in seeds:
            for value in SCALING_DATA_VALUES:
                specs.append(
                    DatasetSpec(
                        name=dataset_name,
                        suite=base["suite"],
                        n_train=value,
                        n_test=base["n_test"],
                        seed=seed,
                    )
                )
        return SCALING_DATA_VALUES, specs

    if axis_name in {"parameters", "compute"}:
        for seed in seeds:
            specs.append(
                DatasetSpec(
                    name=dataset_name,
                    suite=base["suite"],
                    n_train=base["n_train"],
                    n_test=base["n_test"],
                    seed=seed,
                )
            )
        if axis_name == "parameters":
            return {
                "mlp": SCALING_MLP_WIDTHS,
                "transformer": SCALING_TRANSFORMER_PATCH_SIZES,
            }, specs
        return {
            "mlp": SCALING_MLP_COMPUTE_DEPTHS,
            "transformer": SCALING_TRANSFORMER_COMPUTE_DEPTHS,
        }, specs

    raise ValueError(f"Unknown scaling axis: {axis_name}")


def run_scaling_suite(dataset_name, mlp_winner, transformer_winner, mlp_config, transformer_config):
    rows = []

    for axis_name in ["data", "parameters", "compute"]:
        scale_values, dataset_specs = scaling_specs_for_axis(dataset_name, axis_name, SCALING_SEEDS)
        if axis_name == "data":
            for spec in dataset_specs:
                bundle = load_dataset_bundle(spec)
                mlp_row = run_closed_form_mlp(bundle, mlp_winner, mlp_config, spec.seed)
                mlp_row["axis"] = axis_name
                mlp_row["scale_value"] = spec.n_train
                rows.append(mlp_row)
                mlp_bp = run_backprop_mlp(bundle, mlp_config, spec.seed)
                mlp_bp["axis"] = axis_name
                mlp_bp["scale_value"] = spec.n_train
                rows.append(mlp_bp)

                tr_row = run_closed_form_transformer(bundle, transformer_winner, transformer_config, spec.seed)
                tr_row["axis"] = axis_name
                tr_row["scale_value"] = spec.n_train
                rows.append(tr_row)
                tr_bp = run_backprop_transformer(bundle, transformer_config, spec.seed)
                tr_bp["axis"] = axis_name
                tr_bp["scale_value"] = spec.n_train
                rows.append(tr_bp)

        elif axis_name == "parameters":
            for spec in dataset_specs:
                bundle = load_dataset_bundle(spec)
                for width in scale_values["mlp"]:
                    mlp_cfg = MLPEvalConfig(**{**asdict(mlp_config), "width": width})
                    mlp_row = run_closed_form_mlp(bundle, mlp_winner, mlp_cfg, spec.seed)
                    mlp_row["axis"] = axis_name
                    mlp_row["scale_value"] = width
                    rows.append(mlp_row)
                    mlp_bp = run_backprop_mlp(bundle, mlp_cfg, spec.seed)
                    mlp_bp["axis"] = axis_name
                    mlp_bp["scale_value"] = width
                    rows.append(mlp_bp)

                for patch_size in scale_values["transformer"]:
                    tr_cfg = TransformerEvalConfig(**{**asdict(transformer_config), "patch_size": patch_size})
                    tr_row = run_closed_form_transformer(bundle, transformer_winner, tr_cfg, spec.seed)
                    tr_row["axis"] = axis_name
                    tr_row["scale_value"] = patch_size
                    rows.append(tr_row)
                    tr_bp = run_backprop_transformer(bundle, tr_cfg, spec.seed)
                    tr_bp["axis"] = axis_name
                    tr_bp["scale_value"] = patch_size
                    rows.append(tr_bp)

        elif axis_name == "compute":
            for spec in dataset_specs:
                bundle = load_dataset_bundle(spec)
                for depth in scale_values["mlp"]:
                    mlp_cfg = MLPEvalConfig(**{**asdict(mlp_config), "depth": depth})
                    mlp_row = run_closed_form_mlp(bundle, mlp_winner, mlp_cfg, spec.seed)
                    mlp_row["axis"] = axis_name
                    mlp_row["scale_value"] = depth
                    rows.append(mlp_row)
                    mlp_bp = run_backprop_mlp(bundle, mlp_cfg, spec.seed)
                    mlp_bp["axis"] = axis_name
                    mlp_bp["scale_value"] = depth
                    rows.append(mlp_bp)

                for depth in scale_values["transformer"]:
                    tr_cfg = TransformerEvalConfig(**{**asdict(transformer_config), "depth": depth})
                    tr_row = run_closed_form_transformer(bundle, transformer_winner, tr_cfg, spec.seed)
                    tr_row["axis"] = axis_name
                    tr_row["scale_value"] = depth
                    rows.append(tr_row)
                    tr_bp = run_backprop_transformer(bundle, tr_cfg, spec.seed)
                    tr_bp["axis"] = axis_name
                    tr_bp["scale_value"] = depth
                    rows.append(tr_bp)

    return rows


def build_dataset_specs(dataset_names, seeds):
    specs = []
    for seed in seeds:
        for dataset_name in dataset_names:
            base = DATASET_REGISTRY[dataset_name]
            specs.append(
                DatasetSpec(
                    name=dataset_name,
                    suite=base["suite"],
                    n_train=base["n_train"],
                    n_test=base["n_test"],
                    seed=seed,
                )
            )
    return specs


def main():
    supported_datasets = sorted(set(DEFAULT_DATASETS + ["mnist", "fashion_mnist", "cifar10"]))
    parser = argparse.ArgumentParser(description="Broader closed-form vs backprop evaluation suite.")
    parser.add_argument("--datasets", nargs="+", choices=supported_datasets + ["all"], default=["all"])
    parser.add_argument("--run-selection", action="store_true")
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--skip-scaling", action="store_true")
    parser.add_argument("--reuse-winners-from", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    dataset_names = DEFAULT_DATASETS if args.datasets == ["all"] else args.datasets
    dataset_specs = build_dataset_specs(dataset_names, MAIN_SEEDS)
    mlp_config = MLPEvalConfig()
    transformer_config = TransformerEvalConfig()

    mlp_selection = []
    transformer_selection = []
    mlp_winner = None
    transformer_winner = None
    winner_source = None

    if args.reuse_winners_from is not None:
        reuse_path = resolve_json_path(args.reuse_winners_from)
        payload = json.loads(reuse_path.read_text(encoding="utf-8"))
        mlp_winner = payload["winners"]["mlp"]
        transformer_winner = payload["winners"]["transformer"]
        selection_payload = payload.get("selection") or {}
        mlp_selection = selection_payload.get("mlp") or []
        transformer_selection = selection_payload.get("transformer") or []
        winner_source = f"reuse:{reuse_path.name}"
    elif args.run_selection:
        selection_mlp_rows, selection_transformer_rows = run_closed_form_selection(dataset_specs, mlp_config, transformer_config)
        mlp_selection = aggregate_by_method(selection_mlp_rows)
        transformer_selection = aggregate_by_method(selection_transformer_rows)
        mlp_winner = mlp_selection[0]["closed_form_method"]
        transformer_winner = transformer_selection[0]["closed_form_method"]
        winner_source = "selection_runs"
    else:
        inferred = infer_winners_from_existing_logs()
        mlp_winner = inferred["mlp_winner"]
        transformer_winner = inferred["transformer_winner"]
        mlp_selection = inferred["mlp_log_summary"]
        transformer_selection = inferred["transformer_log_summary"]
        winner_source = "existing_logs"

    main_rows = []
    main_summary = []
    analytics_tables = {"run_table": [], "ood_table": []}
    if not args.skip_main:
        main_rows = run_main_comparison(
            dataset_specs,
            mlp_config,
            transformer_config,
            mlp_winner,
            transformer_winner,
        )
        main_summary = summarize_main_results(main_rows)
        analytics_tables = build_analytics_tables(main_rows)

    scaling_rows = []
    scaling_aggregated = []
    scaling_fits = []
    if not args.skip_scaling:
        scaling_rows = run_scaling_suite(
            dataset_name=DEFAULT_SCALING_DATASET,
            mlp_winner=mlp_winner,
            transformer_winner=transformer_winner,
            mlp_config=mlp_config,
            transformer_config=transformer_config,
        )
        scaling_aggregated, scaling_fits = summarize_scaling_rows(scaling_rows)

    if args.json_out is None:
        if args.skip_main and not args.skip_scaling:
            json_name = "broader_eval_suite_scaling.json"
        elif args.skip_scaling and not args.skip_main:
            json_name = "broader_eval_suite_main.json"
        else:
            json_name = "broader_eval_suite.json"
    else:
        json_name = "broader_eval_suite.json"
    json_path = default_json_path(json_name) if args.json_out is None else resolve_json_path(args.json_out)
    run_table_path = default_json_path("broader_eval_suite_run_table.jsonl")
    ood_table_path = default_json_path("broader_eval_suite_ood_table.jsonl")
    selection_table_path = default_json_path("broader_eval_suite_selection_table.jsonl")
    scaling_table_path = default_json_path("broader_eval_suite_scaling_table.jsonl")
    scaling_summary_path = default_json_path("broader_eval_suite_scaling_summary.jsonl")
    scaling_fit_path = default_json_path("broader_eval_suite_scaling_fit_table.jsonl")
    main_plot_path = default_plot_path("broader_eval_suite_summary.png")
    scaling_plot_path = default_plot_path("broader_eval_suite_scaling.png")

    main_plot_ok = bool(main_summary) and plot_main_results(main_summary, main_plot_path)
    scaling_plot_ok = bool(scaling_aggregated) and plot_scaling_results(scaling_aggregated, scaling_fits, scaling_plot_path)
    selection_table = [
        {"architecture": "mlp", "winner_source": winner_source, **row}
        for row in mlp_selection
    ] + [
        {"architecture": "transformer", "winner_source": winner_source, **row}
        for row in transformer_selection
    ]

    payload = {
        "config": {
            "datasets": dataset_names,
            "main_seeds": MAIN_SEEDS,
            "scaling_seeds": SCALING_SEEDS,
            "scaling_dataset": DEFAULT_SCALING_DATASET,
            "run_selection": args.run_selection,
            "skip_main": args.skip_main,
            "skip_scaling": args.skip_scaling,
            "winner_source": winner_source,
            "mlp_config": asdict(mlp_config),
            "transformer_config": asdict(transformer_config),
            "mlp_candidates": MLP_CANDIDATES,
            "transformer_candidates": TRANSFORMER_CANDIDATES,
        },
        "selection": {
            "mlp": mlp_selection,
            "transformer": transformer_selection,
        },
        "winners": {
            "mlp": mlp_winner,
            "transformer": transformer_winner,
        },
        "main_results": main_rows,
        "main_summary": main_summary,
        "analytics_tables": analytics_tables,
        "scaling_results": scaling_rows,
        "scaling_summary": scaling_aggregated,
        "scaling_fits": scaling_fits,
        "artifacts": {
            "run_table": str(run_table_path),
            "ood_table": str(ood_table_path),
            "selection_table": str(selection_table_path),
            "scaling_table": str(scaling_table_path),
            "scaling_summary": str(scaling_summary_path),
            "scaling_fit_table": str(scaling_fit_path),
            "main_plot": str(main_plot_path) if main_plot_ok else None,
            "scaling_plot": str(scaling_plot_path) if scaling_plot_ok else None,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if analytics_tables["run_table"]:
        run_table_path.write_text("\n".join(json.dumps(row) for row in analytics_tables["run_table"]) + "\n", encoding="utf-8")
    if analytics_tables["ood_table"]:
        ood_table_path.write_text("\n".join(json.dumps(row) for row in analytics_tables["ood_table"]) + "\n", encoding="utf-8")
    if selection_table:
        selection_table_path.write_text("\n".join(json.dumps(row) for row in selection_table) + "\n", encoding="utf-8")
    if scaling_rows:
        scaling_table_path.write_text("\n".join(json.dumps(row) for row in scaling_rows) + "\n", encoding="utf-8")
    if scaling_aggregated:
        scaling_summary_path.write_text("\n".join(json.dumps(row) for row in scaling_aggregated) + "\n", encoding="utf-8")
    if scaling_fits:
        scaling_fit_path.write_text("\n".join(json.dumps(row) for row in scaling_fits) + "\n", encoding="utf-8")

    print(f"Saved json to {json_path}")
    print(f"MLP winner: {mlp_winner}")
    print(f"Transformer winner: {transformer_winner}")
    for row in main_summary:
        print(
            f"{row['architecture']:11s} {row['model']:10s} {row['dataset']:14s} "
            f"acc={row['mean_accuracy']:.4f} +/- {row['ci95_accuracy']:.4f}"
        )
    for fit in scaling_fits:
        print(
            f"scaling {fit['architecture']:11s} {fit['model']:10s} {fit['axis']:10s} "
            f"slope={fit['slope']:.3f} R2={fit['r2']:.3f} range={fit['range_ratio']:.1f}x"
        )


if __name__ == "__main__":
    main()
