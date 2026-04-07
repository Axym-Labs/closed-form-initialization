import argparse
import copy
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
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
from project_paths import default_json_path, default_plot_path, repo_relative_path, resolve_json_path


MAIN_SEEDS = [7, 11, 19]
SCALING_SEEDS = [7, 11, 19]
SCALING_DATA_VALUES = [1000, 2000, 4000, 8000, 16000, 32000, 64000]
SCALING_TRANSFORMER_TEXT_DIMS = [16, 32, 64, 128, 192]
SCALING_TRANSFORMER_CONTEXT_LENGTHS = [8, 16, 32, 48]

MLP_CANDIDATES = [
    "closed-form-barlow",
    "paper-cca-shared",
    "whitened-shared-pca",
]
DEFAULT_MLP_WINNER = "closed-form-barlow"
DEFAULT_MLP_CONFIG_OVERRIDES = {
    "dual_mapping": True,
    "output_source": "post-hidden",
    "center_after_hidden": False,
}

TRANSFORMER_CANDIDATES = ["spectral-self"]
DEFAULT_TRANSFORMER_WINNER = "spectral-self"
DEFAULT_TRANSFORMER_CONFIG_OVERRIDES = {
    "analytic_num_heads": 2,
    "attention_target": "mean",
    "attention_power_iters": 1,
    "attention_num_bags": 1,
    "attention_bag_fraction": 1.0,
}

DEFAULT_DATASETS = ["covtype", "cifar100", "qnli", "wikitext2_next_token"]
DEFAULT_SCALING_DATASET = "wikitext2_next_token_scale"
PLOT_SUBDIR = "broader_eval_suite"
SCALING_AXES = ["data", "parameters"]
FINE_TUNE_MODEL_NAME = "closed-form+backprop-ft"
CONTEXT_LENGTH_AXIS_NAME = "compute (context-length)"

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+|[.,!?;:]")
TEXT_PAD = 0
TEXT_UNK = 1


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    suite: str
    n_train: int
    n_test: int
    seed: int
    task_type: str = "classification"
    image_size: int = 32
    text_vocab_size: int = 512
    text_seq_len: int = 32
    text_embed_dim: int = 64
    text_drop_prob: float = 0.2
    text_dataset_name: str | None = None
    text_dataset_config: str | None = None
    text_fields: tuple[str, ...] = ("text",)
    label_field: str = "label"
    eval_split: str | None = None
    next_token_stride: int = 4


@dataclass(frozen=True)
class MLPEvalConfig:
    width: int = 507
    depth: int = 3
    activation: str = "relu"
    lambda_reg: float = 1.0
    head_reg: float = 100.0
    epochs: int = 8
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dual_mapping: bool = True
    output_source: str = "post-hidden"
    center_after_hidden: bool = False


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
    task_type: str
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
    "covtype": {
        "suite": "feature-mask",
        "n_train": 12000,
        "n_test": 4000,
        "task_type": "classification",
    },
    "svhn": {
        "suite": "random-affine",
        "n_train": 6000,
        "n_test": 2000,
        "task_type": "classification",
    },
    "cifar100": {
        "suite": "random-affine",
        "n_train": 8000,
        "n_test": 2000,
        "task_type": "classification",
    },
    "qnli": {
        "suite": "token-mask",
        "n_train": 10000,
        "n_test": 3000,
        "task_type": "classification",
        "text_dataset_name": "glue",
        "text_dataset_config": "qnli",
        "text_fields": ("question", "sentence"),
        "label_field": "label",
        "eval_split": "validation",
        "text_vocab_size": 8192,
        "text_seq_len": 80,
        "text_embed_dim": 64,
        "text_drop_prob": 0.12,
    },
    "wikitext2_next_token": {
        "suite": "token-mask",
        "n_train": 16000,
        "n_test": 3000,
        "task_type": "next_token",
        "text_dataset_name": "wikitext",
        "text_dataset_config": "wikitext-2-raw-v1",
        "text_fields": ("text",),
        "text_vocab_size": 4096,
        "text_seq_len": 64,
        "text_embed_dim": 64,
        "text_drop_prob": 0.1,
        "next_token_stride": 3,
    },
    "wikitext2_next_token_scale": {
        "suite": "token-mask",
        "n_train": 16000,
        "n_test": 8000,
        "task_type": "next_token",
        "text_dataset_name": "wikitext",
        "text_dataset_config": "wikitext-2-raw-v1",
        "text_fields": ("text",),
        "text_vocab_size": 4096,
        "text_seq_len": 32,
        "text_embed_dim": 64,
        "text_drop_prob": 0.1,
        "next_token_stride": 3,
    },
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def one_hot(y, num_classes, dtype=np.float64):
    eye = np.eye(num_classes, dtype=dtype)
    return eye[y]


def ridge_regression(X, Y, reg):
    gram = X.T @ X
    rhs = X.T @ Y
    return np.linalg.solve(gram + reg * np.eye(gram.shape[0], dtype=gram.dtype), rhs)


def evaluate_logits(logits, y):
    return float((np.argmax(logits, axis=1) == y).mean())


def evaluate_cross_entropy(logits, y):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-np.mean(log_probs[np.arange(labels.shape[0]), labels]))


def fit_logit_temperature(logits, y, max_samples=4096, seed=0):
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(y, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[0] != labels.shape[0] or logits.shape[0] == 0:
        return 1.0
    if logits.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(logits.shape[0], size=max_samples, replace=False)
        logits = logits[idx]
        labels = labels[idx]
    logits_t = torch.from_numpy(logits)
    labels_t = torch.from_numpy(labels)
    log_scale = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_scale], lr=0.5, max_iter=25, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits_t * torch.exp(log_scale), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_scale.detach()).cpu().item())


def scaling_metrics_from_logits(logits, y, task_type):
    accuracy = evaluate_logits(logits, y)
    metrics = {
        "classifier_accuracy": accuracy,
        "scaling_metric_name": "error_rate",
        "scaling_metric_label": "Error rate",
        "scaling_metric_value": float(1.0 - accuracy),
        "scaling_fit_type": "power-law",
    }
    if task_type == "next_token":
        cross_entropy = evaluate_cross_entropy(logits, y)
        metrics.update(
            {
                "validation_cross_entropy": cross_entropy,
                "validation_perplexity": float(math.exp(min(cross_entropy, 30.0))),
                "scaling_metric_name": "validation_cross_entropy",
                "scaling_metric_label": "Validation cross-entropy",
                "scaling_metric_value": cross_entropy,
                "scaling_fit_type": "log-linear",
            }
        )
    return metrics


def mean_std(values):
    vals = np.asarray(values, dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=0))


def confidence_interval(values):
    vals = np.asarray(values, dtype=np.float64)
    if vals.size <= 1:
        return 0.0
    return float(1.96 * vals.std(ddof=1) / np.sqrt(vals.size))


def make_train_loader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def compute_matched_step_budget(total_budget, init_budget, finetune_epoch_budget, steps_per_epoch, max_epochs=None):
    if steps_per_epoch <= 0 or finetune_epoch_budget <= 0.0:
        return 0, 0.0
    total_budget = float(max(total_budget, 0.0))
    remaining = float(max(total_budget - max(init_budget, 0.0), 0.0))
    if remaining <= 0.0:
        return 0, float(max(init_budget, 0.0))
    step_budget = finetune_epoch_budget / steps_per_epoch
    max_steps = None if max_epochs is None else steps_per_epoch * max_epochs
    budget_limited_steps = int(math.floor(remaining / max(step_budget, 1e-12)))
    used_steps = budget_limited_steps if max_steps is None else min(max_steps, budget_limited_steps)
    used_budget = float(init_budget + used_steps * step_budget)
    return used_steps, used_budget


def run_training_steps(train_loader, total_steps, step_fn):
    if total_steps <= 0:
        return []
    epoch_stats = []
    epoch_losses = []
    steps_per_epoch = max(len(train_loader), 1)
    iterator = iter(train_loader)
    for step_idx in range(total_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        loss_value = float(step_fn(batch))
        epoch_losses.append(loss_value)
        if (step_idx + 1) % steps_per_epoch == 0:
            epoch_stats.append({"loss": float(np.mean(epoch_losses)), "steps": len(epoch_losses)})
            epoch_losses = []
    if epoch_losses:
        epoch_stats.append({"loss": float(np.mean(epoch_losses)), "steps": len(epoch_losses)})
    return epoch_stats


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


def tokenize_text(text):
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def join_text_fields(record, text_fields):
    parts = []
    for field in text_fields:
        value = record[field]
        if value:
            parts.append(str(value))
    return " ".join(parts)


def build_vocab_from_texts(texts, vocab_size):
    counter = Counter()
    for text in texts:
        counter.update(tokenize_text(text))
    most_common = [token for token, _ in counter.most_common(max(0, vocab_size - 2))]
    return {token: idx + 2 for idx, token in enumerate(most_common)}


def encode_texts_to_ids(texts, vocab, seq_len):
    encoded = np.zeros((len(texts), seq_len), dtype=np.int64)
    for row_idx, text in enumerate(texts):
        tokens = [vocab.get(token, TEXT_UNK) for token in tokenize_text(text)]
        tokens = tokens[:seq_len]
        if tokens:
            encoded[row_idx, : len(tokens)] = tokens
    return encoded


@lru_cache(maxsize=8)
def load_text_classification_tokenized(dataset_name, dataset_config, text_fields, label_field, eval_split, vocab_size, seq_len):
    dataset = load_dataset(dataset_name, dataset_config) if dataset_config is not None else load_dataset(dataset_name)
    train_texts = [join_text_fields(record, text_fields) for record in dataset["train"]]
    resolved_eval_split = eval_split
    if resolved_eval_split is None:
        resolved_eval_split = "test" if "test" in dataset and label_field in dataset["test"].column_names else "validation"
    test_texts = [join_text_fields(record, text_fields) for record in dataset[resolved_eval_split]]
    train_labels = np.asarray(dataset["train"][label_field], dtype=np.int64)
    test_labels = np.asarray(dataset[resolved_eval_split][label_field], dtype=np.int64)
    vocab = build_vocab_from_texts(train_texts, vocab_size)
    return {
        "train_ids": encode_texts_to_ids(train_texts, vocab, seq_len),
        "test_ids": encode_texts_to_ids(test_texts, vocab, seq_len),
        "train_labels": train_labels,
        "test_labels": test_labels,
    }


@lru_cache(maxsize=4)
def load_next_token_text_tokenized(dataset_name, dataset_config, text_fields, vocab_size, seq_len, stride):
    dataset = load_dataset(dataset_name, dataset_config) if dataset_config is not None else load_dataset(dataset_name)
    train_texts = [join_text_fields(record, text_fields) for record in dataset["train"]]
    val_split = "validation" if "validation" in dataset else "test"
    eval_texts = [join_text_fields(record, text_fields) for record in dataset[val_split]]
    vocab = build_vocab_from_texts(train_texts, vocab_size)

    def encode_stream(texts):
        stream = []
        for text in texts:
            stream.extend(vocab.get(token, TEXT_UNK) for token in tokenize_text(text))
        return np.asarray(stream, dtype=np.int64)

    def make_examples(stream):
        contexts = []
        labels = []
        max_start = max(0, len(stream) - seq_len - 1)
        for start in range(0, max_start, stride):
            window = stream[start : start + seq_len + 1]
            if len(window) < seq_len + 1:
                break
            contexts.append(window[:seq_len])
            labels.append(window[seq_len])
        return np.asarray(contexts, dtype=np.int64), np.asarray(labels, dtype=np.int64)

    train_stream = encode_stream(train_texts)
    eval_stream = encode_stream(eval_texts)
    train_ids, train_labels = make_examples(train_stream)
    test_ids, test_labels = make_examples(eval_stream)
    return {
        "train_ids": train_ids,
        "test_ids": test_ids,
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
            task_type=spec.task_type,
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
            task_type=spec.task_type,
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
            task_type=spec.task_type,
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
            task_type=spec.task_type,
            train_raw=dataset["xtr_img"].astype(np.float32),
            test_raw=dataset["xte_img"].astype(np.float32),
            ytr=dataset["ytr"].astype(np.int64),
            yte=dataset["yte"].astype(np.int64),
            num_classes=int(max(dataset["ytr"].max(), dataset["yte"].max()) + 1),
            image_size=dataset["xtr_img"].shape[-1],
            image_mean=dataset["xtr_img"].mean(axis=0, keepdims=True).astype(np.float32),
        )

    if spec.text_dataset_name is not None and spec.task_type == "classification":
        tokenized = load_text_classification_tokenized(
            spec.text_dataset_name,
            spec.text_dataset_config,
            spec.text_fields,
            spec.label_field,
            spec.eval_split,
            spec.text_vocab_size,
            spec.text_seq_len,
        )
        idx_tr = sample_indices(len(tokenized["train_ids"]), spec.n_train, spec.seed)
        idx_te = sample_indices(len(tokenized["test_ids"]), spec.n_test, spec.seed + 1)
        embedding = build_text_embedding(spec.text_vocab_size, spec.text_embed_dim, seed=0)
        pos = make_1d_sincos_pos_embed(spec.text_embed_dim, spec.text_seq_len)
        return RawDatasetBundle(
            name=spec.name,
            suite=spec.suite,
            modality="text",
            task_type=spec.task_type,
            train_raw=tokenized["train_ids"][idx_tr],
            test_raw=tokenized["test_ids"][idx_te],
            ytr=tokenized["train_labels"][idx_tr],
            yte=tokenized["test_labels"][idx_te],
            num_classes=int(max(tokenized["train_labels"].max(), tokenized["test_labels"].max()) + 1),
            text_embedding=embedding,
            text_pos=pos,
            text_drop_prob=spec.text_drop_prob,
        )

    if spec.text_dataset_name is not None and spec.task_type == "next_token":
        tokenized = load_next_token_text_tokenized(
            spec.text_dataset_name,
            spec.text_dataset_config,
            spec.text_fields,
            spec.text_vocab_size,
            spec.text_seq_len,
            spec.next_token_stride,
        )
        idx_tr = sample_indices(len(tokenized["train_ids"]), spec.n_train, spec.seed)
        idx_te = sample_indices(len(tokenized["test_ids"]), spec.n_test, spec.seed + 1)
        embedding = build_text_embedding(spec.text_vocab_size, spec.text_embed_dim, seed=0)
        pos = make_1d_sincos_pos_embed(spec.text_embed_dim, spec.text_seq_len)
        return RawDatasetBundle(
            name=spec.name,
            suite=spec.suite,
            modality="text",
            task_type=spec.task_type,
            train_raw=tokenized["train_ids"][idx_tr],
            test_raw=tokenized["test_ids"][idx_te],
            ytr=tokenized["train_labels"][idx_tr],
            yte=tokenized["test_labels"][idx_te],
            num_classes=spec.text_vocab_size,
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
        base_tr = flatten_text_tokens(bundle.train_raw, bundle.text_embedding, bundle.text_pos)
        base_te = flatten_text_tokens(bundle.test_raw, bundle.text_embedding, bundle.text_pos)
        view1_tr = flatten_text_tokens(
            augment_text_ids_numpy(bundle.train_raw, seed + 101, bundle.text_drop_prob),
            bundle.text_embedding,
            bundle.text_pos,
        )
        view2_tr = flatten_text_tokens(
            augment_text_ids_numpy(bundle.train_raw, seed + 102, bundle.text_drop_prob),
            bundle.text_embedding,
            bundle.text_pos,
        )
        view1_te = flatten_text_tokens(
            augment_text_ids_numpy(bundle.test_raw, seed + 202, bundle.text_drop_prob),
            bundle.text_embedding,
            bundle.text_pos,
        )
        view2_te = flatten_text_tokens(
            augment_text_ids_numpy(bundle.test_raw, seed + 203, bundle.text_drop_prob),
            bundle.text_embedding,
            bundle.text_pos,
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
    tokens = tcc.patchify_numpy(images.astype(np.float32) - mean_image.astype(np.float32), patch_size).astype(np.float32)
    grid = int(round(math.sqrt(tokens.shape[1])))
    pos = tcc.make_2d_sincos_pos_embed(tokens.shape[-1], grid).astype(np.float32)
    return tokens + pos[None, :, :]


def build_text_tokens(ids, embedding, pos):
    return embedding[ids].astype(np.float32) + pos[None, :, :].astype(np.float32)


def flatten_text_tokens(ids, embedding, pos):
    tokens = build_text_tokens(ids, embedding, pos)
    return tokens.reshape(tokens.shape[0], -1)


def build_tabular_tokens(features, embedding, pos):
    return features[:, :, None].astype(np.float32) * embedding[None, :, :].astype(np.float32) + pos[None, :, :].astype(np.float32)


def build_transformer_arrays(bundle: RawDatasetBundle, seed: int, patch_size: int, include_test_views: bool = True):
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
        if include_test_views:
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
        else:
            view1_te = None
            view2_te = None
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
        if include_test_views:
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
        else:
            view1_te = None
            view2_te = None
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
        if include_test_views:
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
        else:
            view1_te = None
            view2_te = None
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


def center_train_test_with_mean(train_array, test_array):
    mean = train_array.mean(axis=0, keepdims=True)
    return train_array - mean, test_array - mean, mean


def build_base_mlp_features(bundle: RawDatasetBundle, raw_inputs):
    if bundle.modality == "image":
        return flatten_images(raw_inputs, bundle.image_mean)
    if bundle.modality == "text":
        return flatten_text_tokens(raw_inputs, bundle.text_embedding, bundle.text_pos)
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


def make_augmented_raw_view(bundle: RawDatasetBundle, raw_inputs, seed: int):
    if bundle.modality == "image":
        return cifar_shared.apply_augmentation(raw_inputs, bundle.suite, np.random.default_rng(seed)).astype(np.float32)
    if bundle.modality == "text":
        return augment_text_ids_numpy(raw_inputs, seed, bundle.text_drop_prob)
    if bundle.modality == "tabular":
        return augment_tabular_numpy(raw_inputs, seed, bundle.tabular_drop_prob, bundle.tabular_noise_std).astype(np.float32)
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


def cosine_alignment(a, b):
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    return np.sum(a_norm * b_norm, axis=1)


def view_alignment_summary(repr1, repr2):
    if len(repr1) == 0:
        return {
            "view_alignment_cosine": 0.0,
            "view_alignment_gap": 0.0,
        }
    same = cosine_alignment(repr1, repr2)
    perm = np.roll(np.arange(len(repr2)), 1)
    shuffled = cosine_alignment(repr1, repr2[perm])
    return {
        "view_alignment_cosine": float(np.mean(same)),
        "view_alignment_gap": float(np.mean(same - shuffled)),
    }


def augmentation_prediction_summary(base_logits, logits1, logits2):
    if len(base_logits) == 0:
        return {
            "augmentation_prediction_agreement": 0.0,
            "base_view_prediction_agreement": 0.0,
        }
    base_pred = np.argmax(base_logits, axis=1)
    pred1 = np.argmax(logits1, axis=1)
    pred2 = np.argmax(logits2, axis=1)
    return {
        "augmentation_prediction_agreement": float(np.mean(pred1 == pred2)),
        "base_view_prediction_agreement": float(0.5 * (np.mean(base_pred == pred1) + np.mean(base_pred == pred2))),
    }


def analysis_sample_count(total, task_type):
    base = 256 if task_type == "classification" else 384
    return min(base, total)


def sample_analysis_indices(total, task_type, seed):
    count = analysis_sample_count(total, task_type)
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if total <= count:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=count, replace=False).astype(np.int64))


def linear_cka(features_x, features_y):
    if len(features_x) == 0 or len(features_y) == 0:
        return 0.0
    x = np.asarray(features_x, dtype=np.float64)
    y = np.asarray(features_y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    gram_x = x @ x.T
    gram_y = y @ y.T
    numerator = float(np.sum(gram_x * gram_y))
    denom_x = float(np.sqrt(np.sum(gram_x * gram_x)))
    denom_y = float(np.sqrt(np.sum(gram_y * gram_y)))
    return float(numerator / max(denom_x * denom_y, 1e-12))


def labels_to_one_hot(y, num_classes):
    return np.eye(num_classes, dtype=np.float64)[np.asarray(y, dtype=np.int64)]


def summarize_layer_geometry(layer_reprs, labels, num_classes):
    if not layer_reprs:
        return []
    input_repr = np.asarray(layer_reprs[0], dtype=np.float64)
    label_repr = labels_to_one_hot(labels, num_classes)
    summary = []
    for idx, layer_repr in enumerate(layer_reprs):
        repr_arr = np.asarray(layer_repr, dtype=np.float64)
        summary.append(
            {
                "layer": idx,
                "cka_to_input": linear_cka(input_repr, repr_arr),
                "cka_to_labels": linear_cka(repr_arr, label_repr),
                "effective_rank": effective_rank(repr_arr),
            }
        )
    return summary


def format_umap_labels(labels, task_type):
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return []
    if task_type != "next_token" and len(np.unique(labels)) <= 20:
        return [str(int(label)) for label in labels]
    counter = Counter(labels.tolist())
    top = {label for label, _ in counter.most_common(8)}
    return [str(int(label)) if int(label) in top else "other" for label in labels]


def compute_umap_points(features, labels, task_type, seed):
    if len(features) < 3:
        return []
    try:
        import umap  # type: ignore
    except Exception:
        return []
    n_neighbors = max(2, min(30, len(features) // 8))
    n_neighbors = min(n_neighbors, max(2, len(features) - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.15,
        metric="cosine",
        random_state=seed,
    )
    coords = reducer.fit_transform(np.asarray(features, dtype=np.float32))
    grouped_labels = format_umap_labels(labels, task_type)
    return [
        {
            "point_index": int(idx),
            "x": float(coords[idx, 0]),
            "y": float(coords[idx, 1]),
            "label": int(labels[idx]),
            "group_label": grouped_labels[idx],
        }
        for idx in range(len(coords))
    ]


def pool_sequence_numpy(tokens, task_type):
    if task_type == "next_token":
        return tokens[:, -1, :]
    return tokens.mean(axis=1)


def pool_sequence_torch(tokens, task_type):
    if task_type == "next_token":
        return tokens[:, -1, :]
    return tokens.mean(dim=1)


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


def build_analysis_payload(bundle, seed, predict_logits_fn, encode_final_fn, encode_layers_fn, patch_size):
    ood_results = []
    for shift_name, shifted_inputs in make_ood_variants(bundle, seed).items():
        logits = predict_logits_fn(shifted_inputs)
        ood_results.append({"shift": shift_name, "accuracy": evaluate_logits(logits, bundle.yte)})

    train_repr = encode_final_fn(bundle.train_raw)
    test_repr = encode_final_fn(bundle.test_raw)

    subset_idx = sample_analysis_indices(len(bundle.test_raw), bundle.task_type, seed + 4101)
    analysis_raw = bundle.test_raw[subset_idx]
    analysis_labels = bundle.yte[subset_idx]
    layer_reprs = encode_layers_fn(analysis_raw)
    layer_geometry = summarize_layer_geometry(layer_reprs, analysis_labels, bundle.num_classes)
    umap_points = compute_umap_points(layer_reprs[-1], analysis_labels, bundle.task_type, seed + 5101) if layer_reprs else []

    aug_view1 = make_augmented_raw_view(bundle, analysis_raw, seed + 4001)
    aug_view2 = make_augmented_raw_view(bundle, analysis_raw, seed + 4002)
    base_logits = predict_logits_fn(analysis_raw)
    view1_logits = predict_logits_fn(aug_view1)
    view2_logits = predict_logits_fn(aug_view2)

    interpretability = {
        "effective_rank": effective_rank(test_repr),
        "centroid_margin": centroid_margin(train_repr, bundle.ytr, test_repr, bundle.yte),
        **occlusion_summary(bundle, bundle.test_raw, predict_logits_fn, patch_size=patch_size),
        **view_alignment_summary(encode_final_fn(aug_view1), encode_final_fn(aug_view2)),
        **augmentation_prediction_summary(base_logits, view1_logits, view2_logits),
    }
    if layer_geometry:
        interpretability["final_cka_to_input"] = layer_geometry[-1]["cka_to_input"]
        interpretability["final_cka_to_labels"] = layer_geometry[-1]["cka_to_labels"]

    return {
        "ood_results": ood_results,
        "interpretability": interpretability,
        "layerwise_geometry": layer_geometry,
        "umap_points": umap_points,
    }


def fit_closed_form_mlp_state(bundle: RawDatasetBundle, method_name: str, config: MLPEvalConfig, seed: int):
    set_seed(seed)
    base_tr, base_te, view1_tr, view2_tr, view1_te, view2_te = build_mlp_arrays(bundle, seed)
    train_arrays, test_arrays, initial_mean, initial_scale = normalize_hidden_with_stats(
        [base_tr, view1_tr, view2_tr],
        [base_te, view1_te, view2_te],
    )
    base_tr, view1_tr, view2_tr = train_arrays
    base_te, view1_te, view2_te = test_arrays

    ytr_onehot = one_hot(bundle.ytr, bundle.num_classes)
    yhat_tr = np.zeros_like(ytr_onehot)
    yhat_te = np.zeros((len(bundle.yte), bundle.num_classes), dtype=np.float64)
    layers = []
    activation_param_count = 0
    output_param_count = 0
    layer_states = []
    output_map = None

    start = time.perf_counter()
    for layer_idx in range(config.depth):
        pre_output_map = None
        post_output_map = None
        base_center_mean = None

        if config.dual_mapping and config.output_source == "pre-hidden":
            pre_output_map = ridge_regression(base_tr, ytr_onehot - yhat_tr, reg=config.head_reg)
            output_param_count += int(pre_output_map.size)
            yhat_tr = yhat_tr + base_tr @ pre_output_map
            yhat_te = yhat_te + base_te @ pre_output_map

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

        if config.center_after_hidden:
            base_tr, base_te, base_center_mean = center_train_test_with_mean(base_tr, base_te)
            view1_tr, view1_te, _ = center_train_test_with_mean(view1_tr, view1_te)
            view2_tr, view2_te, _ = center_train_test_with_mean(view2_tr, view2_te)

        if config.dual_mapping and config.output_source == "post-hidden":
            post_output_map = ridge_regression(base_tr, ytr_onehot - yhat_tr, reg=config.head_reg)
            output_param_count += int(post_output_map.size)
            yhat_tr = yhat_tr + base_tr @ post_output_map
            yhat_te = yhat_te + base_te @ post_output_map

        train_arrays, test_arrays, norm_mean, norm_scale = normalize_hidden_with_stats(
            [base_tr, view1_tr, view2_tr],
            [base_te, view1_te, view2_te],
        )
        base_tr, view1_tr, view2_tr = train_arrays
        base_te, view1_te, view2_te = test_arrays

        if not config.dual_mapping:
            output_map = ridge_regression(base_tr, ytr_onehot, reg=config.head_reg)
            output_param_count = int(output_map.size)
            logits_te = base_te @ output_map
        else:
            logits_te = yhat_te

        layer_states.append(
            {
                "fitted": fitted,
                "pre_output_map": pre_output_map,
                "post_output_map": post_output_map,
                "base_center_mean": base_center_mean,
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

    return {
        "bundle": bundle,
        "method_name": method_name,
        "config": config,
        "seed": seed,
        "initial_mean": initial_mean,
        "initial_scale": initial_scale,
        "layer_states": layer_states,
        "layers": layers,
        "yhat_te": yhat_te,
        "output_map": output_map,
        "activation_param_count": activation_param_count,
        "output_param_count": output_param_count,
        "fit_time_sec": fit_time,
    }


def run_closed_form_mlp(bundle: RawDatasetBundle, method_name: str, config: MLPEvalConfig, seed: int, collect_analysis: bool = False):
    state = fit_closed_form_mlp_state(bundle, method_name, config, seed)
    initial_mean = state["initial_mean"]
    initial_scale = state["initial_scale"]
    layer_states = state["layer_states"]
    output_map = state["output_map"]

    analysis = None
    if collect_analysis:
        def encode_layers(raw_inputs):
            hidden = build_base_mlp_features(bundle, raw_inputs)
            hidden = (hidden - initial_mean) / initial_scale
            collected = [hidden]
            for state in layer_states:
                fitted = state["fitted"]
                hidden = cfbt.apply_layer(hidden, fitted["transform_base"], activation=config.activation)
                if state["base_center_mean"] is not None:
                    hidden = hidden - state["base_center_mean"]
                hidden = (hidden - state["norm_mean"]) / state["norm_scale"]
                collected.append(hidden)
            return collected

        def encode_raw(raw_inputs):
            return encode_layers(raw_inputs)[-1]

        def predict_logits(raw_inputs):
            hidden = build_base_mlp_features(bundle, raw_inputs)
            hidden = (hidden - initial_mean) / initial_scale
            if config.dual_mapping:
                logits = np.zeros((len(hidden), bundle.num_classes), dtype=np.float64)
                for state in layer_states:
                    if state["pre_output_map"] is not None:
                        logits = logits + hidden @ state["pre_output_map"]
                    hidden = cfbt.apply_layer(hidden, state["fitted"]["transform_base"], activation=config.activation)
                    if state["base_center_mean"] is not None:
                        hidden = hidden - state["base_center_mean"]
                    if state["post_output_map"] is not None:
                        logits = logits + hidden @ state["post_output_map"]
                    hidden = (hidden - state["norm_mean"]) / state["norm_scale"]
                return logits
            for state in layer_states:
                hidden = cfbt.apply_layer(hidden, state["fitted"]["transform_base"], activation=config.activation)
                if state["base_center_mean"] is not None:
                    hidden = hidden - state["base_center_mean"]
                hidden = (hidden - state["norm_mean"]) / state["norm_scale"]
            return hidden @ output_map

        analysis = build_analysis_payload(
            bundle,
            seed,
            predict_logits,
            encode_raw,
            encode_layers,
            patch_size=8,
        )

    return {
        "architecture": "mlp",
        "model": "closed-form",
        "dataset": bundle.name,
        "suite": bundle.suite,
        "seed": seed,
        "closed_form_method": method_name,
        "classifier_accuracy": (
            evaluate_logits(state["yhat_te"], bundle.yte) if config.dual_mapping else state["layers"][-1]["classifier_accuracy"]
        ),
        "layers": state["layers"],
        "hidden_param_count": state["activation_param_count"],
        "output_param_count": state["output_param_count"],
        "total_parameter_count": state["activation_param_count"] + state["output_param_count"],
        "fit_time_sec": state["fit_time_sec"],
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


def apply_mlp_activation_torch(x, activation):
    if activation == "relu":
        return torch.relu(x)
    if activation == "gelu":
        return F.gelu(x)
    raise ValueError(f"Unsupported MLP activation for fine-tuning: {activation}")


def estimate_mlp_compute_proxy(bundle: RawDatasetBundle, config: MLPEvalConfig, model_kind: str):
    input_dim = int(build_base_mlp_features(bundle, bundle.train_raw[:1]).shape[1])
    train_count = int(len(bundle.ytr))
    eval_count = int(len(bundle.yte))
    dims = [input_dim] + [config.width] * config.depth
    hidden_forward = sum(dims[idx] * dims[idx + 1] for idx in range(config.depth))
    per_layer_head_dim = [dims[idx] if config.output_source == "pre-hidden" else dims[idx + 1] for idx in range(config.depth)]

    if model_kind == "backprop":
        per_example = hidden_forward + dims[-1] * bundle.num_classes
        return float(config.epochs * train_count * per_example)

    if model_kind == "fine-tune":
        head_forward = sum(dim * bundle.num_classes for dim in per_layer_head_dim) if config.dual_mapping else dims[-1] * bundle.num_classes
        per_example = hidden_forward + head_forward
        return float(train_count * per_example)

    if model_kind == "closed-form":
        current_dim = input_dim
        fit_views = 2.0 * train_count
        apply_views = 3.0 * train_count + 3.0 * eval_count
        total = 0.0
        for _ in range(config.depth):
            next_dim = min(config.width, current_dim)
            total += fit_views * current_dim * next_dim
            total += apply_views * current_dim * next_dim
            head_dim = current_dim if config.output_source == "pre-hidden" else next_dim
            if config.dual_mapping:
                total += train_count * head_dim * bundle.num_classes
            current_dim = next_dim
        if not config.dual_mapping:
            total += train_count * current_dim * bundle.num_classes
        return float(total)

    raise ValueError(f"Unknown MLP compute proxy model kind: {model_kind}")


class ClosedFormFineTuneMLP(nn.Module):
    def __init__(self, input_dim, num_classes, state, activation):
        super().__init__()
        self.num_classes = num_classes
        self.activation = activation
        self.hidden = nn.ModuleList()
        self.pre_heads = nn.ModuleDict()
        self.post_heads = nn.ModuleDict()
        self.final_head = None

        self.register_buffer("initial_mean", torch.from_numpy(state["initial_mean"]).float())
        self.register_buffer("initial_scale", torch.from_numpy(state["initial_scale"]).float())

        current_dim = input_dim
        for layer_idx, layer_state in enumerate(state["layer_states"]):
            fitted = layer_state["fitted"]
            transform = torch.from_numpy(fitted["transform_base"]).float()
            linear = nn.Linear(current_dim, transform.shape[1], bias=False)
            linear.weight.data.copy_(transform.T)
            self.hidden.append(linear)

            pre_map = layer_state["pre_output_map"]
            if pre_map is not None:
                head = nn.Linear(current_dim, num_classes, bias=False)
                head.weight.data.copy_(torch.from_numpy(pre_map.T).float())
                self.pre_heads[str(layer_idx)] = head

            post_map = layer_state["post_output_map"]
            next_dim = transform.shape[1]
            if post_map is not None:
                head = nn.Linear(next_dim, num_classes, bias=False)
                head.weight.data.copy_(torch.from_numpy(post_map.T).float())
                self.post_heads[str(layer_idx)] = head

            center = layer_state["base_center_mean"]
            center_tensor = torch.from_numpy(center).float() if center is not None else torch.zeros((0,), dtype=torch.float32)
            self.register_buffer(f"base_center_mean_{layer_idx}", center_tensor)
            self.register_buffer(f"norm_mean_{layer_idx}", torch.from_numpy(layer_state["norm_mean"]).float())
            self.register_buffer(f"norm_scale_{layer_idx}", torch.from_numpy(layer_state["norm_scale"]).float())
            current_dim = next_dim

        if state["output_map"] is not None:
            head = nn.Linear(current_dim, num_classes, bias=False)
            head.weight.data.copy_(torch.from_numpy(state["output_map"].T).float())
            self.final_head = head

    def encode_layers(self, x):
        hidden = (x - self.initial_mean) / self.initial_scale
        collected = [hidden]
        cumulative = hidden.new_zeros((hidden.shape[0], self.num_classes))
        depth_logits = []
        for layer_idx, layer in enumerate(self.hidden):
            pre_key = str(layer_idx)
            pre_head = self.pre_heads[pre_key] if pre_key in self.pre_heads else None
            if pre_head is not None:
                cumulative = cumulative + pre_head(hidden)
            hidden = apply_mlp_activation_torch(layer(hidden), self.activation)
            center = getattr(self, f"base_center_mean_{layer_idx}")
            if center.numel() > 0:
                hidden = hidden - center
            post_key = str(layer_idx)
            post_head = self.post_heads[post_key] if post_key in self.post_heads else None
            if post_head is not None:
                cumulative = cumulative + post_head(hidden)
            hidden = (hidden - getattr(self, f"norm_mean_{layer_idx}")) / getattr(self, f"norm_scale_{layer_idx}")
            collected.append(hidden)
            if len(self.pre_heads) or len(self.post_heads):
                depth_logits.append(cumulative)
        if self.final_head is not None:
            final_logits = self.final_head(hidden)
            depth_logits = [final_logits]
        return collected, depth_logits

    def encode(self, x):
        return self.encode_layers(x)[0][-1]

    def forward(self, x):
        _, depth_logits = self.encode_layers(x)
        return depth_logits


def run_closed_form_backprop_finetune_mlp(bundle: RawDatasetBundle, method_name: str, config: MLPEvalConfig, seed: int, collect_analysis: bool = False):
    state = fit_closed_form_mlp_state(bundle, method_name, config, seed)
    xtr_np = build_base_mlp_features(bundle, bundle.train_raw).astype(np.float32)
    xte_np = build_base_mlp_features(bundle, bundle.test_raw).astype(np.float32)
    xtr = torch.from_numpy(xtr_np)
    xte = torch.from_numpy(xte_np)
    ytr = torch.from_numpy(bundle.ytr).long()
    train_loader = make_train_loader(TensorDataset(xtr, ytr), batch_size=config.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClosedFormFineTuneMLP(xtr_np.shape[1], bundle.num_classes, state, config.activation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    total_budget = estimate_mlp_compute_proxy(bundle, config, "backprop")
    init_budget = estimate_mlp_compute_proxy(bundle, config, "closed-form")
    finetune_epoch_budget = estimate_mlp_compute_proxy(bundle, config, "fine-tune")
    fine_tune_steps, used_budget = compute_matched_step_budget(
        total_budget,
        init_budget,
        finetune_epoch_budget,
        len(train_loader),
    )

    def train_step(batch):
        xb, yb = batch
        xb = xb.to(device)
        yb = yb.to(device)
        depth_logits = model(xb)
        loss = sum(criterion(logits, yb) for logits in depth_logits) / max(len(depth_logits), 1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss.item()

    start = time.perf_counter()
    epoch_stats = run_training_steps(train_loader, fine_tune_steps, train_step)
    fine_tune_time = time.perf_counter() - start
    fit_time = state["fit_time_sec"] + fine_tune_time

    model.eval()
    with torch.no_grad():
        depth_logits = model(xte.to(device))
        pred = depth_logits[-1].argmax(dim=1).cpu().numpy()
        layers = [
            {
                "depth": depth_idx,
                "classifier_accuracy": float((logits.argmax(dim=1).cpu().numpy() == bundle.yte).mean()),
            }
            for depth_idx, logits in enumerate(depth_logits, start=1)
        ]

    analysis = None
    if collect_analysis:
        def predict_logits(raw_inputs):
            with torch.no_grad():
                feats = torch.from_numpy(build_base_mlp_features(bundle, raw_inputs).astype(np.float32)).to(device)
                return model(feats)[-1].cpu().numpy()

        def encode_repr(raw_inputs):
            with torch.no_grad():
                feats = torch.from_numpy(build_base_mlp_features(bundle, raw_inputs).astype(np.float32)).to(device)
                return model.encode(feats).cpu().numpy().astype(np.float64)

        def encode_layers(raw_inputs):
            with torch.no_grad():
                feats = torch.from_numpy(build_base_mlp_features(bundle, raw_inputs).astype(np.float32)).to(device)
                return [layer.cpu().numpy().astype(np.float64) for layer in model.encode_layers(feats)[0]]

        analysis = build_analysis_payload(
            bundle,
            seed,
            predict_logits,
            encode_repr,
            encode_layers,
            patch_size=8,
        )

    hidden_param_count = int(sum(p.numel() for p in model.hidden.parameters()))
    output_heads = list(model.pre_heads.parameters()) + list(model.post_heads.parameters())
    if model.final_head is not None:
        output_heads += list(model.final_head.parameters())
    output_param_count = int(sum(p.numel() for p in output_heads))
    return {
        "architecture": "mlp",
        "model": FINE_TUNE_MODEL_NAME,
        "dataset": bundle.name,
        "suite": bundle.suite,
        "seed": seed,
        "closed_form_method": method_name,
        "classifier_accuracy": float((pred == bundle.yte).mean()),
        "layers": layers,
        "hidden_param_count": hidden_param_count,
        "output_param_count": output_param_count,
        "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "fit_time_sec": fit_time,
        "closed_form_init_time_sec": state["fit_time_sec"],
        "fine_tune_time_sec": fine_tune_time,
        "fine_tune_steps": fine_tune_steps,
        "fine_tune_effective_epochs": float(fine_tune_steps / max(len(train_loader), 1)),
        "compute_proxy": used_budget,
        "compute_unit": "ops_proxy",
        "epoch_stats": epoch_stats,
        "config": {**asdict(config), "compute_matched_to": "backprop"},
        "analysis": analysis,
    }


def run_backprop_mlp(bundle: RawDatasetBundle, config: MLPEvalConfig, seed: int, collect_analysis: bool = False):
    set_seed(seed)
    base_tr, base_te, _, _, _, _ = build_mlp_arrays(bundle, seed)
    xtr = torch.from_numpy(base_tr).float()
    xte = torch.from_numpy(base_te).float()
    ytr = torch.from_numpy(bundle.ytr).long()
    yte = torch.from_numpy(bundle.yte).long()

    train_loader = make_train_loader(TensorDataset(xtr, ytr), batch_size=config.batch_size, shuffle=True)
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
            optimizer.zero_grad(set_to_none=True)
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
        def features_from_raw(raw_inputs):
            return build_base_mlp_features(bundle, raw_inputs)

        def predict_logits(raw_inputs):
            with torch.no_grad():
                feats = torch.from_numpy(features_from_raw(raw_inputs)).float().to(device)
                return model(feats).cpu().numpy()

        def encode_repr(raw_inputs):
            with torch.no_grad():
                feats = torch.from_numpy(features_from_raw(raw_inputs)).float().to(device)
                return model.encode(feats).cpu().numpy().astype(np.float64)

        def encode_layers(raw_inputs):
            with torch.no_grad():
                feats = torch.from_numpy(features_from_raw(raw_inputs)).float().to(device)
                hidden = feats
                collected = [hidden.cpu().numpy().astype(np.float64)]
                for layer in model.hidden:
                    hidden = torch.relu(layer(hidden))
                    collected.append(hidden.cpu().numpy().astype(np.float64))
                return collected

        analysis = build_analysis_payload(
            bundle,
            seed,
            predict_logits,
            encode_repr,
            encode_layers,
            patch_size=8,
        )

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


def fit_closed_form_transformer_state(bundle: RawDatasetBundle, attention_kind: str, config: TransformerEvalConfig, seed: int):
    set_seed(seed)
    base_tr, base_te, view1_tr, view2_tr, _, _ = build_transformer_arrays(
        bundle, seed, config.patch_size, include_test_views=False
    )
    attn_cfg = make_attention_config(base_tr, attention_kind, config, seed)
    ytr_onehot = one_hot(bundle.ytr, bundle.num_classes, dtype=base_tr.dtype)
    yhat_tr = np.zeros_like(ytr_onehot)
    yhat_te = np.zeros((len(bundle.yte), bundle.num_classes), dtype=base_tr.dtype)

    layers = []
    output_param_count = 0
    hidden_param_count = 0
    layer_states = []

    start = time.perf_counter()
    for layer_idx in range(config.depth):
        pooled_tr = pool_sequence_numpy(base_tr, bundle.task_type)
        pooled_te = pool_sequence_numpy(base_te, bundle.task_type)
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
    raw_yhat_te = yhat_te.copy()
    logit_temperature = 1.0
    if bundle.task_type == "next_token":
        logit_temperature = fit_logit_temperature(yhat_tr, bundle.ytr, seed=seed + 1701)
        yhat_tr = yhat_tr * logit_temperature
        yhat_te = yhat_te * logit_temperature

    return {
        "bundle": bundle,
        "attention_kind": attention_kind,
        "config": config,
        "seed": seed,
        "attn_cfg": attn_cfg,
        "layer_states": layer_states,
        "layers": layers,
        "yhat_te": yhat_te,
        "raw_yhat_te": raw_yhat_te,
        "logit_temperature": float(logit_temperature),
        "hidden_param_count": hidden_param_count,
        "output_param_count": output_param_count,
        "fit_time_sec": fit_time,
    }


def run_closed_form_transformer(bundle: RawDatasetBundle, attention_kind: str, config: TransformerEvalConfig, seed: int, collect_analysis: bool = False):
    state = fit_closed_form_transformer_state(bundle, attention_kind, config, seed)
    attn_cfg = state["attn_cfg"]
    layer_states = state["layer_states"]
    metric_summary = scaling_metrics_from_logits(state["yhat_te"], bundle.yte, bundle.task_type)
    raw_metric_summary = scaling_metrics_from_logits(state["raw_yhat_te"], bundle.yte, bundle.task_type)

    analysis = None
    if collect_analysis:
        def encode_token_layers(raw_inputs):
            tokens = build_base_transformer_tokens(bundle, raw_inputs, config.patch_size)
            collected = [pool_sequence_numpy(tokens, bundle.task_type)]
            for state in layer_states:
                tokens = tcc.apply_attention_block(tokens, attn_cfg, state["att_model"])
                flat = tokens.reshape(-1, tokens.shape[-1])
                ffn = cfbt.apply_activation(flat @ state["ffn_model"]["transform_base"], "relu")
                tokens = tcc.token_layer_norm(tokens + ffn.reshape(tokens.shape))
                collected.append(pool_sequence_numpy(tokens, bundle.task_type))
            return collected

        def encode_raw(raw_inputs):
            return encode_token_layers(raw_inputs)[-1]

        def predict_logits(raw_inputs):
            tokens = build_base_transformer_tokens(bundle, raw_inputs, config.patch_size)
            logits = np.zeros((tokens.shape[0], bundle.num_classes), dtype=tokens.dtype)
            for state in layer_states:
                pooled = pool_sequence_numpy(tokens, bundle.task_type)
                logits = logits + pooled @ state["out_map"]
                tokens = tcc.apply_attention_block(tokens, attn_cfg, state["att_model"])
                flat = tokens.reshape(-1, tokens.shape[-1])
                ffn = cfbt.apply_activation(flat @ state["ffn_model"]["transform_base"], "relu")
                tokens = tcc.token_layer_norm(tokens + ffn.reshape(tokens.shape))
            return logits * float(state_ref["logit_temperature"])

        state_ref = state

        analysis = build_analysis_payload(
            bundle,
            seed,
            predict_logits,
            encode_raw,
            encode_token_layers,
            patch_size=max(1, config.patch_size),
        )

    return {
        "architecture": "transformer",
        "model": "closed-form",
        "dataset": bundle.name,
        "suite": bundle.suite,
        "seed": seed,
        "closed_form_method": attention_kind,
        "classifier_accuracy": metric_summary["classifier_accuracy"],
        "validation_cross_entropy": metric_summary.get("validation_cross_entropy"),
        "validation_perplexity": metric_summary.get("validation_perplexity"),
        "raw_validation_cross_entropy": raw_metric_summary.get("validation_cross_entropy"),
        "raw_validation_perplexity": raw_metric_summary.get("validation_perplexity"),
        "logit_temperature": float(state["logit_temperature"]),
        "scaling_metric_name": metric_summary["scaling_metric_name"],
        "scaling_metric_label": metric_summary["scaling_metric_label"],
        "scaling_metric_value": metric_summary["scaling_metric_value"],
        "scaling_fit_type": metric_summary["scaling_fit_type"],
        "layers": state["layers"],
        "hidden_param_count": state["hidden_param_count"],
        "output_param_count": state["output_param_count"],
        "total_parameter_count": state["hidden_param_count"] + state["output_param_count"],
        "fit_time_sec": state["fit_time_sec"],
        "config": asdict(config),
        "analysis": analysis,
    }


def token_layer_norm_torch(tokens, eps=1e-5):
    return F.layer_norm(tokens, (tokens.shape[-1],), eps=eps)


class TrainableSpectralSelfAttention(nn.Module):
    def __init__(self, att_model):
        super().__init__()
        self.sigma_inv_sqrt = nn.Parameter(torch.from_numpy(att_model["sigma_inv_sqrt"]).float())
        self.projection_heads = nn.ParameterList(
            [nn.Parameter(torch.from_numpy(head).float()) for head in att_model["projection_heads"]]
        )
        self.output_map = nn.Parameter(torch.from_numpy(att_model["output_map"]).float())
        self.mix_scale = nn.Parameter(torch.tensor(float(att_model.get("mix_scale", 0.0)), dtype=torch.float32))
        self.residual_mode = bool(att_model.get("residual_mode", False))
        self.center_values = bool(att_model.get("center_values", False))
        self.whiten_values = bool(att_model.get("whiten_values", False))
        self.score_mode = att_model.get("score_mode", "raw")
        self.register_buffer(
            "score_scales",
            torch.tensor(
                [float(head_spec.get("score_scale", 1.0)) for head_spec in att_model["head_layout"]],
                dtype=torch.float32,
            ),
        )

    def forward(self, tokens):
        if self.score_mode in {None, "raw"}:
            score_tokens = tokens
        elif self.score_mode == "token-centered":
            score_tokens = tokens - tokens.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unsupported trainable score_mode: {self.score_mode}")

        whitened_scores = score_tokens @ self.sigma_inv_sqrt
        values = tokens @ self.sigma_inv_sqrt if self.whiten_values else tokens
        if self.center_values:
            values = values - values.mean(dim=1, keepdim=True)

        contexts = []
        for head_idx, projection in enumerate(self.projection_heads):
            queries = F.normalize(whitened_scores @ projection, dim=-1, eps=1e-8)
            keys = F.normalize(whitened_scores @ projection, dim=-1, eps=1e-8)
            scores = self.score_scales[head_idx] * (
                torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(max(int(projection.shape[1]), 1))
            )
            weights = torch.softmax(scores, dim=-1)
            contexts.append(torch.matmul(weights, values))
        context = torch.cat(contexts, dim=-1)
        attended = context @ self.output_map
        if self.residual_mode:
            return tokens + self.mix_scale * attended
        alpha = torch.clamp(self.mix_scale, 0.0, 1.0)
        return alpha * tokens + (1.0 - alpha) * attended


class ClosedFormSpectralTransformerLayer(nn.Module):
    def __init__(self, layer_state):
        super().__init__()
        self.attention = TrainableSpectralSelfAttention(layer_state["att_model"])
        transform = torch.from_numpy(layer_state["ffn_model"]["transform_base"]).float()
        self.ffn = nn.Linear(transform.shape[0], transform.shape[1], bias=False)
        self.ffn.weight.data.copy_(transform.T)

    def forward(self, tokens):
        hidden = token_layer_norm_torch(self.attention(tokens))
        ffn = torch.relu(self.ffn(hidden))
        return token_layer_norm_torch(hidden + ffn)


class ClosedFormFineTuneTransformer(nn.Module):
    def __init__(self, token_dim, num_classes, state, task_type="classification"):
        super().__init__()
        self.blocks = nn.ModuleList([ClosedFormSpectralTransformerLayer(layer_state) for layer_state in state["layer_states"]])
        self.output_heads = nn.ModuleList()
        for layer_state in state["layer_states"]:
            out_map = torch.from_numpy(layer_state["out_map"]).float()
            head = nn.Linear(token_dim, num_classes, bias=False)
            head.weight.data.copy_(out_map.T)
            self.output_heads.append(head)
        self.task_type = task_type

    def encode_layers(self, tokens):
        hidden = tokens
        collected = [pool_sequence_torch(hidden, self.task_type)]
        cumulative = None
        depth_logits = []
        for block, head in zip(self.blocks, self.output_heads):
            pooled = pool_sequence_torch(hidden, self.task_type)
            logits = head(pooled)
            cumulative = logits if cumulative is None else cumulative + logits
            depth_logits.append(cumulative)
            hidden = block(hidden)
            collected.append(pool_sequence_torch(hidden, self.task_type))
        return collected, depth_logits

    def encode(self, tokens):
        return self.encode_layers(tokens)[0][-1]

    def forward(self, tokens):
        _, depth_logits = self.encode_layers(tokens)
        return depth_logits


class TokenTransformerClassifier(nn.Module):
    def __init__(self, token_dim, num_heads, depth, num_classes, mlp_ratio, task_type="classification"):
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
        self.task_type = task_type

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
            pooled = pool_sequence_torch(hidden, self.task_type)
            logits = head(pooled)
            cumulative = logits if cumulative is None else cumulative + logits
            depth_logits.append(cumulative)
        return depth_logits


def prepare_transformer_training_context(bundle: RawDatasetBundle, config: TransformerEvalConfig, device):
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

    return train_ds, test_raw, token_dim, num_heads, to_tokens


def run_closed_form_backprop_finetune_transformer(
    bundle: RawDatasetBundle, attention_kind: str, config: TransformerEvalConfig, seed: int, collect_analysis: bool = False
):
    state = fit_closed_form_transformer_state(bundle, attention_kind, config, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, test_raw, token_dim, _, to_tokens = prepare_transformer_training_context(bundle, config, device)
    train_loader = make_train_loader(train_ds, batch_size=config.batch_size, shuffle=True)
    model = ClosedFormFineTuneTransformer(token_dim, bundle.num_classes, state, task_type=bundle.task_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    total_budget = estimate_transformer_compute_proxy(bundle, config, "backprop")
    init_budget = estimate_transformer_compute_proxy(bundle, config, "closed-form", attention_kind)
    finetune_epoch_budget = estimate_transformer_compute_proxy(bundle, config, FINE_TUNE_MODEL_NAME, attention_kind)
    fine_tune_steps, used_budget = compute_matched_step_budget(
        total_budget,
        init_budget,
        finetune_epoch_budget,
        len(train_loader),
    )

    def train_step(batch):
        xb, yb = batch
        tokens = to_tokens(xb, augment=True)
        yb = yb.to(device)
        depth_logits = model(tokens)
        loss = sum(criterion(logits, yb) for logits in depth_logits) / max(len(depth_logits), 1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss.item()

    start = time.perf_counter()
    epoch_stats = run_training_steps(train_loader, fine_tune_steps, train_step)
    fine_tune_time = time.perf_counter() - start
    fit_time = state["fit_time_sec"] + fine_tune_time

    model.eval()
    with torch.no_grad():
        tokens_te = to_tokens(test_raw, augment=False)
        depth_logits = model(tokens_te)
        final_logits = depth_logits[-1]
        final_logits_np = final_logits.cpu().numpy()
        pred = final_logits.argmax(dim=1).cpu().numpy()
        layers = [
            {
                "depth": depth_idx,
                "classifier_accuracy": float((logits.argmax(dim=1).cpu().numpy() == bundle.yte).mean()),
            }
            for depth_idx, logits in enumerate(depth_logits, start=1)
        ]
    metric_summary = scaling_metrics_from_logits(final_logits_np, bundle.yte, bundle.task_type)

    analysis = None
    if collect_analysis:
        def raw_to_tokens(raw_inputs):
            if bundle.modality == "image":
                tensor = torch.from_numpy(raw_inputs).float().to(device)
            elif bundle.modality == "text":
                tensor = torch.from_numpy(raw_inputs).long().to(device)
            else:
                tensor = torch.from_numpy(raw_inputs).float().to(device)
            return to_tokens(tensor, augment=False)

        def predict_logits(raw_inputs):
            with torch.no_grad():
                return model(raw_to_tokens(raw_inputs))[-1].cpu().numpy()

        def encode_repr(raw_inputs):
            with torch.no_grad():
                return model.encode(raw_to_tokens(raw_inputs)).cpu().numpy().astype(np.float64)

        def encode_layers(raw_inputs):
            with torch.no_grad():
                return [layer.cpu().numpy().astype(np.float64) for layer in model.encode_layers(raw_to_tokens(raw_inputs))[0]]

        analysis = build_analysis_payload(
            bundle,
            seed,
            predict_logits,
            encode_repr,
            encode_layers,
            patch_size=max(1, config.patch_size),
        )

    return {
        "architecture": "transformer",
        "model": FINE_TUNE_MODEL_NAME,
        "dataset": bundle.name,
        "suite": bundle.suite,
        "seed": seed,
        "closed_form_method": attention_kind,
        "classifier_accuracy": metric_summary["classifier_accuracy"],
        "validation_cross_entropy": metric_summary.get("validation_cross_entropy"),
        "validation_perplexity": metric_summary.get("validation_perplexity"),
        "scaling_metric_name": metric_summary["scaling_metric_name"],
        "scaling_metric_label": metric_summary["scaling_metric_label"],
        "scaling_metric_value": metric_summary["scaling_metric_value"],
        "scaling_fit_type": metric_summary["scaling_fit_type"],
        "layers": layers,
        "hidden_param_count": int(sum(p.numel() for p in model.blocks.parameters())),
        "output_param_count": int(sum(p.numel() for p in model.output_heads.parameters())),
        "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "fit_time_sec": fit_time,
        "closed_form_init_time_sec": state["fit_time_sec"],
        "fine_tune_time_sec": fine_tune_time,
        "fine_tune_steps": fine_tune_steps,
        "fine_tune_effective_epochs": float(fine_tune_steps / max(len(train_loader), 1)),
        "compute_proxy": used_budget,
        "compute_unit": "ops_proxy",
        "epoch_stats": epoch_stats,
        "config": {**asdict(config), "compute_matched_to": "backprop"},
        "analysis": analysis,
    }


def run_backprop_transformer(bundle: RawDatasetBundle, config: TransformerEvalConfig, seed: int, collect_analysis: bool = False):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, test_raw, token_dim, num_heads, to_tokens = prepare_transformer_training_context(bundle, config, device)
    train_loader = make_train_loader(train_ds, batch_size=config.batch_size, shuffle=True)
    model = TokenTransformerClassifier(token_dim, num_heads, config.depth, bundle.num_classes, config.mlp_ratio, task_type=bundle.task_type).to(device)
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
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_stats.append({"loss": float(np.mean(losses))})
    fit_time = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        tokens_te = to_tokens(test_raw, augment=False)
        depth_logits = model(tokens_te)
        final_logits = depth_logits[-1]
        final_logits_np = final_logits.cpu().numpy()
        pred = final_logits.argmax(dim=1).cpu().numpy()
        layers = []
        for depth_idx, logits in enumerate(depth_logits, start=1):
            layers.append(
                {
                    "depth": depth_idx,
                    "classifier_accuracy": float((logits.argmax(dim=1).cpu().numpy() == bundle.yte).mean()),
                }
            )
    metric_summary = scaling_metrics_from_logits(final_logits_np, bundle.yte, bundle.task_type)

    analysis = None
    if collect_analysis:
        def raw_to_tokens(raw_inputs):
            if bundle.modality == "image":
                tensor = torch.from_numpy(raw_inputs).float().to(device)
            elif bundle.modality == "text":
                tensor = torch.from_numpy(raw_inputs).long().to(device)
            else:
                tensor = torch.from_numpy(raw_inputs).float().to(device)
            return to_tokens(tensor, augment=False)

        def predict_logits(raw_inputs):
            with torch.no_grad():
                return model(raw_to_tokens(raw_inputs))[-1].cpu().numpy()

        def encode_repr(raw_inputs):
            with torch.no_grad():
                hidden = model.encode(raw_to_tokens(raw_inputs))
                return pool_sequence_torch(hidden, bundle.task_type).cpu().numpy().astype(np.float64)

        def encode_layers(raw_inputs):
            with torch.no_grad():
                hidden = raw_to_tokens(raw_inputs)
                collected = [pool_sequence_torch(hidden, bundle.task_type).cpu().numpy().astype(np.float64)]
                for block in model.blocks:
                    hidden = block(hidden)
                    collected.append(pool_sequence_torch(hidden, bundle.task_type).cpu().numpy().astype(np.float64))
                return collected

        analysis = build_analysis_payload(
            bundle,
            seed,
            predict_logits,
            encode_repr,
            encode_layers,
            patch_size=max(1, config.patch_size),
        )

    return {
        "architecture": "transformer",
        "model": "backprop",
        "dataset": bundle.name,
        "suite": bundle.suite,
        "seed": seed,
        "classifier_accuracy": metric_summary["classifier_accuracy"],
        "validation_cross_entropy": metric_summary.get("validation_cross_entropy"),
        "validation_perplexity": metric_summary.get("validation_perplexity"),
        "scaling_metric_name": metric_summary["scaling_metric_name"],
        "scaling_metric_label": metric_summary["scaling_metric_label"],
        "scaling_metric_value": metric_summary["scaling_metric_value"],
        "scaling_fit_type": metric_summary["scaling_fit_type"],
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
    mlp_records = []
    transformer_records = []

    for path in base.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        rows = []
        top_config = {}
        if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
            rows = data["results"]
            top_config = data.get("config") or {}
        elif isinstance(data, dict) and "classifier_accuracy" in data:
            rows = [data]
            top_config = data.get("config") or {}

        for row in rows:
            method = row.get("layer_method")
            acc = row.get("classifier_accuracy")
            if (
                method in MLP_CANDIDATES
                and acc is not None
                and row.get("dataset") == "cifar100"
                and row.get("suite") == "random-affine"
            ):
                mlp_records.append(
                    {
                        "method": method,
                        "accuracy": float(acc),
                        "config": {
                            "dual_mapping": bool(row.get("dual_mapping", top_config.get("dual_mapping", DEFAULT_MLP_CONFIG_OVERRIDES["dual_mapping"]))),
                            "output_source": (
                                row.get("output_source")
                                or top_config.get("output_source")
                                or DEFAULT_MLP_CONFIG_OVERRIDES["output_source"]
                            ),
                            "center_after_hidden": bool(
                                row.get("center_after_hidden", top_config.get("center_after_hidden", DEFAULT_MLP_CONFIG_OVERRIDES["center_after_hidden"]))
                            ),
                        },
                    }
                )

            model_name = row.get("model", "")
            if isinstance(model_name, str) and model_name.startswith("closed-form-transformer:") and acc is not None:
                kind = model_name.split(":", 1)[1]
                row_config = row.get("config") or top_config
                if kind in TRANSFORMER_CANDIDATES:
                    if (
                        row_config.get("dataset") == "cifar100"
                        and row_config.get("suite") == "random-affine"
                        and int(row_config.get("depth", 3)) == 3
                        and int(row_config.get("patch_size", 8)) == 8
                        and int(row_config.get("n_train", 10000)) >= 10000
                        and int(row_config.get("n_test", 2000)) >= 2000
                    ):
                        transformer_records.append(
                            {
                                "method": kind,
                                "accuracy": float(acc),
                                "config": {
                                    "analytic_num_heads": int(row_config.get("analytic_num_heads", DEFAULT_TRANSFORMER_CONFIG_OVERRIDES["analytic_num_heads"])),
                                    "attention_target": row_config.get("attention_target", DEFAULT_TRANSFORMER_CONFIG_OVERRIDES["attention_target"]),
                                    "attention_rank": int(row_config.get("attention_rank", 0)),
                                    "num_landmarks": int(row_config.get("num_landmarks", 8)),
                                    "local_sigma": float(row_config.get("local_sigma", 1.5)),
                                    "attention_power_iters": int(row_config.get("attention_power_iters", DEFAULT_TRANSFORMER_CONFIG_OVERRIDES["attention_power_iters"])),
                                    "attention_num_bags": int(row_config.get("attention_num_bags", DEFAULT_TRANSFORMER_CONFIG_OVERRIDES["attention_num_bags"])),
                                    "attention_bag_fraction": float(
                                        row_config.get("attention_bag_fraction", DEFAULT_TRANSFORMER_CONFIG_OVERRIDES["attention_bag_fraction"])
                                    ),
                                },
                            }
                        )

    def choose(records, fallback_method, fallback_config):
        grouped = defaultdict(list)
        for record in records:
            key = (record["method"], json.dumps(record["config"], sort_keys=True))
            grouped[key].append(record)
        if not grouped:
            return fallback_method, fallback_config, []

        summary = []
        for (method, _), items in grouped.items():
            vals = [item["accuracy"] for item in items]
            summary.append(
                {
                    "closed_form_method": method,
                    "config": items[0]["config"],
                    "mean_accuracy": float(np.mean(vals)),
                    "std_accuracy": float(np.std(vals, ddof=0)),
                    "best_accuracy": float(np.max(vals)),
                    "num_logs": len(vals),
                    "selection_score": float(np.mean(vals) - 0.35 * np.std(vals, ddof=0) + 0.004 * np.log1p(len(vals))),
                }
            )
        supported = [row for row in summary if row["num_logs"] >= 3]
        if not supported:
            supported = [row for row in summary if row["num_logs"] >= 2]
        source = supported if supported else summary
        source.sort(
            key=lambda row: (
                row["selection_score"],
                row["mean_accuracy"],
                row["best_accuracy"],
                row["num_logs"],
            ),
            reverse=True,
        )
        summary.sort(
            key=lambda row: (
                row["selection_score"],
                row["mean_accuracy"],
                row["best_accuracy"],
                row["num_logs"],
            ),
            reverse=True,
        )
        winner = source[0]
        return winner["closed_form_method"], winner["config"], summary

    mlp_winner, mlp_overrides, mlp_summary = choose(mlp_records, DEFAULT_MLP_WINNER, DEFAULT_MLP_CONFIG_OVERRIDES)
    transformer_winner, transformer_overrides, transformer_summary = choose(
        transformer_records,
        DEFAULT_TRANSFORMER_WINNER,
        DEFAULT_TRANSFORMER_CONFIG_OVERRIDES,
    )
    robust_transformer_rows = [row for row in transformer_summary if row["num_logs"] >= 3]
    if robust_transformer_rows:
        best_mean = max(row["mean_accuracy"] for row in robust_transformer_rows)
        near_best = [row for row in robust_transformer_rows if best_mean - row["mean_accuracy"] <= 0.002]

        def transformer_compute_hint(row):
            config = row["config"]
            kind = row["closed_form_method"]
            score_multiplier = config.get("attention_power_iters", 1) if "score-" in kind else 1
            bag_multiplier = config.get("attention_num_bags", 1) if "bagged" in kind else 1
            return int(score_multiplier) * int(bag_multiplier)

        near_best.sort(
            key=lambda row: (
                transformer_compute_hint(row),
                -row["mean_accuracy"],
                -row["best_accuracy"],
                -row["num_logs"],
            )
        )
        transformer_winner = near_best[0]["closed_form_method"]
        transformer_overrides = near_best[0]["config"]
    return {
        "mlp_winner": mlp_winner,
        "mlp_config_overrides": mlp_overrides,
        "transformer_winner": transformer_winner,
        "transformer_config_overrides": transformer_overrides,
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
    layer_table = []
    latent_table = []
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
                "view_alignment_cosine": interpretability.get("view_alignment_cosine"),
                "view_alignment_gap": interpretability.get("view_alignment_gap"),
                "augmentation_prediction_agreement": interpretability.get("augmentation_prediction_agreement"),
                "base_view_prediction_agreement": interpretability.get("base_view_prediction_agreement"),
                "final_cka_to_input": interpretability.get("final_cka_to_input"),
                "final_cka_to_labels": interpretability.get("final_cka_to_labels"),
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
        for item in analysis.get("layerwise_geometry") or []:
            layer_table.append(
                {
                    "run_id": run_id,
                    "architecture": row["architecture"],
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "seed": row["seed"],
                    "layer": item["layer"],
                    "cka_to_input": item["cka_to_input"],
                    "cka_to_labels": item["cka_to_labels"],
                    "effective_rank": item["effective_rank"],
                }
            )
        for item in analysis.get("umap_points") or []:
            latent_table.append(
                {
                    "run_id": run_id,
                    "architecture": row["architecture"],
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "seed": row["seed"],
                    "point_index": item["point_index"],
                    "x": item["x"],
                    "y": item["y"],
                    "label": item["label"],
                    "group_label": item["group_label"],
                }
            )
    return {
        "run_table": run_table,
        "ood_table": ood_table,
        "layer_table": layer_table,
        "latent_table": latent_table,
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


def fit_log_linear(x_values, y_values):
    xs = np.asarray(x_values, dtype=np.float64)
    ys = np.asarray(y_values, dtype=np.float64)
    mask = (xs > 0.0) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 3:
        return None
    log_x = np.log(xs)
    slope, intercept = np.polyfit(log_x, ys, 1)
    pred = slope * log_x + intercept
    ss_res = np.sum((ys - pred) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "x_min": float(xs.min()),
        "x_max": float(xs.max()),
        "range_ratio": float(xs.max() / xs.min()),
    }


def estimate_transformer_compute_proxy(
    bundle: RawDatasetBundle,
    config: TransformerEvalConfig,
    model_kind: str,
    closed_form_method: str | None = None,
    implementation: str = "current",
):
    if bundle.modality == "image":
        token_dim = int(bundle.train_raw.shape[1] * config.patch_size * config.patch_size)
        num_tokens = int((bundle.image_size // config.patch_size) ** 2)
    elif bundle.modality == "text":
        token_dim = int(bundle.text_embedding.shape[1])
        num_tokens = int(bundle.train_raw.shape[1])
    elif bundle.modality == "tabular":
        token_dim = int(bundle.tabular_token_embedding.shape[1])
        num_tokens = int(bundle.train_raw.shape[1])
    else:
        raise ValueError(f"Unsupported modality for compute proxy: {bundle.modality}")

    train_count = int(len(bundle.ytr))
    num_heads = choose_num_heads(token_dim, preferred=config.num_heads)
    analytic_heads = max(1, min(config.analytic_num_heads, num_heads))
    rank = max(1, config.attention_rank if config.attention_rank > 0 else config.num_landmarks * analytic_heads)
    hidden_ffn = max(int(token_dim * config.mlp_ratio), token_dim)
    eval_count = int(len(bundle.yte))

    if model_kind == "backprop":
        per_layer_cost = (
            4.0 * num_tokens * token_dim * token_dim
            + 2.0 * num_tokens * num_tokens * token_dim
            + 2.0 * num_tokens * token_dim * hidden_ffn
        )
        return float(config.epochs * train_count * config.depth * (per_layer_cost + token_dim * bundle.num_classes))

    if model_kind == FINE_TUNE_MODEL_NAME:
        head_count = analytic_heads
        per_layer_cost = (
            num_tokens * token_dim * token_dim
            + num_tokens * token_dim * rank
            + num_tokens * num_tokens * rank
            + head_count * num_tokens * num_tokens * token_dim
            + num_tokens * head_count * token_dim * token_dim
            + num_tokens * token_dim * token_dim
            + token_dim * bundle.num_classes
        )
        return float(train_count * config.depth * per_layer_cost)

    if implementation not in {"current", "legacy-search"}:
        raise ValueError(f"Unknown transformer compute implementation: {implementation}")

    bag_multiplier = config.attention_num_bags if closed_form_method and "bagged" in closed_form_method else 1
    iter_multiplier = config.attention_power_iters if closed_form_method and "score-" in closed_form_method else 1
    fit_views = 2.0 * train_count
    apply_views = 3.0 * train_count + (1.0 if implementation == "current" else 3.0) * eval_count
    score_projection_cost = num_tokens * token_dim * rank
    score_matrix_cost = num_tokens * num_tokens * rank
    value_mix_cost = num_tokens * num_tokens * token_dim
    attention_build_cost = fit_views * (score_projection_cost + score_matrix_cost + value_mix_cost)
    power_basis_cost = config.depth * bag_multiplier * iter_multiplier * fit_views * (
        score_projection_cost + score_matrix_cost
    )
    search_candidates = 9.0 if implementation == "legacy-search" and closed_form_method == "score-self-power-gain" else 1.0
    analytic_scale_cost = (
        config.depth * fit_views * (score_projection_cost + score_matrix_cost)
        if implementation == "current" and closed_form_method == "score-self-power-gain"
        else 0.0
    )
    attention_fit_cost = config.depth * bag_multiplier * search_candidates * attention_build_cost
    attention_apply_cost = config.depth * apply_views * (score_projection_cost + score_matrix_cost + value_mix_cost)
    ffn_fit_cost = config.depth * fit_views * num_tokens * token_dim * token_dim
    ffn_apply_cost = config.depth * apply_views * num_tokens * token_dim * token_dim
    head_fit_cost = config.depth * (
        train_count * token_dim * bundle.num_classes + token_dim * token_dim * bundle.num_classes
    )
    head_apply_cost = config.depth * (train_count + eval_count) * token_dim * bundle.num_classes
    return float(
        power_basis_cost
        + analytic_scale_cost
        + attention_fit_cost
        + attention_apply_cost
        + ffn_fit_cost
        + ffn_apply_cost
        + head_fit_cost
        + head_apply_cost
    )


def summarize_scaling_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["architecture"], row["model"], row["axis"], row["scale_value"])].append(row)

    aggregated = []
    for (architecture, model, axis, scale_value), items in sorted(grouped.items()):
        accs = [item["classifier_accuracy"] for item in items]
        params = [item["total_parameter_count"] for item in items]
        times = [item["fit_time_sec"] for item in items]
        computes = [item["compute_proxy"] for item in items]
        metric_name = items[0].get("scaling_metric_name", "error_rate")
        metric_label = items[0].get("scaling_metric_label", "Error rate")
        fit_type = items[0].get("scaling_fit_type", "power-law")
        metric_vals = [float(item.get("scaling_metric_value", 1.0 - item["classifier_accuracy"])) for item in items]
        aggregated.append(
            {
                "architecture": architecture,
                "model": model,
                "axis": axis,
                "scale_value": scale_value,
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs, ddof=0)),
                "mean_scaling_metric": float(np.mean(metric_vals)),
                "std_scaling_metric": float(np.std(metric_vals, ddof=0)),
                "scaling_metric_name": metric_name,
                "scaling_metric_label": metric_label,
                "scaling_fit_type": fit_type,
                "mean_parameter_count": float(np.mean(params)),
                "mean_fit_time_sec": float(np.mean(times)),
                "mean_compute_proxy": float(np.mean(computes)),
                "compute_unit": items[0].get("compute_unit", "compute_proxy"),
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
                elif axis in {"compute", CONTEXT_LENGTH_AXIS_NAME}:
                    x_vals = [row["mean_compute_proxy"] for row in rows_axis]
                else:
                    continue
                metric_name = rows_axis[0].get("scaling_metric_name", "error_rate")
                metric_label = rows_axis[0].get("scaling_metric_label", "Error rate")
                fit_type = rows_axis[0].get("scaling_fit_type", "power-law")
                y_vals = [row["mean_scaling_metric"] for row in rows_axis]
                fit = fit_log_linear(x_vals, y_vals) if fit_type == "log-linear" else fit_power_law(x_vals, y_vals)
                if fit is None:
                    continue
                fits.append(
                    {
                        "architecture": architecture,
                        "model": model,
                        "axis": axis,
                        "metric_name": metric_name,
                        "metric_label": metric_label,
                        "fit_type": fit_type,
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
    present_axes = {row["axis"] for row in aggregated_rows}
    axes_order = [axis_name for axis_name in [*SCALING_AXES, CONTEXT_LENGTH_AXIS_NAME] if axis_name in present_axes]
    if not axes_order:
        axes_order = sorted(present_axes)
    architectures = sorted({row["architecture"] for row in aggregated_rows})
    fig, axs = plt.subplots(len(architectures), len(axes_order), figsize=(12, 3.8 * max(len(architectures), 1)))
    axs = np.atleast_2d(axs)

    for row_idx, architecture in enumerate(architectures):
        for col_idx, axis_name in enumerate(axes_order):
            ax = axs[row_idx, col_idx]
            subset = [row for row in aggregated_rows if row["architecture"] == architecture and row["axis"] == axis_name]
            fit_subset = [row for row in fit_rows if row["architecture"] == architecture and row["axis"] == axis_name]
            if not subset:
                ax.set_visible(False)
                continue
            for model in sorted({row["model"] for row in subset}):
                rows_model = [row for row in subset if row["model"] == model]
                if axis_name == "data":
                    x_vals = [row["scale_value"] for row in rows_model]
                elif axis_name == "parameters":
                    x_vals = [row["mean_parameter_count"] for row in rows_model]
                else:
                    x_vals = [row["mean_compute_proxy"] for row in rows_model]
                y_vals = [row["mean_scaling_metric"] for row in rows_model]
                ax.plot(x_vals, y_vals, marker="o", label=model)
            metric_label = subset[0].get("scaling_metric_label", "Error rate")
            fit_type = subset[0].get("scaling_fit_type", "power-law")
            for fit_idx, fit in enumerate(fit_subset):
                ax.text(
                    0.02,
                    0.95 - 0.1 * fit_idx,
                    f"{fit['model']}: slope={fit['slope']:.3f}, R2={fit['r2']:.2f}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8,
                )
            ax.set_xscale("log")
            if fit_type == "power-law":
                ax.set_yscale("log")
            ax.set_title(f"{architecture} {axis_name}")
            ax.set_xlabel("compute proxy" if axis_name in {"compute", CONTEXT_LENGTH_AXIS_NAME} else axis_name)
            ax.set_ylabel(metric_label)
            ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def plot_ood_results(run_rows, ood_rows, output_path):
    plt = maybe_import_pyplot()
    if plt is None or not ood_rows or not run_rows:
        return False
    datasets = sorted({row["dataset"] for row in ood_rows})
    model_order = [("mlp", "closed-form"), ("mlp", "backprop"), ("transformer", "closed-form"), ("transformer", "backprop")]
    base_grouped = defaultdict(list)
    grouped = defaultdict(list)
    for row in run_rows:
        base_grouped[(row["dataset"], row["architecture"], row["model"])].append(row["classifier_accuracy"])
    for row in ood_rows:
        grouped[(row["dataset"], row["architecture"], row["model"], row["shift"])].append(row["accuracy"])

    fig, axs = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 4.2), squeeze=False)
    for ax, dataset in zip(axs[0], datasets):
        shifts = sorted({key[3] for key in grouped if key[0] == dataset})
        condition_names = ["in-distribution"] + shifts
        x = np.arange(len(model_order))
        width = min(0.18, 0.76 / max(len(condition_names), 1))
        for cond_idx, condition_name in enumerate(condition_names):
            means = []
            for architecture, model in model_order:
                if condition_name == "in-distribution":
                    vals = base_grouped.get((dataset, architecture, model), [])
                else:
                    vals = grouped.get((dataset, architecture, model, condition_name), [])
                means.append(float(np.mean(vals)) if vals else np.nan)
            ax.bar(
                x + (cond_idx - 0.5 * (len(condition_names) - 1)) * width,
                means,
                width=width,
                label=condition_name,
            )
        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{architecture}\n{model}" for architecture, model in model_order])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Accuracy")
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=min(len(labels), 4))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def plot_layer_geometry(layer_rows, output_path):
    plt = maybe_import_pyplot()
    if plt is None or not layer_rows:
        return False
    datasets = sorted({row["dataset"] for row in layer_rows})
    model_order = [("mlp", "closed-form"), ("mlp", "backprop"), ("transformer", "closed-form"), ("transformer", "backprop")]
    grouped = defaultdict(list)
    for row in layer_rows:
        grouped[(row["dataset"], row["architecture"], row["model"], row["layer"])].append(row)

    fig, axs = plt.subplots(2, len(datasets), figsize=(4.2 * len(datasets), 7), squeeze=False, sharex=False)
    for col_idx, dataset in enumerate(datasets):
        for architecture, model in model_order:
            layers = sorted({key[3] for key in grouped if key[0] == dataset and key[1] == architecture and key[2] == model})
            if not layers:
                continue
            cka_input = [float(np.mean([item["cka_to_input"] for item in grouped[(dataset, architecture, model, layer)]])) for layer in layers]
            cka_labels = [float(np.mean([item["cka_to_labels"] for item in grouped[(dataset, architecture, model, layer)]])) for layer in layers]
            axs[0, col_idx].plot(layers, cka_input, marker="o", label=f"{architecture}-{model}")
            axs[1, col_idx].plot(layers, cka_labels, marker="o", label=f"{architecture}-{model}")
        axs[0, col_idx].set_title(dataset)
        axs[0, col_idx].set_ylabel("CKA to input")
        axs[1, col_idx].set_ylabel("CKA to labels")
        axs[1, col_idx].set_xlabel("Layer")
        axs[0, col_idx].set_ylim(0.0, 1.05)
        axs[1, col_idx].set_ylim(0.0, 1.05)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def plot_umap_results(latent_rows, output_path):
    plt = maybe_import_pyplot()
    if plt is None or not latent_rows:
        return False
    from matplotlib.lines import Line2D  # type: ignore

    def group_sort_key(label):
        if label == "other":
            return (2, 10**9, label)
        try:
            return (0, int(label), label)
        except Exception:
            return (1, label)

    datasets = sorted({row["dataset"] for row in latent_rows})
    model_order = [("mlp", "closed-form"), ("mlp", "backprop"), ("transformer", "closed-form"), ("transformer", "backprop")]
    dataset_groups = {
        dataset: sorted({row["group_label"] for row in latent_rows if row["dataset"] == dataset}, key=group_sort_key)
        for dataset in datasets
    }
    fig, axs = plt.subplots(len(datasets), len(model_order), figsize=(3.3 * len(model_order), 3.4 * len(datasets)), squeeze=False)

    for row_idx, dataset in enumerate(datasets):
        groups = dataset_groups[dataset]
        cmap = plt.get_cmap("tab10" if len(groups) <= 10 else "tab20")
        color_map = {group_label: cmap(group_idx % cmap.N) for group_idx, group_label in enumerate(groups)}
        for col_idx, (architecture, model) in enumerate(model_order):
            ax = axs[row_idx, col_idx]
            subset = [row for row in latent_rows if row["dataset"] == dataset and row["architecture"] == architecture and row["model"] == model]
            if not subset:
                ax.set_visible(False)
                continue
            for group_label in groups:
                items = [row for row in subset if row["group_label"] == group_label]
                if not items:
                    continue
                ax.scatter(
                    [row["x"] for row in items],
                    [row["y"] for row in items],
                    s=9,
                    alpha=0.75,
                    color=color_map[group_label],
                )
            if row_idx == 0:
                ax.set_title(f"{architecture}-{model}")
            if col_idx == 0:
                ax.set_ylabel(dataset)
            ax.set_xticks([])
            ax.set_yticks([])
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color_map[group_label],
                markeredgecolor=color_map[group_label],
                markersize=5,
                label=group_label,
            )
            for group_label in groups
        ]
        axs[row_idx, 0].legend(
            handles=legend_handles,
            title=f"{dataset} labels",
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02, float(len(model_order)), 0.25),
            mode="expand",
            ncol=min(len(groups), 6),
            borderaxespad=0.0,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=2.0)
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
                        task_type=base.get("task_type", "classification"),
                        text_vocab_size=base.get("text_vocab_size", DatasetSpec.text_vocab_size),
                        text_seq_len=base.get("text_seq_len", DatasetSpec.text_seq_len),
                        text_embed_dim=base.get("text_embed_dim", DatasetSpec.text_embed_dim),
                        text_drop_prob=base.get("text_drop_prob", DatasetSpec.text_drop_prob),
                        text_dataset_name=base.get("text_dataset_name"),
                        text_dataset_config=base.get("text_dataset_config"),
                        text_fields=tuple(base.get("text_fields", DatasetSpec.text_fields)),
                        label_field=base.get("label_field", DatasetSpec.label_field),
                        eval_split=base.get("eval_split", DatasetSpec.eval_split),
                        next_token_stride=base.get("next_token_stride", DatasetSpec.next_token_stride),
                    )
                )
        return SCALING_DATA_VALUES, specs

    if axis_name in {"parameters", "compute", CONTEXT_LENGTH_AXIS_NAME}:
        for seed in seeds:
            specs.append(
                DatasetSpec(
                    name=dataset_name,
                    suite=base["suite"],
                    n_train=base["n_train"],
                    n_test=base["n_test"],
                    seed=seed,
                    task_type=base.get("task_type", "classification"),
                    text_vocab_size=base.get("text_vocab_size", DatasetSpec.text_vocab_size),
                    text_seq_len=base.get("text_seq_len", DatasetSpec.text_seq_len),
                    text_embed_dim=base.get("text_embed_dim", DatasetSpec.text_embed_dim),
                    text_drop_prob=base.get("text_drop_prob", DatasetSpec.text_drop_prob),
                    text_dataset_name=base.get("text_dataset_name"),
                    text_dataset_config=base.get("text_dataset_config"),
                    text_fields=tuple(base.get("text_fields", DatasetSpec.text_fields)),
                    label_field=base.get("label_field", DatasetSpec.label_field),
                    eval_split=base.get("eval_split", DatasetSpec.eval_split),
                    next_token_stride=base.get("next_token_stride", DatasetSpec.next_token_stride),
                )
            )
        if axis_name == "parameters":
            return SCALING_TRANSFORMER_TEXT_DIMS, specs
        return SCALING_TRANSFORMER_CONTEXT_LENGTHS, specs

    raise ValueError(f"Unknown scaling axis: {axis_name}")


def run_scaling_suite(dataset_name, transformer_winner, transformer_config):
    rows = []

    for axis_name in SCALING_AXES:
        scale_values, dataset_specs = scaling_specs_for_axis(dataset_name, axis_name, SCALING_SEEDS)
        if axis_name == "data":
            for spec in dataset_specs:
                bundle = load_dataset_bundle(spec)
                tr_row = run_closed_form_transformer(bundle, transformer_winner, transformer_config, spec.seed)
                tr_row["axis"] = axis_name
                tr_row["scale_value"] = spec.n_train
                tr_row["compute_proxy"] = estimate_transformer_compute_proxy(bundle, transformer_config, "closed-form", transformer_winner)
                tr_row["compute_unit"] = "ops_proxy"
                rows.append(tr_row)
                tr_bp = run_backprop_transformer(bundle, transformer_config, spec.seed)
                tr_bp["axis"] = axis_name
                tr_bp["scale_value"] = spec.n_train
                tr_bp["compute_proxy"] = estimate_transformer_compute_proxy(bundle, transformer_config, "backprop")
                tr_bp["compute_unit"] = "ops_proxy"
                rows.append(tr_bp)
                tr_ft = run_closed_form_backprop_finetune_transformer(bundle, transformer_winner, transformer_config, spec.seed)
                tr_ft["axis"] = axis_name
                tr_ft["scale_value"] = spec.n_train
                rows.append(tr_ft)

        elif axis_name == "parameters":
            for spec in dataset_specs:
                for embed_dim in scale_values:
                    spec_dim = replace(spec, text_embed_dim=embed_dim)
                    bundle = load_dataset_bundle(spec_dim)
                    tr_cfg = TransformerEvalConfig(**{**asdict(transformer_config)})
                    tr_row = run_closed_form_transformer(bundle, transformer_winner, tr_cfg, spec.seed)
                    tr_row["axis"] = axis_name
                    tr_row["scale_value"] = embed_dim
                    tr_row["compute_proxy"] = estimate_transformer_compute_proxy(bundle, tr_cfg, "closed-form", transformer_winner)
                    tr_row["compute_unit"] = "ops_proxy"
                    rows.append(tr_row)
                    tr_bp = run_backprop_transformer(bundle, tr_cfg, spec.seed)
                    tr_bp["axis"] = axis_name
                    tr_bp["scale_value"] = embed_dim
                    tr_bp["compute_proxy"] = estimate_transformer_compute_proxy(bundle, tr_cfg, "backprop")
                    tr_bp["compute_unit"] = "ops_proxy"
                    rows.append(tr_bp)
                    tr_ft = run_closed_form_backprop_finetune_transformer(bundle, transformer_winner, tr_cfg, spec.seed)
                    tr_ft["axis"] = axis_name
                    tr_ft["scale_value"] = embed_dim
                    rows.append(tr_ft)

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
                    task_type=base.get("task_type", "classification"),
                    text_vocab_size=base.get("text_vocab_size", DatasetSpec.text_vocab_size),
                    text_seq_len=base.get("text_seq_len", DatasetSpec.text_seq_len),
                    text_embed_dim=base.get("text_embed_dim", DatasetSpec.text_embed_dim),
                    text_drop_prob=base.get("text_drop_prob", DatasetSpec.text_drop_prob),
                    text_dataset_name=base.get("text_dataset_name"),
                    text_dataset_config=base.get("text_dataset_config"),
                    text_fields=tuple(base.get("text_fields", DatasetSpec.text_fields)),
                    label_field=base.get("label_field", DatasetSpec.label_field),
                    eval_split=base.get("eval_split", DatasetSpec.eval_split),
                    next_token_stride=base.get("next_token_stride", DatasetSpec.next_token_stride),
                )
            )
    return specs


def main():
    supported_datasets = sorted(set(DATASET_REGISTRY) | {"mnist", "fashion_mnist", "cifar10"})
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
    mlp_config_overrides = {}
    transformer_config_overrides = {}
    winner_source = None

    if args.reuse_winners_from is not None:
        reuse_path = resolve_json_path(args.reuse_winners_from)
        payload = json.loads(reuse_path.read_text(encoding="utf-8"))
        mlp_winner = payload["winners"]["mlp"]
        transformer_winner = payload["winners"]["transformer"]
        mlp_config_overrides = payload["winners"].get("mlp_config_overrides") or {}
        transformer_config_overrides = payload["winners"].get("transformer_config_overrides") or {}
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
        mlp_config_overrides = inferred.get("mlp_config_overrides") or {}
        transformer_config_overrides = inferred.get("transformer_config_overrides") or {}
        mlp_selection = inferred["mlp_log_summary"]
        transformer_selection = inferred["transformer_log_summary"]
        winner_source = "existing_logs"

    if mlp_config_overrides:
        mlp_config = replace(mlp_config, **mlp_config_overrides)
    if transformer_config_overrides:
        transformer_config = replace(transformer_config, **transformer_config_overrides)

    main_rows = []
    main_summary = []
    analytics_tables = {"run_table": [], "ood_table": [], "layer_table": [], "latent_table": []}
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
            transformer_winner=transformer_winner,
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
    layer_table_path = default_json_path("broader_eval_suite_layer_table.jsonl")
    latent_table_path = default_json_path("broader_eval_suite_latent_umap_table.jsonl")
    selection_table_path = default_json_path("broader_eval_suite_selection_table.jsonl")
    scaling_table_path = default_json_path("broader_eval_suite_scaling_table.jsonl")
    scaling_summary_path = default_json_path("broader_eval_suite_scaling_summary.jsonl")
    scaling_fit_path = default_json_path("broader_eval_suite_scaling_fit_table.jsonl")
    main_plot_path = default_plot_path(f"{PLOT_SUBDIR}/broader_eval_suite_summary.png")
    scaling_plot_path = default_plot_path(f"{PLOT_SUBDIR}/broader_eval_suite_scaling.png")
    ood_plot_path = default_plot_path(f"{PLOT_SUBDIR}/broader_eval_suite_ood.png")
    layer_plot_path = default_plot_path(f"{PLOT_SUBDIR}/broader_eval_suite_interpretability.png")
    umap_plot_path = default_plot_path(f"{PLOT_SUBDIR}/broader_eval_suite_umap.png")

    main_plot_ok = bool(main_summary) and plot_main_results(main_summary, main_plot_path)
    scaling_plot_ok = bool(scaling_aggregated) and plot_scaling_results(scaling_aggregated, scaling_fits, scaling_plot_path)
    ood_plot_ok = bool(analytics_tables["run_table"]) and bool(analytics_tables["ood_table"]) and plot_ood_results(
        analytics_tables["run_table"],
        analytics_tables["ood_table"],
        ood_plot_path,
    )
    layer_plot_ok = bool(analytics_tables["layer_table"]) and plot_layer_geometry(analytics_tables["layer_table"], layer_plot_path)
    umap_plot_ok = bool(analytics_tables["latent_table"]) and plot_umap_results(analytics_tables["latent_table"], umap_plot_path)
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
            "scaling_axes": SCALING_AXES,
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
            "mlp_config_overrides": mlp_config_overrides,
            "transformer": transformer_winner,
            "transformer_config_overrides": transformer_config_overrides,
        },
        "main_results": main_rows,
        "main_summary": main_summary,
        "analytics_tables": analytics_tables,
        "scaling_results": scaling_rows,
        "scaling_summary": scaling_aggregated,
        "scaling_fits": scaling_fits,
        "artifacts": {
            "run_table": repo_relative_path(run_table_path),
            "ood_table": repo_relative_path(ood_table_path),
            "layer_table": repo_relative_path(layer_table_path),
            "latent_umap_table": repo_relative_path(latent_table_path),
            "selection_table": repo_relative_path(selection_table_path),
            "scaling_table": repo_relative_path(scaling_table_path),
            "scaling_summary": repo_relative_path(scaling_summary_path),
            "scaling_fit_table": repo_relative_path(scaling_fit_path),
            "main_plot": repo_relative_path(main_plot_path) if main_plot_ok else None,
            "scaling_plot": repo_relative_path(scaling_plot_path) if scaling_plot_ok else None,
            "ood_plot": repo_relative_path(ood_plot_path) if ood_plot_ok else None,
            "interpretability_plot": repo_relative_path(layer_plot_path) if layer_plot_ok else None,
            "umap_plot": repo_relative_path(umap_plot_path) if umap_plot_ok else None,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if analytics_tables["run_table"]:
        run_table_path.write_text("\n".join(json.dumps(row) for row in analytics_tables["run_table"]) + "\n", encoding="utf-8")
    if analytics_tables["ood_table"]:
        ood_table_path.write_text("\n".join(json.dumps(row) for row in analytics_tables["ood_table"]) + "\n", encoding="utf-8")
    if analytics_tables["layer_table"]:
        layer_table_path.write_text("\n".join(json.dumps(row) for row in analytics_tables["layer_table"]) + "\n", encoding="utf-8")
    if analytics_tables["latent_table"]:
        latent_table_path.write_text("\n".join(json.dumps(row) for row in analytics_tables["latent_table"]) + "\n", encoding="utf-8")
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
            f"{fit['metric_name']} {fit['fit_type']} slope={fit['slope']:.3f} "
            f"R2={fit['r2']:.3f} range={fit['range_ratio']:.1f}x"
        )


if __name__ == "__main__":
    main()
