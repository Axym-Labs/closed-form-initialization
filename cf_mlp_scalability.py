import argparse
import json
import math
import pickle
import random
import tarfile
import time
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.datasets import fetch_covtype, make_classification


REG_EPS = 1e-4


@dataclass(frozen=True)
class SweepPoint:
    dataset: str
    axis: str
    scale_value: int
    seed: int
    n_train: int
    n_test: int
    input_dim: int
    width: int
    depth: int
    num_classes: int
    lambda_reg: float = 1.0
    head_reg: float = 100.0
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    view_drop_prob: float = 0.15
    view_noise_std: float = 0.10


def one_hot(y, num_classes):
    eye = np.eye(num_classes, dtype=np.float64)
    return eye[y]


def softmax_cross_entropy(logits, y):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(np.clip(shifted, -60.0, 60.0))
    probs = exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), 1e-12)
    loss = -np.log(np.maximum(probs[np.arange(y.shape[0]), y], 1e-12)).mean()
    grad = probs
    grad[np.arange(y.shape[0]), y] -= 1.0
    grad /= y.shape[0]
    return float(loss), grad


def accuracy_from_logits(logits, y):
    return float((np.argmax(logits, axis=1) == y).mean())


def standardize_train_test(xtr, xte):
    mean = xtr.mean(axis=0, keepdims=True)
    std = np.maximum(xtr.std(axis=0, keepdims=True), 1e-6)
    return (xtr - mean) / std, (xte - mean) / std


def normalize_hidden_with_stats(train_arrays, test_arrays):
    mean = sum(arr.mean(axis=0, keepdims=True) for arr in train_arrays) / len(train_arrays)
    centered_train = [arr - mean for arr in train_arrays]
    centered_test = [arr - mean for arr in test_arrays]
    avg_var = sum(np.mean(arr * arr, axis=0, keepdims=True) for arr in centered_train) / len(centered_train)
    scale = np.sqrt(np.maximum(avg_var, 1e-6))
    return [arr / scale for arr in centered_train], [arr / scale for arr in centered_test], mean, scale


def build_views(x, seed, drop_prob, noise_std):
    rng = np.random.default_rng(seed)

    def one_view():
        z = x.copy()
        mask = rng.random(z.shape) < drop_prob
        z[mask] = 0.0
        z += rng.standard_normal(z.shape) * noise_std
        return z.astype(np.float64)

    return one_view(), one_view()


def resize_images_exact_width(images, width):
    channels = images.shape[1]
    target_side = max(1, int(round(np.sqrt(width / channels))))
    zoom = (1.0, 1.0, target_side / images.shape[2], target_side / images.shape[3])
    resized = ndimage.zoom(images, zoom=zoom, order=1).astype(np.float32)
    flat = resized.reshape(resized.shape[0], -1)
    if flat.shape[1] == width:
        return flat
    src = np.linspace(0.0, 1.0, flat.shape[1], dtype=np.float64)
    dst = np.linspace(0.0, 1.0, width, dtype=np.float64)
    return np.stack([np.interp(dst, src, row) for row in flat], axis=0).astype(np.float32)


def random_crop_flip_translate(images, seed, pad=4, max_shift=2):
    rng = np.random.default_rng(seed)
    n, c, h, w = images.shape
    padded = np.pad(images, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")
    augmented = np.empty_like(images)
    ys = rng.integers(0, 2 * pad + 1, size=n)
    xs = rng.integers(0, 2 * pad + 1, size=n)
    dys = rng.integers(-max_shift, max_shift + 1, size=n)
    dxs = rng.integers(-max_shift, max_shift + 1, size=n)
    flips = rng.random(n) < 0.5
    for idx in range(n):
        crop = padded[idx, :, ys[idx] : ys[idx] + h, xs[idx] : xs[idx] + w]
        if flips[idx]:
            crop = crop[:, :, ::-1]
        shifted = np.zeros_like(crop)
        dy = int(dys[idx])
        dx = int(dxs[idx])
        src_r0 = max(0, -dy)
        src_r1 = h - max(0, dy)
        dst_r0 = max(0, dy)
        src_c0 = max(0, -dx)
        src_c1 = w - max(0, dx)
        dst_c0 = max(0, dx)
        shifted[:, dst_r0 : dst_r0 + (src_r1 - src_r0), dst_c0 : dst_c0 + (src_c1 - src_c0)] = crop[
            :, src_r0:src_r1, src_c0:src_c1
        ]
        augmented[idx] = shifted
    return augmented


def ridge_regression(x, y, reg):
    gram = x.T @ x
    rhs = x.T @ y
    return np.linalg.solve(gram + reg * np.eye(gram.shape[0], dtype=np.float64), rhs)


def relu(x):
    return np.maximum(x, 0.0)


def compute_paired_stats(h1, h2):
    mean = 0.5 * (h1.mean(axis=0, keepdims=True) + h2.mean(axis=0, keepdims=True))
    h1c = h1 - mean
    h2c = h2 - mean
    sigma1 = (h1c.T @ h1c) / h1.shape[0]
    sigma2 = (h2c.T @ h2c) / h2.shape[0]
    delta = ((h1c - h2c).T @ (h1c - h2c)) / h1.shape[0]
    sigma_bar = 0.5 * (sigma1 + sigma2)
    return 0.5 * (sigma_bar + sigma_bar.T), 0.5 * (delta + delta.T)


def sqrt_and_inv_sqrt_psd(matrix):
    evals, evecs = np.linalg.eigh(0.5 * (matrix + matrix.T))
    evals = np.maximum(evals, REG_EPS)
    sqrt = (evecs * np.sqrt(evals)) @ evecs.T
    inv_sqrt = (evecs / np.sqrt(evals)) @ evecs.T
    return sqrt, inv_sqrt


def fit_cf_transform(view1, view2, width, lambda_reg):
    dim = view1.shape[1]
    out_dim = min(width, dim)
    sigma_bar, delta = compute_paired_stats(view1, view2)
    sigma_sqrt, sigma_inv_sqrt = sqrt_and_inv_sqrt_psd(sigma_bar)
    m_matrix = sigma_inv_sqrt @ delta @ sigma_inv_sqrt
    m_matrix = 0.5 * (m_matrix + m_matrix.T)
    eigvals, eigvecs = np.linalg.eigh(m_matrix)
    gains = lambda_reg / (np.maximum(eigvals, 0.0) + lambda_reg)

    if width >= dim:
        g_matrix = (eigvecs * gains) @ eigvecs.T
        transform = sigma_sqrt @ g_matrix @ sigma_inv_sqrt
        kept_gains = gains
    else:
        order = np.argsort(gains)[::-1][:out_dim]
        modes = eigvecs[:, order]
        kept_gains = gains[order]
        transform = sigma_inv_sqrt @ (modes * kept_gains)

    return {
        "transform": transform.astype(np.float64),
        "max_whitened_delta": float(np.max(eigvals)),
        "min_whitened_delta": float(np.min(eigvals)),
        "mean_gain": float(np.mean(kept_gains)),
        "min_gain": float(np.min(kept_gains)),
    }


def architecture_dims(input_dim, width, depth):
    dims = [input_dim]
    current = input_dim
    for _ in range(depth):
        current = min(width, current)
        dims.append(current)
    return dims


def estimate_cf_flops(point):
    dims = architecture_dims(point.input_dim, point.width, point.depth)
    total = 0.0
    train_count = float(point.n_train)
    test_count = float(point.n_test)
    classes = float(point.num_classes)
    for layer_idx in range(point.depth):
        d = float(dims[layer_idx])
        k = float(dims[layer_idx + 1])
        stats = 3.0 * train_count * d * d
        eigens = 20.0 * d * d * d
        transform_build = d * d * k
        apply = (3.0 * train_count + test_count) * d * k
        head = train_count * k * k + train_count * k * classes + k * k * k
        total += stats + eigens + transform_build + apply + head
    return float(total)


def estimate_backprop_step_flops(point):
    dims = architecture_dims(point.input_dim, point.width, point.depth)
    hidden = sum(dims[idx] * dims[idx + 1] for idx in range(point.depth))
    heads = sum(dims[idx + 1] * point.num_classes for idx in range(point.depth))
    return float(6.0 * point.batch_size * (hidden + heads))


def fit_cf_mlp(point, xtr, ytr, xte, yte, view1_tr=None, view2_tr=None, view1_te=None, view2_te=None):
    if view1_tr is None or view2_tr is None:
        view1_tr, view2_tr = build_views(xtr, point.seed + 101, point.view_drop_prob, point.view_noise_std)
    if view1_te is None or view2_te is None:
        view1_te, view2_te = build_views(xte, point.seed + 202, point.view_drop_prob, point.view_noise_std)
    train_arrays, test_arrays, initial_mean, initial_scale = normalize_hidden_with_stats(
        [xtr, view1_tr, view2_tr],
        [xte, view1_te, view2_te],
    )
    base_tr, view1_tr, view2_tr = train_arrays
    base_te, view1_te, view2_te = test_arrays

    y_onehot = one_hot(ytr, point.num_classes)
    yhat_tr = np.zeros_like(y_onehot)
    yhat_te = np.zeros((yte.shape[0], point.num_classes), dtype=np.float64)
    layers = []
    start = time.perf_counter()

    for layer_idx in range(point.depth):
        fitted = fit_cf_transform(view1_tr, view2_tr, point.width, point.lambda_reg)
        transform = fitted["transform"]
        base_tr = relu(base_tr @ transform)
        base_te = relu(base_te @ transform)
        view1_tr = relu(view1_tr @ transform)
        view2_tr = relu(view2_tr @ transform)
        view1_te = relu(view1_te @ transform)
        view2_te = relu(view2_te @ transform)

        out_map = ridge_regression(base_tr, y_onehot - yhat_tr, point.head_reg)
        yhat_tr = yhat_tr + base_tr @ out_map
        yhat_te = yhat_te + base_te @ out_map

        layers.append(
            {
                "depth": layer_idx + 1,
                "accuracy": accuracy_from_logits(yhat_te, yte),
                "max_whitened_delta": fitted["max_whitened_delta"],
                "min_whitened_delta": fitted["min_whitened_delta"],
                "mean_gain": fitted["mean_gain"],
                "min_gain": fitted["min_gain"],
            }
        )

        train_arrays, test_arrays, _, _ = normalize_hidden_with_stats(
            [base_tr, view1_tr, view2_tr],
            [base_te, view1_te, view2_te],
        )
        base_tr, view1_tr, view2_tr = train_arrays
        base_te, view1_te, view2_te = test_arrays

    elapsed = time.perf_counter() - start
    return {
        "model": "cf-mlp",
        "accuracy": accuracy_from_logits(yhat_te, yte),
        "cross_entropy": softmax_cross_entropy(yhat_te.copy(), yte)[0],
        "fit_time_sec": float(elapsed),
        "layers": layers,
        "initial_mean_norm": float(np.linalg.norm(initial_mean)),
        "initial_scale_mean": float(np.mean(initial_scale)),
    }


def init_backprop_params(point, rng):
    dims = architecture_dims(point.input_dim, point.width, point.depth)
    weights = []
    heads = []
    for idx in range(point.depth):
        fan_in = dims[idx]
        fan_out = dims[idx + 1]
        weights.append((rng.standard_normal((fan_in, fan_out)) * math.sqrt(2.0 / fan_in)).astype(np.float64))
        heads.append((rng.standard_normal((fan_out, point.num_classes)) * math.sqrt(1.0 / fan_out)).astype(np.float64))
    return weights, heads


def forward_residual_mlp(x, weights, heads):
    h = x
    hidden_inputs = [x]
    preacts = []
    hiddens = []
    logits = []
    cumulative = np.zeros((x.shape[0], heads[0].shape[1]), dtype=np.float64)
    for w, v in zip(weights, heads):
        z = h @ w
        h = relu(z)
        cumulative = cumulative + h @ v
        hidden_inputs.append(h)
        preacts.append(z)
        hiddens.append(h)
        logits.append(cumulative.copy())
    return hidden_inputs, preacts, hiddens, logits


def backprop_step(xb, yb, weights, heads, opt_state, step_idx, lr, weight_decay):
    hidden_inputs, preacts, hiddens, logits = forward_residual_mlp(xb, weights, heads)
    depth = len(weights)
    losses = []
    grad_logits = []
    for out in logits:
        loss, grad = softmax_cross_entropy(out.copy(), yb)
        losses.append(loss)
        grad_logits.append(grad / depth)

    grad_heads = [np.zeros_like(v) for v in heads]
    grad_hidden_direct = [np.zeros_like(h) for h in hiddens]
    grad_cumulative = np.zeros_like(logits[-1])
    for idx in range(depth - 1, -1, -1):
        grad_cumulative = grad_cumulative + grad_logits[idx]
        grad_heads[idx] += hiddens[idx].T @ grad_cumulative
        grad_hidden_direct[idx] += grad_cumulative @ heads[idx].T

    grad_weights = [np.zeros_like(w) for w in weights]
    grad_next = np.zeros_like(hiddens[-1])
    for idx in range(depth - 1, -1, -1):
        grad_h = grad_hidden_direct[idx] + grad_next
        grad_z = grad_h * (preacts[idx] > 0.0)
        grad_weights[idx] = hidden_inputs[idx].T @ grad_z
        grad_next = grad_z @ weights[idx].T

    params = weights + heads
    grads = grad_weights + grad_heads
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for param_idx, (param, grad) in enumerate(zip(params, grads)):
        if not np.all(np.isfinite(grad)):
            grad = np.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 10.0:
            grad *= 10.0 / grad_norm
        m = opt_state["m"][param_idx]
        v = opt_state["v"][param_idx]
        m *= beta1
        m += (1.0 - beta1) * grad
        v *= beta2
        v += (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**step_idx)
        v_hat = v / (1.0 - beta2**step_idx)
        param *= 1.0 - lr * weight_decay
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return float(np.mean(losses))


def fit_backprop_residual_mlp(point, xtr, ytr, xte, yte, train_norm_mean, train_norm_scale, cf_budget):
    xtr_norm = (xtr - train_norm_mean) / train_norm_scale
    xte_norm = (xte - train_norm_mean) / train_norm_scale
    rng = np.random.default_rng(point.seed + 404)
    weights, heads = init_backprop_params(point, rng)
    opt_state = {
        "m": [np.zeros_like(p) for p in weights + heads],
        "v": [np.zeros_like(p) for p in weights + heads],
    }
    step_flops = estimate_backprop_step_flops(point)
    max_steps = max(1, int(math.floor(cf_budget / max(step_flops, 1.0))))
    steps_per_epoch = max(1, int(math.ceil(point.n_train / point.batch_size)))
    indices = np.arange(point.n_train)
    losses = []
    start = time.perf_counter()
    cursor = 0
    rng.shuffle(indices)
    for step_idx in range(1, max_steps + 1):
        if cursor + point.batch_size > point.n_train:
            rng.shuffle(indices)
            cursor = 0
        batch_idx = indices[cursor : cursor + point.batch_size]
        cursor += point.batch_size
        loss = backprop_step(
            xtr_norm[batch_idx],
            ytr[batch_idx],
            weights,
            heads,
            opt_state,
            step_idx,
            point.lr,
            point.weight_decay,
        )
        losses.append(loss)
    elapsed = time.perf_counter() - start
    _, _, _, depth_logits = forward_residual_mlp(xte_norm, weights, heads)
    final_logits = depth_logits[-1]
    layer_acc = [accuracy_from_logits(out, yte) for out in depth_logits]
    return {
        "model": "backprop-residual-mlp",
        "accuracy": accuracy_from_logits(final_logits, yte),
        "cross_entropy": softmax_cross_entropy(final_logits.copy(), yte)[0],
        "fit_time_sec": float(elapsed),
        "steps": int(max_steps),
        "effective_epochs": float(max_steps / steps_per_epoch),
        "mean_train_loss": float(np.mean(losses[-min(len(losses), steps_per_epoch) :])),
        "step_flops_proxy": float(step_flops),
        "used_flops_proxy": float(max_steps * step_flops),
        "layer_accuracy": layer_acc,
    }


_SYNTH_CACHE = {}
_COVTYPE_CACHE = {}
_CIFAR100_CACHE = {}
_TINY_IMAGENET_CACHE = {}
_POINT_DATA_CACHE = {}


def ssl_view_policy(dataset):
    if dataset.endswith("_simclr"):
        return "simclr_cifar"
    if dataset.endswith("_barlow"):
        return "barlow_twins"
    return "mild_crop"


def ensure_cifar100(root=Path("data")):
    shared = Path("/home/davwis/main/data/cifar-100/cifar-100-python")
    if shared.exists():
        return shared
    root.mkdir(parents=True, exist_ok=True)
    extracted = root / "cifar-100-python"
    if extracted.exists():
        return extracted
    archive = root / "cifar-100-python.tar.gz"
    if not archive.exists():
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        print(f"downloading CIFAR100 from {url}", flush=True)
        urllib.request.urlretrieve(url, archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)
    return extracted


def load_cifar100_full():
    if "full" in _CIFAR100_CACHE:
        return _CIFAR100_CACHE["full"]
    root = ensure_cifar100()

    def load_split(name):
        with (root / name).open("rb") as handle:
            data = pickle.load(handle, encoding="latin1")
        x = data["data"].reshape(-1, 3, 32, 32).astype(np.float64) / 255.0
        y = np.asarray(data["fine_labels"], dtype=np.int64)
        return x, y

    xtr, ytr = load_split("train")
    xte, yte = load_split("test")
    _CIFAR100_CACHE["full"] = (xtr, ytr, xte, yte)
    return _CIFAR100_CACHE["full"]


def _pil_from_chw_float(image, image_module):
    arr = np.clip(np.transpose(image, (1, 2, 0)) * 255.0, 0.0, 255.0).astype(np.uint8)
    return image_module.fromarray(arr)


class _PILSSLTransform:
    def __init__(self, image_size, policy, view_index):
        self.image_size = int(image_size)
        self.policy = policy
        self.view_index = int(view_index)
        self.jitter_strength = 0.5 if policy == "simclr_cifar" else 1.0

    def _resample_bicubic(self, image):
        return getattr(getattr(image, "Resampling", image), "BICUBIC", 3)

    def _random_resized_crop(self, image):
        width, height = image.size
        area = float(width * height)
        log_ratio = (math.log(3.0 / 4.0), math.log(4.0 / 3.0))
        for _ in range(10):
            target_area = random.uniform(0.08, 1.0) * area
            aspect = math.exp(random.uniform(*log_ratio))
            crop_w = int(round(math.sqrt(target_area * aspect)))
            crop_h = int(round(math.sqrt(target_area / aspect)))
            if 0 < crop_w <= width and 0 < crop_h <= height:
                left = random.randint(0, width - crop_w)
                top = random.randint(0, height - crop_h)
                image = image.crop((left, top, left + crop_w, top + crop_h))
                return image.resize((self.image_size, self.image_size), self._resample_bicubic(image))
        crop = min(width, height)
        left = (width - crop) // 2
        top = (height - crop) // 2
        image = image.crop((left, top, left + crop, top + crop))
        return image.resize((self.image_size, self.image_size), self._resample_bicubic(image))

    def _color_jitter(self, image):
        from PIL import Image, ImageEnhance

        strength = self.jitter_strength
        factors = [
            ("brightness", random.uniform(max(0.0, 1.0 - 0.8 * strength), 1.0 + 0.8 * strength)),
            ("contrast", random.uniform(max(0.0, 1.0 - 0.8 * strength), 1.0 + 0.8 * strength)),
            ("saturation", random.uniform(max(0.0, 1.0 - 0.8 * strength), 1.0 + 0.8 * strength)),
            ("hue", random.uniform(-0.2 * strength, 0.2 * strength)),
        ]
        random.shuffle(factors)
        for kind, value in factors:
            if kind == "brightness":
                image = ImageEnhance.Brightness(image).enhance(value)
            elif kind == "contrast":
                image = ImageEnhance.Contrast(image).enhance(value)
            elif kind == "saturation":
                image = ImageEnhance.Color(image).enhance(value)
            else:
                hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
                hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + int(round(value * 255.0))) % 256).astype(np.uint8)
                image = Image.fromarray(hsv, mode="HSV").convert("RGB")
        return image

    def __call__(self, image):
        from PIL import ImageFilter, ImageOps

        image = self._random_resized_crop(image)
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
        if random.random() < 0.8:
            image = self._color_jitter(image)
        if random.random() < 0.2:
            image = ImageOps.grayscale(image).convert("RGB")
        if self.policy == "barlow_twins":
            kernel_size = max(3, int(round(0.1 * self.image_size)))
            if kernel_size % 2 == 0:
                kernel_size += 1
            blur_p = 1.0 if self.view_index == 0 else 0.1
            if random.random() < blur_p:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))
            solarize_p = 0.0 if self.view_index == 0 else 0.2
            if random.random() < solarize_p:
                image = ImageOps.solarize(image, threshold=128)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1)).astype(np.float32)


def _ssl_transform(image_size, policy, view_index):
    try:
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
    except Exception:
        return _PILSSLTransform(image_size, policy, view_index)

    crop = transforms.RandomResizedCrop(
        image_size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=InterpolationMode.BICUBIC,
    )
    jitter_strength = 0.5 if policy == "simclr_cifar" else 1.0
    color = transforms.RandomApply(
        [
            transforms.ColorJitter(
                brightness=0.8 * jitter_strength,
                contrast=0.8 * jitter_strength,
                saturation=0.8 * jitter_strength,
                hue=0.2 * jitter_strength,
            )
        ],
        p=0.8,
    )
    ops = [
        crop,
        transforms.RandomHorizontalFlip(p=0.5),
        color,
        transforms.RandomGrayscale(p=0.2),
    ]
    if policy == "barlow_twins":
        kernel_size = max(3, int(round(0.1 * image_size)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        blur_p = 1.0 if view_index == 0 else 0.1
        solarize_p = 0.0 if view_index == 0 else 0.2
        ops.extend(
            [
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))],
                    p=blur_p,
                ),
                transforms.RandomSolarize(threshold=128, p=solarize_p),
            ]
        )
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def ssl_random_crop_color_views(images, seed, policy, view_index):
    if policy == "mild_crop":
        return random_crop_flip_translate(images, seed)
    from PIL import Image

    transform = _ssl_transform(images.shape[2], policy, view_index)
    out = np.empty_like(images, dtype=np.float32)
    for idx, image in enumerate(images):
        sample_seed = int(seed + 1009 * idx + 9176 * view_index)
        random.seed(sample_seed)
        np.random.seed(sample_seed % (2**32 - 1))
        try:
            import torch

            torch.manual_seed(sample_seed)
        except Exception:
            pass
        augmented = transform(_pil_from_chw_float(image, Image))
        if hasattr(augmented, "detach"):
            out[idx] = augmented.detach().cpu().numpy().astype(np.float32)
        else:
            out[idx] = np.asarray(augmented, dtype=np.float32)
    return out


def cifar100_data(point):
    cache_key = (point.dataset, point.seed, point.n_train, point.n_test, point.input_dim)
    if cache_key in _POINT_DATA_CACHE:
        return _POINT_DATA_CACHE[cache_key]
    xtr_all, ytr_all, xte_all, yte_all = load_cifar100_full()
    rng = np.random.default_rng(point.seed)
    idx_tr = rng.choice(xtr_all.shape[0], size=point.n_train, replace=False)
    idx_te = rng.choice(xte_all.shape[0], size=point.n_test, replace=False)
    xtr_img = xtr_all[idx_tr]
    xte_img = xte_all[idx_te]
    ytr = ytr_all[idx_tr]
    yte = yte_all[idx_te]

    xtr_flat = resize_images_exact_width(xtr_img, point.input_dim)
    xte_flat = resize_images_exact_width(xte_img, point.input_dim)
    mean = xtr_flat.mean(axis=0, keepdims=True)
    xtr = (xtr_flat - mean).astype(np.float32)
    xte = (xte_flat - mean).astype(np.float32)

    policy = ssl_view_policy(point.dataset)
    view1_tr = (
        resize_images_exact_width(ssl_random_crop_color_views(xtr_img, point.seed + 101, policy, 0), point.input_dim)
        - mean
    ).astype(np.float32)
    view2_tr = (
        resize_images_exact_width(ssl_random_crop_color_views(xtr_img, point.seed + 102, policy, 1), point.input_dim)
        - mean
    ).astype(np.float32)
    view1_te = (
        resize_images_exact_width(ssl_random_crop_color_views(xte_img, point.seed + 201, policy, 0), point.input_dim)
        - mean
    ).astype(np.float32)
    view2_te = (
        resize_images_exact_width(ssl_random_crop_color_views(xte_img, point.seed + 202, policy, 1), point.input_dim)
        - mean
    ).astype(np.float32)

    data = (xtr, ytr, xte, yte, view1_tr, view2_tr, view1_te, view2_te)
    _POINT_DATA_CACHE[cache_key] = data
    return data


def tiny_imagenet_index():
    if "index" in _TINY_IMAGENET_CACHE:
        return _TINY_IMAGENET_CACHE["index"]
    from PIL import Image  # imported lazily so non-image smoke tests do not require pillow

    root = Path("/home/davwis/main/data/tiny-imagenet-200")
    wnids = [line.strip() for line in (root / "wnids.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    label_map = {wnid: idx for idx, wnid in enumerate(wnids)}
    train_items = []
    for wnid in wnids:
        image_dir = root / "train" / wnid / "images"
        for path in sorted(image_dir.glob("*.JPEG")):
            train_items.append((path, label_map[wnid]))
    val_annotations = {}
    for line in (root / "val" / "val_annotations.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            val_annotations[parts[0]] = label_map[parts[1]]
    val_items = [(root / "val" / "images" / name, label) for name, label in sorted(val_annotations.items())]
    _TINY_IMAGENET_CACHE["index"] = (train_items, val_items, Image)
    return _TINY_IMAGENET_CACHE["index"]


def load_image_items(items, indices, image_module):
    images = np.empty((len(indices), 3, 64, 64), dtype=np.float32)
    labels = np.empty((len(indices),), dtype=np.int64)
    for out_idx, item_idx in enumerate(indices):
        path, label = items[int(item_idx)]
        with image_module.open(path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        images[out_idx] = np.transpose(arr, (2, 0, 1))
        labels[out_idx] = label
    return images, labels


def tiny_imagenet_data(point):
    cache_key = (point.dataset, point.seed, point.n_train, point.n_test, point.input_dim)
    if cache_key in _POINT_DATA_CACHE:
        return _POINT_DATA_CACHE[cache_key]
    train_items, val_items, image_module = tiny_imagenet_index()
    rng = np.random.default_rng(point.seed)
    idx_tr = rng.choice(len(train_items), size=point.n_train, replace=False)
    idx_te = rng.choice(len(val_items), size=point.n_test, replace=False)
    xtr_img, ytr = load_image_items(train_items, idx_tr, image_module)
    xte_img, yte = load_image_items(val_items, idx_te, image_module)

    xtr_flat = resize_images_exact_width(xtr_img, point.input_dim)
    xte_flat = resize_images_exact_width(xte_img, point.input_dim)
    mean = xtr_flat.mean(axis=0, keepdims=True)
    xtr = (xtr_flat - mean).astype(np.float32)
    xte = (xte_flat - mean).astype(np.float32)

    policy = ssl_view_policy(point.dataset)
    view1_tr = (
        resize_images_exact_width(ssl_random_crop_color_views(xtr_img, point.seed + 101, policy, 0), point.input_dim)
        - mean
    ).astype(np.float32)
    view2_tr = (
        resize_images_exact_width(ssl_random_crop_color_views(xtr_img, point.seed + 102, policy, 1), point.input_dim)
        - mean
    ).astype(np.float32)
    view1_te = (
        resize_images_exact_width(ssl_random_crop_color_views(xte_img, point.seed + 201, policy, 0), point.input_dim)
        - mean
    ).astype(np.float32)
    view2_te = (
        resize_images_exact_width(ssl_random_crop_color_views(xte_img, point.seed + 202, policy, 1), point.input_dim)
        - mean
    ).astype(np.float32)

    data = (xtr, ytr, xte, yte, view1_tr, view2_tr, view1_te, view2_te)
    _POINT_DATA_CACHE[cache_key] = data
    return data


def synthetic_data(seed, max_train, n_test, input_dim, num_classes):
    key = (seed, max_train, n_test, input_dim, num_classes)
    if key not in _SYNTH_CACHE:
        x, y = make_classification(
            n_samples=max_train + n_test,
            n_features=input_dim,
            n_informative=max(8, input_dim // 3),
            n_redundant=max(4, input_dim // 6),
            n_repeated=0,
            n_classes=num_classes,
            n_clusters_per_class=2,
            class_sep=1.1,
            flip_y=0.03,
            random_state=seed,
        )
        x = x.astype(np.float64)
        y = y.astype(np.int64)
        xtr = x[:max_train]
        ytr = y[:max_train]
        xte = x[max_train:]
        yte = y[max_train:]
        xtr, xte = standardize_train_test(xtr, xte)
        _SYNTH_CACHE[key] = (xtr, ytr, xte, yte)
    return _SYNTH_CACHE[key]


def covtype_data(seed, max_train, n_test):
    key = (seed, max_train, n_test)
    if key not in _COVTYPE_CACHE:
        bunch = fetch_covtype(data_home="./data", download_if_missing=True)
        x = bunch.data.astype(np.float64)
        y = bunch.target.astype(np.int64) - 1
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.shape[0], size=max_train + n_test, replace=False)
        xtr = x[idx[:max_train]]
        ytr = y[idx[:max_train]]
        xte = x[idx[max_train:]]
        yte = y[idx[max_train:]]
        xtr, xte = standardize_train_test(xtr, xte)
        _COVTYPE_CACHE[key] = (xtr, ytr, xte, yte)
    return _COVTYPE_CACHE[key]


def load_point_data(point):
    if point.dataset in {
        "cifar100",
        "cifar100_fullres",
        "cifar100_resolution",
        "cifar100_fullres_width",
        "cifar100_fullres_lambda",
        "cifar100_simclr",
        "cifar100_barlow",
    }:
        return cifar100_data(point)
    if point.dataset in {"tinyimagenet200", "tinyimagenet200_simclr", "tinyimagenet200_barlow"}:
        return tiny_imagenet_data(point)
    if point.dataset == "synthetic256":
        xtr_all, ytr_all, xte, yte = synthetic_data(
            point.seed,
            max_train=32768,
            n_test=4096,
            input_dim=point.input_dim,
            num_classes=point.num_classes,
        )
        xtr = xtr_all[: point.n_train]
        ytr = ytr_all[: point.n_train]
        xte = xte[: point.n_test]
        yte = yte[: point.n_test]
        view1_tr, view2_tr = build_views(xtr, point.seed + 101, point.view_drop_prob, point.view_noise_std)
        view1_te, view2_te = build_views(xte, point.seed + 202, point.view_drop_prob, point.view_noise_std)
        return xtr, ytr, xte, yte, view1_tr, view2_tr, view1_te, view2_te
    if point.dataset == "covtype":
        xtr_all, ytr_all, xte, yte = covtype_data(point.seed, max_train=16384, n_test=4096)
        xtr = xtr_all[: point.n_train]
        ytr = ytr_all[: point.n_train]
        xte = xte[: point.n_test]
        yte = yte[: point.n_test]
        view1_tr, view2_tr = build_views(xtr, point.seed + 101, point.view_drop_prob, point.view_noise_std)
        view1_te, view2_te = build_views(xte, point.seed + 202, point.view_drop_prob, point.view_noise_std)
        return xtr, ytr, xte, yte, view1_tr, view2_tr, view1_te, view2_te
    raise ValueError(f"Unknown dataset: {point.dataset}")


def run_point(point):
    xtr, ytr, xte, yte, view1_tr, view2_tr, view1_te, view2_te = load_point_data(point)
    _, _, norm_mean, norm_scale = normalize_hidden_with_stats([xtr, view1_tr, view2_tr], [xte, view1_te, view2_te])
    cf_budget = estimate_cf_flops(point)
    start = time.perf_counter()
    cf = fit_cf_mlp(point, xtr, ytr, xte, yte, view1_tr, view2_tr, view1_te, view2_te)
    bp = fit_backprop_residual_mlp(point, xtr, ytr, xte, yte, norm_mean, norm_scale, cf_budget)
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
    return rows


def make_sweep(quick=False):
    seeds = [7] if quick else [7, 11, 19]
    data_values = [1000, 6000] if quick else [1000, 3000, 6000, 12000, 25000, 50000]
    widths = [256, 512] if quick else [128, 256, 384, 512]
    depths = [2, 3] if quick else [2, 3, 6, 9, 12, 18]
    points = []
    for seed in seeds:
        for n_train in data_values:
            points.append(
                SweepPoint(
                    dataset="cifar100",
                    axis="data",
                    scale_value=n_train,
                    seed=seed,
                    n_train=n_train,
                    n_test=1000,
                    input_dim=512,
                    width=512,
                    depth=3,
                    num_classes=100,
                )
            )
        for width in widths:
            points.append(
                SweepPoint(
                    dataset="cifar100",
                    axis="width",
                    scale_value=width,
                    seed=seed,
                    n_train=6000,
                    n_test=1000,
                    input_dim=512,
                    width=width,
                    depth=3,
                    num_classes=100,
                )
            )
        for depth in depths:
            points.append(
                SweepPoint(
                    dataset="cifar100",
                    axis="depth",
                    scale_value=depth,
                    seed=seed,
                    n_train=6000,
                    n_test=1000,
                    input_dim=512,
                    width=512,
                    depth=depth,
                    num_classes=100,
                )
            )
    return points


def make_harder_sweep(quick=False):
    seeds = [7] if quick else [7, 11, 19]
    points = []
    cifar_data_values = [6000, 50000] if quick else [6000, 25000, 50000]
    tiny_data_values = [10000, 50000] if quick else [10000, 50000, 100000]
    for seed in seeds:
        for n_train in cifar_data_values:
            points.append(
                SweepPoint(
                    dataset="cifar100_fullres",
                    axis="data_fullres",
                    scale_value=n_train,
                    seed=seed,
                    n_train=n_train,
                    n_test=1000,
                    input_dim=3072,
                    width=512,
                    depth=3,
                    num_classes=100,
                )
            )
        for depth in ([3, 9] if quick else [3, 9, 18]):
            points.append(
                SweepPoint(
                    dataset="cifar100_fullres",
                    axis="depth_fullres",
                    scale_value=depth,
                    seed=seed,
                    n_train=6000,
                    n_test=1000,
                    input_dim=3072,
                    width=512,
                    depth=depth,
                    num_classes=100,
                )
            )
        for n_train in tiny_data_values:
            points.append(
                SweepPoint(
                    dataset="tinyimagenet200",
                    axis="data_tiny",
                    scale_value=n_train,
                    seed=seed,
                    n_train=n_train,
                    n_test=10000,
                    input_dim=1024,
                    width=512,
                    depth=3,
                    num_classes=200,
                )
            )
        for depth in ([3, 9] if quick else [3, 9, 18]):
            points.append(
                SweepPoint(
                    dataset="tinyimagenet200",
                    axis="depth_tiny",
                    scale_value=depth,
                    seed=seed,
                    n_train=50000,
                    n_test=10000,
                    input_dim=1024,
                    width=512,
                    depth=depth,
                    num_classes=200,
                )
            )
    return points


def make_fullres_diagnostic_sweep(quick=False):
    seeds = [7] if quick else [7, 11, 19]
    input_dims = [512, 3072] if quick else [512, 1024, 2048, 3072]
    widths = [512, 1024] if quick else [256, 512, 1024, 2048, 3072]
    lambdas = [0.1, 1.0, 10.0] if quick else [0.01, 0.1, 1.0, 10.0, 100.0]
    points = []
    for seed in seeds:
        for n_train in ([6000] if quick else [6000, 50000]):
            for input_dim in input_dims:
                points.append(
                    SweepPoint(
                        dataset="cifar100_resolution",
                        axis=f"input_dim_n{n_train}",
                        scale_value=input_dim,
                        seed=seed,
                        n_train=n_train,
                        n_test=1000,
                        input_dim=input_dim,
                        width=512,
                        depth=3,
                        num_classes=100,
                    )
                )
        for width in widths:
            points.append(
                SweepPoint(
                    dataset="cifar100_fullres_width",
                    axis="fullres_width",
                    scale_value=width,
                    seed=seed,
                    n_train=6000,
                    n_test=1000,
                    input_dim=3072,
                    width=width,
                    depth=3,
                    num_classes=100,
                )
            )
        for lambda_reg in lambdas:
            points.append(
                SweepPoint(
                    dataset="cifar100_fullres_lambda",
                    axis="fullres_lambda",
                    scale_value=lambda_reg,
                    seed=seed,
                    n_train=6000,
                    n_test=1000,
                    input_dim=3072,
                    width=512,
                    depth=3,
                    num_classes=100,
                    lambda_reg=lambda_reg,
                )
            )
    return points


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["axis"], row["scale_value"], row["model"])].append(row)
    agg = []
    for (dataset, axis, scale_value, model), items in sorted(grouped.items()):
        acc = np.asarray([item["accuracy"] for item in items], dtype=np.float64)
        err = np.asarray([item["error_rate"] for item in items], dtype=np.float64)
        ce = np.asarray([item["cross_entropy"] for item in items], dtype=np.float64)
        agg.append(
            {
                "dataset": dataset,
                "axis": axis,
                "scale_value": scale_value,
                "model": model,
                "runs": len(items),
                "mean_accuracy": float(acc.mean()),
                "std_accuracy": float(acc.std(ddof=0)),
                "mean_error_rate": float(err.mean()),
                "std_error_rate": float(err.std(ddof=0)),
                "mean_cross_entropy": float(ce.mean()),
                "mean_fit_time_sec": float(np.mean([item["fit_time_sec"] for item in items])),
                "mean_cf_flops_proxy": float(np.mean([item["cf_flops_proxy"] for item in items])),
            }
        )
    return agg


def paired_summary(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], row["axis"], row["scale_value"], row["seed"])
        grouped[key][row["model"]] = row
    paired = []
    for (dataset, axis, scale_value, seed), models in sorted(grouped.items()):
        if "cf-mlp" not in models or "backprop-residual-mlp" not in models:
            continue
        cf = models["cf-mlp"]
        bp = models["backprop-residual-mlp"]
        paired.append(
            {
                "dataset": dataset,
                "axis": axis,
                "scale_value": scale_value,
                "seed": seed,
                "cf_accuracy": cf["accuracy"],
                "backprop_accuracy": bp["accuracy"],
                "accuracy_gap_cf_minus_backprop": cf["accuracy"] - bp["accuracy"],
                "cf_error_rate": cf["error_rate"],
                "backprop_error_rate": bp["error_rate"],
                "error_ratio_cf_over_backprop": cf["error_rate"] / max(bp["error_rate"], 1e-12),
                "cf_fit_time_sec": cf["fit_time_sec"],
                "backprop_fit_time_sec": bp["fit_time_sec"],
                "backprop_steps": bp.get("steps", 0),
                "backprop_effective_epochs": bp.get("effective_epochs", 0.0),
                "cf_flops_proxy": cf["cf_flops_proxy"],
                "backprop_used_flops_proxy": bp.get("used_flops_proxy", 0.0),
            }
        )
    return paired


def trend_summary(paired):
    grouped = defaultdict(list)
    for row in paired:
        grouped[(row["dataset"], row["axis"])].append(row)
    trends = []
    for (dataset, axis), items in sorted(grouped.items()):
        by_scale = defaultdict(list)
        for item in items:
            by_scale[item["scale_value"]].append(item)
        scales = sorted(by_scale)
        mean_gaps = [float(np.mean([r["accuracy_gap_cf_minus_backprop"] for r in by_scale[s]])) for s in scales]
        mean_ratios = [float(np.mean([r["error_ratio_cf_over_backprop"] for r in by_scale[s]])) for s in scales]
        x = np.log(np.asarray(scales, dtype=np.float64))
        if len(scales) >= 2:
            gap_slope = float(np.polyfit(x, np.asarray(mean_gaps), 1)[0])
            ratio_slope = float(np.polyfit(x, np.asarray(mean_ratios), 1)[0])
        else:
            gap_slope = 0.0
            ratio_slope = 0.0
        trends.append(
            {
                "dataset": dataset,
                "axis": axis,
                "scales": scales,
                "mean_accuracy_gaps_cf_minus_backprop": mean_gaps,
                "mean_error_ratios_cf_over_backprop": mean_ratios,
                "smallest_scale_gap": mean_gaps[0],
                "largest_scale_gap": mean_gaps[-1],
                "gap_change_largest_minus_smallest": mean_gaps[-1] - mean_gaps[0],
                "gap_slope_vs_log_scale": gap_slope,
                "largest_scale_error_ratio": mean_ratios[-1],
                "error_ratio_slope_vs_log_scale": ratio_slope,
            }
        )
    return trends


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def markdown_report(agg_rows, paired_rows, trends):
    datasets = sorted({row["dataset"] for row in paired_rows})
    if datasets == ["cifar100"]:
        title = "CF-MLP CIFAR100 Scalability Results"
        scope = "CIFAR100-only"
    else:
        title = "CF-MLP Harder Image Scalability Results"
        scope = "Harder real-image"
    lines = [
        f"# {title}",
        "",
        f"{scope} equal-FLOP comparison of the analytic CF-MLP against a backprop-trained residual MLP with the same depth stream and cumulative post-layer residual heads.",
        "",
        "Images are converted to flattened MLP features and CF paired views use NumPy random crop/flip/translation augmentations. FLOP proxy includes closed-form paired-statistic covariance work, eigensolves, transform application, and ridge-head fitting; backprop is trained for the largest integer number of minibatch Adam steps not exceeding that proxy budget.",
        "",
        "## Paired Gap by Scale",
        "",
        "| Dataset | Axis | Scale | CF acc | Backprop acc | CF - BP | CF/BP error | BP steps |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_key = defaultdict(list)
    for row in paired_rows:
        by_key[(row["dataset"], row["axis"], row["scale_value"])].append(row)
    for (dataset, axis, scale), items in sorted(by_key.items()):
        cf_acc = np.mean([r["cf_accuracy"] for r in items])
        bp_acc = np.mean([r["backprop_accuracy"] for r in items])
        gap = np.mean([r["accuracy_gap_cf_minus_backprop"] for r in items])
        ratio = np.mean([r["error_ratio_cf_over_backprop"] for r in items])
        steps = np.mean([r["backprop_steps"] for r in items])
        lines.append(f"| {dataset} | {axis} | {scale} | {cf_acc:.4f} | {bp_acc:.4f} | {gap:+.4f} | {ratio:.3f} | {steps:.1f} |")

    lines.extend(
        [
            "",
            "## Trend Tests",
            "",
            "| Dataset | Axis | Small-scale gap | Large-scale gap | Gap change | Large-scale CF/BP error |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in trends:
        lines.append(
            f"| {row['dataset']} | {row['axis']} | {row['smallest_scale_gap']:+.4f} | {row['largest_scale_gap']:+.4f} | {row['gap_change_largest_minus_smallest']:+.4f} | {row['largest_scale_error_ratio']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="CIFAR100 CF-MLP scalability sweep against equal-FLOP residual backprop MLP.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_scalability_tests/artifacts"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    points = make_sweep(quick=args.quick)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, point in enumerate(points, start=1):
        print(
            f"[{idx}/{len(points)}] {point.dataset} axis={point.axis} scale={point.scale_value} seed={point.seed} "
            f"n={point.n_train} width={point.width} depth={point.depth}",
            flush=True,
        )
        point_rows = run_point(point)
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
