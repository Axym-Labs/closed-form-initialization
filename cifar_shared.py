from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

SINGLE_TRANSLATION_DX = 3
SINGLE_TRANSLATION_DY = 3
BLOCK_GRID_SPLITS = 3
RANDOM_TRANSLATION_MAX = 3
RANDOM_AFFINE_MAX_DEG = 12.0
RANDOM_AFFINE_MAX_SHIFT = 0.125
RANDOM_AFFINE_MIN_SCALE = 0.9
RANDOM_AFFINE_MAX_SCALE = 1.1


def images_to_flat(images):
    return images.reshape(images.shape[0], -1)


def _resize_images_exact_width(images, width):
    channels = images.shape[1]
    target_side = max(1, int(round(np.sqrt(width / channels))))
    tensor = torch.from_numpy(images).float()
    resized = F.interpolate(tensor, size=(target_side, target_side), mode="bilinear", align_corners=False)
    flat = resized.numpy().reshape(resized.shape[0], -1).astype(np.float64)
    if flat.shape[1] == width:
        return flat

    src = np.linspace(0.0, 1.0, flat.shape[1], dtype=np.float64)
    dst = np.linspace(0.0, 1.0, width, dtype=np.float64)
    return np.stack([np.interp(dst, src, row) for row in flat], axis=0)


@lru_cache(maxsize=16)
def _load_cifar_cached(dataset_name, n_train, n_test, seed, width):
    rng = np.random.default_rng(seed)
    dataset_cls = {
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
    }[dataset_name]

    train_ds = dataset_cls(root="./data", train=True, download=True)
    test_ds = dataset_cls(root="./data", train=False, download=True)

    xtr = train_ds.data.astype(np.float64) / 255.0
    xte = test_ds.data.astype(np.float64) / 255.0
    ytr = np.asarray(train_ds.targets)
    yte = np.asarray(test_ds.targets)

    idx_tr = rng.choice(len(xtr), size=n_train, replace=False)
    idx_te = rng.choice(len(xte), size=n_test, replace=False)
    xtr = np.transpose(xtr[idx_tr], (0, 3, 1, 2))
    xte = np.transpose(xte[idx_te], (0, 3, 1, 2))
    ytr = ytr[idx_tr]
    yte = yte[idx_te]

    xtr_flat = _resize_images_exact_width(xtr, width)
    xte_flat = _resize_images_exact_width(xte, width)
    mean = xtr_flat.mean(axis=0, keepdims=True)
    xtr_flat = xtr_flat - mean
    xte_flat = xte_flat - mean

    return {
        "xtr_img": xtr,
        "xte_img": xte,
        "xtr": xtr_flat,
        "xte": xte_flat,
        "mean": mean.astype(np.float64),
        "ytr": ytr,
        "yte": yte,
        "width": width,
    }


def load_cifar_numpy(dataset_name, n_train, n_test, seed, width):
    cached = _load_cifar_cached(dataset_name, n_train, n_test, seed, width)
    return {
        "xtr_img": cached["xtr_img"].copy(),
        "xte_img": cached["xte_img"].copy(),
        "xtr": cached["xtr"].copy(),
        "xte": cached["xte"].copy(),
        "mean": cached["mean"].copy(),
        "ytr": cached["ytr"].copy(),
        "yte": cached["yte"].copy(),
        "width": cached["width"],
    }


def _random_crop(images, rng, pad=4):
    n, c, h, w = images.shape
    padded = np.pad(images, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")
    crops = np.empty_like(images)
    max_y = 2 * pad
    max_x = 2 * pad
    ys = rng.integers(0, max_y + 1, size=n)
    xs = rng.integers(0, max_x + 1, size=n)
    for i in range(n):
        crops[i] = padded[i, :, ys[i] : ys[i] + h, xs[i] : xs[i] + w]
    return crops


def _horizontal_flip(images, rng, p=0.5):
    flipped = images.copy()
    mask = rng.random(images.shape[0]) < p
    flipped[mask] = flipped[mask, :, :, ::-1]
    return flipped


def _shift_images(images, dx, dy):
    shifted = np.zeros_like(images)
    src_r0 = max(0, -dy)
    src_r1 = images.shape[2] - max(0, dy)
    dst_r0 = max(0, dy)
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    src_c0 = max(0, -dx)
    src_c1 = images.shape[3] - max(0, dx)
    dst_c0 = max(0, dx)
    dst_c1 = dst_c0 + (src_c1 - src_c0)
    shifted[:, :, dst_r0:dst_r1, dst_c0:dst_c1] = images[:, :, src_r0:src_r1, src_c0:src_c1]
    return shifted


def _random_small_translation(images, rng, max_shift=RANDOM_TRANSLATION_MAX):
    shifted = np.empty_like(images)
    dxs = rng.integers(-max_shift, max_shift + 1, size=images.shape[0])
    dys = rng.integers(-max_shift, max_shift + 1, size=images.shape[0])
    for i, (dx, dy) in enumerate(zip(dxs, dys)):
        shifted[i] = _shift_images(images[i : i + 1], dx=int(dx), dy=int(dy))[0]
    return shifted


def _random_affine(images, rng):
    n, _, h, w = images.shape
    theta = np.zeros((n, 2, 3), dtype=np.float32)
    for i in range(n):
        angle = np.deg2rad(rng.uniform(-RANDOM_AFFINE_MAX_DEG, RANDOM_AFFINE_MAX_DEG))
        scale = rng.uniform(RANDOM_AFFINE_MIN_SCALE, RANDOM_AFFINE_MAX_SCALE)
        tx = rng.uniform(-RANDOM_AFFINE_MAX_SHIFT, RANDOM_AFFINE_MAX_SHIFT)
        ty = rng.uniform(-RANDOM_AFFINE_MAX_SHIFT, RANDOM_AFFINE_MAX_SHIFT)
        c = float(np.cos(angle) * scale)
        s = float(np.sin(angle) * scale)
        theta[i] = np.array([[c, -s, tx], [s, c, ty]], dtype=np.float32)

    tensor = torch.from_numpy(images).float()
    theta_t = torch.from_numpy(theta)
    grid = F.affine_grid(theta_t, tensor.size(), align_corners=False)
    warped = F.grid_sample(tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return warped.numpy().astype(np.float64)


def _grid_block_specs(num_splits, h, w):
    row_edges = np.linspace(0, h, num_splits + 1, dtype=int)
    col_edges = np.linspace(0, w, num_splits + 1, dtype=int)
    specs = []
    for i in range(num_splits):
        for j in range(num_splits):
            specs.append(
                (
                    row_edges[i],
                    col_edges[j],
                    row_edges[i + 1] - row_edges[i],
                    col_edges[j + 1] - col_edges[j],
                )
            )
    return specs


def _block_mask_images(images, top, left, height, width):
    masked = images.copy()
    masked[:, :, top : top + height, left : left + width] = 0.0
    return masked


def _legacy_transform_specs(suite_name, h, w):
    if suite_name == "single-translation":
        return [("shift", (SINGLE_TRANSLATION_DX, SINGLE_TRANSLATION_DY))]
    if suite_name == "block-masking":
        return [("block-mask", spec) for spec in _grid_block_specs(BLOCK_GRID_SPLITS, h=h, w=w)]
    raise ValueError(f"Unsupported legacy CIFAR suite: {suite_name}")


def _apply_legacy_spec(images, spec):
    kind, params = spec
    if kind == "identity":
        return images.copy()
    if kind == "shift":
        dx, dy = params
        return _shift_images(images, dx=dx, dy=dy)
    if kind == "block-mask":
        return _block_mask_images(images, *params)
    raise ValueError(f"Unknown legacy transform kind: {kind}")


def apply_augmentation(images, suite_name, rng):
    if suite_name == "random-crop":
        return _random_crop(images, rng)
    if suite_name == "crop-flip":
        return _horizontal_flip(_random_crop(images, rng), rng)
    if suite_name == "random-small-translation":
        return _random_small_translation(images, rng)
    if suite_name == "random-affine":
        return _random_affine(images, rng)
    if suite_name == "single-translation":
        return _shift_images(images, dx=SINGLE_TRANSLATION_DX, dy=SINGLE_TRANSLATION_DY)
    if suite_name == "block-masking":
        specs = _grid_block_specs(BLOCK_GRID_SPLITS, h=images.shape[2], w=images.shape[3])
        spec = specs[int(rng.integers(len(specs)))]
        return _block_mask_images(images, *spec)
    raise ValueError(f"Unsupported CIFAR suite: {suite_name}")


def sample_same_class_pairs(X, y, seed, repeats=1):
    rng = np.random.default_rng(seed)
    class_to_indices = {}
    for idx, label in enumerate(y):
        class_to_indices.setdefault(int(label), []).append(idx)

    left = []
    right = []
    for _ in range(repeats):
        partner_idx = np.empty(X.shape[0], dtype=np.int64)
        for idx, label in enumerate(y):
            candidates = class_to_indices[int(label)]
            if len(candidates) == 1:
                partner_idx[idx] = candidates[0]
                continue
            chosen = idx
            while chosen == idx:
                chosen = candidates[int(rng.integers(len(candidates)))]
            partner_idx[idx] = chosen
        left.append(X.copy())
        right.append(X[partner_idx])
    return np.concatenate(left, axis=0), np.concatenate(right, axis=0)


def sample_pair_views(images, suite_name, seed, width, repeats=1, mean=None):
    rng = np.random.default_rng(seed)
    left = []
    right = []
    for _ in range(repeats):
        if suite_name in {"single-translation", "block-masking"}:
            # Preserve the historical semantics for the legacy linear suites:
            # transforms act on mean-centered images, not on raw images followed
            # by mean subtraction. For masking/zero-padded shifts these are not
            # equivalent because A(x - mu) != A(x) - mu when A is not identity.
            if mean is not None and images.shape[1] * images.shape[2] * images.shape[3] == mean.shape[1]:
                centered_images = images - mean.reshape(1, images.shape[1], images.shape[2], images.shape[3])
                mean_after_resize = None
            else:
                centered_images = images
                mean_after_resize = mean
            specs = [("identity", None)] + _legacy_transform_specs(suite_name, h=images.shape[2], w=images.shape[3])
            idx1 = rng.integers(len(specs), size=images.shape[0])
            idx2 = rng.integers(len(specs), size=images.shape[0])
            view1 = np.empty_like(images)
            view2 = np.empty_like(images)
            for spec_idx, spec in enumerate(specs):
                mask1 = idx1 == spec_idx
                mask2 = idx2 == spec_idx
                if np.any(mask1):
                    view1[mask1] = _apply_legacy_spec(centered_images[mask1], spec)
                if np.any(mask2):
                    view2[mask2] = _apply_legacy_spec(centered_images[mask2], spec)
        elif suite_name == "random-small-translation":
            if mean is not None and images.shape[1] * images.shape[2] * images.shape[3] == mean.shape[1]:
                centered_images = images - mean.reshape(1, images.shape[1], images.shape[2], images.shape[3])
                mean_after_resize = None
            else:
                centered_images = images
                mean_after_resize = mean
            view1 = _random_small_translation(centered_images, rng)
            view2 = _random_small_translation(centered_images, rng)
        else:
            view1 = apply_augmentation(images, suite_name, rng)
            view2 = apply_augmentation(images, suite_name, rng)
            mean_after_resize = mean
        left.append(_resize_images_exact_width(view1, width))
        right.append(_resize_images_exact_width(view2, width))
    left = np.concatenate(left, axis=0)
    right = np.concatenate(right, axis=0)
    if mean_after_resize is not None:
        left = left - mean_after_resize
        right = right - mean_after_resize
    return left, right
