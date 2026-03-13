# mnist_linear_augmentation_suites.py
#
# Hard-whitened linear VICReg experiment on MNIST with:
# - d = 32
# - expectation over k augmentations
# - reusable augmentation-matrix builders
# - image-specific augmentation suite
# - masking augmentation suite
# - comparison against same-d PCA
#
# Objective:
#   min_Y tr(Y N Y^T)   s.t.   Y Y^T = I_d
# where
#   N = Sigma_bar^{-1/2} Delta_avg Sigma_bar^{-1/2}
# and
#   Delta_avg = E_k[(I - A_k) Sigma_x (I - A_k)^T]
#
# We compare:
# - oracle eigenspace solution
# - projected gradient on the Stiefel constraint
# - PCA baseline (same latent dim)
#
# Requirements:
#   pip install numpy scipy matplotlib scikit-learn torchvision torch

import argparse
import numpy as np
from scipy.linalg import eigh, svd
from inspect import signature
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from torchvision import datasets
from torchvision.transforms import ToTensor

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ============================================================
# config
# ============================================================
SEED = 7
N_TRAIN = 12000
N_TEST = 3000
D_LATENT = 32
ITERS = 120
EVAL_EVERY = 5
REG_EPS = 1e-4

DEFAULT_SUITE_NAME = "image"   # "image", "translation", "single-translation", "random-masking", "block-masking", "blurring", "rotation", or "random-crops"

# image-specific suite params
SHIFT_PIXELS = [-2, -1, 1, 2]
SINGLE_TRANSLATION_DX = 3
SINGLE_TRANSLATION_DY = 3
BLUR_ALPHA = [0.15, 0.25]
ANISO_ALPHA = [0.10, 0.18]
DIAG_ALPHA = [0.08]
ROTATION_ANGLES_DEG = [-15.0, -7.5, 7.5, 15.0]

# masking suite params
RANDOM_MASK_RATE = 0.25
RANDOM_MASK_COUNT = 9
BLOCK_GRID_SPLITS = 3

# crop suite params
RANDOM_CROP_COUNT = 6
RANDOM_CROP_MIN_SIZE = 22
RANDOM_CROP_MAX_SIZE = 26

PROBE_MAX_ITER = 2000
PROJ_GRAD_STEP_SCALE = 0.45

rng = np.random.default_rng(SEED)


# ============================================================
# plotting style
# ============================================================
if plt is not None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
    })


# ============================================================
# data
# ============================================================
def load_mnist_numpy():
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=ToTensor())
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=ToTensor())

    Xtr = train_ds.data.numpy().reshape(-1, 28 * 28).astype(np.float64) / 255.0
    ytr = train_ds.targets.numpy()
    Xte = test_ds.data.numpy().reshape(-1, 28 * 28).astype(np.float64) / 255.0
    yte = test_ds.targets.numpy()

    idx_tr = rng.choice(len(Xtr), size=N_TRAIN, replace=False)
    idx_te = rng.choice(len(Xte), size=N_TEST, replace=False)

    Xtr, ytr = Xtr[idx_tr], ytr[idx_tr]
    Xte, yte = Xte[idx_te], yte[idx_te]

    mu = Xtr.mean(axis=0, keepdims=True)
    Xtr = Xtr - mu
    Xte = Xte - mu
    return Xtr, ytr, Xte, yte


# ============================================================
# reusable matrix helpers
# ============================================================
def flatten_index(r, c, width=28):
    return r * width + c


def image_grid_basis(h=28, w=28):
    return h, w, h * w


def shift_matrix_2d(dx, dy, h=28, w=28):
    """Linear operator for zero-padded translation."""
    p = h * w
    A = np.zeros((p, p), dtype=np.float64)
    for r in range(h):
        for c in range(w):
            rr = r - dy
            cc = c - dx
            if 0 <= rr < h and 0 <= cc < w:
                out_idx = flatten_index(r, c, w)
                in_idx = flatten_index(rr, cc, w)
                A[out_idx, in_idx] = 1.0
    return A


def bilinear_row_weights(src_r, src_c, h=28, w=28):
    weights = []
    r0 = int(np.floor(src_r))
    c0 = int(np.floor(src_c))
    fr = src_r - r0
    fc = src_c - c0
    for dr, wr in ((0, 1.0 - fr), (1, fr)):
        rr = r0 + dr
        if not (0 <= rr < h):
            continue
        for dc, wc in ((0, 1.0 - fc), (1, fc)):
            cc = c0 + dc
            if not (0 <= cc < w):
                continue
            weight = wr * wc
            if weight > 0.0:
                weights.append((rr, cc, weight))
    return weights


def rotation_matrix_2d(angle_deg, h=28, w=28):
    p = h * w
    A = np.zeros((p, p), dtype=np.float64)
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cy = 0.5 * (h - 1)
    cx = 0.5 * (w - 1)

    for r in range(h):
        for c in range(w):
            out_idx = flatten_index(r, c, w)
            y = r - cy
            x = c - cx
            src_c = cos_t * x + sin_t * y + cx
            src_r = -sin_t * x + cos_t * y + cy
            for rr, cc, weight in bilinear_row_weights(src_r, src_c, h=h, w=w):
                in_idx = flatten_index(rr, cc, w)
                A[out_idx, in_idx] += weight
    return A


def crop_resize_matrix_2d(top, left, crop_h, crop_w, h=28, w=28):
    p = h * w
    A = np.zeros((p, p), dtype=np.float64)
    scale_r = crop_h / h
    scale_c = crop_w / w

    for r in range(h):
        for c in range(w):
            out_idx = flatten_index(r, c, w)
            src_r = top + (r + 0.5) * scale_r - 0.5
            src_c = left + (c + 0.5) * scale_c - 0.5
            for rr, cc, weight in bilinear_row_weights(src_r, src_c, h=h, w=w):
                in_idx = flatten_index(rr, cc, w)
                A[out_idx, in_idx] += weight
    return A


def row_stochastic_from_kernel_offsets(offsets, weights, h=28, w=28):
    """Builds a local linear operator with zero-padding and row renormalization."""
    p = h * w
    A = np.zeros((p, p), dtype=np.float64)
    for r in range(h):
        for c in range(w):
            out_idx = flatten_index(r, c, w)
            row_sum = 0.0
            for (dr, dc), ww in zip(offsets, weights):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    in_idx = flatten_index(rr, cc, w)
                    A[out_idx, in_idx] += ww
                    row_sum += ww
            if row_sum > 0:
                A[out_idx, :] /= row_sum
            else:
                A[out_idx, out_idx] = 1.0
    return A


def convex_mix(A, B, alpha):
    return (1.0 - alpha) * A + alpha * B


def identity_matrix(p):
    return np.eye(p, dtype=np.float64)


def random_pixel_mask_matrix(mask_rate, h=28, w=28, rng=None):
    """Diagonal keep-mask. Zeroes random pixels."""
    if rng is None:
        rng = np.random.default_rng()
    p = h * w
    keep = (rng.random(p) > mask_rate).astype(np.float64)
    return np.diag(keep)


def block_mask_matrix(top, left, height, width, h=28, w=28):
    """Diagonal keep-mask with one contiguous zero block."""
    p = h * w
    keep = np.ones(p, dtype=np.float64)
    for r in range(top, min(top + height, h)):
        for c in range(left, min(left + width, w)):
            keep[flatten_index(r, c, w)] = 0.0
    return np.diag(keep)


def grid_block_specs(num_splits, h=28, w=28):
    row_edges = np.linspace(0, h, num_splits + 1, dtype=int)
    col_edges = np.linspace(0, w, num_splits + 1, dtype=int)
    specs = []
    for i in range(num_splits):
        for j in range(num_splits):
            top = row_edges[i]
            left = col_edges[j]
            height = row_edges[i + 1] - row_edges[i]
            width = col_edges[j + 1] - col_edges[j]
            specs.append((top, left, height, width))
    return specs


# ============================================================
# augmentation suite builders
# ============================================================
def build_image_augmentation_suite(h=28, w=28):
    """Translation + blur/smoothing suite. All operators are linear."""
    _, _, p = image_grid_basis(h, w)
    I = identity_matrix(p)
    mats = []

    # translations
    for s in SHIFT_PIXELS:
        mats.append(shift_matrix_2d(dx=s, dy=0, h=h, w=w))
        mats.append(shift_matrix_2d(dx=0, dy=s, h=h, w=w))

    # isotropic 4-neighbor smoothing
    neigh4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    S4 = row_stochastic_from_kernel_offsets(
        offsets=[(0, 0)] + neigh4,
        weights=[1.0] + [1.0] * 4,
        h=h, w=w
    )
    for alpha in BLUR_ALPHA:
        mats.append(convex_mix(I, S4, alpha))

    # anisotropic horizontal / vertical smoothing
    Sh = row_stochastic_from_kernel_offsets(
        offsets=[(0, 0), (0, -1), (0, 1)],
        weights=[1.0, 1.0, 1.0],
        h=h, w=w
    )
    Sv = row_stochastic_from_kernel_offsets(
        offsets=[(0, 0), (-1, 0), (1, 0)],
        weights=[1.0, 1.0, 1.0],
        h=h, w=w
    )
    for alpha in ANISO_ALPHA:
        mats.append(convex_mix(I, Sh, alpha))
        mats.append(convex_mix(I, Sv, alpha))

    # diagonal smoothing
    Sd = row_stochastic_from_kernel_offsets(
        offsets=[(0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)],
        weights=[1.0, 1.0, 1.0, 1.0, 1.0],
        h=h, w=w
    )
    for alpha in DIAG_ALPHA:
        mats.append(convex_mix(I, Sd, alpha))

    return mats


def build_blurring_augmentation_suite(h=28, w=28):
    """Simple isotropic blur operators only."""
    _, _, p = image_grid_basis(h, w)
    I = identity_matrix(p)
    mats = []

    neigh4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    S4 = row_stochastic_from_kernel_offsets(
        offsets=[(0, 0)] + neigh4,
        weights=[1.0] + [1.0] * 4,
        h=h,
        w=w,
    )
    for alpha in BLUR_ALPHA:
        mats.append(convex_mix(I, S4, alpha))
    return mats


def build_rotation_augmentation_suite(h=28, w=28):
    mats = []
    for angle in ROTATION_ANGLES_DEG:
        mats.append(rotation_matrix_2d(angle, h=h, w=w))
    return mats


def build_random_crop_augmentation_suite(h=28, w=28, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    mats = []
    for _ in range(RANDOM_CROP_COUNT):
        crop_h = int(rng.integers(RANDOM_CROP_MIN_SIZE, RANDOM_CROP_MAX_SIZE + 1))
        crop_w = int(rng.integers(RANDOM_CROP_MIN_SIZE, RANDOM_CROP_MAX_SIZE + 1))
        top = int(rng.integers(0, h - crop_h + 1))
        left = int(rng.integers(0, w - crop_w + 1))
        mats.append(crop_resize_matrix_2d(top, left, crop_h, crop_w, h=h, w=w))
    return mats


def build_translation_augmentation_suite(h=28, w=28):
    """Translation-only suite using the configured pixel shifts."""
    mats = []
    for s in SHIFT_PIXELS:
        mats.append(shift_matrix_2d(dx=s, dy=0, h=h, w=w))
        mats.append(shift_matrix_2d(dx=0, dy=s, h=h, w=w))
    return mats


def build_single_translation_augmentation_suite(h=28, w=28):
    """A single fixed translation operator."""
    return [shift_matrix_2d(dx=SINGLE_TRANSLATION_DX, dy=SINGLE_TRANSLATION_DY, h=h, w=w)]


def build_random_masking_augmentation_suite(h=28, w=28, rng=None):
    """Random pixel masking with a fixed mask rate."""
    if rng is None:
        rng = np.random.default_rng()
    mats = []
    for _ in range(RANDOM_MASK_COUNT):
        mats.append(random_pixel_mask_matrix(RANDOM_MASK_RATE, h=h, w=w, rng=rng))
    return mats


def build_block_masking_augmentation_suite(h=28, w=28):
    """Mask each cell in a 3x3 partition of the image, one cell at a time."""
    mats = []
    for top, left, hh, ww in grid_block_specs(BLOCK_GRID_SPLITS, h=h, w=w):
        mats.append(block_mask_matrix(top, left, hh, ww, h=h, w=w))
    return mats


def build_augmentation_suite(name, h=28, w=28, rng=None):
    if name == "image":
        return build_image_augmentation_suite(h=h, w=w)
    if name == "blurring":
        return build_blurring_augmentation_suite(h=h, w=w)
    if name == "translation":
        return build_translation_augmentation_suite(h=h, w=w)
    if name == "single-translation":
        return build_single_translation_augmentation_suite(h=h, w=w)
    if name == "random-masking":
        return build_random_masking_augmentation_suite(h=h, w=w, rng=rng)
    if name == "block-masking":
        return build_block_masking_augmentation_suite(h=h, w=w)
    if name == "rotation":
        return build_rotation_augmentation_suite(h=h, w=w)
    if name == "random-crops":
        return build_random_crop_augmentation_suite(h=h, w=w, rng=rng)
    raise ValueError(f"Unknown suite name: {name}")


# ============================================================
# spectral problem construction
# ============================================================
def covariance(X):
    return (X.T @ X) / X.shape[0]


def make_problem_expectation_over_k(X, aug_mats, reg_eps=1e-4):
    """
    A1 = I, A2 = A_k, averaged over k:
      Sigma_bar = E_k[ 0.5 (Sigma_x + A_k Sigma_x A_k^T) ]
      Delta_avg = E_k[ (I - A_k) Sigma_x (I - A_k)^T ]
    """
    Sigma_x = covariance(X)
    p = Sigma_x.shape[0]
    I = np.eye(p, dtype=np.float64)

    Sigma_bar = np.zeros_like(Sigma_x)
    Delta_avg = np.zeros_like(Sigma_x)

    for A in aug_mats:
        Sigma_bar += 0.5 * (Sigma_x + A @ Sigma_x @ A.T)
        M = I - A
        Delta_avg += M @ Sigma_x @ M.T

    Sigma_bar /= len(aug_mats)
    Delta_avg /= len(aug_mats)

    evals, evecs = eigh(Sigma_bar)
    evals = np.maximum(evals, reg_eps)
    Sigma_inv_sqrt = (evecs / np.sqrt(evals)) @ evecs.T

    N = Sigma_inv_sqrt @ Delta_avg @ Sigma_inv_sqrt
    N = 0.5 * (N + N.T)
    return Sigma_x, Sigma_bar, Delta_avg, Sigma_inv_sqrt, N


# ============================================================
# hard-whitened optimization
# ============================================================
def objective(Y, N):
    return float(np.trace(Y @ N @ Y.T))


def random_stiefel(d, p, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    M = rng.standard_normal((d, p))
    U, _, Vt = svd(M, full_matrices=False)
    return U @ Vt


def tangent_grad(Y, N):
    G = 2.0 * Y @ N
    sym = 0.5 * (G @ Y.T + Y @ G.T)
    return G - sym @ Y


def retract_polar(B):
    U, _, Vt = svd(B, full_matrices=False)
    return U @ Vt


def principal_angle_distance(Y, Y_star):
    s = svd(Y @ Y_star.T, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float(np.sqrt(np.sum(1.0 - s**2)))


def exact_solution(N, d):
    evals, evecs = eigh(N)
    Y_star = evecs[:, :d].T
    f_star = float(evals[:d].sum())
    return Y_star, evals, f_star


# ============================================================
# evaluation
# ============================================================
def fit_linear_probe(Ztr, ytr, Zte, yte):
    kwargs = {
        "max_iter": PROBE_MAX_ITER,
        "solver": "lbfgs",
        "n_jobs": None,
    }
    if "multi_class" in signature(LogisticRegression).parameters:
        kwargs["multi_class"] = "multinomial"

    clf = LogisticRegression(**kwargs)
    clf.fit(Ztr, ytr)
    return float((clf.predict(Zte) == yte).mean())


def compute_pca_baseline(Xtr, Xte, ytr, yte, d):
    pca = PCA(n_components=d, svd_solver="randomized", random_state=SEED)
    Ztr = pca.fit_transform(Xtr)
    Zte = pca.transform(Xte)
    acc = fit_linear_probe(Ztr, ytr, Zte, yte)
    return acc


def compute_no_aug_whiten_only_baseline(Xtr, Xte, ytr, yte, Sigma_x, d):
    """
    Baseline for the degenerate A=I case:
    choose a random whitened subspace using Sigma_x^{-1/2}.
    """
    evals, evecs = eigh(Sigma_x)
    evals = np.maximum(evals, REG_EPS)
    Sigma_inv_sqrt = (evecs / np.sqrt(evals)) @ evecs.T
    Y = random_stiefel(d, Xtr.shape[1], rng=rng)
    W = Y @ Sigma_inv_sqrt
    Ztr = Xtr @ W.T
    Zte = Xte @ W.T
    acc = fit_linear_probe(Ztr, ytr, Zte, yte)
    return acc


# ============================================================
# experiment
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Hard-whitened linear VICReg on MNIST.")
    parser.add_argument(
        "--suite",
        choices=["image", "translation", "single-translation", "random-masking", "block-masking", "blurring", "rotation", "random-crops"],
        default=DEFAULT_SUITE_NAME,
        help="Augmentation suite to evaluate.",
    )
    return parser.parse_args()


def run_experiment(suite_name):
    Xtr, ytr, Xte, yte = load_mnist_numpy()
    h, w, _ = image_grid_basis()

    aug_mats = build_augmentation_suite(suite_name, h=h, w=w, rng=rng)

    Sigma_x, Sigma_bar, Delta_avg, Sigma_inv_sqrt, N = make_problem_expectation_over_k(
        Xtr, aug_mats, reg_eps=REG_EPS
    )

    Y_star, evals_N, f_star = exact_solution(N, D_LATENT)
    W_star = Y_star @ Sigma_inv_sqrt
    Ztr_star = Xtr @ W_star.T
    Zte_star = Xte @ W_star.T
    acc_star = fit_linear_probe(Ztr_star, ytr, Zte_star, yte)

    # PCA baseline
    acc_pca = compute_pca_baseline(Xtr, Xte, ytr, yte, D_LATENT)

    # degenerate whiten-only baseline
    acc_whiten_only = compute_no_aug_whiten_only_baseline(
        Xtr, Xte, ytr, yte, Sigma_x, D_LATENT
    )

    # Projected-gradient optimization is intentionally disabled.
    # We keep the old implementation commented here for reference, but
    # downstream comparisons now focus only on the oracle eigenspace and
    # the non-iterative baselines.
    #
    # Y = random_stiefel(D_LATENT, p, rng=rng)
    # lmax = float(max(evals_N[-1], 1e-8))
    # eta = PROJ_GRAD_STEP_SCALE / lmax
    #
    # hist_it = []
    # hist_gap = []
    # hist_subspace = []
    # hist_acc = []
    #
    # for t in range(ITERS + 1):
    #     if t % EVAL_EVERY == 0 or t == ITERS:
    #         W = Y @ Sigma_inv_sqrt
    #         Ztr = Xtr @ W.T
    #         Zte = Xte @ W.T
    #         acc = fit_linear_probe(Ztr, ytr, Zte, yte)
    #         gap = objective(Y, N) - f_star
    #         dist = principal_angle_distance(Y, Y_star)
    #
    #         hist_it.append(t)
    #         hist_gap.append(gap)
    #         hist_subspace.append(dist)
    #         hist_acc.append(acc)
    #
    #     if t == ITERS:
    #         break
    #
    #     G = tangent_grad(Y, N)
    #     Y = retract_polar(Y - eta * G)

    labels = ["Oracle eigenspace", "PCA baseline", "Whiten-only baseline"]
    accuracies = [acc_star, acc_pca, acc_whiten_only]

    # ========================================================
    # plots
    # ========================================================
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.6), constrained_layout=True)
    ax.bar(labels, accuracies, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Linear probe accuracy")
    ax.set_ylabel("Test accuracy")
    ax.tick_params(axis="x", rotation=12)

    fig.suptitle(
        f"Hard-whitened linear VICReg on MNIST  |  suite={suite_name}  |  d={D_LATENT}",
        y=1.03, fontsize=10
    )
    plt.show()
    plt.savefig(f"vicreg_mnist_aug_suite_{suite_name}.png", dpi=150)

    # concise summary
    print("=" * 68)
    print("Experiment summary")
    print("=" * 68)
    print(f"latent dim d                : {D_LATENT}")
    print(f"train / test samples        : {N_TRAIN} / {N_TEST}")
    print(f"augmentation suite          : {suite_name}")
    print(f"number of augmentations     : {len(aug_mats)}")
    print("projected gradient          : disabled")
    print(f"oracle objective            : {f_star:.6e}")
    print(f"oracle probe accuracy       : {acc_star:.4f}")
    print(f"PCA probe accuracy          : {acc_pca:.4f}")
    print(f"whiten-only probe accuracy  : {acc_whiten_only:.4f}")
    # print(f"projected-grad step         : {eta:.3e}")
    # print(f"final objective gap         : {hist_gap[-1]:.6e}")
    # print(f"final subspace error        : {hist_subspace[-1]:.6e}")
    # print(f"final probe accuracy        : {hist_acc[-1]:.4f}")

    # optional: inspect eigengap around d
    if D_LATENT < len(evals_N):
        gap = evals_N[D_LATENT] - evals_N[D_LATENT - 1]
        print(f"eigengap lambda[d+1]-lambda[d] : {gap:.6e}")


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.suite)
