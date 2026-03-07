# simplified_vicreg_mnist.py
#
# Lean experiment for the hard-whitened linear VICReg objective on MNIST:
#
#   min_Y tr(Y N Y^T)   s.t.   Y Y^T = I_d
#
# where N = Sigma^{-1/2} Delta Sigma^{-1/2}.
#
# It:
# 1) loads MNIST,
# 2) builds two fixed linear views x1 = A1 x, x2 = A2 x,
# 3) estimates Sigma_bar and Delta,
# 4) solves the exact bottom-eigenspace problem,
# 5) runs projected gradient on the Stiefel constraint,
# 6) plots objective gap, subspace error, and downstream linear accuracy.
#
# Requirements:
#   pip install numpy scipy matplotlib scikit-learn torch torchvision

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.linear_model import RidgeClassifier
from torchvision import datasets
from torchvision.transforms import ToTensor


# ----------------------------
# config
# ----------------------------
SEED = 7
N_TRAIN = 12000
N_TEST = 3000
D_LATENT = 32
ITERS = 80
EVAL_EVERY = 5
RIDGE_ALPHA = 1.0
REG_EPS = 1e-4
TRANSLATION_SHIFT_Y = 3
TRANSLATION_SHIFT_X = 3
DEVICE_DTYPE = np.float64


# ----------------------------
# style
# ----------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 140,
})


# ----------------------------
# data
# ----------------------------
def load_mnist_numpy():
    rng = np.random.default_rng(SEED)
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=ToTensor())
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=ToTensor())

    Xtr = train_ds.data.numpy().reshape(-1, 28 * 28).astype(DEVICE_DTYPE) / 255.0
    ytr = train_ds.targets.numpy()
    Xte = test_ds.data.numpy().reshape(-1, 28 * 28).astype(DEVICE_DTYPE) / 255.0
    yte = test_ds.targets.numpy()

    # deterministic subsample
    idx_tr = rng.choice(len(Xtr), size=N_TRAIN, replace=False)
    idx_te = rng.choice(len(Xte), size=N_TEST, replace=False)

    Xtr, ytr = Xtr[idx_tr], ytr[idx_tr]
    Xte, yte = Xte[idx_te], yte[idx_te]

    # center using train mean
    mu = Xtr.mean(axis=0, keepdims=True)
    Xtr = Xtr - mu
    Xte = Xte - mu
    return Xtr, ytr, Xte, yte


# ----------------------------
# fixed linear views
# ----------------------------
def make_linear_views(p, shift_y, shift_x):
    side = int(np.sqrt(p))
    if side * side != p:
        raise ValueError(f"Expected square images, got flattened dimension {p}")

    # A1 = I
    A1 = np.eye(p, dtype=DEVICE_DTYPE)

    # A2 = fixed spatial shift with zero padding
    A2 = np.zeros((p, p), dtype=DEVICE_DTYPE)
    for row in range(side):
        for col in range(side):
            src = row * side + col
            tgt_row = row + shift_y
            tgt_col = col + shift_x
            if 0 <= tgt_row < side and 0 <= tgt_col < side:
                tgt = tgt_row * side + tgt_col
                A2[tgt, src] = 1.0

    return A1, A2


# ----------------------------
# covariance objects
# ----------------------------
def covariance(X):
    return (X.T @ X) / X.shape[0]


def make_problem(X, A1, A2, reg_eps=1e-4):
    X1 = X @ A1.T
    X2 = X @ A2.T

    S1 = covariance(X1)
    S2 = covariance(X2)
    Sigma_bar = 0.5 * (S1 + S2)

    DX = X @ (A1 - A2).T
    Delta = covariance(DX)

    # regularize Sigma_bar for numerical stability
    evals, evecs = eigh(Sigma_bar)
    evals = np.maximum(evals, reg_eps)
    Sigma_inv_sqrt = (evecs / np.sqrt(evals)) @ evecs.T

    N = Sigma_inv_sqrt @ Delta @ Sigma_inv_sqrt
    N = 0.5 * (N + N.T)  # enforce symmetry
    return Sigma_bar, Sigma_inv_sqrt, Delta, N


# ----------------------------
# hard-whitened objective
# ----------------------------
def objective(Y, N):
    return float(np.trace(Y @ N @ Y.T))


def random_stiefel(d, p, seed):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((d, p), dtype=DEVICE_DTYPE)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt  # YY^T = I_d


def tangent_grad(Y, N):
    # Euclidean gradient of tr(Y N Y^T) is 2 Y N
    G = 2.0 * Y @ N
    # project to tangent space of YY^T = I
    Sym = 0.5 * (G @ Y.T + Y @ G.T)
    return G - Sym @ Y


def retract_polar(B):
    U, _, Vt = np.linalg.svd(B, full_matrices=False)
    return U @ Vt


def principal_angle_distance(Y, Y_star):
    # rows of Y and Y_star are orthonormal
    s = np.linalg.svd(Y @ Y_star.T, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float(np.sqrt(np.sum(1.0 - s**2)))


def fit_linear_probe(Ztr, ytr, Zte, yte):
    clf = RidgeClassifier(alpha=RIDGE_ALPHA)
    clf.fit(Ztr, ytr)
    return float((clf.predict(Zte) == yte).mean())


# ----------------------------
# exact optimum
# ----------------------------
def exact_solution(N, d):
    evals, evecs = eigh(N)  # ascending
    Y_star = evecs[:, :d].T
    f_star = evals[:d].sum()
    return Y_star, evals, float(f_star)


# ----------------------------
# experiment
# ----------------------------
def format_experiment_label(shift_y, shift_x):
    if shift_y == 0 and shift_x == 0:
        return "No translation"
    return f"Translation ({shift_y:+d}, {shift_x:+d}) px"


def run_experiment(Xtr, ytr, Xte, yte, d_latent, shift_y, shift_x):
    p = Xtr.shape[1]

    A1, A2 = make_linear_views(p, shift_y, shift_x)
    Sigma_bar, Sigma_inv_sqrt, Delta, N = make_problem(Xtr, A1, A2, reg_eps=REG_EPS)

    Y_star, evals_N, f_star = exact_solution(N, d_latent)

    # map to encoder W = Y Sigma^{-1/2}
    W_star = Y_star @ Sigma_inv_sqrt
    Ztr_star = Xtr @ W_star.T
    Zte_star = Xte @ W_star.T
    acc_star = fit_linear_probe(Ztr_star, ytr, Zte_star, yte)

    # projected gradient
    Y = random_stiefel(d_latent, p, seed=SEED + 1)

    # simple safe step size from spectral norm
    lmax = float(evals_N[-1])
    eta = 0.5 / max(lmax, 1e-8)

    hist_it = []
    hist_gap = []
    hist_subspace = []
    hist_acc = []

    for t in range(ITERS + 1):
        if t % EVAL_EVERY == 0 or t == ITERS:
            W = Y @ Sigma_inv_sqrt
            Ztr = Xtr @ W.T
            Zte = Xte @ W.T
            acc = fit_linear_probe(Ztr, ytr, Zte, yte)
            gap = objective(Y, N) - f_star
            dist = principal_angle_distance(Y, Y_star)

            hist_it.append(t)
            hist_gap.append(gap)
            hist_subspace.append(dist)
            hist_acc.append(acc)

        if t == ITERS:
            break

        G = tangent_grad(Y, N)
        Y = retract_polar(Y - eta * G)

    return {
        "label": format_experiment_label(shift_y, shift_x),
        "shift_y": shift_y,
        "shift_x": shift_x,
        "eta": eta,
        "f_star": f_star,
        "acc_star": acc_star,
        "hist_it": hist_it,
        "hist_gap": hist_gap,
        "hist_subspace": hist_subspace,
        "hist_acc": hist_acc,
    }


def main():
    Xtr, ytr, Xte, yte = load_mnist_numpy()
    experiments = [
        (0, 0),
        (TRANSLATION_SHIFT_Y, TRANSLATION_SHIFT_X),
    ]
    results = [
        run_experiment(Xtr, ytr, Xte, yte, D_LATENT, shift_y, shift_x)
        for shift_y, shift_x in experiments
    ]

    # ----------------------------
    # plots
    # ----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), constrained_layout=True)

    # objective gap
    for result in results:
        axes[0].plot(
            result["hist_it"],
            np.maximum(result["hist_gap"], 1e-16),
            marker="o",
            lw=1.6,
            ms=3.8,
            label=result["label"],
        )
    axes[0].axhline(1e-16, linestyle=":", lw=1.2, label="Oracle optimum")
    axes[0].set_yscale("log")
    axes[0].set_title("Objective gap")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$f(Y_t)-f(Y_\star)$")
    axes[0].legend(frameon=False, fontsize=8)

    # subspace error
    for result in results:
        axes[1].plot(
            result["hist_it"],
            np.maximum(result["hist_subspace"], 1e-16),
            marker="o",
            lw=1.6,
            ms=3.8,
            label=result["label"],
        )
    axes[1].set_yscale("log")
    axes[1].set_title("Subspace error")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$\|\sin\Theta(\mathcal{S}_t,\mathcal{S}_\star)\|_F$")
    axes[1].legend(frameon=False, fontsize=8)

    # downstream accuracy
    for result in results:
        axes[2].plot(
            result["hist_it"],
            result["hist_acc"],
            marker="o",
            lw=1.6,
            ms=3.8,
            label=result["label"],
        )
        axes[2].axhline(result["acc_star"], linestyle=":", lw=1.2)
    axes[2].set_title("Linear probe accuracy")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Test accuracy")
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("Hard-whitened linear VICReg on MNIST (d = 32)", y=1.03, fontsize=10)
    plt.show()

    # concise terminal summary
    print(f"latent dim d          : {D_LATENT}")
    print(f"train / test samples  : {N_TRAIN} / {N_TEST}")
    for result in results:
        print(f"[{result['label']}]")
        print(f"projected-grad step   : {result['eta']:.3e}")
        print(f"oracle objective      : {result['f_star']:.6e}")
        print(f"oracle probe accuracy : {result['acc_star']:.4f}")
        print(f"final objective gap   : {result['hist_gap'][-1]:.6e}")
        print(f"final subspace error  : {result['hist_subspace'][-1]:.6e}")
        print(f"final probe accuracy  : {result['hist_acc'][-1]:.4f}")


if __name__ == "__main__":
    main()
