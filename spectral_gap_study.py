import json
from inspect import signature
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from torchvision import datasets
from torchvision.transforms import ToTensor

import test2


SEED = 7
N_TRAIN = 12000
N_TEST = 3000
REG_EPS = 1e-4
PROBE_MAX_ITER = 2000

SHALLOW_D = 32
SHALLOW_SUITES = [
    "single-translation",
    "blurring",
    "random-masking",
    "block-masking",
]
GRAPH_PAIR_NAME = "1nn-graph"
GRAPH_MEAN_PAIR_NAME = "5nn-mean-graph"
GRAPH_MEAN_K = 5
GRAPH_MUTUAL_PAIR_NAME = "mutual-10nn-graph"
GRAPH_AFFINITY_MEAN_PAIR_NAME = "affinity-10nn-mean-graph"
GRAPH_AFFINITY_K = 10
SKETCH_PAIR_NAME = "sign-64-sketch"
SPARSE_SKETCH_PAIR_NAME = "sparse-64-sketch"
SKETCH_DIM = 64
SKETCH_COUNT = 8
SPARSE_SKETCH_NNZ = 8
RESIDUAL_BRANCH_RANK = 128
RESIDUAL_DEPTH = 3
METHODS = [
    "pca",
    "pca_surplus",
    "logdet_surplus",
    "hard_whitened_invariance",
    "shared_covariance",
    "auto_fisher",
]
LAYER_DIMS = [256, 64, 32]
LAYERWISE_SETTINGS = [
    {"name": "single-translation", "pair_mode": "suite"},
    {"name": "blurring", "pair_mode": "suite"},
    {"name": "1nn-graph-refresh", "pair_mode": "graph_refresh"},
    {"name": "5nn-mean-graph-refresh", "pair_mode": "graph_mean_refresh"},
    {"name": "mutual-10nn-graph-refresh", "pair_mode": "graph_mutual_refresh"},
    {"name": "affinity-10nn-mean-graph-refresh", "pair_mode": "graph_affinity_mean_refresh"},
    {"name": "sign-64-sketch-refresh", "pair_mode": "sketch_refresh"},
    {"name": "sparse-64-sketch-refresh", "pair_mode": "sparse_sketch_refresh"},
]
RESIDUAL_SETTINGS = [
    {"name": "sign-64-sketch-residual", "pair_name": SKETCH_PAIR_NAME},
    {"name": "sparse-64-sketch-residual", "pair_name": SPARSE_SKETCH_PAIR_NAME},
]


def covariance(X):
    return (X.T @ X) / X.shape[0]


def cross_covariance(X, Y):
    return (X.T @ Y) / X.shape[0]


def load_mnist_numpy():
    rng = np.random.default_rng(SEED)
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


def make_suite_views(X, suite_name):
    rng = np.random.default_rng(SEED)
    mats = test2.build_augmentation_suite(suite_name, h=28, w=28, rng=rng)
    return [X @ A.T for A in mats]


def make_graph_neighbor_view(X):
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(X)
    indices = nn.kneighbors(return_distance=False)
    return [X[indices[:, 1]]]


def make_graph_mean_view(X, k):
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    indices = nn.kneighbors(return_distance=False)
    neighbors = indices[:, 1:]
    return [X[neighbors].mean(axis=1)]


def knn_graph(X, k):
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(return_distance=True)
    return distances[:, 1:], indices[:, 1:]


def make_mutual_knn_view(X, k):
    distances, indices = knn_graph(X, k)
    neighbor_sets = [set(row.tolist()) for row in indices]
    selected = np.empty(X.shape[0], dtype=np.int64)
    for i in range(X.shape[0]):
        choice = indices[i, 0]
        for j in indices[i]:
            if i in neighbor_sets[j]:
                choice = j
                break
        selected[i] = choice
    return [X[selected]]


def make_affinity_mean_view(X, k):
    distances, indices = knn_graph(X, k)
    local_scale = np.maximum(distances[:, -1], 1e-8)
    neighbor_scales = local_scale[indices]
    denom = np.maximum(local_scale[:, None] * neighbor_scales, 1e-12)
    weights = np.exp(-(distances ** 2) / denom)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return [np.einsum("nk,nkd->nd", weights, X[indices])]


def make_sign_sketch(p, m, rng):
    return rng.choice(np.array([-1.0, 1.0]), size=(m, p)).astype(np.float64) / np.sqrt(m)


def make_sparse_sketch(p, m, rng, nnz_per_row):
    sketch = np.zeros((m, p), dtype=np.float64)
    scale = 1.0 / np.sqrt(nnz_per_row)
    for row in range(m):
        cols = rng.choice(p, size=nnz_per_row, replace=False)
        signs = rng.choice(np.array([-1.0, 1.0]), size=nnz_per_row)
        sketch[row, cols] = signs * scale
    return sketch


def linear_mmse_reconstruction_operator(sigma, sketch, ridge):
    middle = sketch @ sigma @ sketch.T + ridge * np.eye(sketch.shape[0], dtype=np.float64)
    return sigma @ sketch.T @ np.linalg.inv(middle) @ sketch


def make_sketch_reconstruction_views(X, sketch_kind, sketch_dim=SKETCH_DIM, sketch_count=SKETCH_COUNT):
    sigma = covariance(X)
    p = X.shape[1]
    ridge = 1e-3 * float(np.trace(sigma) / p)
    rng = np.random.default_rng(SEED)
    views = []
    for _ in range(sketch_count):
        if sketch_kind == "sign":
            sketch = make_sign_sketch(p, sketch_dim, rng)
        elif sketch_kind == "sparse":
            sketch = make_sparse_sketch(p, sketch_dim, rng, SPARSE_SKETCH_NNZ)
        else:
            raise ValueError(f"Unknown sketch kind: {sketch_kind}")
        operator = linear_mmse_reconstruction_operator(sigma, sketch, ridge)
        views.append(X @ operator.T)
    return views


def center_pair_family(base, family):
    mean = base.mean(axis=0, keepdims=True)
    if family:
        mean = (mean + sum(view.mean(axis=0, keepdims=True) for view in family)) / (len(family) + 1)
    base_c = base - mean
    family_c = [view - mean for view in family]
    return base_c, family_c


def compute_pair_statistics(base, family):
    base_c, family_c = center_pair_family(base, family)
    sigma = covariance(base_c)

    sigma_bar = np.zeros_like(sigma)
    delta = np.zeros_like(sigma)
    shared = np.zeros_like(sigma)

    for view in family_c:
        sigma_view = covariance(view)
        sigma_bar += 0.5 * (sigma + sigma_view)
        diff = base_c - view
        delta += covariance(diff)
        shared += 0.5 * (cross_covariance(base_c, view) + cross_covariance(view, base_c))

    sigma_bar /= len(family_c)
    delta /= len(family_c)
    shared /= len(family_c)
    commutator = sigma @ delta - delta @ sigma
    commutator_ratio = float(
        np.linalg.norm(commutator, ord="fro")
        / max(np.linalg.norm(sigma, ord="fro") * np.linalg.norm(delta, ord="fro"), 1e-12)
    )
    return {
        "sigma": sigma,
        "sigma_bar": sigma_bar,
        "delta": delta,
        "shared": shared,
        "delta_trace_ratio": float(np.trace(delta) / max(np.trace(sigma), 1e-12)),
        "commutator_ratio": commutator_ratio,
    }


def fit_pca(stats, d):
    evals, evecs = eigh(stats["sigma"])
    basis = evecs[:, -d:]
    return {
        "name": "pca",
        "W": basis.T,
        "basis": basis,
    }


def fit_hard_whitened_invariance(stats, d):
    sigma_bar = stats["sigma_bar"]
    delta = stats["delta"]

    evals_sigma, evecs_sigma = eigh(sigma_bar)
    evals_sigma = np.maximum(evals_sigma, REG_EPS)
    sigma_inv_sqrt = (evecs_sigma / np.sqrt(evals_sigma)) @ evecs_sigma.T

    n_matrix = sigma_inv_sqrt @ delta @ sigma_inv_sqrt
    n_matrix = 0.5 * (n_matrix + n_matrix.T)
    evals_n, evecs_n = eigh(n_matrix)
    y = evecs_n[:, :d].T
    w = y @ sigma_inv_sqrt
    basis = orthonormal_basis(w.T)
    return {
        "name": "hard_whitened_invariance",
        "W": w,
        "basis": basis,
    }


def fit_pca_surplus(stats, d):
    sigma = 0.5 * (stats["sigma"] + stats["sigma"].T)
    delta = 0.5 * (stats["delta"] + stats["delta"].T)
    p = sigma.shape[0]

    evals, evecs = eigh(sigma)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    basis = evecs[:, order]
    delta_diag = np.sum(basis * (delta @ basis), axis=0)
    delta_diag = np.maximum(delta_diag, 0.0)

    sigma_scale = float(np.trace(sigma) / p)
    delta_scale = float(np.trace(delta) / p + 1e-6 * sigma_scale)
    scores = np.log1p(evals / max(sigma_scale, 1e-12)) - np.log1p(delta_diag / max(delta_scale, 1e-12))
    selected = np.argsort(scores)[-d:][::-1]
    subspace = basis[:, selected]
    return {
        "name": "pca_surplus",
        "W": subspace.T,
        "basis": subspace,
        "scores": scores[selected].tolist(),
        "sigma_scale": sigma_scale,
        "delta_scale": delta_scale,
    }


def fit_logdet_surplus(stats, d):
    sigma = 0.5 * (stats["sigma"] + stats["sigma"].T)
    delta = 0.5 * (stats["delta"] + stats["delta"].T)
    p = sigma.shape[0]

    sigma_floor = float(np.trace(sigma) / p + REG_EPS)
    delta_floor = float(np.trace(delta) / p + 1e-6 * sigma_floor)
    sigma_gain = sigma @ np.linalg.inv(sigma + sigma_floor * np.eye(p, dtype=np.float64))
    delta_gain = delta @ np.linalg.inv(delta + delta_floor * np.eye(p, dtype=np.float64))
    score_matrix = 0.5 * ((sigma_gain - delta_gain) + (sigma_gain - delta_gain).T)
    evals, evecs = eigh(score_matrix)
    basis = evecs[:, -d:]
    return {
        "name": "logdet_surplus",
        "W": basis.T,
        "basis": basis,
        "sigma_floor": sigma_floor,
        "delta_floor": delta_floor,
    }


def fit_shared_covariance(stats, d):
    shared = 0.5 * (stats["shared"] + stats["shared"].T)
    evals, evecs = eigh(shared)
    basis = evecs[:, -d:]
    return {
        "name": "shared_covariance",
        "W": basis.T,
        "basis": basis,
    }


def fit_auto_fisher(stats, d):
    sigma = stats["sigma"]
    delta = 0.5 * (stats["delta"] + stats["delta"].T)
    p = sigma.shape[0]
    delta_scale = float(np.trace(delta) / p)
    sigma_scale = float(np.trace(sigma) / p)
    floor = delta_scale + 1e-6 * sigma_scale
    denom = delta + floor * np.eye(p, dtype=np.float64)
    evals, evecs = eigh(sigma, denom)
    basis = orthonormal_basis(evecs[:, -d:])
    return {
        "name": "auto_fisher",
        "W": basis.T,
        "basis": basis,
        "floor": floor,
    }


def fit_method(method_name, stats, d):
    if method_name == "pca":
        return fit_pca(stats, d)
    if method_name == "pca_surplus":
        return fit_pca_surplus(stats, d)
    if method_name == "logdet_surplus":
        return fit_logdet_surplus(stats, d)
    if method_name == "hard_whitened_invariance":
        return fit_hard_whitened_invariance(stats, d)
    if method_name == "shared_covariance":
        return fit_shared_covariance(stats, d)
    if method_name == "auto_fisher":
        return fit_auto_fisher(stats, d)
    raise ValueError(f"Unknown method: {method_name}")


def orthonormal_basis(columns):
    q, _ = np.linalg.qr(columns, mode="reduced")
    return q


def standardize_train_test(ztr, zte):
    mu = ztr.mean(axis=0, keepdims=True)
    std = ztr.std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    return (ztr - mu) / std, (zte - mu) / std


def fit_linear_probe(ztr, ytr, zte, yte):
    kwargs = {
        "max_iter": PROBE_MAX_ITER,
        "solver": "lbfgs",
        "n_jobs": None,
    }
    if "multi_class" in signature(LogisticRegression).parameters:
        kwargs["multi_class"] = "multinomial"
    clf = LogisticRegression(**kwargs)
    clf.fit(ztr, ytr)
    return float((clf.predict(zte) == yte).mean())


def evaluate_projection(model, stats, xtr, ytr, xte, yte):
    ztr = xtr @ model["W"].T
    zte = xte @ model["W"].T
    ztr, zte = standardize_train_test(ztr, zte)
    acc = fit_linear_probe(ztr, ytr, zte, yte)

    sigma = stats["sigma"]
    shared = stats["shared"]
    delta = stats["delta"]
    basis = model["basis"]
    pca_top = float(np.sort(np.linalg.eigvalsh(sigma))[-basis.shape[1] :].sum())
    retained = float(np.trace(basis.T @ sigma @ basis))
    shared_energy = float(np.trace(basis.T @ shared @ basis))
    delta_energy = float(np.trace(basis.T @ delta @ basis))
    return {
        "probe_accuracy": acc,
        "retained_variance": retained,
        "retained_vs_pca": retained / max(pca_top, 1e-12),
        "shared_energy": shared_energy,
        "delta_energy": delta_energy,
    }


def run_shallow_study(xtr, ytr, xte, yte):
    results = []
    for suite_name in SHALLOW_SUITES:
        family_tr = make_suite_views(xtr, suite_name)
        stats = compute_pair_statistics(xtr, family_tr)
        for method_name in METHODS:
            model = fit_method(method_name, stats, SHALLOW_D)
            metrics = evaluate_projection(model, stats, xtr, ytr, xte, yte)
            row = {
                "suite": suite_name,
                "method": method_name,
                "d": SHALLOW_D,
                "delta_trace_ratio": stats["delta_trace_ratio"],
                "commutator_ratio": stats["commutator_ratio"],
            }
            row.update(metrics)
            if "floor" in model:
                row["floor"] = model["floor"]
            results.append(row)
    return results


def run_graph_study(xtr, ytr, xte, yte):
    graph_builders = [
        (GRAPH_PAIR_NAME, lambda data: make_graph_neighbor_view(data)),
        (GRAPH_MEAN_PAIR_NAME, lambda data: make_graph_mean_view(data, GRAPH_MEAN_K)),
        (GRAPH_MUTUAL_PAIR_NAME, lambda data: make_mutual_knn_view(data, GRAPH_AFFINITY_K)),
        (GRAPH_AFFINITY_MEAN_PAIR_NAME, lambda data: make_affinity_mean_view(data, GRAPH_AFFINITY_K)),
    ]
    results = []
    for pair_name, builder in graph_builders:
        family_tr = builder(xtr)
        stats = compute_pair_statistics(xtr, family_tr)
        for method_name in METHODS:
            model = fit_method(method_name, stats, SHALLOW_D)
            metrics = evaluate_projection(model, stats, xtr, ytr, xte, yte)
            row = {
                "suite": pair_name,
                "method": method_name,
                "d": SHALLOW_D,
                "delta_trace_ratio": stats["delta_trace_ratio"],
                "commutator_ratio": stats["commutator_ratio"],
            }
            row.update(metrics)
            if "floor" in model:
                row["floor"] = model["floor"]
            results.append(row)
    return results


def run_channel_study(xtr, ytr, xte, yte):
    channel_builders = [
        (SKETCH_PAIR_NAME, lambda data: make_sketch_reconstruction_views(data, "sign")),
        (SPARSE_SKETCH_PAIR_NAME, lambda data: make_sketch_reconstruction_views(data, "sparse")),
    ]
    results = []
    for pair_name, builder in channel_builders:
        family_tr = builder(xtr)
        stats = compute_pair_statistics(xtr, family_tr)
        for method_name in METHODS:
            model = fit_method(method_name, stats, SHALLOW_D)
            metrics = evaluate_projection(model, stats, xtr, ytr, xte, yte)
            row = {
                "suite": pair_name,
                "method": method_name,
                "d": SHALLOW_D,
                "delta_trace_ratio": stats["delta_trace_ratio"],
                "commutator_ratio": stats["commutator_ratio"],
            }
            row.update(metrics)
            if "floor" in model:
                row["floor"] = model["floor"]
            results.append(row)
    return results


def relu(X):
    return np.maximum(X, 0.0)


def apply_layer_transform(model, base, family):
    base_next = base @ model["W"].T
    family_next = [view @ model["W"].T for view in family]
    return base_next, family_next


def apply_residual_block(model, base):
    projector = model["basis"] @ model["basis"].T
    return base + base @ projector


def postprocess_hidden(base_tr, base_te, family_tr, family_te):
    base_tr = relu(base_tr)
    base_te = relu(base_te)
    family_tr = [relu(view) for view in family_tr]
    family_te = [relu(view) for view in family_te]

    mu = base_tr.mean(axis=0, keepdims=True)
    std = base_tr.std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)

    base_tr = (base_tr - mu) / std
    base_te = (base_te - mu) / std
    family_tr = [(view - mu) / std for view in family_tr]
    family_te = [(view - mu) / std for view in family_te]
    return base_tr, base_te, family_tr, family_te


def greedy_layerwise_features(xtr, xte, suite_name, method_name, dims):
    if suite_name == GRAPH_PAIR_NAME:
        family_tr = make_graph_neighbor_view(xtr)
        family_te = None
    elif suite_name == GRAPH_MEAN_PAIR_NAME:
        family_tr = make_graph_mean_view(xtr, GRAPH_MEAN_K)
        family_te = None
    elif suite_name == GRAPH_MUTUAL_PAIR_NAME:
        family_tr = make_mutual_knn_view(xtr, GRAPH_AFFINITY_K)
        family_te = None
    elif suite_name == GRAPH_AFFINITY_MEAN_PAIR_NAME:
        family_tr = make_affinity_mean_view(xtr, GRAPH_AFFINITY_K)
        family_te = None
    elif suite_name == SKETCH_PAIR_NAME:
        family_tr = make_sketch_reconstruction_views(xtr, "sign")
        family_te = None
    elif suite_name == SPARSE_SKETCH_PAIR_NAME:
        family_tr = make_sketch_reconstruction_views(xtr, "sparse")
        family_te = None
    else:
        family_tr = make_suite_views(xtr, suite_name)
        family_te = make_suite_views(xte, suite_name)
    base_tr = xtr.copy()
    base_te = xte.copy()
    layers = []

    for layer_idx, width in enumerate(dims):
        stats = compute_pair_statistics(base_tr, family_tr)
        model = fit_method(method_name, stats, width)
        base_tr, family_tr = apply_layer_transform(model, base_tr, family_tr)
        if family_te is None:
            base_te = base_te @ model["W"].T
        else:
            base_te, family_te = apply_layer_transform(model, base_te, family_te)

        layer_metrics = {
            "layer": layer_idx + 1,
            "width": width,
            "retained_variance": float(np.trace(model["basis"].T @ stats["sigma"] @ model["basis"])),
            "shared_energy": float(np.trace(model["basis"].T @ stats["shared"] @ model["basis"])),
            "delta_energy": float(np.trace(model["basis"].T @ stats["delta"] @ model["basis"])),
            "commutator_ratio": stats["commutator_ratio"],
        }
        if "floor" in model:
            layer_metrics["floor"] = model["floor"]
        layers.append(layer_metrics)

        if layer_idx < len(dims) - 1:
            if family_te is None:
                base_tr = relu(base_tr)
                base_te = relu(base_te)

                mu = base_tr.mean(axis=0, keepdims=True)
                std = base_tr.std(axis=0, keepdims=True)
                std = np.where(std > 1e-6, std, 1.0)

                base_tr = (base_tr - mu) / std
                base_te = (base_te - mu) / std
                if suite_name == GRAPH_PAIR_NAME:
                    family_tr = make_graph_neighbor_view(base_tr)
                elif suite_name == GRAPH_MEAN_PAIR_NAME:
                    family_tr = make_graph_mean_view(base_tr, GRAPH_MEAN_K)
                elif suite_name == GRAPH_MUTUAL_PAIR_NAME:
                    family_tr = make_mutual_knn_view(base_tr, GRAPH_AFFINITY_K)
                elif suite_name == GRAPH_AFFINITY_MEAN_PAIR_NAME:
                    family_tr = make_affinity_mean_view(base_tr, GRAPH_AFFINITY_K)
                elif suite_name == SKETCH_PAIR_NAME:
                    family_tr = make_sketch_reconstruction_views(base_tr, "sign")
                elif suite_name == SPARSE_SKETCH_PAIR_NAME:
                    family_tr = make_sketch_reconstruction_views(base_tr, "sparse")
                else:
                    raise ValueError(f"Unknown refresh suite: {suite_name}")
            else:
                base_tr, base_te, family_tr, family_te = postprocess_hidden(
                    base_tr, base_te, family_tr, family_te
                )

    base_tr, base_te = standardize_train_test(base_tr, base_te)
    return base_tr, base_te, layers


def run_layerwise_study(xtr, ytr, xte, yte):
    results = []
    for setting in LAYERWISE_SETTINGS:
        suite_name = setting["name"]
        if setting["pair_mode"] == "graph_refresh":
            pair_name = GRAPH_PAIR_NAME
        elif setting["pair_mode"] == "graph_mean_refresh":
            pair_name = GRAPH_MEAN_PAIR_NAME
        elif setting["pair_mode"] == "graph_mutual_refresh":
            pair_name = GRAPH_MUTUAL_PAIR_NAME
        elif setting["pair_mode"] == "graph_affinity_mean_refresh":
            pair_name = GRAPH_AFFINITY_MEAN_PAIR_NAME
        elif setting["pair_mode"] == "sketch_refresh":
            pair_name = SKETCH_PAIR_NAME
        elif setting["pair_mode"] == "sparse_sketch_refresh":
            pair_name = SPARSE_SKETCH_PAIR_NAME
        else:
            pair_name = suite_name
        for method_name in METHODS:
            ztr, zte, layers = greedy_layerwise_features(
                xtr, xte, suite_name=pair_name, method_name=method_name, dims=LAYER_DIMS
            )
            acc = fit_linear_probe(ztr, ytr, zte, yte)
            results.append(
                {
                    "suite": suite_name,
                    "pair_mode": setting["pair_mode"],
                    "method": method_name,
                    "layer_dims": LAYER_DIMS,
                    "probe_accuracy": acc,
                    "layers": layers,
                    "first_layer_commutator_ratio": layers[0].get("commutator_ratio"),
                }
            )
    return results


def build_channel_family(data, pair_name):
    if pair_name == SKETCH_PAIR_NAME:
        return make_sketch_reconstruction_views(data, "sign")
    if pair_name == SPARSE_SKETCH_PAIR_NAME:
        return make_sketch_reconstruction_views(data, "sparse")
    raise ValueError(f"Unknown channel pair: {pair_name}")


def run_residual_study(xtr, ytr, xte, yte):
    results = []
    for setting in RESIDUAL_SETTINGS:
        pair_name = setting["pair_name"]
        for method_name in METHODS:
            base_tr = xtr.copy()
            base_te = xte.copy()
            blocks = []

            for block_idx in range(RESIDUAL_DEPTH):
                family_tr = build_channel_family(base_tr, pair_name)
                stats = compute_pair_statistics(base_tr, family_tr)
                model = fit_method(method_name, stats, RESIDUAL_BRANCH_RANK)

                base_tr = apply_residual_block(model, base_tr)
                base_te = apply_residual_block(model, base_te)

                base_tr = relu(base_tr)
                base_te = relu(base_te)
                base_tr, base_te = standardize_train_test(base_tr, base_te)

                blocks.append(
                    {
                        "block": block_idx + 1,
                        "branch_rank": RESIDUAL_BRANCH_RANK,
                        "retained_variance": float(np.trace(model["basis"].T @ stats["sigma"] @ model["basis"])),
                        "shared_energy": float(np.trace(model["basis"].T @ stats["shared"] @ model["basis"])),
                        "delta_energy": float(np.trace(model["basis"].T @ stats["delta"] @ model["basis"])),
                        "commutator_ratio": stats["commutator_ratio"],
                    }
                )

            final_family = build_channel_family(base_tr, pair_name)
            final_stats = compute_pair_statistics(base_tr, final_family)
            final_model = fit_method(method_name, final_stats, SHALLOW_D)
            ztr = base_tr @ final_model["W"].T
            zte = base_te @ final_model["W"].T
            ztr, zte = standardize_train_test(ztr, zte)
            acc = fit_linear_probe(ztr, ytr, zte, yte)

            results.append(
                {
                    "suite": setting["name"],
                    "pair_name": pair_name,
                    "method": method_name,
                    "branch_rank": RESIDUAL_BRANCH_RANK,
                    "depth": RESIDUAL_DEPTH,
                    "final_dim": SHALLOW_D,
                    "probe_accuracy": acc,
                    "blocks": blocks,
                    "final_commutator_ratio": final_stats["commutator_ratio"],
                }
            )
    return results


def summarize_shallow(rows):
    lines = ["Shallow spectral study (d = 32)", "-" * 72]
    for suite_name in SHALLOW_SUITES:
        lines.append(f"[suite={suite_name}]")
        suite_rows = [row for row in rows if row["suite"] == suite_name]
        suite_rows.sort(key=lambda row: row["probe_accuracy"], reverse=True)
        for row in suite_rows:
            lines.append(
                f"{row['method']:>24} | acc={row['probe_accuracy']:.4f} | "
                f"retained_vs_pca={row['retained_vs_pca']:.3f} | "
                f"shared={row['shared_energy']:.4f} | delta={row['delta_energy']:.4f} | "
                f"comm={row['commutator_ratio']:.3f}"
            )
        lines.append("")
    return "\n".join(lines).strip()


def summarize_layerwise(rows):
    lines = ["Layerwise analytic study", "-" * 72]
    for setting in LAYERWISE_SETTINGS:
        suite_name = setting["name"]
        lines.append(f"[suite={suite_name}, pair_mode={setting['pair_mode']}, dims={LAYER_DIMS}]")
        suite_rows = [row for row in rows if row["suite"] == suite_name]
        suite_rows.sort(key=lambda row: row["probe_accuracy"], reverse=True)
        for row in suite_rows:
            lines.append(f"{row['method']:>24} | acc={row['probe_accuracy']:.4f}")
        lines.append("")
    return "\n".join(lines).strip()


def summarize_graph(rows):
    lines = ["Graph-pair spectral study", "-" * 72]
    for pair_name in [
        GRAPH_PAIR_NAME,
        GRAPH_MEAN_PAIR_NAME,
        GRAPH_MUTUAL_PAIR_NAME,
        GRAPH_AFFINITY_MEAN_PAIR_NAME,
    ]:
        lines.append(f"[pair={pair_name}]")
        ordered = sorted(
            [row for row in rows if row["suite"] == pair_name],
            key=lambda row: row["probe_accuracy"],
            reverse=True,
        )
        for row in ordered:
                lines.append(
                    f"{row['method']:>24} | acc={row['probe_accuracy']:.4f} | "
                    f"retained_vs_pca={row['retained_vs_pca']:.3f} | "
                    f"shared={row['shared_energy']:.4f} | delta={row['delta_energy']:.4f} | "
                    f"comm={row['commutator_ratio']:.3f}"
                )
        lines.append("")
    return "\n".join(lines)


def summarize_channel(rows):
    lines = ["Channel-pair spectral study", "-" * 72]
    for pair_name in [SKETCH_PAIR_NAME, SPARSE_SKETCH_PAIR_NAME]:
        lines.append(f"[pair={pair_name}]")
        ordered = sorted(
            [row for row in rows if row["suite"] == pair_name],
            key=lambda row: row["probe_accuracy"],
            reverse=True,
        )
        for row in ordered:
            lines.append(
                f"{row['method']:>24} | acc={row['probe_accuracy']:.4f} | "
                f"retained_vs_pca={row['retained_vs_pca']:.3f} | "
                f"shared={row['shared_energy']:.4f} | delta={row['delta_energy']:.4f} | "
                f"comm={row['commutator_ratio']:.3f}"
            )
        lines.append("")
    return "\n".join(lines).strip()


def summarize_residual(rows):
    lines = ["Residual analytic study", "-" * 72]
    for setting in RESIDUAL_SETTINGS:
        suite_name = setting["name"]
        lines.append(
            f"[suite={suite_name}, branch_rank={RESIDUAL_BRANCH_RANK}, depth={RESIDUAL_DEPTH}]"
        )
        suite_rows = sorted(
            [row for row in rows if row["suite"] == suite_name],
            key=lambda row: row["probe_accuracy"],
            reverse=True,
        )
        for row in suite_rows:
            lines.append(
                f"{row['method']:>24} | acc={row['probe_accuracy']:.4f} | "
                f"final_comm={row['final_commutator_ratio']:.3f}"
            )
        lines.append("")
    return "\n".join(lines).strip()


def main():
    xtr, ytr, xte, yte = load_mnist_numpy()
    shallow = run_shallow_study(xtr, ytr, xte, yte)
    graph = run_graph_study(xtr, ytr, xte, yte)
    channel = run_channel_study(xtr, ytr, xte, yte)
    layerwise = run_layerwise_study(xtr, ytr, xte, yte)
    residual = run_residual_study(xtr, ytr, xte, yte)

    payload = {
        "config": {
            "seed": SEED,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "shallow_d": SHALLOW_D,
            "layer_dims": LAYER_DIMS,
            "methods": METHODS,
            "shallow_suites": SHALLOW_SUITES,
            "layerwise_settings": LAYERWISE_SETTINGS,
            "graph_pair_name": GRAPH_PAIR_NAME,
            "graph_mean_pair_name": GRAPH_MEAN_PAIR_NAME,
            "graph_mean_k": GRAPH_MEAN_K,
            "graph_mutual_pair_name": GRAPH_MUTUAL_PAIR_NAME,
            "graph_affinity_mean_pair_name": GRAPH_AFFINITY_MEAN_PAIR_NAME,
            "graph_affinity_k": GRAPH_AFFINITY_K,
            "residual_settings": RESIDUAL_SETTINGS,
            "residual_branch_rank": RESIDUAL_BRANCH_RANK,
            "residual_depth": RESIDUAL_DEPTH,
        },
        "shallow": shallow,
        "graph": graph,
        "channel": channel,
        "layerwise": layerwise,
        "residual": residual,
    }

    output_path = Path("spectral_gap_results.json")
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(summarize_shallow(shallow))
    print()
    print(summarize_graph(graph))
    print()
    print(summarize_channel(channel))
    print()
    print(summarize_layerwise(layerwise))
    print()
    print(summarize_residual(residual))
    print()
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
