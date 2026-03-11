import argparse
import json
import time
from pathlib import Path

import numpy as np
from inspect import signature
from sklearn.linear_model import LogisticRegression
from torchvision import datasets

import closed_form_barlow_twins as cfbt
import closed_form_barlow_twins_cifar as cfbt_cifar
from project_paths import resolve_json_path, resolve_plot_path

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


SEED = 7
MEMORY_COUNT = 512
ITERATIONS = 5
BETA = 1.5
FINAL_D = 32
PROBE_MAX_ITER = 2000
RIDGE_FACTOR = 0.1
VIS_SAMPLES = 8

VARIANTS = [
    "supervised-labelpatch",
    "unsupervised",
    "unsupervised-info-ridge",
]


def load_dataset(dataset_name, n_train=None, n_test=None):
    rng = np.random.default_rng(SEED)
    if dataset_name == "mnist":
        train_ds = datasets.MNIST(root="./data", train=True, download=True)
        test_ds = datasets.MNIST(root="./data", train=False, download=True)

        xtr_raw = train_ds.data.numpy().astype(np.float64) / 255.0
        xte_raw = test_ds.data.numpy().astype(np.float64) / 255.0
        ytr = train_ds.targets.numpy()
        yte = test_ds.targets.numpy()

        n_train = cfbt.N_TRAIN if n_train is None else n_train
        n_test = cfbt.N_TEST if n_test is None else n_test
        idx_tr = rng.choice(len(xtr_raw), size=n_train, replace=False)
        idx_te = rng.choice(len(xte_raw), size=n_test, replace=False)
        xtr_raw = xtr_raw[idx_tr]
        ytr = ytr[idx_tr]
        xte_raw = xte_raw[idx_te]
        yte = yte[idx_te]
        channels, height, width = 1, 28, 28
    elif dataset_name in {"cifar10", "cifar100"}:
        n_train = cfbt_cifar.N_TRAIN if n_train is None else n_train
        n_test = cfbt_cifar.N_TEST if n_test is None else n_test
        dataset_cls = {
            "cifar10": datasets.CIFAR10,
            "cifar100": datasets.CIFAR100,
        }[dataset_name]
        train_ds = dataset_cls(root="./data", train=True, download=True)
        test_ds = dataset_cls(root="./data", train=False, download=True)

        xtr_raw = train_ds.data.astype(np.float64) / 255.0
        xte_raw = test_ds.data.astype(np.float64) / 255.0
        ytr = np.asarray(train_ds.targets)
        yte = np.asarray(test_ds.targets)

        idx_tr = rng.choice(len(xtr_raw), size=n_train, replace=False)
        idx_te = rng.choice(len(xte_raw), size=n_test, replace=False)
        xtr_raw = xtr_raw[idx_tr]
        ytr = ytr[idx_tr]
        xte_raw = xte_raw[idx_te]
        yte = yte[idx_te]

        xtr_raw = np.transpose(xtr_raw, (0, 3, 1, 2))
        xte_raw = np.transpose(xte_raw, (0, 3, 1, 2))
        channels, height, width = 3, 32, 32
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    xtr_flat = xtr_raw.reshape(xtr_raw.shape[0], -1)
    xte_flat = xte_raw.reshape(xte_raw.shape[0], -1)
    mean = xtr_flat.mean(axis=0, keepdims=True)
    xtr = xtr_flat - mean
    xte = xte_flat - mean
    return {
        "xtr": xtr,
        "ytr": ytr,
        "xte": xte,
        "yte": yte,
        "xtr_raw": xtr_raw,
        "xte_raw": xte_raw,
        "mean": mean,
        "channels": channels,
        "height": height,
        "width": width,
    }


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


def factor_patch_area(num_classes, height, width):
    best = None
    for patch_h in range(1, height + 1):
        patch_w = int(np.ceil(num_classes / patch_h))
        if patch_w > width:
            continue
        extra = patch_h * patch_w - num_classes
        aspect = abs(patch_h - patch_w)
        score = (extra, aspect)
        if best is None or score < best[:2]:
            best = (extra, aspect, patch_h, patch_w)
    if best is None:
        raise ValueError("Could not fit label patch into image.")
    return best[2], best[3]


def overwrite_label_patch(flat, labels, channels, height, width, scale):
    num_classes = int(np.max(labels) + 1)
    patch_h, patch_w = factor_patch_area(num_classes, height, width)
    images = flat.reshape(flat.shape[0], channels, height, width).copy()
    images[:, 0, :patch_h, :patch_w] = 0.0
    onehot = np.eye(num_classes, dtype=np.float64)[labels] * scale
    patch = np.zeros((flat.shape[0], patch_h * patch_w), dtype=np.float64)
    patch[:, :num_classes] = onehot
    images[:, 0, :patch_h, :patch_w] = patch.reshape(flat.shape[0], patch_h, patch_w)
    return images.reshape(flat.shape[0], -1), (patch_h, patch_w)


def zero_label_patch(flat, channels, height, width, patch_shape):
    patch_h, patch_w = patch_shape
    images = flat.reshape(flat.shape[0], channels, height, width).copy()
    images[:, 0, :patch_h, :patch_w] = 0.0
    return images.reshape(flat.shape[0], -1)


def decode_label_patch(flat, channels, height, width, num_classes, patch_shape):
    patch_h, patch_w = patch_shape
    images = flat.reshape(flat.shape[0], channels, height, width)
    patch = images[:, 0, :patch_h, :patch_w].reshape(flat.shape[0], -1)
    return np.argmax(patch[:, :num_classes], axis=1)


def negentropy_feature_weights(X):
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    z = (X - mu) / std
    kurt = np.mean(z ** 4, axis=0)
    score = np.abs(kurt - 3.0)
    center = np.median(score)
    mad = np.median(np.abs(score - center))
    scale = 1.4826 * mad if mad > 1e-6 else 1.0
    weights = 0.5 + 1.0 / (1.0 + np.exp(-(score - center) / scale))
    return weights.astype(np.float64)


def select_memory_indices(n_train, memory_count):
    rng = np.random.default_rng(SEED + 101)
    return rng.choice(n_train, size=min(memory_count, n_train), replace=False)


def build_memory_system(X_mem, ridge_lambda):
    memories = X_mem.T
    gram = memories.T @ memories
    if ridge_lambda > 0.0:
        gram = gram + ridge_lambda * np.eye(gram.shape[0], dtype=np.float64)
        gram_inv = np.linalg.solve(gram, np.eye(gram.shape[0], dtype=np.float64))
    else:
        gram_inv = np.linalg.pinv(gram)
    return {
        "memories": memories,
        "gram_inv": gram_inv,
        "ridge_lambda": float(ridge_lambda),
    }


def retrieve_features(queries, system, iterations, beta):
    state = queries.copy()
    memories = system["memories"]
    gram_inv = system["gram_inv"]
    for _ in range(iterations):
        coeff = (state @ memories) @ gram_inv
        state = np.tanh(beta * (coeff @ memories.T))
    coeff = (state @ memories) @ gram_inv
    return state, coeff


def top_pca_projection(Xtr, d):
    cov = cfbt.covariance(Xtr)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    return evecs[:, order[:d]]


def reconstruct_for_plot(flat_state, mean, channels, height, width, inverse_scale=None):
    state = flat_state.copy()
    if inverse_scale is not None:
        state = state / inverse_scale
    state = state + mean
    state = state.reshape(state.shape[0], channels, height, width)
    if channels == 1:
        return np.clip(state[:, 0], 0.0, 1.0)
    state = np.transpose(state, (0, 2, 3, 1))
    return np.clip(state, 0.0, 1.0)


def plot_retrieval_grid(dataset_name, variant, originals, queries, retrieved, y_true, y_pred, plot_path, slot_pred=None):
    if plt is None:
        return
    num_samples = originals.shape[0]
    fig, axes = plt.subplots(3, num_samples, figsize=(1.6 * num_samples, 4.8))
    if num_samples == 1:
        axes = np.array(axes).reshape(3, 1)

    row_titles = ["Original", "Query", "Retrieved"]
    for row, imgs in enumerate([originals, queries, retrieved]):
        for col in range(num_samples):
            ax = axes[row, col]
            if imgs.ndim == 3:
                ax.imshow(imgs[col], cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(imgs[col], vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_titles[row])
            if row == 0:
                title = f"y={y_true[col]} / pred={y_pred[col]}"
                if slot_pred is not None:
                    title += f"\nslot={slot_pred[col]}"
                ax.set_title(title)
    fig.suptitle(f"Hopfield retrieval | {dataset_name} | {variant}")
    fig.tight_layout()
    plot_path = resolve_plot_path(plot_path)
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def prepare_variant(dataset, variant, memory_count):
    xtr = dataset["xtr"]
    ytr = dataset["ytr"]
    xte = dataset["xte"]
    channels = dataset["channels"]
    height = dataset["height"]
    width = dataset["width"]
    num_classes = int(np.max(ytr) + 1)
    mem_idx = select_memory_indices(len(xtr), memory_count)

    if variant == "supervised-labelpatch":
        label_scale = 2.0
        mem_raw, patch_shape = overwrite_label_patch(xtr[mem_idx], ytr[mem_idx], channels, height, width, label_scale)
        xtr_query = zero_label_patch(xtr, channels, height, width, patch_shape)
        xte_query = zero_label_patch(xte, channels, height, width, patch_shape)
        ridge_lambda = 0.0
        inverse_scale = None
        slot_meta = {
            "patch_shape": patch_shape,
            "num_classes": num_classes,
        }
    elif variant == "unsupervised":
        mem_raw = xtr[mem_idx]
        xtr_query = xtr
        xte_query = xte
        ridge_lambda = 0.0
        inverse_scale = None
        slot_meta = None
    elif variant == "unsupervised-info-ridge":
        weights = negentropy_feature_weights(xtr)
        mem_raw = xtr[mem_idx] * weights
        xtr_query = xtr * weights
        xte_query = xte * weights
        gram = mem_raw.T @ mem_raw
        ridge_lambda = RIDGE_FACTOR * float(np.trace(gram) / gram.shape[0])
        inverse_scale = weights
        slot_meta = None
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return {
        "memories": mem_raw,
        "xtr_query": xtr_query,
        "xte_query": xte_query,
        "mem_idx": mem_idx,
        "ridge_lambda": ridge_lambda,
        "inverse_scale": inverse_scale,
        "slot_meta": slot_meta,
    }


def compute_baselines(xtr, ytr, xte, yte):
    raw_tr, raw_te = cfbt.standardize_train_test(xtr, xte)
    raw_linear = fit_linear_probe(raw_tr, ytr, raw_te, yte)

    pca_basis = top_pca_projection(xtr, FINAL_D)
    pca_tr = xtr @ pca_basis
    pca_te = xte @ pca_basis
    pca_tr, pca_te = cfbt.standardize_train_test(pca_tr, pca_te)
    pca32 = fit_linear_probe(pca_tr, ytr, pca_te, yte)
    return {
        "raw_linear_probe_accuracy": raw_linear,
        "raw_pca32_probe_accuracy": pca32,
    }


def run_variant(dataset_name, variant, memory_count, iterations, beta, plot_prefix):
    dataset = load_dataset(dataset_name)
    xtr = dataset["xtr"]
    ytr = dataset["ytr"]
    xte = dataset["xte"]
    yte = dataset["yte"]
    prepared = prepare_variant(dataset, variant, memory_count)

    fit_start = time.perf_counter()
    system = build_memory_system(prepared["memories"], prepared["ridge_lambda"])
    fit_time = time.perf_counter() - fit_start

    infer_start = time.perf_counter()
    ztr_state, ztr_coeff = retrieve_features(prepared["xtr_query"], system, iterations=iterations, beta=beta)
    zte_state, zte_coeff = retrieve_features(prepared["xte_query"], system, iterations=iterations, beta=beta)
    infer_time = time.perf_counter() - infer_start

    probe_tr, probe_te = cfbt.standardize_train_test(ztr_state, zte_state)
    state_acc = fit_linear_probe(probe_tr, ytr, probe_te, yte)

    coeff_tr, coeff_te = cfbt.standardize_train_test(ztr_coeff, zte_coeff)
    coeff_acc = fit_linear_probe(coeff_tr, ytr, coeff_te, yte)

    state_basis = top_pca_projection(ztr_state, FINAL_D)
    state_pca_tr = ztr_state @ state_basis
    state_pca_te = zte_state @ state_basis
    state_pca_tr, state_pca_te = cfbt.standardize_train_test(state_pca_tr, state_pca_te)
    state_pca_acc = fit_linear_probe(state_pca_tr, ytr, state_pca_te, yte)

    baselines = compute_baselines(xtr, ytr, xte, yte)

    slot_acc = None
    slot_pred = None
    if prepared["slot_meta"] is not None:
        slot_pred = decode_label_patch(
            zte_state,
            dataset["channels"],
            dataset["height"],
            dataset["width"],
            prepared["slot_meta"]["num_classes"],
            prepared["slot_meta"]["patch_shape"],
        )
        slot_acc = float((slot_pred == yte).mean())

    rng = np.random.default_rng(SEED + 303)
    vis_idx = rng.choice(len(xte), size=min(VIS_SAMPLES, len(xte)), replace=False)
    orig_plot = reconstruct_for_plot(
        xte[vis_idx],
        dataset["mean"],
        dataset["channels"],
        dataset["height"],
        dataset["width"],
    )
    query_plot = reconstruct_for_plot(
        prepared["xte_query"][vis_idx],
        dataset["mean"],
        dataset["channels"],
        dataset["height"],
        dataset["width"],
        inverse_scale=prepared["inverse_scale"],
    )
    retrieved_plot = reconstruct_for_plot(
        zte_state[vis_idx],
        dataset["mean"],
        dataset["channels"],
        dataset["height"],
        dataset["width"],
        inverse_scale=prepared["inverse_scale"],
    )
    probe_kwargs = {
        "max_iter": PROBE_MAX_ITER,
        "solver": "lbfgs",
        "n_jobs": None,
    }
    if "multi_class" in signature(LogisticRegression).parameters:
        probe_kwargs["multi_class"] = "multinomial"
    clf = LogisticRegression(**probe_kwargs)
    clf.fit(probe_tr, ytr)
    pred_labels = clf.predict(probe_te)[vis_idx]
    plot_name = f"{plot_prefix}_{dataset_name}_{variant}.png"
    plot_retrieval_grid(
        dataset_name=dataset_name,
        variant=variant,
        originals=orig_plot,
        queries=query_plot,
        retrieved=retrieved_plot,
        y_true=yte[vis_idx],
        y_pred=pred_labels,
        plot_path=Path(plot_name),
        slot_pred=slot_pred[vis_idx] if slot_pred is not None else None,
    )

    return {
        "dataset": dataset_name,
        "variant": variant,
        "memory_count": memory_count,
        "iterations": iterations,
        "beta": beta,
        "fit_time_sec": fit_time,
        "inference_time_sec": infer_time,
        "state_probe_accuracy": state_acc,
        "coefficient_probe_accuracy": coeff_acc,
        "state_pca32_probe_accuracy": state_pca_acc,
        "slot_decode_accuracy": slot_acc,
        "raw_linear_probe_accuracy": baselines["raw_linear_probe_accuracy"],
        "raw_pca32_probe_accuracy": baselines["raw_pca32_probe_accuracy"],
        "plot_path": str(resolve_plot_path(Path(plot_name))),
        "ridge_lambda": prepared["ridge_lambda"],
        "note": (
            "Hopfield-style associative memory with a capped memory bank and a pseudoinverse-family closed-form memory matrix. "
            "The reported state accuracy uses a multinomial linear probe on the retrieved state after a fixed number of tanh retrieval steps."
        ),
    }


def default_json_name(dataset_name, variant):
    return f"hopfield_pseudoinverse_{dataset_name}_{variant}.json"


def main():
    parser = argparse.ArgumentParser(description="Hopfield pseudoinverse analytic baselines with linear readouts.")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100", "all"], default="all")
    parser.add_argument("--variant", choices=VARIANTS + ["all"], default="all")
    parser.add_argument("--memory-count", type=int, default=MEMORY_COUNT)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--save-json", action="store_true")
    args = parser.parse_args()

    datasets_to_run = ["mnist", "cifar10", "cifar100"] if args.dataset == "all" else [args.dataset]
    variants_to_run = VARIANTS if args.variant == "all" else [args.variant]

    all_results = []
    for dataset_name in datasets_to_run:
        print(f"Hopfield pseudoinverse | dataset={dataset_name}")
        for variant in variants_to_run:
            result = run_variant(
                dataset_name=dataset_name,
                variant=variant,
                memory_count=args.memory_count,
                iterations=args.iterations,
                beta=args.beta,
                plot_prefix="hopfield_retrieval",
            )
            all_results.append(result)
            slot_msg = (
                f" | slot={result['slot_decode_accuracy']:.4f}" if result["slot_decode_accuracy"] is not None else ""
            )
            print(
                f"  {variant:>24s} | state={result['state_probe_accuracy']:.4f} | coeff={result['coefficient_probe_accuracy']:.4f} "
                f"| state-pca32={result['state_pca32_probe_accuracy']:.4f} | raw-pca32={result['raw_pca32_probe_accuracy']:.4f} "
                f"| raw-linear={result['raw_linear_probe_accuracy']:.4f}{slot_msg}"
            )
            if args.save_json:
                json_path = resolve_json_path(Path(default_json_name(dataset_name, variant)))
                json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(f"    saved json to {json_path}")

    if len(all_results) > 1 and args.save_json:
        summary_path = resolve_json_path(Path("hopfield_pseudoinverse_summary.json"))
        summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"saved summary to {summary_path}")


if __name__ == "__main__":
    main()
