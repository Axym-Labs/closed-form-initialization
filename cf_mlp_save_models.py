import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cf_mlp_layer_mechanistic import forward_with_hiddens
from cf_mlp_representation import collect_cf_state, ridge_map_np, apply_map_np, softmax_ce_np, tensors_from_arrays
from cf_mlp_scalability import SweepPoint, accuracy_from_logits, estimate_backprop_step_flops, estimate_cf_flops, load_point_data
from cf_mlp_scalability_gpu import init_backprop_params, normalize_hidden_with_stats_torch


def cpu_list(params):
    return [param.detach().cpu() for param in params]


def eval_supervised(point, xte_norm, yte, weights, heads):
    with torch.no_grad():
        _, parts, logits = forward_with_hiddens(xte_norm, weights, heads)
        final = logits[-1]
        layer_acc = [float((out.argmax(dim=1) == yte).float().mean().detach().cpu().item()) for out in logits]
        part_acc = [float((out.argmax(dim=1) == yte).float().mean().detach().cpu().item()) for out in parts]
        return {
            "accuracy": float((final.argmax(dim=1) == yte).float().mean().detach().cpu().item()),
            "cross_entropy": float(F.cross_entropy(final, yte).detach().cpu().item()),
            "layer_accuracy": layer_acc,
            "single_layer_accuracy": part_acc,
        }


def forward_hiddens_only(x, weights):
    h = x
    hiddens = []
    for weight in weights:
        h = torch.relu(h @ weight)
        hiddens.append(h)
    return hiddens


def train_supervised(point, arrays, device, steps=None, epochs=None):
    tensors = tensors_from_arrays(arrays, device)
    xtr, ytr, xte, yte, view1_tr, view2_tr, *_ = tensors
    _, _, norm_mean, norm_scale = normalize_hidden_with_stats_torch([xtr, view1_tr, view2_tr], [xte])
    xtr_norm = (xtr - norm_mean) / norm_scale
    xte_norm = (xte - norm_mean) / norm_scale
    weights, heads = init_backprop_params(point, device)
    params = list(weights.parameters()) + list(heads.parameters())
    optimizer = torch.optim.AdamW(params, lr=point.lr, weight_decay=point.weight_decay)
    steps_per_epoch = max(1, int(math.ceil(point.n_train / point.batch_size)))
    if steps is None:
        steps = int(epochs * steps_per_epoch)
    permutation = torch.randperm(point.n_train, device=device)
    cursor = 0
    losses = []
    checkpoints = set(range(steps_per_epoch, steps + 1, steps_per_epoch))
    checkpoints.add(steps)
    history = []
    best = None
    torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(1, steps + 1):
        if cursor + point.batch_size > point.n_train:
            permutation = torch.randperm(point.n_train, device=device)
            cursor = 0
        batch_idx = permutation[cursor : cursor + point.batch_size]
        cursor += point.batch_size
        _, _, logits = forward_with_hiddens(xtr_norm[batch_idx], weights, heads)
        loss = sum(F.cross_entropy(out, ytr[batch_idx]) for out in logits) / len(logits)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        if step in checkpoints:
            metrics = eval_supervised(point, xte_norm, yte, weights, heads)
            history.append(
                {
                    "step": step,
                    "epoch": step / steps_per_epoch,
                    "mean_train_loss": float(np.mean(losses[-min(len(losses), steps_per_epoch) :])),
                    **metrics,
                }
            )
            print(
                f"supervised step={step} epoch={step/steps_per_epoch:.2f} acc={metrics['accuracy']:.4f} loss={history[-1]['mean_train_loss']:.3f}",
                flush=True,
            )
            if best is None or metrics["accuracy"] > best["metrics"]["accuracy"]:
                best = {
                    "step": step,
                    "epoch": step / steps_per_epoch,
                    "metrics": metrics,
                    "weights": cpu_list(weights),
                    "heads": cpu_list(heads),
                }
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    metrics = eval_supervised(point, xte_norm, yte, weights, heads)
    state = {
        "weights": cpu_list(weights),
        "heads": cpu_list(heads),
        "norm_mean": norm_mean.detach().cpu(),
        "norm_scale": norm_scale.detach().cpu(),
        "history": history,
        "best_checkpoint_by_eval_accuracy": best,
        "metrics": metrics,
        "steps": steps,
        "effective_epochs": steps / steps_per_epoch,
        "fit_time_sec": elapsed,
    }
    del tensors
    torch.cuda.empty_cache()
    return state


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_loss(z1, z2, lambd):
    z1 = (z1 - z1.mean(dim=0)) / torch.clamp(z1.std(dim=0), min=1e-4)
    z2 = (z2 - z2.mean(dim=0)) / torch.clamp(z2.std(dim=0), min=1e-4)
    c = (z1.T @ z2) / z1.shape[0]
    on_diag = torch.diagonal(c).add_(-1.0).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    return on_diag + lambd * off_diag


def linear_probe_final(point, xtr_norm, ytr, xte_norm, yte, weights, reg):
    with torch.no_grad():
        htr = forward_hiddens_only(xtr_norm, weights)
        hte = forward_hiddens_only(xte_norm, weights)
    ftr = htr[-1].detach().cpu().numpy().astype(np.float32)
    fte = hte[-1].detach().cpu().numpy().astype(np.float32)
    mean = ftr.mean(axis=0, keepdims=True)
    std = np.maximum(ftr.std(axis=0, keepdims=True), 1e-6)
    ftr = (ftr - mean) / std
    fte = (fte - mean) / std
    ytr_np = ytr.detach().cpu().numpy().astype(np.int64)
    yte_np = yte.detach().cpu().numpy().astype(np.int64)
    y_onehot = np.eye(point.num_classes, dtype=np.float32)[ytr_np]
    probe = ridge_map_np(ftr, y_onehot, reg=reg, fit_bias=True)
    train_logits = apply_map_np(ftr, probe, fit_bias=True)
    test_logits = apply_map_np(fte, probe, fit_bias=True)
    return {
        "probe_weight": torch.from_numpy(probe.astype(np.float32)),
        "feature_mean": torch.from_numpy(mean.astype(np.float32)),
        "feature_std": torch.from_numpy(std.astype(np.float32)),
        "train_accuracy": accuracy_from_logits(train_logits, ytr_np),
        "test_accuracy": accuracy_from_logits(test_logits, yte_np),
        "train_ce": softmax_ce_np(train_logits, ytr_np),
        "test_ce": softmax_ce_np(test_logits, yte_np),
    }


def train_barlow(point, arrays, device, epochs, bt_batch_size, projector_dim, lambd, probe_reg):
    tensors = tensors_from_arrays(arrays, device)
    xtr, ytr, xte, yte, view1_tr, view2_tr, *_ = tensors
    _, _, norm_mean, norm_scale = normalize_hidden_with_stats_torch([xtr, view1_tr, view2_tr], [xte])
    xtr_norm = (xtr - norm_mean) / norm_scale
    xte_norm = (xte - norm_mean) / norm_scale
    view1_norm = (view1_tr - norm_mean) / norm_scale
    view2_norm = (view2_tr - norm_mean) / norm_scale
    weights, _ = init_backprop_params(point, device)
    final_dim = min(point.width, point.input_dim)
    projector = torch.nn.Parameter(torch.randn((final_dim, projector_dim), device=device) * math.sqrt(2.0 / final_dim))
    params = list(weights.parameters()) + [projector]
    optimizer = torch.optim.AdamW(params, lr=point.lr, weight_decay=point.weight_decay)
    steps_per_epoch = max(1, int(math.ceil(point.n_train / bt_batch_size)))
    total_steps = int(epochs * steps_per_epoch)
    permutation = torch.randperm(point.n_train, device=device)
    cursor = 0
    history = []
    losses = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(1, total_steps + 1):
        if cursor + bt_batch_size > point.n_train:
            permutation = torch.randperm(point.n_train, device=device)
            cursor = 0
        batch_idx = permutation[cursor : cursor + bt_batch_size]
        cursor += bt_batch_size
        h1 = forward_hiddens_only(view1_norm[batch_idx], weights)
        h2 = forward_hiddens_only(view2_norm[batch_idx], weights)
        z1 = h1[-1] @ projector
        z2 = h2[-1] @ projector
        loss = barlow_loss(z1, z2, lambd)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        if step % steps_per_epoch == 0 or step == total_steps:
            history.append(
                {
                    "step": step,
                    "epoch": step / steps_per_epoch,
                    "mean_barlow_loss": float(np.mean(losses[-min(len(losses), steps_per_epoch) :])),
                }
            )
            print(f"barlow step={step} epoch={step/steps_per_epoch:.2f} loss={history[-1]['mean_barlow_loss']:.3f}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    probe = linear_probe_final(point, xtr_norm, ytr, xte_norm, yte, weights, probe_reg)
    state = {
        "weights": cpu_list(weights),
        "projector": projector.detach().cpu(),
        "norm_mean": norm_mean.detach().cpu(),
        "norm_scale": norm_scale.detach().cpu(),
        "history": history,
        "linear_probe": probe,
        "steps": total_steps,
        "effective_epochs": total_steps / steps_per_epoch,
        "fit_time_sec": elapsed,
        "bt_batch_size": bt_batch_size,
        "bt_lambda": lambd,
    }
    del tensors
    torch.cuda.empty_cache()
    return state


def main():
    parser = argparse.ArgumentParser(description="Save CF, equal-FLOP BP, full BP, and Barlow Twins models for CF-MLP representation work.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/models_resized_seed7"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.40)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--full-epochs", type=float, default=30.0)
    parser.add_argument("--barlow-epochs", type=float, default=30.0)
    parser.add_argument("--bt-batch-size", type=int, default=512)
    parser.add_argument("--projector-dim", type=int, default=512)
    parser.add_argument("--bt-lambda", type=float, default=0.005)
    parser.add_argument("--probe-reg", type=float, default=100.0)
    args = parser.parse_args()

    torch.set_num_threads(args.torch_threads)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    point = SweepPoint(
        dataset="cifar100_fullres_width",
        axis="saved_models",
        scale_value=args.width,
        seed=args.seed,
        n_train=args.n_train,
        n_test=args.n_test,
        input_dim=args.input_dim,
        width=args.width,
        depth=args.depth,
        num_classes=100,
    )
    arrays = load_point_data(point)
    meta = {
        "point": point.__dict__,
        "device": device_name,
        "note": "CIFAR100 resized-input regime when input_dim=512; same-data-instance positives from fixed augmented views.",
    }

    print("saving CF closed-form model", flush=True)
    cf_state = collect_cf_state(point, device, device_name)
    cf_save = {
        **meta,
        "model_type": "cf_closed_form_depth_path_with_supervised_residual_heads",
        "transforms": [torch.from_numpy(t) for t in cf_state["transforms"]],
        "heads": [torch.from_numpy(h) for h in cf_state["heads"]],
        "layer_rows": cf_state["layer_rows"],
        "fit_time_sec": cf_state["fit_time_sec"],
    }
    torch.save(cf_save, args.out_dir / "01_cf_closed_form.pt")

    cf_budget = estimate_cf_flops(point)
    equal_steps = max(1, int(math.floor(cf_budget / max(estimate_backprop_step_flops(point), 1.0))))
    print(f"training equal-FLOP supervised BP steps={equal_steps}", flush=True)
    bp_equal = train_supervised(point, arrays, device, steps=equal_steps)
    torch.save({**meta, "model_type": "backprop_supervised_equal_flop", **bp_equal}, args.out_dir / "02_backprop_equal_flop.pt")

    print(f"training full supervised BP epochs={args.full_epochs}", flush=True)
    bp_full = train_supervised(point, arrays, device, epochs=args.full_epochs)
    torch.save({**meta, "model_type": "backprop_supervised_full", **bp_full}, args.out_dir / "03_backprop_full.pt")

    print(f"training Barlow Twins BP epochs={args.barlow_epochs}", flush=True)
    bt_state = train_barlow(point, arrays, device, args.barlow_epochs, args.bt_batch_size, args.projector_dim, args.bt_lambda, args.probe_reg)
    torch.save({**meta, "model_type": "backprop_barlow_twins", **bt_state}, args.out_dir / "04_barlow_twins.pt")

    summary = {
        "cf_final_accuracy": cf_state["layer_rows"][-1]["cumulative_accuracy"],
        "backprop_equal_accuracy": bp_equal["metrics"]["accuracy"],
        "backprop_equal_epochs": bp_equal["effective_epochs"],
        "backprop_full_accuracy": bp_full["metrics"]["accuracy"],
        "backprop_full_epochs": bp_full["effective_epochs"],
        "barlow_linear_probe_test_accuracy": bt_state["linear_probe"]["test_accuracy"],
        "barlow_epochs": bt_state["effective_epochs"],
        "model_files": [
            "01_cf_closed_form.pt",
            "02_backprop_equal_flop.pt",
            "03_backprop_full.pt",
            "04_barlow_twins.pt",
        ],
    }
    (args.out_dir / "summary.json").write_text(__import__("json").dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
