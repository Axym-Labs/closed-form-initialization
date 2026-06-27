import argparse
import csv
import gc
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cf_mlp_barlow_clean import barlow_loss, load_tensors, normalized_inputs, parse_config
from cf_mlp_bpbt_spectral_diagnostic import agreement_spectrum_metrics, transition_metrics
from cf_mlp_bt_objective_by_layer import bt_hidden_metrics, find_residual_bt_model
from cf_mlp_clean_readouts import linear_classifier_readout, pca512_readout
from cf_mlp_layer_mechanistic import covariance_spectrum, standardize_many
from cf_mlp_residual_barlow import BP_BT_ACTIVATION, BP_BT_ACTIVATION_ALPHA, leaky_gelu
from cf_mlp_scalability import SweepPoint, write_jsonl


def point_for(args, depth, seed):
    return SweepPoint(
        dataset=args.dataset,
        axis="residual_bt_route3",
        scale_value=args.width,
        seed=seed,
        n_train=args.n_train,
        n_test=args.n_test,
        input_dim=args.input_dim,
        width=args.width,
        depth=depth,
        num_classes=args.num_classes,
    )


def residual_layer(h, weight, activation_alpha, residual_scale, use_layernorm):
    branch = leaky_gelu(h @ weight, activation_alpha)
    pre_norm = h + float(residual_scale) * branch
    out = F.layer_norm(pre_norm, (pre_norm.shape[-1],)) if use_layernorm else pre_norm
    return branch, pre_norm, out


def forward_residual_stages(x, weights, activation_alpha, residual_scale, use_layernorm):
    h = x
    stages = []
    for weight in weights:
        inp = h
        branch, pre_norm, h = residual_layer(inp, weight, activation_alpha, residual_scale, use_layernorm)
        stages.append({"input": inp, "branch": branch, "pre_norm": pre_norm, "output": h})
    return stages


def collect_residual_reps(point, tensors, state, device):
    norm_mean = state["norm_mean"].to(device)
    norm_scale = state["norm_scale"].to(device)
    normed = normalized_inputs(tensors, norm_mean, norm_scale)
    weights = [param.to(device) for param in state["weights"]]
    with torch.no_grad():
        train_stages = forward_residual_stages(
            normed["xtr"], weights, state["activation_alpha"], state["residual_scale"], state["layernorm"]
        )
        test_stages = forward_residual_stages(
            normed["xte"], weights, state["activation_alpha"], state["residual_scale"], state["layernorm"]
        )
        view1_stages = forward_residual_stages(
            normed["view1_tr"], weights, state["activation_alpha"], state["residual_scale"], state["layernorm"]
        )
        view2_stages = forward_residual_stages(
            normed["view2_tr"], weights, state["activation_alpha"], state["residual_scale"], state["layernorm"]
        )
    raw_train = [stage["output"].detach().cpu().numpy().astype(np.float32) for stage in train_stages]
    raw_test = [stage["output"].detach().cpu().numpy().astype(np.float32) for stage in test_stages]
    raw_view1 = [stage["output"].detach().cpu().numpy().astype(np.float32) for stage in view1_stages]
    raw_view2 = [stage["output"].detach().cpu().numpy().astype(np.float32) for stage in view2_stages]
    pathnorm_train = []
    pathnorm_test = []
    pathnorm_view1 = []
    pathnorm_view2 = []
    for htr, hte, hv1, hv2 in zip(raw_train, raw_test, raw_view1, raw_view2):
        htr_s, hte_s, hv1_s, hv2_s = standardize_many(htr, hte, hv1, hv2)
        pathnorm_train.append(htr_s)
        pathnorm_test.append(hte_s)
        pathnorm_view1.append(hv1_s)
        pathnorm_view2.append(hv2_s)
    return {
        "normed": normed,
        "weights": weights,
        "train_stages": train_stages,
        "test_stages": test_stages,
        "view1_stages": view1_stages,
        "view2_stages": view2_stages,
        "pathnorm_train": pathnorm_train,
        "pathnorm_test": pathnorm_test,
        "pathnorm_view1_train": pathnorm_view1,
        "pathnorm_view2_train": pathnorm_view2,
    }


def shared_difference_metrics_np(view1, view2):
    x1 = np.asarray(view1, dtype=np.float64)
    x2 = np.asarray(view2, dtype=np.float64)
    shared = 0.5 * (x1 + x2)
    diff = 0.5 * (x1 - x2)
    shared = shared - shared.mean(axis=0, keepdims=True)
    diff = diff - diff.mean(axis=0, keepdims=True)
    shared_trace = float(np.mean(shared * shared))
    diff_trace = float(np.mean(diff * diff))
    return {
        "shared_trace_per_dim": shared_trace,
        "diff_trace_per_dim": diff_trace,
        "shared_diff_ratio": shared_trace / max(diff_trace, 1e-12),
        "diff_fraction": diff_trace / max(shared_trace + diff_trace, 1e-12),
    }


def tensor_np(x):
    return x.detach().cpu().numpy().astype(np.float32)


def rms_torch(x):
    return float(torch.sqrt(torch.mean(x * x)).detach().cpu().item())


def row_cosine_torch(a, b):
    num = torch.sum(a * b, dim=1)
    den = torch.linalg.vector_norm(a, dim=1) * torch.linalg.vector_norm(b, dim=1)
    return float(torch.mean(num / torch.clamp(den, min=1e-12)).detach().cpu().item())


def stage_metrics(args, point, model, layer_idx, train_stage, view1_stage, view2_stage, prev_train_np, device):
    input_v1 = tensor_np(view1_stage["input"])
    input_v2 = tensor_np(view2_stage["input"])
    output_v1 = tensor_np(view1_stage["output"])
    output_v2 = tensor_np(view2_stage["output"])
    output_train = tensor_np(train_stage["output"])
    input_train = tensor_np(train_stage["input"])
    branch_train = train_stage["branch"]
    update_train = train_stage["output"] - train_stage["input"]
    pre_update_train = train_stage["pre_norm"] - train_stage["input"]

    input_bt = bt_hidden_metrics(input_v1, input_v2, args.bt_lambda)
    output_bt = bt_hidden_metrics(output_v1, output_v2, args.bt_lambda)
    input_sd = shared_difference_metrics_np(input_v1, input_v2)
    output_sd = shared_difference_metrics_np(output_v1, output_v2)

    row = {
        "model": model,
        "dataset": point.dataset,
        "seed": point.seed,
        "depth": point.depth,
        "width": point.width,
        "layer": layer_idx + 1,
        "input_bt_total_per_dim": input_bt["bt_total_per_dim"],
        "output_bt_total_per_dim": output_bt["bt_total_per_dim"],
        "delta_bt_total_per_dim": output_bt["bt_total_per_dim"] - input_bt["bt_total_per_dim"],
        "input_corr_diag_mean": input_bt["corr_diag_mean"],
        "output_corr_diag_mean": output_bt["corr_diag_mean"],
        "delta_corr_diag_mean": output_bt["corr_diag_mean"] - input_bt["corr_diag_mean"],
        "input_weighted_off_per_dim": input_bt["bt_weighted_off_diag_per_dim"],
        "output_weighted_off_per_dim": output_bt["bt_weighted_off_diag_per_dim"],
        "input_shared_diff_ratio": input_sd["shared_diff_ratio"],
        "output_shared_diff_ratio": output_sd["shared_diff_ratio"],
        "delta_shared_diff_ratio": output_sd["shared_diff_ratio"] - input_sd["shared_diff_ratio"],
        "input_diff_fraction": input_sd["diff_fraction"],
        "output_diff_fraction": output_sd["diff_fraction"],
        "branch_rms": rms_torch(branch_train),
        "input_rms": rms_torch(train_stage["input"]),
        "pre_update_rms": rms_torch(pre_update_train),
        "postnorm_update_rms": rms_torch(update_train),
        "branch_over_input_rms": rms_torch(branch_train) / max(rms_torch(train_stage["input"]), 1e-12),
        "pre_update_over_input_rms": rms_torch(pre_update_train) / max(rms_torch(train_stage["input"]), 1e-12),
        "postnorm_update_over_input_rms": rms_torch(update_train) / max(rms_torch(train_stage["input"]), 1e-12),
        "branch_input_row_cosine": row_cosine_torch(branch_train, train_stage["input"]),
        "update_input_row_cosine": row_cosine_torch(update_train, train_stage["input"]),
    }
    row.update({f"output_{key}": value for key, value in covariance_spectrum(output_train).items()})
    row.update({f"agreement_{key}": value for key, value in agreement_spectrum_metrics(output_v1, output_v2, args, device).items()})
    if prev_train_np is None:
        row.update(
            {
                "prev_to_cur_cka": float("nan"),
                "prev_to_cur_forward_r2": float("nan"),
                "prev_to_cur_reverse_r2": float("nan"),
                "prev_to_cur_sym_r2": float("nan"),
                "prev_to_cur_linear_novelty": float("nan"),
            }
        )
    else:
        row.update(transition_metrics(prev_train_np, output_train, args, device))
    return row


def all_stage_rows(args, point, model, reps, device):
    rows = []
    prev_train_np = None
    for idx, (train_stage, view1_stage, view2_stage) in enumerate(
        zip(reps["train_stages"], reps["view1_stages"], reps["view2_stages"])
    ):
        row = stage_metrics(args, point, model, idx, train_stage, view1_stage, view2_stage, prev_train_np, device)
        rows.append(row)
        prev_train_np = tensor_np(train_stage["output"])
    return rows


def frob_cosine(a, b):
    af = a.reshape(-1)
    bf = b.reshape(-1)
    return float((torch.dot(af, bf) / torch.clamp(torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf), min=1e-12)).detach().cpu().item())


def local_hidden_bt_loss(h1, h2, bt_lambda):
    return barlow_loss(h1, h2, bt_lambda) / h1.shape[1]


def forward_from_layer(h, weights, start_idx, activation_alpha, residual_scale, use_layernorm):
    out = h
    for weight in weights[start_idx:]:
        _, _, out = residual_layer(out, weight, activation_alpha, residual_scale, use_layernorm)
    return out


def gradient_alignment_rows(args, point, model, state, reps, device):
    weights = reps["weights"]
    projector = state.get("projector")
    if isinstance(projector, list):
        final_projector = projector[-1].to(device)
    else:
        final_projector = projector.to(device)
    rows = []
    n = min(args.grad_samples, reps["normed"]["view1_tr"].shape[0])
    idx = torch.linspace(0, reps["normed"]["view1_tr"].shape[0] - 1, n, device=device).long()

    for layer_idx, (v1_stage, v2_stage) in enumerate(zip(reps["view1_stages"], reps["view2_stages"])):
        in1 = v1_stage["input"].index_select(0, idx).detach()
        in2 = v2_stage["input"].index_select(0, idx).detach()
        out1_actual = v1_stage["output"].index_select(0, idx).detach()
        out2_actual = v2_stage["output"].index_select(0, idx).detach()
        upd1 = out1_actual - in1
        upd2 = out2_actual - in2

        loc1 = in1.clone().requires_grad_(True)
        loc2 = in2.clone().requires_grad_(True)
        local_loss = local_hidden_bt_loss(loc1, loc2, args.bt_lambda)
        local_grads = torch.autograd.grad(local_loss, [loc1, loc2], retain_graph=False)
        local_neg1 = -local_grads[0].detach()
        local_neg2 = -local_grads[1].detach()

        fin1 = in1.clone().requires_grad_(True)
        fin2 = in2.clone().requires_grad_(True)
        final_h1 = forward_from_layer(
            fin1, weights, layer_idx, state["activation_alpha"], state["residual_scale"], state["layernorm"]
        )
        final_h2 = forward_from_layer(
            fin2, weights, layer_idx, state["activation_alpha"], state["residual_scale"], state["layernorm"]
        )
        final_loss = barlow_loss(final_h1 @ final_projector, final_h2 @ final_projector, args.bt_lambda) / final_projector.shape[1]
        final_grads = torch.autograd.grad(final_loss, [fin1, fin2], retain_graph=False)
        final_neg1 = -final_grads[0].detach()
        final_neg2 = -final_grads[1].detach()

        update = torch.cat([upd1, upd2], dim=0)
        local_neg = torch.cat([local_neg1, local_neg2], dim=0)
        final_neg = torch.cat([final_neg1, final_neg2], dim=0)
        rows.append(
            {
                "model": model,
                "dataset": point.dataset,
                "seed": point.seed,
                "depth": point.depth,
                "layer": layer_idx + 1,
                "grad_samples": int(n),
                "local_bt_loss_at_input_per_dim": float(local_loss.detach().cpu().item()),
                "final_projector_loss_from_input_per_projdim": float(final_loss.detach().cpu().item()),
                "update_vs_neg_local_grad_cosine": frob_cosine(update, local_neg),
                "update_vs_neg_final_grad_cosine": frob_cosine(update, final_neg),
                "local_vs_final_neg_grad_cosine": frob_cosine(local_neg, final_neg),
                "update_norm": float(torch.linalg.vector_norm(update).detach().cpu().item()),
                "neg_local_grad_norm": float(torch.linalg.vector_norm(local_neg).detach().cpu().item()),
                "neg_final_grad_norm": float(torch.linalg.vector_norm(final_neg).detach().cpu().item()),
                "local_first_order_delta": float(torch.sum(-local_neg * update).detach().cpu().item()),
                "final_first_order_delta": float(torch.sum(-final_neg * update).detach().cpu().item()),
            }
        )
    return rows


def readout_summary(point, model, state, reps, ytr, yte, args):
    layer_rows = []
    for idx, (xtr, xte) in enumerate(zip(reps["pathnorm_train"], reps["pathnorm_test"])):
        row = {
            "model": model,
            "seed": point.seed,
            "dataset": point.dataset,
            "depth": point.depth,
            "layer": idx + 1,
            "setup": "layer_hidden_512",
        }
        row.update(linear_classifier_readout(xtr, xte, ytr, yte, args.probe_reg))
        row.update(covariance_spectrum(xtr))
        layer_rows.append(row)
    last = {
        "model": model,
        "seed": point.seed,
        "dataset": point.dataset,
        "depth": point.depth,
        "setup": "last_layer_512",
    }
    last.update(linear_classifier_readout(reps["pathnorm_train"][-1], reps["pathnorm_test"][-1], ytr, yte, args.probe_reg))
    all_tr = np.concatenate(reps["pathnorm_train"], axis=1)
    all_te = np.concatenate(reps["pathnorm_test"], axis=1)
    all_pca = {
        "model": model,
        "seed": point.seed,
        "dataset": point.dataset,
        "depth": point.depth,
        "setup": "all_layers_pca512",
    }
    all_pca.update(pca512_readout(all_tr, all_te, ytr, yte, args.probe_reg, point.seed, args.pca_dim))
    best = max(layer_rows, key=lambda row: row["test_accuracy"])
    summary = {
        "model": model,
        "seed": point.seed,
        "dataset": point.dataset,
        "depth": point.depth,
        "last_layer_accuracy": last["test_accuracy"],
        "all_pca_accuracy": all_pca["test_accuracy"],
        "best_layer_accuracy": best["test_accuracy"],
        "best_layer": int(best["layer"]),
        "last_effective_rank": layer_rows[-1]["effective_rank"],
    }
    return layer_rows, [last, all_pca], summary


def train_greedy_residual_barlow(point, tensors, device, config, args):
    normed = normalized_inputs(tensors)
    htr = normed["xtr"]
    hte = normed["xte"]
    v1tr = normed["view1_tr"]
    v2tr = normed["view2_tr"]
    v1te = normed["view1_te"]
    v2te = normed["view2_te"]
    weights = []
    projectors = []
    histories = []
    final_dim = min(point.width, point.input_dim)
    steps_per_epoch = max(1, int(math.ceil(point.n_train / config["batch_size"])))
    total_steps = int(math.ceil(config["epochs"] * steps_per_epoch))
    start = time.perf_counter()

    for layer_idx in range(point.depth):
        torch.manual_seed(point.seed + 2404 + layer_idx)
        weight = torch.nn.Parameter(torch.randn((final_dim, final_dim), device=device) * math.sqrt(2.0 / final_dim))
        projector = torch.nn.Parameter(
            torch.randn((final_dim, config["projector_dim"]), device=device) * math.sqrt(2.0 / final_dim)
        )
        if args.branch_init_scale != 1.0:
            with torch.no_grad():
                weight.mul_(float(args.branch_init_scale))
        optimizer = torch.optim.AdamW([weight, projector], lr=config["lr"], weight_decay=config["weight_decay"])
        permutation = torch.randperm(point.n_train, device=device)
        cursor = 0
        losses = []
        layer_history = []
        for step in range(1, total_steps + 1):
            if cursor + config["batch_size"] > point.n_train:
                permutation = torch.randperm(point.n_train, device=device)
                cursor = 0
            batch_idx = permutation[cursor : cursor + config["batch_size"]]
            cursor += config["batch_size"]
            _, _, out1 = residual_layer(v1tr[batch_idx], weight, args.activation_alpha, args.residual_scale, args.layernorm)
            _, _, out2 = residual_layer(v2tr[batch_idx], weight, args.activation_alpha, args.residual_scale, args.layernorm)
            loss = barlow_loss(out1 @ projector, out2 @ projector, config["bt_lambda"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([weight, projector], max_norm=args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            if step % steps_per_epoch == 0 or step == total_steps:
                layer_history.append(
                    {
                        "layer": layer_idx + 1,
                        "step": step,
                        "epoch": step / steps_per_epoch,
                        "mean_barlow_loss": float(np.mean(losses[-min(len(losses), steps_per_epoch) :])),
                    }
                )
        with torch.no_grad():
            _, _, htr = residual_layer(htr, weight, args.activation_alpha, args.residual_scale, args.layernorm)
            _, _, hte = residual_layer(hte, weight, args.activation_alpha, args.residual_scale, args.layernorm)
            _, _, v1tr = residual_layer(v1tr, weight, args.activation_alpha, args.residual_scale, args.layernorm)
            _, _, v2tr = residual_layer(v2tr, weight, args.activation_alpha, args.residual_scale, args.layernorm)
            _, _, v1te = residual_layer(v1te, weight, args.activation_alpha, args.residual_scale, args.layernorm)
            _, _, v2te = residual_layer(v2te, weight, args.activation_alpha, args.residual_scale, args.layernorm)
        weights.append(weight.detach().cpu())
        projectors.append(projector.detach().cpu())
        histories.extend(layer_history)
        print(
            f"greedy residual BT depth={point.depth} layer={layer_idx + 1}/{point.depth} "
            f"loss={layer_history[-1]['mean_barlow_loss']:.3f}",
            flush=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "weights": weights,
        "projector": projectors,
        "norm_mean": normed["norm_mean"].detach().cpu(),
        "norm_scale": normed["norm_scale"].detach().cpu(),
        "history": histories,
        "fit_time_sec": elapsed,
        "config": dict(config),
        "activation": BP_BT_ACTIVATION,
        "activation_alpha": float(args.activation_alpha),
        "residual_scale": float(args.residual_scale),
        "layernorm": bool(args.layernorm),
        "branch_init_scale": float(args.branch_init_scale),
        "model_type": "greedy_residual_backprop_barlow_twins",
    }


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_stage(stage_rows, grad_rows, readout_summaries):
    summaries = []
    keys = sorted({(row["model"], row["depth"], row["seed"]) for row in stage_rows})
    for model, depth, seed in keys:
        rows = sorted(
            [row for row in stage_rows if row["model"] == model and row["depth"] == depth and row["seed"] == seed],
            key=lambda row: row["layer"],
        )
        grads = [row for row in grad_rows if row["model"] == model and row["depth"] == depth and row["seed"] == seed]
        readouts = [row for row in readout_summaries if row["model"] == model and row["depth"] == depth and row["seed"] == seed]
        readout = readouts[0] if readouts else {}
        final = rows[-1]
        summaries.append(
            {
                "model": model,
                "depth": depth,
                "seed": seed,
                "final_output_bt_total_per_dim": final["output_bt_total_per_dim"],
                "first_input_bt_total_per_dim": rows[0]["input_bt_total_per_dim"],
                "bt_improving_step_fraction": float(np.mean([row["delta_bt_total_per_dim"] < 0.0 for row in rows])),
                "final_output_corr_diag_mean": final["output_corr_diag_mean"],
                "first_input_shared_diff_ratio": rows[0]["input_shared_diff_ratio"],
                "final_output_shared_diff_ratio": final["output_shared_diff_ratio"],
                "mean_postnorm_update_over_input_rms": float(np.mean([row["postnorm_update_over_input_rms"] for row in rows])),
                "final_postnorm_update_over_input_rms": final["postnorm_update_over_input_rms"],
                "mean_linear_novelty": float(np.nanmean([row["prev_to_cur_linear_novelty"] for row in rows])),
                "mean_update_vs_neg_local_grad_cosine": float(np.mean([row["update_vs_neg_local_grad_cosine"] for row in grads])) if grads else float("nan"),
                "mean_update_vs_neg_final_grad_cosine": float(np.mean([row["update_vs_neg_final_grad_cosine"] for row in grads])) if grads else float("nan"),
                "mean_local_vs_final_neg_grad_cosine": float(np.mean([row["local_vs_final_neg_grad_cosine"] for row in grads])) if grads else float("nan"),
                "last_layer_accuracy": readout.get("last_layer_accuracy", float("nan")),
                "all_pca_accuracy": readout.get("all_pca_accuracy", float("nan")),
                "best_layer_accuracy": readout.get("best_layer_accuracy", float("nan")),
                "best_layer": readout.get("best_layer", -1),
                "last_effective_rank": readout.get("last_effective_rank", float("nan")),
            }
        )
    return summaries


def fmt(value):
    if isinstance(value, float) and np.isnan(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_report(path, summaries):
    lines = [
        "# Residual BT Route-3 Mechanistic Diagnostic",
        "",
        "This compares end-to-end residual BP-BT against a greedy local residual BP-BT control.",
        "Greedy training uses the same residual block form but optimizes one layer at a time with a BT loss on that layer's output.",
        "",
        "| Model | Depth | Final BT/dim | Step improve frac | Corr diag | Shared/diff | Mean update/input | Mean novelty | Update vs local grad | Update vs final grad | Local vs final grad | Last acc | All PCA | Best layer |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(summaries, key=lambda item: (item["depth"], item["model"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    str(row["depth"]),
                    fmt(row["final_output_bt_total_per_dim"]),
                    fmt(row["bt_improving_step_fraction"]),
                    fmt(row["final_output_corr_diag_mean"]),
                    fmt(row["final_output_shared_diff_ratio"]),
                    fmt(row["mean_postnorm_update_over_input_rms"]),
                    fmt(row["mean_linear_novelty"]),
                    fmt(row["mean_update_vs_neg_local_grad_cosine"]),
                    fmt(row["mean_update_vs_neg_final_grad_cosine"]),
                    fmt(row["mean_local_vs_final_neg_grad_cosine"]),
                    fmt(row["last_layer_accuracy"]),
                    fmt(row["all_pca_accuracy"]),
                    f"{fmt(row['best_layer_accuracy'])} @ {row['best_layer']}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Files: `stage_rows.jsonl/csv`, `gradient_rows.jsonl/csv`, `readout_summary.jsonl/csv`, `summary.jsonl/csv`.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args):
    torch.set_num_threads(args.torch_threads)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
        torch.backends.cudnn.allow_tf32 = not args.no_tf32
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = parse_config(args.config)

    stage_rows = []
    gradient_rows = []
    layer_readouts = []
    setup_readouts = []
    readout_summaries = []
    model_summaries = []

    for depth in args.depths:
        for seed in args.seeds:
            point = point_for(args, depth, seed)
            tensors = load_tensors(point, device)
            ytr = tensors["ytr_np"]
            yte = tensors["yte_np"]

            if "e2e" in args.models:
                model_path = find_residual_bt_model(args.bt_model_dir, depth)
                print(f"route3 collecting e2e residual BP-BT depth={depth} checkpoint={model_path}", flush=True)
                state = torch.load(model_path, map_location="cpu", weights_only=False)
                reps = collect_residual_reps(point, tensors, state, device)
                stage_rows.extend(all_stage_rows(args, point, "e2e_residual_bpbt", reps, device))
                gradient_rows.extend(gradient_alignment_rows(args, point, "e2e_residual_bpbt", state, reps, device))
                lr, sr, rs = readout_summary(point, "e2e_residual_bpbt", state, reps, ytr, yte, args)
                layer_readouts.extend(lr)
                setup_readouts.extend(sr)
                readout_summaries.append(rs)
                del state, reps, lr, sr, rs
                torch.cuda.empty_cache()
                gc.collect()

            if "greedy" in args.models:
                print(f"route3 training greedy residual BP-BT depth={depth} config={config}", flush=True)
                state = train_greedy_residual_barlow(point, tensors, device, config, args)
                model_name = f"greedy_residual_bt_{config['name']}_d{depth}"
                torch.save({**state, "point": asdict(point)}, args.out_dir / f"{model_name}.pt")
                reps = collect_residual_reps(point, tensors, state, device)
                stage_rows.extend(all_stage_rows(args, point, "greedy_residual_bpbt", reps, device))
                gradient_rows.extend(gradient_alignment_rows(args, point, "greedy_residual_bpbt", state, reps, device))
                lr, sr, rs = readout_summary(point, "greedy_residual_bpbt", state, reps, ytr, yte, args)
                layer_readouts.extend(lr)
                setup_readouts.extend(sr)
                readout_summaries.append(rs)
                model_summaries.append(
                    {
                        "model": "greedy_residual_bpbt",
                        "depth": depth,
                        "seed": seed,
                        "fit_time_sec": float(state["fit_time_sec"]),
                        "final_layer_training_loss": float(state["history"][-1]["mean_barlow_loss"]),
                        "epochs_per_layer": float(config["epochs"]),
                    }
                )
                del state, reps, lr, sr, rs
                torch.cuda.empty_cache()
                gc.collect()

            write_jsonl(args.out_dir / "stage_rows.partial.jsonl", stage_rows)
            write_jsonl(args.out_dir / "gradient_rows.partial.jsonl", gradient_rows)
            write_jsonl(args.out_dir / "layer_readouts.partial.jsonl", layer_readouts)
            write_jsonl(args.out_dir / "setup_readouts.partial.jsonl", setup_readouts)
            write_jsonl(args.out_dir / "readout_summary.partial.jsonl", readout_summaries)
            del tensors
            torch.cuda.empty_cache()
            gc.collect()

    summaries = summarize_stage(stage_rows, gradient_rows, readout_summaries)
    write_jsonl(args.out_dir / "stage_rows.jsonl", stage_rows)
    write_jsonl(args.out_dir / "gradient_rows.jsonl", gradient_rows)
    write_jsonl(args.out_dir / "layer_readouts.jsonl", layer_readouts)
    write_jsonl(args.out_dir / "setup_readouts.jsonl", setup_readouts)
    write_jsonl(args.out_dir / "readout_summary.jsonl", readout_summaries)
    write_jsonl(args.out_dir / "model_summary.jsonl", model_summaries)
    write_jsonl(args.out_dir / "summary.jsonl", summaries)
    write_csv(args.out_dir / "stage_rows.csv", stage_rows)
    write_csv(args.out_dir / "gradient_rows.csv", gradient_rows)
    write_csv(args.out_dir / "readout_summary.csv", readout_summaries)
    write_csv(args.out_dir / "summary.csv", summaries)
    write_report(args.out_dir / "report.md", summaries)
    print((args.out_dir / "report.md").read_text(encoding="utf-8"), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Route-3 residual BP-BT mechanistic and greedy-local diagnostic.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_route3_residual_bt_seed7"))
    parser.add_argument("--bt-model-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_residual_barlow_seed7"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.65)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--models", nargs="+", default=["e2e", "greedy"], choices=["e2e", "greedy"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--dataset", default="cifar100_simclr")
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depths", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--config", default="greedy_residual_bt:100:0.001:1024:2048:0.005:0.0001")
    parser.add_argument("--bt-lambda", type=float, default=0.005)
    parser.add_argument("--activation-alpha", type=float, default=BP_BT_ACTIVATION_ALPHA)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--branch-init-scale", type=float, default=1.0)
    parser.add_argument("--layernorm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--grad-samples", type=int, default=2048)
    parser.add_argument("--probe-reg", type=float, default=100.0)
    parser.add_argument("--pca-dim", type=int, default=512)
    parser.add_argument("--max-spectrum-samples", type=int, default=50000)
    parser.add_argument("--max-transition-samples", type=int, default=12000)
    parser.add_argument("--ridge-reg", type=float, default=1e-3)
    parser.add_argument("--spectrum-eps", type=float, default=1e-6)
    parser.add_argument("--cut-thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--soft-lambdas", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.25, 1.0])
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
