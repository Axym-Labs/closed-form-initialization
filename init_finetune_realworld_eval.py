import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset

import broader_eval_suite as bes
import closed_form_barlow_twins as cfbt
import dual_path_residual_cifar as dpr
import transformer_cifar_compare as tcc
from project_paths import default_json_path, default_plot_path, repo_relative_path, resolve_json_path


BENCHMARK_NAME = "init_finetune_realworld_eval"
PLOT_SUBDIR = BENCHMARK_NAME
SEEDS = [7, 11, 19]
ANYTIME_BUDGET_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
LOW_DATA_FRACTIONS = [0.01, 0.10, 1.00]
LOW_DATA_BUDGET_FRACTIONS = [0.10, 1.00]
CE_HEAD_EPOCHS = 24
CE_HEAD_BATCH_SIZE = 512
CE_HEAD_LR = 5e-3
CE_HEAD_WEIGHT_DECAY = 1e-4
TRANSFER_FRACTIONS = LOW_DATA_FRACTIONS
TRANSFORMER_METHOD = "spectral-self"
MLP_METHOD = bes.DEFAULT_MLP_WINNER


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    title: str
    architecture: str
    dataset: bes.DatasetSpec
    mlp_config: bes.MLPEvalConfig | None = None
    transformer_config: bes.TransformerEvalConfig | None = None


@dataclass
class TrainArtifact:
    rows: list[dict]
    final_row: dict
    predict_logits: object
    init_time_sec: float = 0.0
    init_compute_proxy: float = 0.0
    shared_state: dict | None = None
    shared_feature_time_sec: float = 0.0
    shared_feature_compute_proxy: float = 0.0
    shared_feature_cache: dict | None = None


def dataset_spec_from_registry(name: str, **overrides):
    base = dict(bes.DATASET_REGISTRY[name])
    base.update(overrides)
    return bes.DatasetSpec(
        name=name,
        suite=base.pop("suite"),
        n_train=base.pop("n_train"),
        n_test=base.pop("n_test"),
        seed=base.pop("seed", 0),
        **base,
    )


def scenario_registry():
    return [
        ScenarioSpec(
            name="covtype_mlp",
            title="Covtype / MLP",
            architecture="mlp",
            dataset=dataset_spec_from_registry("covtype", seed=0),
            mlp_config=replace(bes.MLPEvalConfig(), epochs=8, batch_size=256),
        ),
        ScenarioSpec(
            name="cifar100_transformer",
            title="CIFAR-100 / Transformer",
            architecture="transformer",
            dataset=dataset_spec_from_registry("cifar100", seed=0),
            transformer_config=replace(bes.TransformerEvalConfig(), epochs=5, batch_size=128, patch_size=8),
        ),
        ScenarioSpec(
            name="qnli_transformer",
            title="QNLI / Transformer",
            architecture="transformer",
            dataset=dataset_spec_from_registry("qnli", seed=0),
            transformer_config=replace(bes.TransformerEvalConfig(), epochs=5, batch_size=128),
        ),
        ScenarioSpec(
            name="wikitext2_next_token_transformer",
            title="WikiText-2 Next Token / Transformer",
            architecture="transformer",
            dataset=dataset_spec_from_registry("wikitext2_next_token", seed=0, n_test=4000),
            transformer_config=replace(bes.TransformerEvalConfig(), epochs=5, batch_size=128),
        ),
    ]


def maybe_import_pyplot():
    return bes.maybe_import_pyplot()


def json_path(name: str):
    return default_json_path(name)


def plot_path(name: str):
    return default_plot_path(Path(PLOT_SUBDIR) / name)


def write_json(path: Path, payload):
    path = resolve_json_path(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows):
    path = resolve_json_path(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def quality_direction(task_type: str):
    return "min" if task_type == "next_token" else "max"


def primary_metric_name(task_type: str):
    return "validation_cross_entropy" if task_type == "next_token" else "classifier_accuracy"


def primary_metric_label(task_type: str):
    return "Validation cross-entropy" if task_type == "next_token" else "Accuracy"


def quality_value(metrics: dict, task_type: str):
    return -float(metrics["validation_cross_entropy"]) if task_type == "next_token" else float(metrics["classifier_accuracy"])


def softmax_numpy(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def expected_calibration_error(logits, y, bins=15):
    probs = softmax_numpy(logits)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    y = np.asarray(y, dtype=np.int64)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (conf >= lower) & (conf <= upper)
        else:
            mask = (conf >= lower) & (conf < upper)
        if not np.any(mask):
            continue
        bucket_acc = correct[mask].mean()
        bucket_conf = conf[mask].mean()
        ece += float(mask.mean()) * abs(bucket_acc - bucket_conf)
    return float(ece)


def evaluate_task_metrics(logits, y, task_type):
    metrics = {
        "classifier_accuracy": float(bes.evaluate_logits(logits, y)),
        "negative_log_likelihood": float(bes.evaluate_cross_entropy(logits, y)),
    }
    if task_type == "classification":
        metrics["expected_calibration_error"] = expected_calibration_error(logits, y)
        metrics["primary_metric_name"] = "classifier_accuracy"
        metrics["primary_metric_label"] = "Accuracy"
        metrics["primary_metric_value"] = metrics["classifier_accuracy"]
    else:
        metrics["validation_cross_entropy"] = metrics["negative_log_likelihood"]
        metrics["validation_perplexity"] = float(math.exp(min(metrics["validation_cross_entropy"], 30.0)))
        metrics["primary_metric_name"] = "validation_cross_entropy"
        metrics["primary_metric_label"] = "Validation cross-entropy"
        metrics["primary_metric_value"] = metrics["validation_cross_entropy"]
    metrics["quality_value"] = quality_value(metrics, task_type)
    metrics["quality_direction"] = quality_direction(task_type)
    return metrics


def mean_std_ci(values):
    vals = np.asarray(values, dtype=np.float64)
    mean = float(vals.mean())
    std = float(vals.std(ddof=0))
    ci = 0.0 if vals.size <= 1 else float(1.96 * vals.std(ddof=1) / math.sqrt(vals.size))
    return mean, std, ci


def normalize_train_arrays_only(train_arrays):
    mean = sum(arr.mean(axis=0, keepdims=True) for arr in train_arrays) / len(train_arrays)
    centered = [arr - mean for arr in train_arrays]
    avg_var = sum(np.mean(arr * arr, axis=0, keepdims=True) for arr in centered) / len(centered)
    scale = np.sqrt(np.maximum(avg_var, 1e-6))
    scaled = [arr / scale for arr in centered]
    return scaled, mean, scale


def fit_encoder_init_mlp_state(bundle: bes.RawDatasetBundle, method_name: str, config: bes.MLPEvalConfig, seed: int):
    bes.set_seed(seed)
    base_tr, _, view1_tr, view2_tr, _, _ = bes.build_mlp_arrays(bundle, seed)
    train_arrays, initial_mean, initial_scale = normalize_train_arrays_only([base_tr, view1_tr, view2_tr])
    base_tr, view1_tr, view2_tr = train_arrays
    layer_states = []
    hidden_param_count = 0
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
        hidden_param_count += int(fitted["transform_base"].size)
        base_tr = cfbt.apply_layer(base_tr, fitted["transform_base"], activation=config.activation)
        view1_tr = cfbt.apply_layer(view1_tr, fitted["transform_view1"], activation=config.activation)
        view2_tr = cfbt.apply_layer(view2_tr, fitted["transform_view2"], activation=config.activation)

        base_center_mean = None
        if config.center_after_hidden:
            base_center_mean = base_tr.mean(axis=0, keepdims=True)
            base_tr = base_tr - base_center_mean
            view1_tr = view1_tr - view1_tr.mean(axis=0, keepdims=True)
            view2_tr = view2_tr - view2_tr.mean(axis=0, keepdims=True)

        train_arrays, norm_mean, norm_scale = normalize_train_arrays_only([base_tr, view1_tr, view2_tr])
        base_tr, view1_tr, view2_tr = train_arrays
        post_map = np.zeros((base_tr.shape[1], bundle.num_classes), dtype=np.float32)
        layer_states.append(
            {
                "fitted": fitted,
                "pre_output_map": None,
                "post_output_map": post_map,
                "base_center_mean": base_center_mean,
                "norm_mean": norm_mean,
                "norm_scale": norm_scale,
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
        "layers": [],
        "yhat_te": np.zeros((len(bundle.yte), bundle.num_classes), dtype=np.float64),
        "output_map": None,
        "activation_param_count": hidden_param_count,
        "output_param_count": 0,
        "fit_time_sec": fit_time,
    }


def fit_encoder_init_transformer_state(
    bundle: bes.RawDatasetBundle, attention_kind: str, config: bes.TransformerEvalConfig, seed: int
):
    bes.set_seed(seed)
    base_tr, _, view1_tr, view2_tr, _, _ = bes.build_transformer_arrays(bundle, seed, config.patch_size, include_test_views=False)
    attn_cfg = bes.make_attention_config(base_tr, attention_kind, config, seed)
    layer_states = []
    hidden_param_count = 0
    start = time.perf_counter()
    for _ in range(config.depth):
        att_model = tcc.fit_attention_block(attn_cfg, view1_tr, view2_tr)
        hidden_param_count += int(att_model["parameter_count"])
        base_tr = tcc.apply_attention_block(base_tr, attn_cfg, att_model)
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
        view1_tr = apply_ffn(view1_tr)
        view2_tr = apply_ffn(view2_tr)
        out_map = np.zeros((base_tr.shape[-1], bundle.num_classes), dtype=np.float32)
        layer_states.append({"out_map": out_map, "att_model": att_model, "ffn_model": ffn_model})
    fit_time = time.perf_counter() - start
    return {
        "bundle": bundle,
        "attention_kind": attention_kind,
        "config": config,
        "seed": seed,
        "attn_cfg": attn_cfg,
        "layer_states": layer_states,
        "layers": [],
        "yhat_te": np.zeros((len(bundle.yte), bundle.num_classes), dtype=np.float64),
        "raw_yhat_te": np.zeros((len(bundle.yte), bundle.num_classes), dtype=np.float64),
        "logit_temperature": 1.0,
        "hidden_param_count": hidden_param_count,
        "output_param_count": 0,
        "fit_time_sec": fit_time,
    }


def estimate_encoder_init_mlp_compute(bundle: bes.RawDatasetBundle, config: bes.MLPEvalConfig):
    input_dim = int(bes.build_base_mlp_features(bundle, bundle.train_raw[:1]).shape[1])
    train_count = int(len(bundle.ytr))
    fit_views = 2.0 * train_count
    apply_views = 3.0 * train_count
    current_dim = input_dim
    total = 0.0
    for _ in range(config.depth):
        next_dim = int(config.width)
        total += fit_views * current_dim * next_dim
        total += apply_views * current_dim * next_dim
        current_dim = next_dim
    return float(total)


def estimate_frozen_mlp_encode_compute(bundle: bes.RawDatasetBundle, config: bes.MLPEvalConfig, total_examples: int):
    input_dim = int(bes.build_base_mlp_features(bundle, bundle.train_raw[:1]).shape[1])
    current_dim = input_dim
    total = 0.0
    for _ in range(config.depth):
        next_dim = int(config.width)
        total += total_examples * current_dim * next_dim
        current_dim = next_dim
    return float(total)


def estimate_encoder_init_transformer_compute(bundle: bes.RawDatasetBundle, config: bes.TransformerEvalConfig):
    if bundle.modality == "image":
        token_dim = int(bundle.train_raw.shape[1] * config.patch_size * config.patch_size)
        num_tokens = int((bundle.image_size // config.patch_size) ** 2)
    elif bundle.modality == "text":
        token_dim = int(bundle.text_embedding.shape[1])
        num_tokens = int(bundle.train_raw.shape[1])
    else:
        token_dim = int(bundle.tabular_token_embedding.shape[1])
        num_tokens = int(bundle.train_raw.shape[1])
    train_count = int(len(bundle.ytr))
    num_heads = bes.choose_num_heads(token_dim, preferred=config.num_heads)
    analytic_heads = max(1, min(config.analytic_num_heads, num_heads))
    rank = max(1, config.attention_rank if config.attention_rank > 0 else config.num_landmarks * analytic_heads)
    fit_views = 2.0 * train_count
    apply_views = 3.0 * train_count
    score_projection_cost = num_tokens * token_dim * rank
    score_matrix_cost = num_tokens * num_tokens * rank
    value_mix_cost = num_tokens * num_tokens * token_dim
    attention_fit_cost = config.depth * fit_views * (score_projection_cost + score_matrix_cost + value_mix_cost)
    attention_apply_cost = config.depth * apply_views * (score_projection_cost + score_matrix_cost + value_mix_cost)
    ffn_fit_cost = config.depth * fit_views * num_tokens * token_dim * token_dim
    ffn_apply_cost = config.depth * apply_views * num_tokens * token_dim * token_dim
    return float(attention_fit_cost + attention_apply_cost + ffn_fit_cost + ffn_apply_cost)


def estimate_frozen_transformer_encode_compute(bundle: bes.RawDatasetBundle, config: bes.TransformerEvalConfig, total_examples: int):
    if bundle.modality == "image":
        token_dim = int(bundle.train_raw.shape[1] * config.patch_size * config.patch_size)
        num_tokens = int((bundle.image_size // config.patch_size) ** 2)
    elif bundle.modality == "text":
        token_dim = int(bundle.text_embedding.shape[1])
        num_tokens = int(bundle.train_raw.shape[1])
    else:
        token_dim = int(bundle.tabular_token_embedding.shape[1])
        num_tokens = int(bundle.train_raw.shape[1])
    num_heads = bes.choose_num_heads(token_dim, preferred=config.num_heads)
    analytic_heads = max(1, min(config.analytic_num_heads, num_heads))
    rank = max(1, config.attention_rank if config.attention_rank > 0 else config.num_landmarks * analytic_heads)
    score_projection_cost = num_tokens * token_dim * rank
    score_matrix_cost = num_tokens * num_tokens * rank
    value_mix_cost = num_tokens * num_tokens * token_dim
    attention_apply_cost = config.depth * total_examples * (score_projection_cost + score_matrix_cost + value_mix_cost)
    ffn_apply_cost = config.depth * total_examples * num_tokens * token_dim * token_dim
    return float(attention_apply_cost + ffn_apply_cost)


def estimate_linear_head_train_compute(train_examples: int, feature_dim: int, num_classes: int, epochs: int):
    return float(2.0 * epochs * train_examples * feature_dim * num_classes)


def checkpoint_step_map(total_steps: int, fractions: list[float]):
    mapping = {}
    if total_steps <= 0:
        return mapping
    for frac in fractions:
        step = max(1, min(total_steps, int(math.ceil(total_steps * frac))))
        mapping[step] = float(step / total_steps)
    mapping[total_steps] = 1.0
    return dict(sorted(mapping.items()))


def subset_indices(total: int, fraction: float, seed: int):
    if fraction >= 1.0:
        return np.arange(total, dtype=np.int64)
    count = max(1, int(round(total * fraction)))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=count, replace=False))


def subset_bundle(bundle: bes.RawDatasetBundle, indices):
    return bes.RawDatasetBundle(
        name=bundle.name,
        suite=bundle.suite,
        modality=bundle.modality,
        task_type=bundle.task_type,
        train_raw=bundle.train_raw[indices],
        test_raw=bundle.test_raw,
        ytr=bundle.ytr[indices],
        yte=bundle.yte,
        num_classes=bundle.num_classes,
        image_size=bundle.image_size,
        image_mean=bundle.image_mean,
        text_embedding=bundle.text_embedding,
        text_pos=bundle.text_pos,
        text_drop_prob=bundle.text_drop_prob,
        tabular_mean=bundle.tabular_mean,
        tabular_std=bundle.tabular_std,
        tabular_token_embedding=bundle.tabular_token_embedding,
        tabular_pos=bundle.tabular_pos,
        tabular_drop_prob=bundle.tabular_drop_prob,
        tabular_noise_std=bundle.tabular_noise_std,
    )


def reset_linear_module(module: nn.Module, seed: int):
    bes.set_seed(seed)
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def reset_mlp_heads(model: bes.ClosedFormFineTuneMLP, seed: int):
    counter = 0
    for head in list(model.pre_heads.values()) + list(model.post_heads.values()):
        reset_linear_module(head, seed + 1000 + counter)
        counter += 1
    if model.final_head is not None:
        reset_linear_module(model.final_head, seed + 1000 + counter)


def reset_transformer_heads(model: bes.ClosedFormFineTuneTransformer, seed: int):
    for idx, head in enumerate(model.output_heads):
        reset_linear_module(head, seed + 2000 + idx)


def raw_tensor_for_bundle(raw_inputs, bundle: bes.RawDatasetBundle):
    if bundle.modality == "text":
        return torch.from_numpy(raw_inputs).long()
    return torch.from_numpy(raw_inputs).float()


def run_backprop_mlp_anytime(bundle, config, seed, scenario_name, data_fraction, budget_fractions):
    bes.set_seed(seed)
    base_tr, base_te, _, _, _, _ = bes.build_mlp_arrays(bundle, seed)
    xtr = torch.from_numpy(base_tr).float()
    xte = torch.from_numpy(base_te).float()
    ytr = torch.from_numpy(bundle.ytr).long()
    train_loader = bes.make_train_loader(TensorDataset(xtr, ytr), batch_size=config.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bes.SupervisedMLP(base_tr.shape[1], config.width, config.depth, bundle.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    total_budget = bes.estimate_mlp_compute_proxy(bundle, config, "backprop")
    total_steps = max(1, config.epochs * len(train_loader))
    step_budget = total_budget / total_steps
    schedule = checkpoint_step_map(total_steps, budget_fractions)
    rows = []
    iterator = iter(train_loader)
    start = time.perf_counter()
    for step_idx in range(1, total_steps + 1):
        try:
            xb, yb = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            xb, yb = next(iterator)
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step_idx not in schedule:
            continue
        model.eval()
        with torch.no_grad():
            logits_te = model(xte.to(device)).cpu().numpy()
        metrics = evaluate_task_metrics(logits_te, bundle.yte, bundle.task_type)
        rows.append(
            {
                "experiment": "anytime",
                "scenario": scenario_name,
                "dataset": bundle.name,
                "architecture": "mlp",
                "model": "backprop",
                "seed": seed,
                "data_fraction": data_fraction,
                "budget_fraction": float(schedule[step_idx]),
                "checkpoint_label": f"{int(round(schedule[step_idx] * 100))}%",
                "total_compute_proxy": float(step_idx * step_budget),
                "wall_clock_sec": float(time.perf_counter() - start),
                "fit_time_sec": float(time.perf_counter() - start),
                "trainable_parameter_count": int(sum(p.numel() for p in model.parameters())),
                "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
                **metrics,
            }
        )
        model.train()

    def predict_logits(raw_inputs):
        with torch.no_grad():
            feats = torch.from_numpy(bes.build_base_mlp_features(bundle, raw_inputs)).float().to(device)
            return model(feats).cpu().numpy()

    return TrainArtifact(rows=rows, final_row=rows[-1], predict_logits=predict_logits)


def run_encoder_init_finetune_mlp_anytime(bundle, config, seed, scenario_name, data_fraction, budget_fractions, state=None):
    if state is None:
        state = fit_encoder_init_mlp_state(bundle, MLP_METHOD, config, seed)
    xtr_np = bes.build_base_mlp_features(bundle, bundle.train_raw).astype(np.float32)
    xte_np = bes.build_base_mlp_features(bundle, bundle.test_raw).astype(np.float32)
    xtr = torch.from_numpy(xtr_np)
    xte = torch.from_numpy(xte_np)
    ytr = torch.from_numpy(bundle.ytr).long()
    train_loader = bes.make_train_loader(TensorDataset(xtr, ytr), batch_size=config.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bes.ClosedFormFineTuneMLP(xtr_np.shape[1], bundle.num_classes, state, config.activation).to(device)
    reset_mlp_heads(model, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    total_budget = bes.estimate_mlp_compute_proxy(bundle, config, "backprop")
    init_budget = estimate_encoder_init_mlp_compute(bundle, config)
    finetune_epoch_budget = bes.estimate_mlp_compute_proxy(bundle, config, "fine-tune")
    total_steps, _ = bes.compute_matched_step_budget(total_budget, init_budget, finetune_epoch_budget, len(train_loader))
    step_budget = 0.0 if len(train_loader) == 0 else finetune_epoch_budget / len(train_loader)
    schedule = checkpoint_step_map(total_steps, budget_fractions)
    rows = []
    model.eval()
    with torch.no_grad():
        init_logits = model(xte.to(device))[-1].cpu().numpy()
    init_metrics = evaluate_task_metrics(init_logits, bundle.yte, bundle.task_type)
    rows.append(
        {
            "experiment": "anytime",
            "scenario": scenario_name,
            "dataset": bundle.name,
            "architecture": "mlp",
            "model": bes.FINE_TUNE_MODEL_NAME,
            "seed": seed,
            "data_fraction": data_fraction,
            "budget_fraction": float(init_budget / max(total_budget, 1e-12)),
            "checkpoint_label": "init",
            "total_compute_proxy": float(init_budget),
            "wall_clock_sec": float(state["fit_time_sec"]),
            "fit_time_sec": float(state["fit_time_sec"]),
            "trainable_parameter_count": int(sum(p.numel() for p in model.parameters())),
            "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
            **init_metrics,
        }
    )
    if total_steps > 0:
        model.train()
        iterator = iter(train_loader)
        train_start = time.perf_counter()
        for step_idx in range(1, total_steps + 1):
            try:
                xb, yb = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                xb, yb = next(iterator)
            xb = xb.to(device)
            yb = yb.to(device)
            depth_logits = model(xb)
            loss = sum(criterion(logits, yb) for logits in depth_logits) / max(len(depth_logits), 1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step_idx not in schedule:
                continue
            model.eval()
            with torch.no_grad():
                logits_te = model(xte.to(device))[-1].cpu().numpy()
            metrics = evaluate_task_metrics(logits_te, bundle.yte, bundle.task_type)
            elapsed = state["fit_time_sec"] + (time.perf_counter() - train_start)
            rows.append(
                {
                    "experiment": "anytime",
                    "scenario": scenario_name,
                    "dataset": bundle.name,
                    "architecture": "mlp",
                    "model": bes.FINE_TUNE_MODEL_NAME,
                    "seed": seed,
                    "data_fraction": data_fraction,
                    "budget_fraction": float((init_budget + step_idx * step_budget) / max(total_budget, 1e-12)),
                    "checkpoint_label": f"{int(round(schedule[step_idx] * 100))}%",
                    "total_compute_proxy": float(init_budget + step_idx * step_budget),
                    "wall_clock_sec": float(elapsed),
                    "fit_time_sec": float(elapsed),
                    "trainable_parameter_count": int(sum(p.numel() for p in model.parameters())),
                    "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
                    **metrics,
                }
            )
            model.train()

    def predict_logits(raw_inputs):
        with torch.no_grad():
            feats = torch.from_numpy(bes.build_base_mlp_features(bundle, raw_inputs).astype(np.float32)).to(device)
            return model(feats)[-1].cpu().numpy()

    return TrainArtifact(
        rows=rows,
        final_row=rows[-1],
        predict_logits=predict_logits,
        init_time_sec=float(state["fit_time_sec"]),
        init_compute_proxy=float(init_budget),
        shared_state=state,
    )


def extract_mlp_layer_features(model, bundle, raw_inputs, batch_size=2048):
    feats = bes.build_base_mlp_features(bundle, raw_inputs).astype(np.float32)
    device = next(model.parameters()).device
    collected = []
    for start in range(0, feats.shape[0], batch_size):
        xb = torch.from_numpy(feats[start : start + batch_size]).to(device)
        with torch.no_grad():
            layer_feats, _ = model.encode_layers(xb)
            collected.append(torch.cat(layer_feats, dim=1).cpu().numpy())
    return np.concatenate(collected, axis=0).astype(np.float32)


def train_linear_head_anytime(
    train_features,
    train_labels,
    test_features,
    test_labels,
    task_type,
    num_classes,
    seed,
    budget_fractions,
    total_compute_offset,
    wall_clock_offset,
    scenario_name,
    dataset_name,
    architecture,
    data_fraction,
    model_name,
    hidden_param_count,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xtr = torch.from_numpy(train_features).float()
    xte = torch.from_numpy(test_features).float()
    ytr = torch.from_numpy(train_labels).long()
    loader = bes.make_train_loader(
        TensorDataset(xtr, ytr),
        batch_size=min(CE_HEAD_BATCH_SIZE, max(1, len(train_features))),
        shuffle=True,
    )
    head = nn.Linear(train_features.shape[1], num_classes).to(device)
    reset_linear_module(head, seed + 3000)
    optimizer = torch.optim.AdamW(head.parameters(), lr=CE_HEAD_LR, weight_decay=CE_HEAD_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    total_steps = max(1, CE_HEAD_EPOCHS * len(loader))
    head_budget = estimate_linear_head_train_compute(len(train_features), train_features.shape[1], num_classes, CE_HEAD_EPOCHS)
    step_budget = head_budget / total_steps
    schedule = checkpoint_step_map(total_steps, budget_fractions)
    rows = []
    iterator = iter(loader)
    train_start = time.perf_counter()
    for step_idx in range(1, total_steps + 1):
        try:
            xb, yb = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            xb, yb = next(iterator)
        xb = xb.to(device)
        yb = yb.to(device)
        logits = head(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step_idx not in schedule:
            continue
        head.eval()
        with torch.no_grad():
            logits_te = head(xte.to(device)).cpu().numpy()
        metrics = evaluate_task_metrics(logits_te, test_labels, task_type)
        elapsed = wall_clock_offset + (time.perf_counter() - train_start)
        rows.append(
            {
                "experiment": "anytime",
                "scenario": scenario_name,
                "dataset": dataset_name,
                "architecture": architecture,
                "model": model_name,
                "seed": seed,
                "data_fraction": data_fraction,
                "budget_fraction": float(schedule[step_idx]),
                "checkpoint_label": f"{int(round(schedule[step_idx] * 100))}%",
                "total_compute_proxy": float(total_compute_offset + step_idx * step_budget),
                "wall_clock_sec": float(elapsed),
                "fit_time_sec": float(elapsed),
                "trainable_parameter_count": int(sum(p.numel() for p in head.parameters())),
                "total_parameter_count": int(hidden_param_count + sum(p.numel() for p in head.parameters())),
                **metrics,
            }
        )
        head.train()

    def predict_logits(feature_matrix):
        with torch.no_grad():
            xb = torch.from_numpy(feature_matrix).float().to(device)
            return head(xb).cpu().numpy()

    return rows, head, head_budget, predict_logits


def run_encoder_init_ce_head_mlp_anytime(bundle, config, seed, scenario_name, data_fraction, budget_fractions, state=None):
    if state is None:
        state = fit_encoder_init_mlp_state(bundle, MLP_METHOD, config, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dim = int(bes.build_base_mlp_features(bundle, bundle.train_raw[:1]).shape[1])
    encoder = bes.ClosedFormFineTuneMLP(base_dim, bundle.num_classes, state, config.activation).to(device)
    feature_start = time.perf_counter()
    train_features = extract_mlp_layer_features(encoder, bundle, bundle.train_raw)
    test_features = extract_mlp_layer_features(encoder, bundle, bundle.test_raw)
    feature_time = time.perf_counter() - feature_start
    feature_compute = estimate_frozen_mlp_encode_compute(bundle, config, len(bundle.ytr) + len(bundle.yte))
    rows, _, _, predict_head = train_linear_head_anytime(
        train_features,
        bundle.ytr,
        test_features,
        bundle.yte,
        bundle.task_type,
        bundle.num_classes,
        seed,
        budget_fractions,
        total_compute_offset=estimate_encoder_init_mlp_compute(bundle, config) + feature_compute,
        wall_clock_offset=state["fit_time_sec"] + feature_time,
        scenario_name=scenario_name,
        dataset_name=bundle.name,
        architecture="mlp",
        data_fraction=data_fraction,
        model_name="closed-form+ce-head",
        hidden_param_count=state["activation_param_count"],
    )

    def predict_logits(raw_inputs):
        feats = extract_mlp_layer_features(encoder, bundle, raw_inputs)
        return predict_head(feats)

    return TrainArtifact(
        rows=rows,
        final_row=rows[-1],
        predict_logits=predict_logits,
        init_time_sec=float(state["fit_time_sec"]),
        init_compute_proxy=float(estimate_encoder_init_mlp_compute(bundle, config)),
        shared_state=state,
        shared_feature_time_sec=float(feature_time),
        shared_feature_compute_proxy=float(feature_compute),
        shared_feature_cache={"train_features": train_features, "test_features": test_features},
    )


def run_backprop_transformer_anytime(bundle, config, seed, scenario_name, data_fraction, budget_fractions):
    bes.set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, test_raw, token_dim, num_heads, to_tokens = bes.prepare_transformer_training_context(bundle, config, device)
    train_loader = bes.make_train_loader(train_ds, batch_size=config.batch_size, shuffle=True)
    model = bes.TokenTransformerClassifier(token_dim, num_heads, config.depth, bundle.num_classes, config.mlp_ratio, task_type=bundle.task_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    total_budget = bes.estimate_transformer_compute_proxy(bundle, config, "backprop")
    total_steps = max(1, config.epochs * len(train_loader))
    step_budget = total_budget / total_steps
    schedule = checkpoint_step_map(total_steps, budget_fractions)
    rows = []
    iterator = iter(train_loader)
    start = time.perf_counter()
    for step_idx in range(1, total_steps + 1):
        try:
            xb, yb = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            xb, yb = next(iterator)
        tokens = to_tokens(xb, augment=True)
        yb = yb.to(device)
        depth_logits = model(tokens)
        loss = sum(criterion(logits, yb) for logits in depth_logits) / len(depth_logits)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step_idx not in schedule:
            continue
        model.eval()
        with torch.no_grad():
            logits_te = model(to_tokens(test_raw, augment=False))[-1].cpu().numpy()
        metrics = evaluate_task_metrics(logits_te, bundle.yte, bundle.task_type)
        rows.append(
            {
                "experiment": "anytime",
                "scenario": scenario_name,
                "dataset": bundle.name,
                "architecture": "transformer",
                "model": "backprop",
                "seed": seed,
                "data_fraction": data_fraction,
                "budget_fraction": float(schedule[step_idx]),
                "checkpoint_label": f"{int(round(schedule[step_idx] * 100))}%",
                "total_compute_proxy": float(step_idx * step_budget),
                "wall_clock_sec": float(time.perf_counter() - start),
                "fit_time_sec": float(time.perf_counter() - start),
                "trainable_parameter_count": int(sum(p.numel() for p in model.parameters())),
                "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
                **metrics,
            }
        )
        model.train()

    def predict_logits(raw_inputs):
        with torch.no_grad():
            tensor = raw_tensor_for_bundle(raw_inputs, bundle).to(device)
            return model(to_tokens(tensor, augment=False))[-1].cpu().numpy()

    return TrainArtifact(rows=rows, final_row=rows[-1], predict_logits=predict_logits)


def run_encoder_init_finetune_transformer_anytime(
    bundle, config, seed, scenario_name, data_fraction, budget_fractions, state=None
):
    if state is None:
        state = fit_encoder_init_transformer_state(bundle, TRANSFORMER_METHOD, config, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, test_raw, token_dim, _, to_tokens = bes.prepare_transformer_training_context(bundle, config, device)
    train_loader = bes.make_train_loader(train_ds, batch_size=config.batch_size, shuffle=True)
    model = bes.ClosedFormFineTuneTransformer(token_dim, bundle.num_classes, state, task_type=bundle.task_type).to(device)
    reset_transformer_heads(model, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    total_budget = bes.estimate_transformer_compute_proxy(bundle, config, "backprop")
    init_budget = estimate_encoder_init_transformer_compute(bundle, config)
    finetune_epoch_budget = bes.estimate_transformer_compute_proxy(bundle, config, bes.FINE_TUNE_MODEL_NAME, TRANSFORMER_METHOD)
    total_steps, _ = bes.compute_matched_step_budget(total_budget, init_budget, finetune_epoch_budget, len(train_loader))
    step_budget = 0.0 if len(train_loader) == 0 else finetune_epoch_budget / len(train_loader)
    schedule = checkpoint_step_map(total_steps, budget_fractions)
    rows = []
    model.eval()
    with torch.no_grad():
        init_logits = model(to_tokens(test_raw, augment=False))[-1].cpu().numpy()
    init_metrics = evaluate_task_metrics(init_logits, bundle.yte, bundle.task_type)
    rows.append(
        {
            "experiment": "anytime",
            "scenario": scenario_name,
            "dataset": bundle.name,
            "architecture": "transformer",
            "model": bes.FINE_TUNE_MODEL_NAME,
            "seed": seed,
            "data_fraction": data_fraction,
            "budget_fraction": float(init_budget / max(total_budget, 1e-12)),
            "checkpoint_label": "init",
            "total_compute_proxy": float(init_budget),
            "wall_clock_sec": float(state["fit_time_sec"]),
            "fit_time_sec": float(state["fit_time_sec"]),
            "trainable_parameter_count": int(sum(p.numel() for p in model.parameters())),
            "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
            **init_metrics,
        }
    )
    if total_steps > 0:
        iterator = iter(train_loader)
        train_start = time.perf_counter()
        model.train()
        for step_idx in range(1, total_steps + 1):
            try:
                xb, yb = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                xb, yb = next(iterator)
            tokens = to_tokens(xb, augment=True)
            yb = yb.to(device)
            depth_logits = model(tokens)
            loss = sum(criterion(logits, yb) for logits in depth_logits) / max(len(depth_logits), 1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step_idx not in schedule:
                continue
            model.eval()
            with torch.no_grad():
                logits_te = model(to_tokens(test_raw, augment=False))[-1].cpu().numpy()
            metrics = evaluate_task_metrics(logits_te, bundle.yte, bundle.task_type)
            elapsed = state["fit_time_sec"] + (time.perf_counter() - train_start)
            rows.append(
                {
                    "experiment": "anytime",
                    "scenario": scenario_name,
                    "dataset": bundle.name,
                    "architecture": "transformer",
                    "model": bes.FINE_TUNE_MODEL_NAME,
                    "seed": seed,
                    "data_fraction": data_fraction,
                    "budget_fraction": float((init_budget + step_idx * step_budget) / max(total_budget, 1e-12)),
                    "checkpoint_label": f"{int(round(schedule[step_idx] * 100))}%",
                    "total_compute_proxy": float(init_budget + step_idx * step_budget),
                    "wall_clock_sec": float(elapsed),
                    "fit_time_sec": float(elapsed),
                    "trainable_parameter_count": int(sum(p.numel() for p in model.parameters())),
                    "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
                    **metrics,
                }
            )
            model.train()

    def predict_logits(raw_inputs):
        with torch.no_grad():
            tensor = raw_tensor_for_bundle(raw_inputs, bundle).to(device)
            return model(to_tokens(tensor, augment=False))[-1].cpu().numpy()

    return TrainArtifact(
        rows=rows,
        final_row=rows[-1],
        predict_logits=predict_logits,
        init_time_sec=float(state["fit_time_sec"]),
        init_compute_proxy=float(init_budget),
        shared_state=state,
    )


def extract_transformer_layer_features(model, bundle, raw_inputs, to_tokens, batch_size):
    device = next(model.parameters()).device
    raw_tensor = raw_tensor_for_bundle(raw_inputs, bundle)
    outputs = []
    for start in range(0, raw_tensor.shape[0], batch_size):
        batch = raw_tensor[start : start + batch_size].to(device)
        with torch.no_grad():
            tokens = to_tokens(batch, augment=False)
            layer_feats, _ = model.encode_layers(tokens)
            outputs.append(torch.cat(layer_feats, dim=1).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def run_encoder_init_ce_head_transformer_anytime(
    bundle, config, seed, scenario_name, data_fraction, budget_fractions, state=None
):
    if state is None:
        state = fit_encoder_init_transformer_state(bundle, TRANSFORMER_METHOD, config, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, token_dim, _, to_tokens = bes.prepare_transformer_training_context(bundle, config, device)
    encoder = bes.ClosedFormFineTuneTransformer(token_dim, bundle.num_classes, state, task_type=bundle.task_type).to(device)
    feature_start = time.perf_counter()
    train_features = extract_transformer_layer_features(encoder, bundle, bundle.train_raw, to_tokens, config.batch_size)
    test_features = extract_transformer_layer_features(encoder, bundle, bundle.test_raw, to_tokens, config.batch_size)
    feature_time = time.perf_counter() - feature_start
    feature_compute = estimate_frozen_transformer_encode_compute(bundle, config, len(bundle.ytr) + len(bundle.yte))
    rows, _, _, predict_head = train_linear_head_anytime(
        train_features,
        bundle.ytr,
        test_features,
        bundle.yte,
        bundle.task_type,
        bundle.num_classes,
        seed,
        budget_fractions,
        total_compute_offset=estimate_encoder_init_transformer_compute(bundle, config) + feature_compute,
        wall_clock_offset=state["fit_time_sec"] + feature_time,
        scenario_name=scenario_name,
        dataset_name=bundle.name,
        architecture="transformer",
        data_fraction=data_fraction,
        model_name="closed-form+ce-head",
        hidden_param_count=state["hidden_param_count"],
    )

    def predict_logits(raw_inputs):
        feats = extract_transformer_layer_features(encoder, bundle, raw_inputs, to_tokens, config.batch_size)
        return predict_head(feats)

    return TrainArtifact(
        rows=rows,
        final_row=rows[-1],
        predict_logits=predict_logits,
        init_time_sec=float(state["fit_time_sec"]),
        init_compute_proxy=float(estimate_encoder_init_transformer_compute(bundle, config)),
        shared_state=state,
        shared_feature_time_sec=float(feature_time),
        shared_feature_compute_proxy=float(feature_compute),
        shared_feature_cache={"train_features": train_features, "test_features": test_features},
    )


def run_ood_for_artifacts(scenario, bundle, seed, artifacts):
    rows = []
    ood_variants = bes.make_ood_variants(bundle, seed)
    for model_name, artifact in artifacts.items():
        id_logits = artifact.predict_logits(bundle.test_raw)
        id_metrics = evaluate_task_metrics(id_logits, bundle.yte, bundle.task_type)
        rows.append(
            {
                "scenario": scenario.name,
                "dataset": bundle.name,
                "architecture": scenario.architecture,
                "model": model_name,
                "seed": seed,
                "variant": "in-distribution",
                **id_metrics,
            }
        )
        for variant_name, raw_inputs in ood_variants.items():
            logits = artifact.predict_logits(raw_inputs)
            metrics = evaluate_task_metrics(logits, bundle.yte, bundle.task_type)
            rows.append(
                {
                    "scenario": scenario.name,
                    "dataset": bundle.name,
                    "architecture": scenario.architecture,
                    "model": model_name,
                    "seed": seed,
                    "variant": variant_name,
                    "delta_primary_metric": float(metrics["primary_metric_value"] - id_metrics["primary_metric_value"]),
                    **metrics,
                }
            )
    return rows


def low_data_subset_runs(scenario, full_bundle, seed):
    rows = []
    for fraction in [frac for frac in LOW_DATA_FRACTIONS if frac < 1.0]:
        idx = subset_indices(len(full_bundle.ytr), fraction, seed + int(round(fraction * 1000)) + 7000)
        bundle = subset_bundle(full_bundle, idx)
        if scenario.architecture == "mlp":
            rows.extend(run_backprop_mlp_anytime(bundle, scenario.mlp_config, seed, scenario.name, fraction, LOW_DATA_BUDGET_FRACTIONS).rows)
            state = fit_encoder_init_mlp_state(bundle, MLP_METHOD, scenario.mlp_config, seed)
            rows.extend(
                run_encoder_init_finetune_mlp_anytime(
                    bundle,
                    scenario.mlp_config,
                    seed,
                    scenario.name,
                    fraction,
                    LOW_DATA_BUDGET_FRACTIONS,
                    state=state,
                ).rows
            )
            rows.extend(
                run_encoder_init_ce_head_mlp_anytime(
                    bundle,
                    scenario.mlp_config,
                    seed,
                    scenario.name,
                    fraction,
                    LOW_DATA_BUDGET_FRACTIONS,
                    state=state,
                ).rows
            )
        else:
            rows.extend(
                run_backprop_transformer_anytime(bundle, scenario.transformer_config, seed, scenario.name, fraction, LOW_DATA_BUDGET_FRACTIONS).rows
            )
            state = fit_encoder_init_transformer_state(bundle, TRANSFORMER_METHOD, scenario.transformer_config, seed)
            rows.extend(
                run_encoder_init_finetune_transformer_anytime(
                    bundle,
                    scenario.transformer_config,
                    seed,
                    scenario.name,
                    fraction,
                    LOW_DATA_BUDGET_FRACTIONS,
                    state=state,
                ).rows
            )
            rows.extend(
                run_encoder_init_ce_head_transformer_anytime(
                    bundle,
                    scenario.transformer_config,
                    seed,
                    scenario.name,
                    fraction,
                    LOW_DATA_BUDGET_FRACTIONS,
                    state=state,
                ).rows
            )
    return rows


def run_shared_init_transfer_mlp(scenario, full_bundle, seed, shared_ft_artifact, shared_ce_artifact):
    config = scenario.mlp_config
    state = shared_ft_artifact.shared_state
    results = []
    shared_init_compute = shared_ft_artifact.init_compute_proxy
    shared_init_time = shared_ft_artifact.init_time_sec
    shared_feature_compute = shared_ce_artifact.shared_feature_compute_proxy
    shared_feature_time = shared_ce_artifact.shared_feature_time_sec
    full_train_features = shared_ce_artifact.shared_feature_cache["train_features"]
    test_features = shared_ce_artifact.shared_feature_cache["test_features"]
    for fraction in TRANSFER_FRACTIONS:
        idx = subset_indices(len(full_bundle.ytr), fraction, seed + int(round(fraction * 1000)) + 9000)
        bundle = subset_bundle(full_bundle, idx)
        xtr_np = bes.build_base_mlp_features(bundle, bundle.train_raw).astype(np.float32)
        ytr = torch.from_numpy(bundle.ytr).long()
        loader = bes.make_train_loader(TensorDataset(torch.from_numpy(xtr_np), ytr), batch_size=config.batch_size, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = bes.ClosedFormFineTuneMLP(xtr_np.shape[1], bundle.num_classes, state, config.activation).to(device)
        reset_mlp_heads(model, seed + int(round(fraction * 1000)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        criterion = nn.CrossEntropyLoss()
        total_budget = bes.estimate_mlp_compute_proxy(bundle, config, "backprop")
        subset_init_budget = estimate_encoder_init_mlp_compute(bundle, config)
        finetune_epoch_budget = bes.estimate_mlp_compute_proxy(bundle, config, "fine-tune")
        steps, _ = bes.compute_matched_step_budget(total_budget, subset_init_budget, finetune_epoch_budget, len(loader))
        step_budget = 0.0 if len(loader) == 0 else finetune_epoch_budget / len(loader)
        iterator = iter(loader)
        start = time.perf_counter()
        for _ in range(steps):
            try:
                xb, yb = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                xb, yb = next(iterator)
            depth_logits = model(xb.to(device))
            loss = sum(criterion(logits, yb.to(device)) for logits in depth_logits) / max(len(depth_logits), 1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits_te = model(torch.from_numpy(bes.build_base_mlp_features(bundle, bundle.test_raw).astype(np.float32)).to(device))[-1].cpu().numpy()
        metrics = evaluate_task_metrics(logits_te, bundle.yte, bundle.task_type)
        results.append(
            {
                "scenario": scenario.name,
                "dataset": bundle.name,
                "architecture": "mlp",
                "model": bes.FINE_TUNE_MODEL_NAME,
                "seed": seed,
                "data_fraction": fraction,
                "transfer_mode": "shared-init",
                "downstream_compute_proxy": float(steps * step_budget),
                "downstream_wall_clock_sec": float(time.perf_counter() - start),
                "shared_compute_proxy": float(shared_init_compute),
                "shared_wall_clock_sec": float(shared_init_time),
                **metrics,
            }
        )

        train_features = full_train_features[idx]
        head_rows, _, head_budget, _ = train_linear_head_anytime(
            train_features,
            bundle.ytr,
            test_features,
            bundle.yte,
            bundle.task_type,
            bundle.num_classes,
            seed + int(round(fraction * 1000)) + 17,
            [1.0],
            total_compute_offset=0.0,
            wall_clock_offset=0.0,
            scenario_name=scenario.name,
            dataset_name=bundle.name,
            architecture="mlp",
            data_fraction=fraction,
            model_name="closed-form+ce-head",
            hidden_param_count=state["activation_param_count"],
        )
        ce_row = dict(head_rows[-1])
        ce_row.update(
            {
                "seed": seed,
                "transfer_mode": "shared-init",
                "downstream_compute_proxy": float(head_budget),
                "downstream_wall_clock_sec": float(ce_row["wall_clock_sec"]),
                "shared_compute_proxy": float(shared_init_compute + shared_feature_compute),
                "shared_wall_clock_sec": float(shared_init_time + shared_feature_time),
            }
        )
        results.append(ce_row)
    return results


def run_shared_init_transfer_transformer(scenario, full_bundle, seed, shared_ft_artifact, shared_ce_artifact):
    config = scenario.transformer_config
    state = shared_ft_artifact.shared_state
    results = []
    shared_init_compute = shared_ft_artifact.init_compute_proxy
    shared_init_time = shared_ft_artifact.init_time_sec
    shared_feature_compute = shared_ce_artifact.shared_feature_compute_proxy
    shared_feature_time = shared_ce_artifact.shared_feature_time_sec
    full_train_features = shared_ce_artifact.shared_feature_cache["train_features"]
    test_features = shared_ce_artifact.shared_feature_cache["test_features"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fraction in TRANSFER_FRACTIONS:
        idx = subset_indices(len(full_bundle.ytr), fraction, seed + int(round(fraction * 1000)) + 9100)
        bundle = subset_bundle(full_bundle, idx)
        train_ds, test_raw, token_dim, _, to_tokens = bes.prepare_transformer_training_context(bundle, config, device)
        loader = bes.make_train_loader(train_ds, batch_size=config.batch_size, shuffle=True)
        model = bes.ClosedFormFineTuneTransformer(token_dim, bundle.num_classes, state, task_type=bundle.task_type).to(device)
        reset_transformer_heads(model, seed + int(round(fraction * 1000)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        criterion = nn.CrossEntropyLoss()
        total_budget = bes.estimate_transformer_compute_proxy(bundle, config, "backprop")
        subset_init_budget = estimate_encoder_init_transformer_compute(bundle, config)
        finetune_epoch_budget = bes.estimate_transformer_compute_proxy(bundle, config, bes.FINE_TUNE_MODEL_NAME, TRANSFORMER_METHOD)
        steps, _ = bes.compute_matched_step_budget(total_budget, subset_init_budget, finetune_epoch_budget, len(loader))
        step_budget = 0.0 if len(loader) == 0 else finetune_epoch_budget / len(loader)
        iterator = iter(loader)
        start = time.perf_counter()
        for _ in range(steps):
            try:
                xb, yb = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                xb, yb = next(iterator)
            depth_logits = model(to_tokens(xb, augment=True))
            loss = sum(criterion(logits, yb.to(device)) for logits in depth_logits) / max(len(depth_logits), 1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits_te = model(to_tokens(test_raw, augment=False))[-1].cpu().numpy()
        metrics = evaluate_task_metrics(logits_te, bundle.yte, bundle.task_type)
        results.append(
            {
                "scenario": scenario.name,
                "dataset": bundle.name,
                "architecture": "transformer",
                "model": bes.FINE_TUNE_MODEL_NAME,
                "seed": seed,
                "data_fraction": fraction,
                "transfer_mode": "shared-init",
                "downstream_compute_proxy": float(steps * step_budget),
                "downstream_wall_clock_sec": float(time.perf_counter() - start),
                "shared_compute_proxy": float(shared_init_compute),
                "shared_wall_clock_sec": float(shared_init_time),
                **metrics,
            }
        )

        train_features = full_train_features[idx]
        head_rows, _, head_budget, _ = train_linear_head_anytime(
            train_features,
            bundle.ytr,
            test_features,
            bundle.yte,
            bundle.task_type,
            bundle.num_classes,
            seed + int(round(fraction * 1000)) + 23,
            [1.0],
            total_compute_offset=0.0,
            wall_clock_offset=0.0,
            scenario_name=scenario.name,
            dataset_name=bundle.name,
            architecture="transformer",
            data_fraction=fraction,
            model_name="closed-form+ce-head",
            hidden_param_count=state["hidden_param_count"],
        )
        ce_row = dict(head_rows[-1])
        ce_row.update(
            {
                "seed": seed,
                "transfer_mode": "shared-init",
                "downstream_compute_proxy": float(head_budget),
                "downstream_wall_clock_sec": float(ce_row["wall_clock_sec"]),
                "shared_compute_proxy": float(shared_init_compute + shared_feature_compute),
                "shared_wall_clock_sec": float(shared_init_time + shared_feature_time),
            }
        )
        results.append(ce_row)
    return results


def aggregate_rows(rows, group_keys, metric_keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    out = []
    for key, items in grouped.items():
        record = {group_keys[idx]: key[idx] for idx in range(len(group_keys))}
        for metric in metric_keys:
            vals = [float(item[metric]) for item in items if item.get(metric) is not None]
            if not vals:
                continue
            mean, std, ci = mean_std_ci(vals)
            record[f"mean_{metric}"] = mean
            record[f"std_{metric}"] = std
            record[f"ci_{metric}"] = ci
        record["count"] = len(items)
        out.append(record)
    return out


def pareto_frontier(points, x_key, task_type):
    maximize = task_type != "next_token"
    sorted_points = sorted(points, key=lambda row: row[x_key])
    frontier = []
    best_quality = -np.inf if maximize else np.inf
    for row in sorted_points:
        value = row["mean_primary_metric_value"]
        if maximize:
            if value > best_quality + 1e-12:
                frontier.append(row)
                best_quality = value
        else:
            if value < best_quality - 1e-12:
                frontier.append(row)
                best_quality = value
    return frontier


def compute_targets(anytime_rows, scenario_specs):
    targets = []
    for scenario in scenario_specs:
        backprop_final = [
            row
            for row in anytime_rows
            if row["scenario"] == scenario.name and row["model"] == "backprop" and row["data_fraction"] == 1.0 and row["checkpoint_label"] == "100%"
        ]
        if not backprop_final:
            continue
        values = [row["primary_metric_value"] for row in backprop_final]
        mean_value = float(np.mean(values))
        target_value = mean_value * (1.05 if scenario.dataset.task_type == "next_token" else 0.95)
        targets.append(
            {
                "scenario": scenario.name,
                "dataset": scenario.dataset.name,
                "task_type": scenario.dataset.task_type,
                "target_metric_name": primary_metric_name(scenario.dataset.task_type),
                "target_metric_value": float(target_value),
            }
        )
    return targets


def compute_to_target_rows(anytime_rows, targets):
    per_seed = []
    target_lookup = {item["scenario"]: item for item in targets}
    grouped = defaultdict(list)
    for row in anytime_rows:
        if row["data_fraction"] != 1.0:
            continue
        grouped[(row["scenario"], row["model"], row["seed"])].append(row)
    for (scenario, model, seed), items in grouped.items():
        target = target_lookup.get(scenario)
        if target is None:
            continue
        direction = quality_direction(target["task_type"])
        target_value = target["target_metric_value"]
        items = sorted(items, key=lambda row: row["total_compute_proxy"])
        reached = None
        for row in items:
            value = row["primary_metric_value"]
            if (direction == "max" and value >= target_value) or (direction == "min" and value <= target_value):
                reached = row
                break
        per_seed.append(
            {
                "scenario": scenario,
                "model": model,
                "seed": seed,
                "reached": reached is not None,
                "target_metric_name": target["target_metric_name"],
                "target_metric_value": float(target_value),
                "compute_to_target": None if reached is None else float(reached["total_compute_proxy"]),
                "wall_clock_to_target": None if reached is None else float(reached["wall_clock_sec"]),
                "budget_fraction_to_target": None if reached is None else float(reached["budget_fraction"]),
            }
        )
    summary = []
    grouped_summary = defaultdict(list)
    for row in per_seed:
        grouped_summary[(row["scenario"], row["model"])].append(row)
    for (scenario, model), items in grouped_summary.items():
        reached_items = [item for item in items if item["reached"]]
        entry = {
            "scenario": scenario,
            "model": model,
            "reach_rate": float(len(reached_items) / len(items)),
            "num_runs": len(items),
        }
        if reached_items:
            for key in ["compute_to_target", "wall_clock_to_target", "budget_fraction_to_target"]:
                mean, std, ci = mean_std_ci([item[key] for item in reached_items])
                entry[f"mean_{key}"] = mean
                entry[f"std_{key}"] = std
                entry[f"ci_{key}"] = ci
        summary.append(entry)
    return per_seed, summary


def matched_budget_summary(anytime_rows):
    final_rows = [row for row in anytime_rows if row["data_fraction"] == 1.0 and row["checkpoint_label"] == "100%"]
    return aggregate_rows(
        final_rows,
        ["scenario", "dataset", "architecture", "model"],
        ["primary_metric_value", "classifier_accuracy", "negative_log_likelihood", "expected_calibration_error", "validation_perplexity", "total_compute_proxy", "wall_clock_sec"],
    )


def seed_sensitivity_summary(anytime_rows):
    final_rows = [row for row in anytime_rows if row["data_fraction"] == 1.0 and row["checkpoint_label"] == "100%"]
    return aggregate_rows(final_rows, ["scenario", "model"], ["primary_metric_value"])


def low_data_summary(rows):
    filtered = [row for row in rows if row["checkpoint_label"] in {"10%", "100%"}]
    return aggregate_rows(
        filtered,
        ["scenario", "dataset", "architecture", "model", "data_fraction", "checkpoint_label"],
        ["primary_metric_value", "total_compute_proxy", "wall_clock_sec"],
    )


def ood_summary(rows):
    return aggregate_rows(rows, ["scenario", "model", "variant"], ["primary_metric_value", "delta_primary_metric"])


def transfer_summary(rows, scenario_specs):
    grouped = defaultdict(list)
    task_lookup = {scenario.name: scenario.dataset.task_type for scenario in scenario_specs}
    for row in rows:
        grouped[(row["scenario"], row["model"], row["seed"], row["transfer_mode"])].append(row)
    per_seed = []
    for (scenario, model, seed, transfer_mode), items in grouped.items():
        task_type = task_lookup[scenario]
        downstream_compute = sum(float(item["downstream_compute_proxy"]) for item in items)
        downstream_time = sum(float(item["downstream_wall_clock_sec"]) for item in items)
        shared_compute = float(items[0]["shared_compute_proxy"]) if items else 0.0
        shared_time = float(items[0]["shared_wall_clock_sec"]) if items else 0.0
        metric_values = [float(item["primary_metric_value"]) for item in items]
        mean_quality = float(np.mean(metric_values))
        per_seed.append(
            {
                "scenario": scenario,
                "model": model,
                "seed": seed,
                "transfer_mode": transfer_mode,
                "mean_primary_metric_value": mean_quality,
                "quality_value": -mean_quality if task_type == "next_token" else mean_quality,
                "total_compute_proxy": float(shared_compute + downstream_compute),
                "total_wall_clock_sec": float(shared_time + downstream_time),
            }
        )
    summary = aggregate_rows(per_seed, ["scenario", "model", "transfer_mode"], ["mean_primary_metric_value", "total_compute_proxy", "total_wall_clock_sec"])
    return per_seed, summary


def build_backprop_transfer_rows(scenario, seed, backprop_artifact, low_rows):
    rows = []
    full_row = backprop_artifact.final_row
    rows.append(
        {
            "scenario": scenario.name,
            "dataset": full_row["dataset"],
            "architecture": scenario.architecture,
            "model": "backprop",
            "seed": seed,
            "data_fraction": 1.0,
            "transfer_mode": "scratch",
            "downstream_compute_proxy": float(full_row["total_compute_proxy"]),
            "downstream_wall_clock_sec": float(full_row["wall_clock_sec"]),
            "shared_compute_proxy": 0.0,
            "shared_wall_clock_sec": 0.0,
            "primary_metric_name": full_row["primary_metric_name"],
            "primary_metric_value": float(full_row["primary_metric_value"]),
        }
    )
    for fraction in [frac for frac in TRANSFER_FRACTIONS if frac < 1.0]:
        subset_row = next(
            row
            for row in low_rows
            if row["scenario"] == scenario.name
            and row["model"] == "backprop"
            and row["seed"] == seed
            and row["data_fraction"] == fraction
            and row["checkpoint_label"] == "100%"
        )
        rows.append(
            {
                "scenario": scenario.name,
                "dataset": subset_row["dataset"],
                "architecture": scenario.architecture,
                "model": "backprop",
                "seed": seed,
                "data_fraction": fraction,
                "transfer_mode": "scratch",
                "downstream_compute_proxy": float(subset_row["total_compute_proxy"]),
                "downstream_wall_clock_sec": float(subset_row["wall_clock_sec"]),
                "shared_compute_proxy": 0.0,
                "shared_wall_clock_sec": 0.0,
                "primary_metric_name": subset_row["primary_metric_name"],
                "primary_metric_value": float(subset_row["primary_metric_value"]),
            }
        )
    return rows


def pareto_summary(anytime_rows, scenario_specs):
    aggregated = aggregate_rows(
        [row for row in anytime_rows if row["data_fraction"] == 1.0],
        ["scenario", "dataset", "architecture", "model", "checkpoint_label"],
        ["primary_metric_value", "total_compute_proxy", "wall_clock_sec"],
    )
    task_lookup = {scenario.name: scenario.dataset.task_type for scenario in scenario_specs}
    frontier_rows = []
    for scenario in scenario_specs:
        scenario_rows = [row for row in aggregated if row["scenario"] == scenario.name]
        for axis in ["mean_total_compute_proxy", "mean_wall_clock_sec"]:
            frontier = pareto_frontier(scenario_rows, axis, task_lookup[scenario.name])
            for row in frontier:
                frontier_rows.append({**row, "pareto_axis": axis})
    return aggregated, frontier_rows


def plot_anytime(anytime_summary, scenario_specs):
    plt = maybe_import_pyplot()
    if plt is None:
        return None
    fig, axes = plt.subplots(len(scenario_specs), 2, figsize=(13, 3.6 * len(scenario_specs)), squeeze=False)
    colors = {"backprop": "#1f77b4", bes.FINE_TUNE_MODEL_NAME: "#d62728", "closed-form+ce-head": "#2ca02c"}
    task_lookup = {scenario.name: scenario.dataset.task_type for scenario in scenario_specs}
    for row_idx, scenario in enumerate(scenario_specs):
        scenario_rows = [row for row in anytime_summary if row["scenario"] == scenario.name]
        for col_idx, axis_name in enumerate(["mean_total_compute_proxy", "mean_wall_clock_sec"]):
            ax = axes[row_idx, col_idx]
            for model in ["backprop", bes.FINE_TUNE_MODEL_NAME, "closed-form+ce-head"]:
                model_rows = [row for row in scenario_rows if row["model"] == model]
                if not model_rows:
                    continue
                model_rows.sort(key=lambda item: item[axis_name])
                ax.plot(
                    [row[axis_name] for row in model_rows],
                    [row["mean_primary_metric_value"] for row in model_rows],
                    marker="o",
                    linewidth=1.8,
                    color=colors[model],
                    label=model if row_idx == 0 and col_idx == 0 else None,
                )
            if task_lookup[scenario.name] == "next_token":
                ax.invert_yaxis()
            ax.set_title(f"{scenario.title} / {'Compute' if col_idx == 0 else 'Wall-clock'}")
            ax.set_xlabel("FLOPs proxy" if col_idx == 0 else "Wall-clock (s)")
            ax.set_ylabel(primary_metric_label(task_lookup[scenario.name]))
            ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = plot_path(f"{BENCHMARK_NAME}_anytime.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_pareto(pareto_rows, frontier_rows, scenario_specs):
    plt = maybe_import_pyplot()
    if plt is None:
        return None
    fig, axes = plt.subplots(len(scenario_specs), 2, figsize=(13, 3.6 * len(scenario_specs)), squeeze=False)
    colors = {"backprop": "#1f77b4", bes.FINE_TUNE_MODEL_NAME: "#d62728", "closed-form+ce-head": "#2ca02c"}
    task_lookup = {scenario.name: scenario.dataset.task_type for scenario in scenario_specs}
    for row_idx, scenario in enumerate(scenario_specs):
        scenario_rows = [row for row in pareto_rows if row["scenario"] == scenario.name]
        for col_idx, axis_name in enumerate(["mean_total_compute_proxy", "mean_wall_clock_sec"]):
            ax = axes[row_idx, col_idx]
            for model in ["backprop", bes.FINE_TUNE_MODEL_NAME, "closed-form+ce-head"]:
                model_rows = [row for row in scenario_rows if row["model"] == model]
                if not model_rows:
                    continue
                ax.scatter(
                    [row[axis_name] for row in model_rows],
                    [row["mean_primary_metric_value"] for row in model_rows],
                    color=colors[model],
                    s=32,
                    alpha=0.8,
                    label=model if row_idx == 0 and col_idx == 0 else None,
                )
                for point in model_rows:
                    ax.annotate(point["checkpoint_label"], (point[axis_name], point["mean_primary_metric_value"]), fontsize=7, alpha=0.8)
            frontier = [row for row in frontier_rows if row["scenario"] == scenario.name and row["pareto_axis"] == axis_name]
            frontier = sorted(frontier, key=lambda row: row[axis_name])
            if frontier:
                ax.plot([row[axis_name] for row in frontier], [row["mean_primary_metric_value"] for row in frontier], color="black", linewidth=1.5, alpha=0.8)
            if task_lookup[scenario.name] == "next_token":
                ax.invert_yaxis()
            ax.set_title(f"{scenario.title} Pareto / {'Compute' if col_idx == 0 else 'Wall-clock'}")
            ax.set_xlabel("FLOPs proxy" if col_idx == 0 else "Wall-clock (s)")
            ax.set_ylabel(primary_metric_label(task_lookup[scenario.name]))
            ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = plot_path(f"{BENCHMARK_NAME}_pareto.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_low_data(low_data_rows, scenario_specs):
    plt = maybe_import_pyplot()
    if plt is None:
        return None
    fig, axes = plt.subplots(len(scenario_specs), 2, figsize=(13, 3.4 * len(scenario_specs)), squeeze=False)
    colors = {"backprop": "#1f77b4", bes.FINE_TUNE_MODEL_NAME: "#d62728", "closed-form+ce-head": "#2ca02c"}
    task_lookup = {scenario.name: scenario.dataset.task_type for scenario in scenario_specs}
    for row_idx, scenario in enumerate(scenario_specs):
        scenario_rows = [row for row in low_data_rows if row["scenario"] == scenario.name]
        for col_idx, checkpoint in enumerate(["10%", "100%"]):
            ax = axes[row_idx, col_idx]
            for model in ["backprop", bes.FINE_TUNE_MODEL_NAME, "closed-form+ce-head"]:
                model_rows = [row for row in scenario_rows if row["model"] == model and row["checkpoint_label"] == checkpoint]
                model_rows.sort(key=lambda row: row["data_fraction"])
                if not model_rows:
                    continue
                ax.plot(
                    [row["data_fraction"] for row in model_rows],
                    [row["mean_primary_metric_value"] for row in model_rows],
                    marker="o",
                    color=colors[model],
                    label=model if row_idx == 0 and col_idx == 0 else None,
                )
            if task_lookup[scenario.name] == "next_token":
                ax.invert_yaxis()
            ax.set_xscale("log")
            ax.set_title(f"{scenario.title} / Budget {checkpoint}")
            ax.set_xlabel("Train-data fraction")
            ax.set_ylabel(primary_metric_label(task_lookup[scenario.name]))
            ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = plot_path(f"{BENCHMARK_NAME}_low_data.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_ood(ood_rows, scenario_specs):
    plt = maybe_import_pyplot()
    if plt is None:
        return None
    fig, axes = plt.subplots(len(scenario_specs), 1, figsize=(12, 3.2 * len(scenario_specs)), squeeze=False)
    colors = {"backprop": "#1f77b4", bes.FINE_TUNE_MODEL_NAME: "#d62728", "closed-form+ce-head": "#2ca02c"}
    task_lookup = {scenario.name: scenario.dataset.task_type for scenario in scenario_specs}
    for row_idx, scenario in enumerate(scenario_specs):
        ax = axes[row_idx, 0]
        scenario_rows = [row for row in ood_rows if row["scenario"] == scenario.name]
        variants = sorted({row["variant"] for row in scenario_rows})
        model_order = ["backprop", bes.FINE_TUNE_MODEL_NAME, "closed-form+ce-head"]
        width = 0.25
        x_positions = np.arange(len(variants))
        for model_idx, model in enumerate(model_order):
            model_rows = {row["variant"]: row for row in scenario_rows if row["model"] == model}
            vals = [model_rows[variant]["mean_primary_metric_value"] if variant in model_rows else np.nan for variant in variants]
            ax.bar(x_positions + (model_idx - 1) * width, vals, width=width, color=colors[model], label=model if row_idx == 0 else None)
        if task_lookup[scenario.name] == "next_token":
            ax.invert_yaxis()
        ax.set_title(f"{scenario.title} / OOD")
        ax.set_xticks(x_positions, variants, rotation=15)
        ax.set_ylabel(primary_metric_label(task_lookup[scenario.name]))
        ax.grid(alpha=0.25, axis="y")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = plot_path(f"{BENCHMARK_NAME}_ood.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_transfer(transfer_rows, scenario_specs):
    plt = maybe_import_pyplot()
    if plt is None:
        return None
    fig, axes = plt.subplots(len(scenario_specs), 2, figsize=(13, 3.2 * len(scenario_specs)), squeeze=False)
    colors = {"backprop": "#1f77b4", bes.FINE_TUNE_MODEL_NAME: "#d62728", "closed-form+ce-head": "#2ca02c"}
    task_lookup = {scenario.name: scenario.dataset.task_type for scenario in scenario_specs}
    for row_idx, scenario in enumerate(scenario_specs):
        scenario_rows = [row for row in transfer_rows if row["scenario"] == scenario.name]
        for col_idx, axis_name in enumerate(["mean_total_compute_proxy", "mean_total_wall_clock_sec"]):
            ax = axes[row_idx, col_idx]
            for model in ["backprop", bes.FINE_TUNE_MODEL_NAME, "closed-form+ce-head"]:
                row = next((item for item in scenario_rows if item["model"] == model), None)
                if row is None:
                    continue
                ax.scatter(row[axis_name], row["mean_mean_primary_metric_value"], color=colors[model], s=65)
                ax.annotate(model, (row[axis_name], row["mean_mean_primary_metric_value"]), fontsize=8)
            if task_lookup[scenario.name] == "next_token":
                ax.invert_yaxis()
            ax.set_title(f"{scenario.title} / Transfer {'Compute' if col_idx == 0 else 'Wall-clock'}")
            ax.set_xlabel("Total FLOPs proxy" if col_idx == 0 else "Total wall-clock (s)")
            ax.set_ylabel(f"Mean {primary_metric_label(task_lookup[scenario.name])}")
            ax.grid(alpha=0.25)
    fig.tight_layout()
    path = plot_path(f"{BENCHMARK_NAME}_transfer.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def run_benchmark(scenario_specs):
    anytime_rows = []
    ood_rows = []
    low_data_rows = []
    transfer_detail_rows = []
    for scenario in scenario_specs:
        bundle = bes.load_dataset_bundle(scenario.dataset)
        for seed in SEEDS:
            if scenario.architecture == "mlp":
                backprop_artifact = run_backprop_mlp_anytime(bundle, scenario.mlp_config, seed, scenario.name, 1.0, ANYTIME_BUDGET_FRACTIONS)
                init_state = fit_encoder_init_mlp_state(bundle, MLP_METHOD, scenario.mlp_config, seed)
                fine_tune_artifact = run_encoder_init_finetune_mlp_anytime(
                    bundle,
                    scenario.mlp_config,
                    seed,
                    scenario.name,
                    1.0,
                    ANYTIME_BUDGET_FRACTIONS,
                    state=init_state,
                )
                ce_head_artifact = run_encoder_init_ce_head_mlp_anytime(
                    bundle,
                    scenario.mlp_config,
                    seed,
                    scenario.name,
                    1.0,
                    ANYTIME_BUDGET_FRACTIONS,
                    state=init_state,
                )
                low_rows = low_data_subset_runs(scenario, bundle, seed)
                transfer_rows = run_shared_init_transfer_mlp(scenario, bundle, seed, fine_tune_artifact, ce_head_artifact)
            else:
                backprop_artifact = run_backprop_transformer_anytime(bundle, scenario.transformer_config, seed, scenario.name, 1.0, ANYTIME_BUDGET_FRACTIONS)
                init_state = fit_encoder_init_transformer_state(bundle, TRANSFORMER_METHOD, scenario.transformer_config, seed)
                fine_tune_artifact = run_encoder_init_finetune_transformer_anytime(
                    bundle,
                    scenario.transformer_config,
                    seed,
                    scenario.name,
                    1.0,
                    ANYTIME_BUDGET_FRACTIONS,
                    state=init_state,
                )
                ce_head_artifact = run_encoder_init_ce_head_transformer_anytime(
                    bundle,
                    scenario.transformer_config,
                    seed,
                    scenario.name,
                    1.0,
                    ANYTIME_BUDGET_FRACTIONS,
                    state=init_state,
                )
                low_rows = low_data_subset_runs(scenario, bundle, seed)
                transfer_rows = run_shared_init_transfer_transformer(scenario, bundle, seed, fine_tune_artifact, ce_head_artifact)

            artifacts = {
                "backprop": backprop_artifact,
                bes.FINE_TUNE_MODEL_NAME: fine_tune_artifact,
                "closed-form+ce-head": ce_head_artifact,
            }
            anytime_rows.extend(backprop_artifact.rows)
            anytime_rows.extend(fine_tune_artifact.rows)
            anytime_rows.extend(ce_head_artifact.rows)
            ood_rows.extend(run_ood_for_artifacts(scenario, bundle, seed, artifacts))
            low_data_rows.extend(low_rows)
            transfer_detail_rows.extend(build_backprop_transfer_rows(scenario, seed, backprop_artifact, low_rows))
            transfer_detail_rows.extend(transfer_rows)

    anytime_summary = aggregate_rows(
        [row for row in anytime_rows if row["data_fraction"] == 1.0],
        ["scenario", "dataset", "architecture", "model", "checkpoint_label"],
        ["primary_metric_value", "total_compute_proxy", "wall_clock_sec"],
    )
    targets = compute_targets(anytime_rows, scenario_specs)
    compute_to_target_detail, compute_to_target_summary = compute_to_target_rows(anytime_rows, targets)
    matched_summary = matched_budget_summary(anytime_rows)
    seed_summary = seed_sensitivity_summary(anytime_rows)
    low_data_agg = low_data_summary(low_data_rows + [row for row in anytime_rows if row["checkpoint_label"] in {"10%", "100%"} and row["data_fraction"] == 1.0])
    ood_agg = ood_summary(ood_rows)
    transfer_seed, transfer_agg = transfer_summary(transfer_detail_rows, scenario_specs)
    pareto_rows, pareto_front = pareto_summary(anytime_rows, scenario_specs)
    return {
        "anytime_rows": anytime_rows,
        "anytime_summary": anytime_summary,
        "targets": targets,
        "compute_to_target_detail": compute_to_target_detail,
        "compute_to_target_summary": compute_to_target_summary,
        "matched_summary": matched_summary,
        "seed_summary": seed_summary,
        "ood_rows": ood_rows,
        "ood_summary": ood_agg,
        "low_data_rows": low_data_rows,
        "low_data_summary": low_data_agg,
        "transfer_rows": transfer_detail_rows,
        "transfer_seed_summary": transfer_seed,
        "transfer_summary": transfer_agg,
        "pareto_rows": pareto_rows,
        "pareto_front": pareto_front,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=json_path(f"{BENCHMARK_NAME}.json"))
    args = parser.parse_args()

    start = time.perf_counter()
    scenarios = scenario_registry()
    payload = run_benchmark(scenarios)
    runtime_sec = float(time.perf_counter() - start)

    anytime_plot = plot_anytime(payload["anytime_summary"], scenarios)
    pareto_plot = plot_pareto(payload["pareto_rows"], payload["pareto_front"], scenarios)
    low_data_plot = plot_low_data(payload["low_data_summary"], scenarios)
    ood_plot = plot_ood(payload["ood_summary"], scenarios)
    transfer_plot = plot_transfer(payload["transfer_summary"], scenarios)

    payload["metadata"] = {
        "benchmark_name": BENCHMARK_NAME,
        "seeds": SEEDS,
        "anytime_budget_fractions": ANYTIME_BUDGET_FRACTIONS,
        "low_data_fractions": LOW_DATA_FRACTIONS,
        "low_data_budget_fractions": LOW_DATA_BUDGET_FRACTIONS,
        "ce_head_epochs": CE_HEAD_EPOCHS,
        "runtime_sec": runtime_sec,
        "scenarios": [
            {
                "name": scenario.name,
                "title": scenario.title,
                "architecture": scenario.architecture,
                "dataset": asdict(scenario.dataset),
                "mlp_config": None if scenario.mlp_config is None else asdict(scenario.mlp_config),
                "transformer_config": None if scenario.transformer_config is None else asdict(scenario.transformer_config),
            }
            for scenario in scenarios
        ],
        "plots": {
            "anytime": repo_relative_path(anytime_plot),
            "pareto": repo_relative_path(pareto_plot),
            "low_data": repo_relative_path(low_data_plot),
            "ood": repo_relative_path(ood_plot),
            "transfer": repo_relative_path(transfer_plot),
        },
    }

    output_path = write_json(args.output, payload)
    write_jsonl(json_path(f"{BENCHMARK_NAME}_anytime_rows.jsonl"), payload["anytime_rows"])
    write_jsonl(json_path(f"{BENCHMARK_NAME}_anytime_summary.jsonl"), payload["anytime_summary"])
    write_jsonl(json_path(f"{BENCHMARK_NAME}_compute_to_target_detail.jsonl"), payload["compute_to_target_detail"])
    write_jsonl(json_path(f"{BENCHMARK_NAME}_compute_to_target_summary.jsonl"), payload["compute_to_target_summary"])
    write_jsonl(json_path(f"{BENCHMARK_NAME}_matched_summary.jsonl"), payload["matched_summary"])
    write_jsonl(json_path(f"{BENCHMARK_NAME}_seed_summary.jsonl"), payload["seed_summary"])
    write_jsonl(json_path(f"{BENCHMARK_NAME}_ood_rows.jsonl"), payload["ood_rows"])
    write_jsonl(json_path(f"{BENCHMARK_NAME}_ood_summary.jsonl"), payload["ood_summary"])
    write_jsonl(json_path(f"{BENCHMARK_NAME}_low_data_rows.jsonl"), payload["low_data_rows"])
    write_jsonl(json_path(f"{BENCHMARK_NAME}_low_data_summary.jsonl"), payload["low_data_summary"])
    write_jsonl(json_path(f"{BENCHMARK_NAME}_transfer_rows.jsonl"), payload["transfer_rows"])
    write_jsonl(json_path(f"{BENCHMARK_NAME}_transfer_summary.jsonl"), payload["transfer_summary"])
    write_jsonl(json_path(f"{BENCHMARK_NAME}_pareto_rows.jsonl"), payload["pareto_rows"])
    write_jsonl(json_path(f"{BENCHMARK_NAME}_pareto_front.jsonl"), payload["pareto_front"])
    print(str(output_path))


if __name__ == "__main__":
    main()
