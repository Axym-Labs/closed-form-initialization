from itertools import combinations, product

import numpy as np
from scipy.linalg import eigh

import closed_form_barlow_twins as cfbt


def _symmetrize(matrix):
    return 0.5 * (matrix + matrix.T)


def _random_orthogonal_basis(dim, width, seed):
    rng = np.random.default_rng(seed)
    rank = min(width, dim)
    basis, _ = np.linalg.qr(rng.standard_normal((dim, rank)))
    return basis[:, :rank]


def softmax_rows(scores):
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(np.clip(shifted, -40.0, 40.0))
    denom = np.maximum(exp_scores.sum(axis=1, keepdims=True), 1e-12)
    return exp_scores / denom


def _softmax_last_axis(scores):
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(np.clip(shifted, -40.0, 40.0))
    denom = np.maximum(exp_scores.sum(axis=-1, keepdims=True), 1e-12)
    return exp_scores / denom


def infer_token_layout(vector_dim):
    side = int(round(np.sqrt(vector_dim / 3.0)))
    if 3 * side * side == vector_dim:
        return {
            "mode": "row-image",
            "vector_dim": vector_dim,
            "side": side,
            "num_tokens": side,
            "token_dim": 3 * side,
            "padded_dim": vector_dim,
        }

    for token_dim in (32, 24, 16, 8):
        if vector_dim % token_dim == 0 and vector_dim // token_dim >= 4:
            return {
                "mode": "chunk",
                "vector_dim": vector_dim,
                "num_tokens": vector_dim // token_dim,
                "token_dim": token_dim,
                "padded_dim": vector_dim,
            }

    token_dim = min(32, vector_dim)
    num_tokens = int(np.ceil(vector_dim / token_dim))
    padded_dim = num_tokens * token_dim
    return {
        "mode": "chunk",
        "vector_dim": vector_dim,
        "num_tokens": num_tokens,
        "token_dim": token_dim,
        "padded_dim": padded_dim,
    }


def vectors_to_tokens(X, layout):
    if layout["mode"] == "row-image":
        side = layout["side"]
        image = X.reshape(X.shape[0], 3, side, side)
        return np.transpose(image, (0, 2, 1, 3)).reshape(X.shape[0], side, 3 * side)

    padded = np.zeros((X.shape[0], layout["padded_dim"]), dtype=np.float64)
    padded[:, : layout["vector_dim"]] = X
    return padded.reshape(X.shape[0], layout["num_tokens"], layout["token_dim"])


def tokens_to_vectors(tokens, layout):
    if layout["mode"] == "row-image":
        side = layout["side"]
        image = tokens.reshape(tokens.shape[0], side, 3, side)
        image = np.transpose(image, (0, 2, 1, 3))
        return image.reshape(tokens.shape[0], layout["vector_dim"])

    flat = tokens.reshape(tokens.shape[0], layout["padded_dim"])
    return flat[:, : layout["vector_dim"]]


def _flatten_tokens(tokens):
    return tokens.reshape(-1, tokens.shape[-1])


def _axis_tokens_from_vectors(X, side, axis):
    image = X.reshape(X.shape[0], 3, side, side)
    if axis == "row":
        return np.transpose(image, (0, 2, 1, 3)).reshape(X.shape[0], side, 3 * side)
    if axis == "col":
        return np.transpose(image, (0, 3, 1, 2)).reshape(X.shape[0], side, 3 * side)
    raise ValueError(f"Unknown axis: {axis}")


def _vectors_from_axis_tokens(tokens, side, axis):
    if axis == "row":
        image = tokens.reshape(tokens.shape[0], side, 3, side)
        image = np.transpose(image, (0, 2, 1, 3))
        return image.reshape(tokens.shape[0], 3 * side * side)
    if axis == "col":
        image = tokens.reshape(tokens.shape[0], side, 3, side)
        image = np.transpose(image, (0, 2, 3, 1))
        return image.reshape(tokens.shape[0], 3 * side * side)
    raise ValueError(f"Unknown axis: {axis}")


def _token_attention_weights(tokens, sigma_inv_sqrt, keys, temperature):
    whitened = tokens @ sigma_inv_sqrt
    flat = _flatten_tokens(whitened)
    scores = temperature * (flat @ keys)
    weights = softmax_rows(scores)
    return weights.reshape(tokens.shape[0], tokens.shape[1], keys.shape[1]), whitened


def _global_attention_weights(X, sigma_inv_sqrt, keys, temperature):
    whitened = X @ sigma_inv_sqrt
    scores = temperature * (whitened @ keys)
    return softmax_rows(scores), whitened


def _normalized_features(X):
    norms = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    return X / norms


def _normalize_last_axis(tensor):
    norms = np.maximum(np.linalg.norm(tensor, axis=-1, keepdims=True), 1e-8)
    return tensor / norms


def _fit_residual_scale(pred1, pred2, target1, target2):
    numerator = np.sum(pred1 * target1) + np.sum(pred2 * target2)
    denominator = np.sum(pred1 * pred1) + np.sum(pred2 * pred2)
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _fit_target_blend(input1, input2, pred1, pred2, target1, target2):
    diff1 = input1 - pred1
    diff2 = input2 - pred2
    rhs1 = target1 - pred1
    rhs2 = target2 - pred2
    numerator = np.sum(diff1 * rhs1) + np.sum(diff2 * rhs2)
    denominator = np.sum(diff1 * diff1) + np.sum(diff2 * diff2)
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _fit_mean_blend(input1, input2, pred1, pred2, target1, target2):
    return _fit_target_blend(input1, input2, pred1, pred2, target1, target2)


def _default_projection_rank(token_dim, num_heads):
    return min(token_dim, 8 * max(1, int(num_heads)))


def _projection_head_indices(total_rank, max_rank, num_heads):
    clipped_rank = int(max(1, min(total_rank, max_rank)))
    head_count = int(max(1, min(num_heads, clipped_rank)))
    return [idxs for idxs in np.array_split(np.arange(clipped_rank), head_count) if len(idxs)]


def _projection_head_indices_interleaved(total_rank, max_rank, num_heads):
    clipped_rank = int(max(1, min(total_rank, max_rank)))
    head_count = int(max(1, min(num_heads, clipped_rank)))
    buckets = [[] for _ in range(head_count)]
    for idx in range(clipped_rank):
        buckets[idx % head_count].append(idx)
    return [np.asarray(bucket, dtype=np.int64) for bucket in buckets if bucket]


def _split_projection_heads(eigvecs, total_rank, num_heads):
    return [eigvecs[:, idxs] for idxs in _projection_head_indices(total_rank, eigvecs.shape[1], num_heads)]


def _split_random_projection_heads(token_dim, total_rank, num_heads, seed):
    projection_rank = int(max(1, min(total_rank, token_dim)))
    basis = _random_orthogonal_basis(token_dim, projection_rank, seed)
    head_indices = _projection_head_indices(projection_rank, basis.shape[1], num_heads)
    return [basis[:, idxs] for idxs in head_indices]


def _relative_position_bias(num_tokens, sigma):
    if sigma is None:
        return None

    side = int(round(np.sqrt(num_tokens)))
    if side * side == num_tokens:
        coords = np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="ij"), axis=-1).reshape(num_tokens, 2)
    else:
        coords = np.stack([np.arange(num_tokens), np.zeros(num_tokens, dtype=np.int64)], axis=1)

    deltas = coords[:, None, :] - coords[None, :, :]
    dist_sq = np.sum(deltas * deltas, axis=-1, dtype=np.float64)
    sigma_sq = max(float(sigma) ** 2, 1e-8)
    return -0.5 * dist_sq / sigma_sq


def _build_spectral_head_layout(attention_kind, head_count):
    if attention_kind in {
        "spectral-self",
        "spectral-self-token-stats",
        "spectral-self-token-centered",
        "score-self-power",
        "score-self-power-raw",
        "mixed-self-objective",
        "token-self-maxent",
        "score-self-power-deflated-gain",
        "score-self-cosine-gain",
        "score-self-power-bagged-gain",
        "score-self-power-bagged-shrink-gain",
        "score-self-power-bagged-consensus-gain",
    }:
        return [{"projection_index": head_idx, "bias_kind": "global"} for head_idx in range(head_count)]
    if attention_kind == "local-spectral":
        return [{"projection_index": head_idx, "bias_kind": "local"} for head_idx in range(head_count)]
    if attention_kind in {"hybrid-spectral", "hybrid-spectral-bt"}:
        if head_count == 1:
            return [
                {"projection_index": 0, "bias_kind": "global"},
                {"projection_index": 0, "bias_kind": "local"},
            ]
        global_heads = max(1, head_count // 2)
        return [
            *({"projection_index": head_idx, "bias_kind": "global"} for head_idx in range(global_heads)),
            *({"projection_index": head_idx, "bias_kind": "local"} for head_idx in range(global_heads, head_count)),
        ]
    raise ValueError(f"Unknown spectral attention kind: {attention_kind}")


def _build_token_targets(tokens1, tokens2, lambda_reg, target_mode):
    if target_mode == "mean":
        target = 0.5 * (tokens1 + tokens2)
        return target, target, False, {}
    if target_mode == "mean-centered":
        target = 0.5 * (tokens1 + tokens2)
        target = target - target.mean(axis=1, keepdims=True)
        return target, target, False, {}
    if target_mode == "residual":
        return 0.5 * (tokens2 - tokens1), 0.5 * (tokens1 - tokens2), True, {}
    if target_mode == "residual-centered":
        target1 = 0.5 * (tokens2 - tokens1)
        target2 = 0.5 * (tokens1 - tokens2)
        target1 = target1 - target1.mean(axis=1, keepdims=True)
        target2 = target2 - target2.mean(axis=1, keepdims=True)
        return target1, target2, True, {}
    if target_mode == "cross":
        return tokens2, tokens1, False, {}
    if target_mode in {"bt", "bt-residual"}:
        flat1 = _flatten_tokens(tokens1)
        flat2 = _flatten_tokens(tokens2)
        teacher_model = cfbt.fit_layer(flat1, flat2, lambda_reg=lambda_reg)
        teacher1 = (flat1 @ teacher_model["transform_base"]).reshape(tokens1.shape)
        teacher2 = (flat2 @ teacher_model["transform_base"]).reshape(tokens2.shape)
        teacher_stats = {
            "teacher_transform_fro": teacher_model["transform_fro"],
            "teacher_distance_to_identity": teacher_model["distance_to_identity"],
            "teacher_max_whitened_delta": teacher_model["max_whitened_delta"],
        }
        if target_mode == "bt":
            return teacher1, teacher2, False, teacher_stats
        return teacher1 - tokens1, teacher2 - tokens2, True, teacher_stats
    raise ValueError(f"Unknown token attention target_mode: {target_mode}")


def _solve_context_output(context1, context2, input1, input2, lambda_reg, target_mode):
    target1, target2, residual_mode, target_stats = _build_token_targets(input1, input2, lambda_reg, target_mode)

    flat_context1 = _flatten_tokens(context1)
    flat_context2 = _flatten_tokens(context2)
    design = np.concatenate([flat_context1, flat_context2], axis=0)
    targets = np.concatenate([_flatten_tokens(target1), _flatten_tokens(target2)], axis=0)
    gram = design.T @ design
    rhs = design.T @ targets
    output_map = np.linalg.solve(
        gram + lambda_reg * np.eye(gram.shape[0], dtype=np.float64),
        rhs,
    )

    pred1 = (flat_context1 @ output_map).reshape(input1.shape)
    pred2 = (flat_context2 @ output_map).reshape(input2.shape)
    if residual_mode:
        mix_scale = _fit_residual_scale(pred1, pred2, target1, target2)
    else:
        mix_scale = _fit_target_blend(input1, input2, pred1, pred2, target1, target2)

    return {
        "output_map": output_map,
        "mix_scale": mix_scale,
        "residual_mode": residual_mode,
        "target_stats": target_stats,
    }


def _attention_solution_loss(context1, context2, input1, input2, solved, lambda_reg, target_mode):
    target1, target2, residual_mode, _ = _build_token_targets(input1, input2, lambda_reg, target_mode)
    pred1 = (_flatten_tokens(context1) @ solved["output_map"]).reshape(input1.shape)
    pred2 = (_flatten_tokens(context2) @ solved["output_map"]).reshape(input2.shape)
    if residual_mode:
        err1 = solved["mix_scale"] * pred1 - target1
        err2 = solved["mix_scale"] * pred2 - target2
    else:
        alpha = solved["mix_scale"]
        out1 = alpha * input1 + (1.0 - alpha) * pred1
        out2 = alpha * input2 + (1.0 - alpha) * pred2
        err1 = out1 - target1
        err2 = out2 - target2
    return float(np.mean(err1 * err1) + np.mean(err2 * err2))


def _spectral_head_contexts(
    query_tokens,
    key_tokens,
    value_tokens,
    sigma_inv_sqrt,
    projection_heads,
    head_layout,
    local_bias=None,
    center_values=False,
    whiten_values=False,
    query_score_tokens=None,
    key_score_tokens=None,
):
    query_score_map = _resolve_score_token_map(query_tokens, head_layout, query_score_tokens)
    key_score_map = _resolve_score_token_map(key_tokens, head_layout, key_score_tokens)
    whitened_q_cache = {}
    whitened_k_cache = {}
    values = value_tokens @ sigma_inv_sqrt if whiten_values else value_tokens
    if center_values:
        values = values - values.mean(axis=1, keepdims=True)
    contexts = []
    for head_spec in head_layout:
        score_mode = head_spec.get("score_mode", "raw")
        projection = projection_heads[head_spec["projection_index"]]
        if score_mode not in whitened_q_cache:
            whitened_q_cache[score_mode] = query_score_map[score_mode] @ sigma_inv_sqrt
        if score_mode not in whitened_k_cache:
            whitened_k_cache[score_mode] = key_score_map[score_mode] @ sigma_inv_sqrt
        queries = _normalize_last_axis(whitened_q_cache[score_mode] @ projection)
        keys = _normalize_last_axis(whitened_k_cache[score_mode] @ projection)
        score_scale = float(head_spec.get("score_scale", 1.0))
        scores = score_scale * np.matmul(queries, np.swapaxes(keys, 1, 2)) / np.sqrt(max(projection.shape[1], 1))
        if (
            head_spec["bias_kind"] == "local"
            and local_bias is not None
            and query_tokens.shape[1] == key_tokens.shape[1]
        ):
            scores = scores + local_bias[None, :, :]
        weights = _softmax_last_axis(scores)
        contexts.append(np.matmul(weights, values))
    return contexts


def _aggregate_head_contexts(contexts, head_weights=None):
    if len(contexts) == 1:
        return contexts[0]

    stacked = np.stack(contexts, axis=0)
    if head_weights is None:
        return stacked.mean(axis=0)

    weights = np.asarray(head_weights, dtype=np.float64)
    weights = weights / np.maximum(weights.sum(), 1e-12)
    return np.tensordot(weights, stacked, axes=(0, 0))


def _asymmetric_head_contexts(query_tokens, key_tokens, value_tokens, query_heads, key_heads, center_values=False):
    values = value_tokens - value_tokens.mean(axis=1, keepdims=True) if center_values else value_tokens
    contexts = []
    for query_projection, key_projection in zip(query_heads, key_heads):
        queries = _normalize_last_axis(query_tokens @ query_projection)
        keys = _normalize_last_axis(key_tokens @ key_projection)
        scores = np.matmul(queries, np.swapaxes(keys, 1, 2)) / np.sqrt(max(query_projection.shape[1], 1))
        weights = _softmax_last_axis(scores)
        contexts.append(np.matmul(weights, values))
    return contexts


def _center_tokens_within_sample(tokens):
    return tokens - tokens.mean(axis=1, keepdims=True)


def _prepare_attention_score_tokens(tokens, score_mode):
    if score_mode in {None, "raw"}:
        return tokens
    if score_mode == "token-centered":
        return _center_tokens_within_sample(tokens)
    raise ValueError(f"Unknown attention score_mode: {score_mode}")


def _resolve_score_token_map(tokens, head_layout, score_tokens):
    if isinstance(score_tokens, dict):
        return score_tokens
    if score_tokens is not None:
        score_modes = {head_spec.get("score_mode", "raw") for head_spec in head_layout}
        if len(score_modes) == 1:
            return {next(iter(score_modes)): score_tokens}
        return {"raw": score_tokens}

    score_modes = {head_spec.get("score_mode", "raw") for head_spec in head_layout}
    return {mode: _prepare_attention_score_tokens(tokens, mode) for mode in score_modes}


def _spectral_attention_context(
    tokens,
    sigma_inv_sqrt,
    projection_heads,
    head_layout,
    local_bias,
    center_values=False,
    whiten_values=False,
    score_tokens=None,
):
    score_token_map = _resolve_score_token_map(tokens, head_layout, score_tokens)
    return np.concatenate(
        _spectral_head_contexts(
            tokens,
            tokens,
            tokens,
            sigma_inv_sqrt,
            projection_heads,
            head_layout,
            local_bias=local_bias,
            center_values=center_values,
            whiten_values=whiten_values,
            query_score_tokens=score_token_map,
            key_score_tokens=score_token_map,
        ),
        axis=-1,
    )


def _operator_attention_context(
    tokens,
    sigma_inv_sqrt,
    shared_basis,
    score_operators,
    head_layout,
    center_values=False,
    whiten_values=False,
    score_tokens=None,
):
    score_token_map = _resolve_score_token_map(tokens, head_layout, score_tokens)
    projected_cache = {}
    values = tokens @ sigma_inv_sqrt if whiten_values else tokens
    if center_values:
        values = values - values.mean(axis=1, keepdims=True)

    contexts = []
    basis_dim = max(shared_basis.shape[1], 1)
    for head_spec in head_layout:
        score_mode = head_spec.get("score_mode", "raw")
        if score_mode not in projected_cache:
            whitened = score_token_map[score_mode] @ sigma_inv_sqrt
            projected_cache[score_mode] = _normalize_last_axis(whitened @ shared_basis)
        projected = projected_cache[score_mode]
        operator = score_operators[head_spec["operator_index"]]
        score_scale = float(head_spec.get("score_scale", 1.0))
        scores = score_scale * np.einsum(
            "nti,ij,nsj->nts",
            projected,
            operator,
            projected,
            optimize=True,
        ) / np.sqrt(basis_dim)
        weights = _softmax_last_axis(scores)
        contexts.append(np.matmul(weights, values))

    return np.concatenate(contexts, axis=-1)


def _fit_self_attention_with_projections(
    tokens1,
    tokens2,
    lambda_reg,
    sigma_sqrt,
    sigma_inv_sqrt,
    projection_heads,
    head_layout,
    attention_kind,
    target_mode,
    local_bias=None,
    local_sigma=None,
    center_values=False,
    whiten_values=False,
    score_mode="raw",
    extra_stats=None,
):
    model, _ = _build_self_attention_model(
        tokens1=tokens1,
        tokens2=tokens2,
        lambda_reg=lambda_reg,
        sigma_sqrt=sigma_sqrt,
        sigma_inv_sqrt=sigma_inv_sqrt,
        projection_heads=projection_heads,
        head_layout=head_layout,
        attention_kind=attention_kind,
        target_mode=target_mode,
        local_bias=local_bias,
        local_sigma=local_sigma,
        center_values=center_values,
        whiten_values=whiten_values,
        score_mode=score_mode,
        extra_stats=extra_stats,
    )
    return model


def _build_self_attention_model(
    tokens1,
    tokens2,
    lambda_reg,
    sigma_sqrt,
    sigma_inv_sqrt,
    projection_heads,
    head_layout,
    attention_kind,
    target_mode,
    local_bias=None,
    local_sigma=None,
    center_values=False,
    whiten_values=False,
    score_mode="raw",
    extra_stats=None,
):
    score_tokens1 = _prepare_attention_score_tokens(tokens1, score_mode) if score_mode != "mixed" else None
    score_tokens2 = _prepare_attention_score_tokens(tokens2, score_mode) if score_mode != "mixed" else None
    context1 = _spectral_attention_context(
        tokens1,
        sigma_inv_sqrt,
        projection_heads,
        head_layout,
        local_bias,
        center_values=center_values,
        whiten_values=whiten_values,
        score_tokens=score_tokens1,
    )
    context2 = _spectral_attention_context(
        tokens2,
        sigma_inv_sqrt,
        projection_heads,
        head_layout,
        local_bias,
        center_values=center_values,
        whiten_values=whiten_values,
        score_tokens=score_tokens2,
    )
    solved = _solve_context_output(
        context1=context1,
        context2=context2,
        input1=tokens1,
        input2=tokens2,
        lambda_reg=lambda_reg,
        target_mode=target_mode,
    )
    loss = _attention_solution_loss(context1, context2, tokens1, tokens2, solved, lambda_reg, target_mode)

    total_projection_rank = int(sum(head.shape[1] for head in projection_heads))
    model = {
        "attention_kind": attention_kind,
        "sigma_inv_sqrt": sigma_inv_sqrt,
        "sigma_sqrt": sigma_sqrt,
        "projection_heads": projection_heads,
        "head_layout": head_layout,
        "local_bias": local_bias,
        "local_sigma": local_sigma if local_bias is not None else None,
        "center_values": center_values,
        "whiten_values": whiten_values,
        "score_mode": score_mode,
        "output_map": solved["output_map"],
        "mix_scale": solved["mix_scale"],
        "residual_mode": solved["residual_mode"],
        "target_mode": target_mode,
        "projection_rank": total_projection_rank,
        "num_heads": len(head_layout),
        "teacher_stats": solved["target_stats"],
        "parameter_count": int(
            sigma_inv_sqrt.size
            + sum(head.size for head in projection_heads)
            + solved["output_map"].size
        ),
    }
    if extra_stats:
        model.update(extra_stats)
    return model, loss


def _fit_self_attention_with_scale_search(
    tokens1,
    tokens2,
    lambda_reg,
    sigma_sqrt,
    sigma_inv_sqrt,
    projection_heads,
    head_layout,
    attention_kind,
    target_mode,
    scale_candidates,
    scale_group_key=None,
    local_bias=None,
    local_sigma=None,
    center_values=False,
    whiten_values=False,
    score_mode="raw",
    extra_stats=None,
):
    groups = ["all"] if scale_group_key is None else sorted({head.get(scale_group_key, "all") for head in head_layout})
    best_model = None
    best_loss = None
    best_scale_map = None
    for combo in product(scale_candidates, repeat=len(groups)):
        scale_map = dict(zip(groups, combo))
        scaled_head_layout = []
        for head in head_layout:
            group = "all" if scale_group_key is None else head.get(scale_group_key, "all")
            scaled_head_layout.append({**head, "score_scale": float(scale_map[group])})
        model, loss = _build_self_attention_model(
            tokens1=tokens1,
            tokens2=tokens2,
            lambda_reg=lambda_reg,
            sigma_sqrt=sigma_sqrt,
            sigma_inv_sqrt=sigma_inv_sqrt,
            projection_heads=projection_heads,
            head_layout=scaled_head_layout,
            attention_kind=attention_kind,
            target_mode=target_mode,
            local_bias=local_bias,
            local_sigma=local_sigma,
            center_values=center_values,
            whiten_values=whiten_values,
            score_mode=score_mode,
            extra_stats=extra_stats,
        )
        if best_loss is None or loss < best_loss:
            best_model = model
            best_loss = loss
            best_scale_map = scale_map
    best_model["selected_score_scales"] = {key: float(val) for key, val in best_scale_map.items()}
    best_model["train_fit_loss"] = float(best_loss)
    return best_model


def _orthogonalize_vector(vector, basis):
    if not basis:
        return vector

    q = np.column_stack(basis)
    return vector - q @ (q.T @ vector)


def _orthogonalize_matrix(matrix, basis):
    if not basis:
        return matrix

    out = matrix
    for base in basis:
        out = out - float(np.sum(out * base)) * base
    return out


def _random_symmetric_matrix(dim, seed):
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((dim, dim))
    return _symmetrize(matrix)


def _flatten_operator_basis(operators):
    if not operators:
        raise ValueError("Need at least one operator to flatten.")
    return np.column_stack([operator.reshape(-1) for operator in operators])


def _unflatten_operator_basis(flat_basis, matrix_dim):
    operators = []
    for idx in range(flat_basis.shape[1]):
        operator = _symmetrize(flat_basis[:, idx].reshape(matrix_dim, matrix_dim))
        operator = _orthogonalize_matrix(operator, operators)
        norm = np.linalg.norm(operator)
        if norm <= 1e-12:
            continue
        operators.append(operator / norm)
    return operators


def _score_alignment_power_basis(
    score_tokens1,
    score_tokens2,
    sigma_inv_sqrt,
    rank,
    num_iters=8,
    seed=0,
    init_basis=None,
):
    z1 = score_tokens1 @ sigma_inv_sqrt
    z2 = score_tokens2 @ sigma_inv_sqrt
    token_count = max(z1.shape[1], 1)
    dim = z1.shape[-1]

    if init_basis is None or init_basis.size == 0:
        init_basis = _random_orthogonal_basis(dim, rank, seed)

    directions = []
    objective_values = []
    for comp in range(int(max(1, rank))):
        if comp < init_basis.shape[1]:
            direction = init_basis[:, comp].astype(np.float64, copy=True)
        else:
            direction = _random_orthogonal_basis(dim, 1, seed + comp)[:, 0]
        direction = _orthogonalize_vector(direction, directions)
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            direction = _random_orthogonal_basis(dim, 1, seed + 101 + comp)[:, 0]
            direction = _orthogonalize_vector(direction, directions)
            norm = np.linalg.norm(direction)
        direction = direction / max(norm, 1e-12)

        for _ in range(max(1, int(num_iters))):
            proj1 = np.einsum("ntd,d->nt", z1, direction, optimize=True)
            proj2 = np.einsum("ntd,d->nt", z2, direction, optimize=True)
            coeff = np.mean(proj1 * proj2, axis=1)
            gradient = (
                np.einsum("n,nt,ntd->d", coeff, proj2, z1, optimize=True)
                + np.einsum("n,nt,ntd->d", coeff, proj1, z2, optimize=True)
            ) / token_count
            gradient = _orthogonalize_vector(gradient, directions)
            grad_norm = np.linalg.norm(gradient)
            if grad_norm <= 1e-12:
                break
            updated = gradient / grad_norm
            if abs(np.dot(updated, direction)) >= 1.0 - 1e-7:
                direction = updated
                break
            direction = updated

        proj1 = np.einsum("ntd,d->nt", z1, direction, optimize=True)
        proj2 = np.einsum("ntd,d->nt", z2, direction, optimize=True)
        coeff = np.mean(proj1 * proj2, axis=1)
        directions.append(direction)
        objective_values.append(float(np.mean(coeff * coeff)))

    return np.column_stack(directions), np.asarray(objective_values, dtype=np.float64)


def _complete_basis_in_complement(dim, rank, seed, previous_basis=None, initial=None):
    pieces = []
    if initial is not None and initial.size:
        pieces.append(initial.astype(np.float64, copy=True))

    rng = np.random.default_rng(seed)
    while sum(piece.shape[1] for piece in pieces) < rank:
        pieces.append(rng.standard_normal((dim, max(rank, 2))))

    candidate = np.concatenate(pieces, axis=1)
    if previous_basis is not None and previous_basis.size:
        candidate = candidate - previous_basis @ (previous_basis.T @ candidate)
    q, _ = np.linalg.qr(candidate)
    if q.shape[1] < rank:
        raise ValueError("Could not build enough orthogonal directions in complement.")
    return q[:, :rank]


def _score_alignment_block_head_bases(
    score_tokens1,
    score_tokens2,
    sigma_inv_sqrt,
    head_dims,
    num_iters=8,
    seed=0,
    init_heads=None,
):
    z1 = score_tokens1 @ sigma_inv_sqrt
    z2 = score_tokens2 @ sigma_inv_sqrt
    token_count = max(z1.shape[1], 1)
    cross_moments = np.einsum("ntd,nte->nde", z1, z2, optimize=True) / token_count
    cross_moments = 0.5 * (cross_moments + np.swapaxes(cross_moments, 1, 2))
    dim = z1.shape[-1]

    previous_basis = np.zeros((dim, 0), dtype=np.float64)
    head_bases = []
    objective_values = []
    for head_idx, head_dim in enumerate(head_dims):
        init_basis = None if init_heads is None or head_idx >= len(init_heads) else init_heads[head_idx]
        basis = _complete_basis_in_complement(
            dim,
            int(max(1, head_dim)),
            seed + 37 * head_idx,
            previous_basis=previous_basis,
            initial=init_basis,
        )

        for _ in range(max(1, int(num_iters))):
            au = np.einsum("nij,jk->nik", cross_moments, basis, optimize=True)
            compressed = np.einsum("di,ndj->nij", basis, au, optimize=True)
            gradient = np.einsum("ndi,nij->dj", au, compressed, optimize=True) / max(cross_moments.shape[0], 1)
            basis = _complete_basis_in_complement(
                dim,
                basis.shape[1],
                seed + 97 * head_idx,
                previous_basis=previous_basis,
                initial=gradient,
            )

        au = np.einsum("nij,jk->nik", cross_moments, basis, optimize=True)
        compressed = np.einsum("di,ndj->nij", basis, au, optimize=True)
        objective_values.append(float(np.mean(np.sum(compressed * compressed, axis=(1, 2)))))
        head_bases.append(basis)
        previous_basis = np.concatenate([previous_basis, basis], axis=1)

    return head_bases, np.asarray(objective_values, dtype=np.float64)


def _score_operator_power_matrices(projected1, projected2, num_heads, num_iters=24, seed=0):
    cross_moments = np.einsum("nti,ntj->nij", projected1, projected2, optimize=True) / max(projected1.shape[1], 1)
    cross_moments_t = np.swapaxes(cross_moments, 1, 2)
    dim = projected1.shape[-1]

    def apply_operator(matrix):
        updated = 0.5 * (
            np.einsum("nij,jk,nlk->il", cross_moments, matrix, cross_moments, optimize=True)
            + np.einsum("nij,jk,nkl->il", cross_moments_t, matrix, cross_moments, optimize=True)
        ) / max(cross_moments.shape[0], 1)
        return _symmetrize(updated)

    operators = []
    objective_values = []
    for comp in range(int(max(1, num_heads))):
        if comp == 0:
            matrix = np.eye(dim, dtype=np.float64)
        elif comp < dim:
            matrix = np.zeros((dim, dim), dtype=np.float64)
            matrix[comp, comp] = 1.0
        else:
            matrix = _random_symmetric_matrix(dim, seed + comp)
        matrix = _orthogonalize_matrix(_symmetrize(matrix), operators)
        norm = np.linalg.norm(matrix)
        if norm <= 1e-12:
            matrix = _orthogonalize_matrix(_random_symmetric_matrix(dim, seed + 101 + comp), operators)
            norm = np.linalg.norm(matrix)
        matrix = matrix / max(norm, 1e-12)

        for _ in range(max(1, int(num_iters))):
            updated = _orthogonalize_matrix(apply_operator(matrix), operators)
            updated = _symmetrize(updated)
            updated_norm = np.linalg.norm(updated)
            if updated_norm <= 1e-12:
                break
            updated = updated / updated_norm
            if abs(float(np.sum(updated * matrix))) >= 1.0 - 1e-7:
                matrix = updated
                break
            matrix = updated

        operator_image = apply_operator(matrix)
        operators.append(matrix)
        objective_values.append(float(np.sum(matrix * operator_image)))

    return operators, np.asarray(objective_values, dtype=np.float64)


def _score_alignment_power_basis_deflated(
    score_tokens1,
    score_tokens2,
    sigma_inv_sqrt,
    rank,
    num_iters=8,
    seed=0,
    init_basis=None,
):
    z1 = (score_tokens1 @ sigma_inv_sqrt).copy()
    z2 = (score_tokens2 @ sigma_inv_sqrt).copy()
    dim = z1.shape[-1]

    if init_basis is None or init_basis.size == 0:
        init_basis = _random_orthogonal_basis(dim, rank, seed)

    directions = []
    objective_values = []
    for comp in range(int(max(1, rank))):
        if comp < init_basis.shape[1]:
            direction = init_basis[:, comp].astype(np.float64, copy=True)
        else:
            direction = _random_orthogonal_basis(dim, 1, seed + comp)[:, 0]
        norm = np.linalg.norm(direction)
        direction = direction / max(norm, 1e-12)

        for _ in range(max(1, int(num_iters))):
            proj1 = np.einsum("ntd,d->nt", z1, direction, optimize=True)
            proj2 = np.einsum("ntd,d->nt", z2, direction, optimize=True)
            coeff = np.mean(proj1 * proj2, axis=1)
            gradient = (
                np.einsum("n,nt,ntd->d", coeff, proj2, z1, optimize=True)
                + np.einsum("n,nt,ntd->d", coeff, proj1, z2, optimize=True)
            ) / max(z1.shape[1], 1)
            grad_norm = np.linalg.norm(gradient)
            if grad_norm <= 1e-12:
                break
            updated = gradient / grad_norm
            if abs(np.dot(updated, direction)) >= 1.0 - 1e-7:
                direction = updated
                break
            direction = updated

        proj1 = np.einsum("ntd,d->nt", z1, direction, optimize=True)
        proj2 = np.einsum("ntd,d->nt", z2, direction, optimize=True)
        directions.append(direction)
        objective_values.append(float(np.mean(np.mean(proj1 * proj2, axis=1) ** 2)))

        z1 = z1 - np.einsum("nt,d->ntd", proj1, direction, optimize=True)
        z2 = z2 - np.einsum("nt,d->ntd", proj2, direction, optimize=True)

    return np.column_stack(directions), np.asarray(objective_values, dtype=np.float64)


def _score_alignment_cosine_basis(
    score_tokens1,
    score_tokens2,
    sigma_inv_sqrt,
    rank,
    num_iters=8,
    seed=0,
    init_basis=None,
):
    z1 = score_tokens1 @ sigma_inv_sqrt
    z2 = score_tokens2 @ sigma_inv_sqrt
    dim = z1.shape[-1]

    if init_basis is None or init_basis.size == 0:
        init_basis = _random_orthogonal_basis(dim, rank, seed)

    directions = []
    objective_values = []
    eps = 1e-8
    for comp in range(int(max(1, rank))):
        if comp < init_basis.shape[1]:
            direction = init_basis[:, comp].astype(np.float64, copy=True)
        else:
            direction = _random_orthogonal_basis(dim, 1, seed + comp)[:, 0]
        direction = _orthogonalize_vector(direction, directions)
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            direction = _random_orthogonal_basis(dim, 1, seed + 211 + comp)[:, 0]
            direction = _orthogonalize_vector(direction, directions)
            norm = np.linalg.norm(direction)
        direction = direction / max(norm, 1e-12)

        for _ in range(max(1, int(num_iters))):
            proj1 = np.einsum("ntd,d->nt", z1, direction, optimize=True)
            proj2 = np.einsum("ntd,d->nt", z2, direction, optimize=True)
            norm1_sq = np.sum(proj1 * proj1, axis=1) + eps
            norm2_sq = np.sum(proj2 * proj2, axis=1) + eps
            inv_norm_prod = 1.0 / np.sqrt(norm1_sq * norm2_sq)
            cosine = np.sum(proj1 * proj2, axis=1) * inv_norm_prod

            grad_cross = (
                np.einsum("n,nt,ntd->d", inv_norm_prod, proj2, z1, optimize=True)
                + np.einsum("n,nt,ntd->d", inv_norm_prod, proj1, z2, optimize=True)
            )
            grad_self = (
                np.einsum("n,nt,ntd->d", cosine / norm1_sq, proj1, z1, optimize=True)
                + np.einsum("n,nt,ntd->d", cosine / norm2_sq, proj2, z2, optimize=True)
            )
            gradient = np.einsum("n,d->d", cosine, grad_cross - grad_self, optimize=True)
            gradient = _orthogonalize_vector(gradient, directions)
            grad_norm = np.linalg.norm(gradient)
            if grad_norm <= 1e-12:
                break
            updated = gradient / grad_norm
            if abs(np.dot(updated, direction)) >= 1.0 - 1e-7:
                direction = updated
                break
            direction = updated

        proj1 = np.einsum("ntd,d->nt", z1, direction, optimize=True)
        proj2 = np.einsum("ntd,d->nt", z2, direction, optimize=True)
        cosine = np.sum(proj1 * proj2, axis=1) / np.sqrt(
            (np.sum(proj1 * proj1, axis=1) + eps) * (np.sum(proj2 * proj2, axis=1) + eps)
        )
        directions.append(direction)
        objective_values.append(float(np.mean(cosine * cosine)))

    return np.column_stack(directions), np.asarray(objective_values, dtype=np.float64)


def _maxent_stable_basis(eigvecs, eigvals, rank, seed=0, pool_factor=4, flatten_exponent=0.25):
    pool_rank = int(max(rank, min(eigvecs.shape[1], pool_factor * max(1, rank))))
    pool_eigvecs = eigvecs[:, :pool_rank]
    pool_eigvals = np.maximum(eigvals[:pool_rank], 1e-8)
    flatten = (pool_eigvals / np.maximum(pool_eigvals[0], 1e-8)) ** (-float(flatten_exponent))
    mixed = pool_eigvecs @ (flatten[:, None] * _random_orthogonal_basis(pool_rank, rank, seed))
    basis, _ = np.linalg.qr(mixed)
    return basis[:, : min(rank, basis.shape[1])]


def _average_projector_basis(bases, rank, weights=None, prior_basis=None, prior_weight=0.0):
    if not bases:
        raise ValueError("Need at least one basis to average.")

    dim = bases[0].shape[0]
    projector = np.zeros((dim, dim), dtype=np.float64)
    if weights is None:
        weights = np.ones(len(bases), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.maximum(weights.sum(), 1e-12)

    for weight, basis in zip(weights, bases):
        q, _ = np.linalg.qr(basis)
        projector = projector + float(weight) * (q @ q.T)

    if prior_basis is not None and prior_weight > 0.0:
        q_prior, _ = np.linalg.qr(prior_basis)
        projector = projector + float(prior_weight) * (q_prior @ q_prior.T)

    projector = _symmetrize(projector)
    evals, evecs = eigh(projector)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    return evecs[:, : min(rank, evecs.shape[1])]


def _align_basis_to_reference(basis, reference_basis):
    if basis.shape[1] != reference_basis.shape[1]:
        raise ValueError("Basis and reference must have same width for alignment.")
    overlap = basis.T @ reference_basis
    u, _, vt = np.linalg.svd(overlap, full_matrices=False)
    return basis @ (u @ vt)


def _average_aligned_bases(bases, reference_basis, weights=None, prior_weight=0.0):
    if not bases:
        raise ValueError("Need at least one basis to average.")

    if weights is None:
        weights = np.ones(len(bases), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.maximum(weights.sum(), 1e-12)

    averaged = np.zeros_like(reference_basis, dtype=np.float64)
    alignment_scores = []
    for weight, basis in zip(weights, bases):
        aligned = _align_basis_to_reference(basis, reference_basis)
        averaged = averaged + float(weight) * aligned
        alignment_scores.append(float(np.mean(np.abs(np.sum(aligned * reference_basis, axis=0)))))

    if prior_weight > 0.0:
        averaged = averaged + float(prior_weight) * reference_basis

    q, _ = np.linalg.qr(averaged)
    q = q[:, : reference_basis.shape[1]]
    q = _align_basis_to_reference(q, reference_basis)
    return q, np.asarray(alignment_scores, dtype=np.float64)


def _normalized_projector_overlap(basis_a, basis_b):
    qa, _ = np.linalg.qr(basis_a)
    qb, _ = np.linalg.qr(basis_b)
    overlap = qa.T @ qb
    return float(np.sum(overlap * overlap) / max(min(qa.shape[1], qb.shape[1]), 1))


def _fit_spectral_attention_from_tokens(
    tokens1,
    tokens2,
    lambda_reg,
    attention_kind,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    local_sigma=1.5,
    center_values=False,
    head_split_mode="contiguous",
    whiten_values=False,
    score_mode="raw",
):
    stats_tokens1 = _prepare_attention_score_tokens(tokens1, score_mode)
    stats_tokens2 = _prepare_attention_score_tokens(tokens2, score_mode)
    flat1 = _flatten_tokens(stats_tokens1)
    flat2 = _flatten_tokens(stats_tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    if head_split_mode == "contiguous":
        head_indices = _projection_head_indices(projection_rank, eigvecs.shape[1], num_heads)
    elif head_split_mode == "interleaved":
        head_indices = _projection_head_indices_interleaved(projection_rank, eigvecs.shape[1], num_heads)
    else:
        raise ValueError(f"Unknown head_split_mode: {head_split_mode}")
    projection_heads = [eigvecs[:, idxs] for idxs in head_indices]
    head_layout = _build_spectral_head_layout(attention_kind, len(projection_heads))
    local_bias = None
    if any(head_spec["bias_kind"] == "local" for head_spec in head_layout):
        local_bias = _relative_position_bias(tokens1.shape[1], local_sigma)

    return _fit_self_attention_with_projections(
        tokens1=tokens1,
        tokens2=tokens2,
        lambda_reg=lambda_reg,
        sigma_sqrt=sigma_sqrt,
        sigma_inv_sqrt=sigma_inv_sqrt,
        projection_heads=projection_heads,
        head_layout=head_layout,
        attention_kind=attention_kind,
        target_mode=target_mode,
        local_bias=local_bias,
        local_sigma=local_sigma,
        center_values=center_values,
        whiten_values=whiten_values,
        score_mode=score_mode,
        extra_stats={
            "head_split_mode": head_split_mode,
            "shared_eigenvalues": eigvals[: int(sum(head.shape[1] for head in projection_heads))],
            "shared_trace": float(np.trace(stats["shared"])),
            "delta_trace": float(np.trace(stats["delta"])),
        },
    )


def _build_operator_attention_model(
    tokens1,
    tokens2,
    lambda_reg,
    sigma_sqrt,
    sigma_inv_sqrt,
    shared_basis,
    score_operators,
    head_layout,
    attention_kind,
    target_mode,
    center_values=False,
    whiten_values=False,
    score_mode="raw",
    extra_stats=None,
):
    score_tokens1 = _prepare_attention_score_tokens(tokens1, score_mode) if score_mode != "mixed" else None
    score_tokens2 = _prepare_attention_score_tokens(tokens2, score_mode) if score_mode != "mixed" else None
    context1 = _operator_attention_context(
        tokens1,
        sigma_inv_sqrt,
        shared_basis,
        score_operators,
        head_layout,
        center_values=center_values,
        whiten_values=whiten_values,
        score_tokens=score_tokens1,
    )
    context2 = _operator_attention_context(
        tokens2,
        sigma_inv_sqrt,
        shared_basis,
        score_operators,
        head_layout,
        center_values=center_values,
        whiten_values=whiten_values,
        score_tokens=score_tokens2,
    )
    solved = _solve_context_output(
        context1=context1,
        context2=context2,
        input1=tokens1,
        input2=tokens2,
        lambda_reg=lambda_reg,
        target_mode=target_mode,
    )
    loss = _attention_solution_loss(context1, context2, tokens1, tokens2, solved, lambda_reg, target_mode)

    model = {
        "attention_kind": attention_kind,
        "sigma_inv_sqrt": sigma_inv_sqrt,
        "sigma_sqrt": sigma_sqrt,
        "shared_basis": shared_basis,
        "score_operators": score_operators,
        "head_layout": head_layout,
        "center_values": center_values,
        "whiten_values": whiten_values,
        "score_mode": score_mode,
        "output_map": solved["output_map"],
        "mix_scale": solved["mix_scale"],
        "residual_mode": solved["residual_mode"],
        "target_mode": target_mode,
        "projection_rank": int(shared_basis.shape[1]),
        "num_heads": len(head_layout),
        "teacher_stats": solved["target_stats"],
        "parameter_count": int(
            sigma_inv_sqrt.size
            + shared_basis.size
            + sum(operator.size for operator in score_operators)
            + solved["output_map"].size
        ),
    }
    if extra_stats:
        model.update(extra_stats)
    return model, loss


def _fit_operator_attention_with_scale_search(
    tokens1,
    tokens2,
    lambda_reg,
    sigma_sqrt,
    sigma_inv_sqrt,
    shared_basis,
    score_operators,
    head_layout,
    attention_kind,
    target_mode,
    scale_candidates,
    scale_group_key=None,
    center_values=False,
    whiten_values=False,
    score_mode="raw",
    extra_stats=None,
):
    groups = ["all"] if scale_group_key is None else sorted({head.get(scale_group_key, "all") for head in head_layout})
    best_model = None
    best_loss = None
    best_scale_map = None
    for combo in product(scale_candidates, repeat=len(groups)):
        scale_map = dict(zip(groups, combo))
        scaled_head_layout = []
        for head in head_layout:
            group = "all" if scale_group_key is None else head.get(scale_group_key, "all")
            scaled_head_layout.append({**head, "score_scale": float(scale_map[group])})
        model, loss = _build_operator_attention_model(
            tokens1=tokens1,
            tokens2=tokens2,
            lambda_reg=lambda_reg,
            sigma_sqrt=sigma_sqrt,
            sigma_inv_sqrt=sigma_inv_sqrt,
            shared_basis=shared_basis,
            score_operators=score_operators,
            head_layout=scaled_head_layout,
            attention_kind=attention_kind,
            target_mode=target_mode,
            center_values=center_values,
            whiten_values=whiten_values,
            score_mode=score_mode,
            extra_stats=extra_stats,
        )
        if best_loss is None or loss < best_loss:
            best_model = model
            best_loss = loss
            best_scale_map = scale_map
    best_model["selected_score_scales"] = {key: float(val) for key, val in best_scale_map.items()}
    best_model["train_fit_loss"] = float(best_loss)
    return best_model


def _fit_score_operator_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=24,
    seed=0,
    basis_score_mode="token-centered",
    score_mode="raw",
    scale_candidates=None,
):
    basis_tokens1 = _prepare_attention_score_tokens(tokens1, basis_score_mode)
    basis_tokens2 = _prepare_attention_score_tokens(tokens2, basis_score_mode)
    flat1 = _flatten_tokens(basis_tokens1)
    flat2 = _flatten_tokens(basis_tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    projection_rank = int(max(1, min(projection_rank, eigvecs.shape[1])))
    shared_basis = eigvecs[:, :projection_rank]

    score_tokens1 = _prepare_attention_score_tokens(tokens1, score_mode)
    score_tokens2 = _prepare_attention_score_tokens(tokens2, score_mode)
    projected1 = _normalize_last_axis(score_tokens1 @ sigma_inv_sqrt @ shared_basis)
    projected2 = _normalize_last_axis(score_tokens2 @ sigma_inv_sqrt @ shared_basis)
    score_operators, objective_values = _score_operator_power_matrices(
        projected1,
        projected2,
        num_heads=max(1, int(num_heads)),
        num_iters=num_power_iters,
        seed=seed,
    )
    head_layout = [
        {
            "operator_index": head_idx,
            "score_mode": score_mode,
        }
        for head_idx in range(len(score_operators))
    ]

    fit_fn = _build_operator_attention_model if scale_candidates is None else _fit_operator_attention_with_scale_search
    fit_kwargs = {
        "tokens1": tokens1,
        "tokens2": tokens2,
        "lambda_reg": lambda_reg,
        "sigma_sqrt": sigma_sqrt,
        "sigma_inv_sqrt": sigma_inv_sqrt,
        "shared_basis": shared_basis,
        "score_operators": score_operators,
        "head_layout": head_layout,
        "attention_kind": "score-operator-self" if scale_candidates is None else "score-operator-self-gain",
        "target_mode": target_mode,
        "center_values": False,
        "whiten_values": False,
        "score_mode": score_mode,
        "extra_stats": {
            "basis_score_mode": basis_score_mode,
            "shared_eigenvalues": eigvals[:projection_rank],
            "shared_trace": float(np.trace(stats["shared"])),
            "delta_trace": float(np.trace(stats["delta"])),
            "operator_power_iterations": int(num_power_iters),
            "operator_seed": int(seed),
            "score_operator_objective_values": objective_values,
        },
    }
    if scale_candidates is not None:
        fit_kwargs["scale_candidates"] = scale_candidates
    result = fit_fn(**fit_kwargs)
    return result[0] if isinstance(result, tuple) else result


def _apply_spectral_attention(tokens, model):
    model_score_mode = model.get("score_mode", "raw")
    score_tokens = None if model_score_mode == "mixed" else _prepare_attention_score_tokens(tokens, model_score_mode)
    context = _spectral_attention_context(
        tokens,
        model["sigma_inv_sqrt"],
        model["projection_heads"],
        model["head_layout"],
        model.get("local_bias"),
        center_values=model.get("center_values", False),
        whiten_values=model.get("whiten_values", False),
        score_tokens=score_tokens,
    )
    attended = (_flatten_tokens(context) @ model["output_map"]).reshape(tokens.shape)
    if model["residual_mode"]:
        return tokens + model.get("mix_scale", 1.0) * attended

    alpha = model.get("mix_scale", 0.0)
    return alpha * tokens + (1.0 - alpha) * attended


def fit_score_operator_self_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=24,
    seed=0,
):
    return _fit_score_operator_attention_from_token_pairs(
        tokens1,
        tokens2,
        lambda_reg=lambda_reg,
        total_rank=total_rank,
        num_heads=num_heads,
        target_mode=target_mode,
        num_power_iters=num_power_iters,
        seed=seed,
    )


def fit_score_operator_self_attention_scaled_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=24,
    seed=0,
    scale_candidates=None,
):
    return _fit_score_operator_attention_from_token_pairs(
        tokens1,
        tokens2,
        lambda_reg=lambda_reg,
        total_rank=total_rank,
        num_heads=num_heads,
        target_mode=target_mode,
        num_power_iters=num_power_iters,
        seed=seed,
        scale_candidates=scale_candidates or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
    )


def fit_score_operator_self_attention_bagged_scaled_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=24,
    seed=0,
    scale_candidates=None,
    num_bags=4,
    bag_fraction=0.7,
):
    basis_tokens1 = _center_tokens_within_sample(tokens1)
    basis_tokens2 = _center_tokens_within_sample(tokens2)
    flat1 = _flatten_tokens(basis_tokens1)
    flat2 = _flatten_tokens(basis_tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    projection_rank = int(max(1, min(projection_rank, eigvecs.shape[1])))
    shared_basis = eigvecs[:, :projection_rank]

    score_tokens1 = tokens1
    score_tokens2 = tokens2
    projected1 = _normalize_last_axis(score_tokens1 @ sigma_inv_sqrt @ shared_basis)
    projected2 = _normalize_last_axis(score_tokens2 @ sigma_inv_sqrt @ shared_basis)

    prior_operators, prior_objective_values = _score_operator_power_matrices(
        projected1,
        projected2,
        num_heads=max(1, int(num_heads)),
        num_iters=num_power_iters,
        seed=seed,
    )
    prior_basis = _flatten_operator_basis(prior_operators)

    num_samples = tokens1.shape[0]
    bag_size = max(num_heads, int(round(float(bag_fraction) * num_samples)))
    rng = np.random.default_rng(seed + 733)

    operator_bases = []
    weights = []
    objective_summaries = []
    for bag_idx in range(int(max(1, num_bags))):
        subset_idx = np.sort(rng.choice(num_samples, size=min(bag_size, num_samples), replace=False))
        bag_operators, objective_values = _score_operator_power_matrices(
            projected1[subset_idx],
            projected2[subset_idx],
            num_heads=max(1, int(num_heads)),
            num_iters=num_power_iters,
            seed=seed + bag_idx,
        )
        operator_bases.append(_flatten_operator_basis(bag_operators))
        weight = max(float(np.mean(objective_values)), 1e-8)
        weights.append(weight)
        objective_summaries.append(weight)

    flat_operator_basis = _average_projector_basis(
        operator_bases,
        rank=max(1, int(num_heads)),
        weights=weights,
        prior_basis=prior_basis,
        prior_weight=0.25,
    )
    score_operators = _unflatten_operator_basis(flat_operator_basis, projection_rank)
    if not score_operators:
        score_operators = prior_operators
    head_layout = [
        {
            "operator_index": head_idx,
            "score_mode": "raw",
        }
        for head_idx in range(len(score_operators))
    ]

    return _fit_operator_attention_with_scale_search(
        tokens1=tokens1,
        tokens2=tokens2,
        lambda_reg=lambda_reg,
        sigma_sqrt=sigma_sqrt,
        sigma_inv_sqrt=sigma_inv_sqrt,
        shared_basis=shared_basis,
        score_operators=score_operators,
        head_layout=head_layout,
        attention_kind="score-operator-self-bagged-gain",
        target_mode=target_mode,
        scale_candidates=scale_candidates or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        score_mode="raw",
        extra_stats={
            "basis_score_mode": "token-centered",
            "shared_eigenvalues": eigvals[:projection_rank],
            "shared_trace": float(np.trace(stats["shared"])),
            "delta_trace": float(np.trace(stats["delta"])),
            "operator_power_iterations": int(num_power_iters),
            "operator_seed": int(seed),
            "score_operator_objective_values": np.asarray(prior_objective_values, dtype=np.float64),
            "bag_count": int(max(1, num_bags)),
            "bag_fraction": float(bag_fraction),
            "bag_seed": int(seed + 733),
            "bag_objective_means": np.asarray(objective_summaries, dtype=np.float64),
        },
    )


def apply_score_operator_attention(tokens, model):
    model_score_mode = model.get("score_mode", "raw")
    score_tokens = None if model_score_mode == "mixed" else _prepare_attention_score_tokens(tokens, model_score_mode)
    context = _operator_attention_context(
        tokens,
        model["sigma_inv_sqrt"],
        model["shared_basis"],
        model["score_operators"],
        model["head_layout"],
        center_values=model.get("center_values", False),
        whiten_values=model.get("whiten_values", False),
        score_tokens=score_tokens,
    )
    attended = (_flatten_tokens(context) @ model["output_map"]).reshape(tokens.shape)
    if model["residual_mode"]:
        return tokens + model.get("mix_scale", 1.0) * attended

    alpha = model.get("mix_scale", 0.0)
    return alpha * tokens + (1.0 - alpha) * attended


def _attention_context_from_model(tokens, model):
    attention_kind = model.get("attention_kind", "")
    if "score-operator" in attention_kind:
        model_score_mode = model.get("score_mode", "raw")
        score_tokens = None if model_score_mode == "mixed" else _prepare_attention_score_tokens(tokens, model_score_mode)
        return _operator_attention_context(
            tokens,
            model["sigma_inv_sqrt"],
            model["shared_basis"],
            model["score_operators"],
            model["head_layout"],
            center_values=model.get("center_values", False),
            whiten_values=model.get("whiten_values", False),
            score_tokens=score_tokens,
        )

    model_score_mode = model.get("score_mode", "raw")
    score_tokens = None if model_score_mode == "mixed" else _prepare_attention_score_tokens(tokens, model_score_mode)
    return _spectral_attention_context(
        tokens,
        model["sigma_inv_sqrt"],
        model["projection_heads"],
        model["head_layout"],
        model.get("local_bias"),
        center_values=model.get("center_values", False),
        whiten_values=model.get("whiten_values", False),
        score_tokens=score_tokens,
    )


def fit_score_kernel_dictionary_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=24,
    seed=0,
):
    projector_heads = max(1, num_heads // 2)
    operator_heads = max(1, num_heads - projector_heads)
    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    projector_rank = max(1, projection_rank // 2)
    operator_rank = max(1, projection_rank - projector_rank)

    projector_model = fit_score_power_bagged_gain_attention_from_token_pairs(
        tokens1,
        tokens2,
        lambda_reg=lambda_reg,
        total_rank=projector_rank,
        num_heads=projector_heads,
        target_mode=target_mode,
        num_power_iters=num_power_iters,
        seed=seed,
    )
    operator_model = fit_score_operator_self_attention_bagged_scaled_from_token_pairs(
        tokens1,
        tokens2,
        lambda_reg=lambda_reg,
        total_rank=operator_rank,
        num_heads=operator_heads,
        target_mode=target_mode,
        num_power_iters=num_power_iters,
        seed=seed + 97,
    )

    context1 = np.concatenate(
        [
            _attention_context_from_model(tokens1, projector_model),
            _attention_context_from_model(tokens1, operator_model),
        ],
        axis=-1,
    )
    context2 = np.concatenate(
        [
            _attention_context_from_model(tokens2, projector_model),
            _attention_context_from_model(tokens2, operator_model),
        ],
        axis=-1,
    )
    solved = _solve_context_output(
        context1=context1,
        context2=context2,
        input1=tokens1,
        input2=tokens2,
        lambda_reg=lambda_reg,
        target_mode=target_mode,
    )
    loss = _attention_solution_loss(context1, context2, tokens1, tokens2, solved, lambda_reg, target_mode)

    sigma_size = projector_model["sigma_inv_sqrt"].size
    projector_size = sum(head.size for head in projector_model["projection_heads"])
    operator_size = operator_model["shared_basis"].size + sum(operator.size for operator in operator_model["score_operators"])
    return {
        "attention_kind": "score-kernel-dictionary",
        "projector_model": projector_model,
        "operator_model": operator_model,
        "output_map": solved["output_map"],
        "mix_scale": solved["mix_scale"],
        "residual_mode": solved["residual_mode"],
        "target_mode": target_mode,
        "projection_rank": int(projector_model.get("projection_rank", 0) + operator_model.get("projection_rank", 0)),
        "num_heads": int(projector_model.get("num_heads", 0) + operator_model.get("num_heads", 0)),
        "shared_eigenvalues": projector_model.get("shared_eigenvalues", operator_model.get("shared_eigenvalues", [])),
        "shared_trace": float(projector_model.get("shared_trace", 0.0)),
        "delta_trace": float(projector_model.get("delta_trace", 0.0)),
        "teacher_stats": solved["target_stats"],
        "selected_score_scales": {
            "projector": projector_model.get("selected_score_scales"),
            "operator": operator_model.get("selected_score_scales"),
        },
        "branch_train_fit_losses": {
            "projector": float(projector_model.get("train_fit_loss", np.nan)),
            "operator": float(operator_model.get("train_fit_loss", np.nan)),
        },
        "train_fit_loss": float(loss),
        "parameter_count": int(
            sigma_size
            + projector_size
            + operator_size
            + solved["output_map"].size
        ),
    }


def apply_score_kernel_dictionary_attention(tokens, model):
    context = np.concatenate(
        [
            _attention_context_from_model(tokens, model["projector_model"]),
            _attention_context_from_model(tokens, model["operator_model"]),
        ],
        axis=-1,
    )
    attended = (_flatten_tokens(context) @ model["output_map"]).reshape(tokens.shape)
    if model["residual_mode"]:
        return tokens + model.get("mix_scale", 1.0) * attended

    alpha = model.get("mix_scale", 0.0)
    return alpha * tokens + (1.0 - alpha) * attended


def fit_score_metric_bagged_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=24,
    seed=0,
    scale_candidates=None,
):
    projector_model = fit_score_power_bagged_gain_attention_from_token_pairs(
        tokens1,
        tokens2,
        lambda_reg=lambda_reg,
        total_rank=total_rank,
        num_heads=num_heads,
        target_mode=target_mode,
        num_power_iters=num_power_iters,
        seed=seed,
        scale_candidates=scale_candidates or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    )

    shared_basis = np.concatenate(projector_model["projection_heads"], axis=1)
    sigma_sqrt = projector_model["sigma_sqrt"]
    sigma_inv_sqrt = projector_model["sigma_inv_sqrt"]
    score_mode = projector_model.get("score_mode", "token-centered")

    score_tokens1 = _prepare_attention_score_tokens(tokens1, score_mode)
    score_tokens2 = _prepare_attention_score_tokens(tokens2, score_mode)
    projected1 = _normalize_last_axis(score_tokens1 @ sigma_inv_sqrt @ shared_basis)
    projected2 = _normalize_last_axis(score_tokens2 @ sigma_inv_sqrt @ shared_basis)

    score_operators = []
    metric_eigenvalues = []
    offset = 0
    for projection_head in projector_model["projection_heads"]:
        width = projection_head.shape[1]
        idxs = np.arange(offset, offset + width)
        cross = np.einsum(
            "nti,ntj->nij",
            projected1[:, :, idxs],
            projected2[:, :, idxs],
            optimize=True,
        ) / max(projected1.shape[1], 1)
        metric_matrix = np.mean(cross * cross, axis=0)
        metric_matrix = _symmetrize(metric_matrix)
        eigvals, eigvecs = eigh(metric_matrix)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        diag_weights = eigvecs[:, 0]
        operator = np.zeros((shared_basis.shape[1], shared_basis.shape[1]), dtype=np.float64)
        operator[idxs, idxs] = diag_weights
        score_operators.append(operator)
        metric_eigenvalues.append(float(eigvals[0]))
        offset += width

    head_layout = [
        {
            "operator_index": head_idx,
            "score_mode": score_mode,
        }
        for head_idx in range(len(score_operators))
    ]
    return _fit_operator_attention_with_scale_search(
        tokens1=tokens1,
        tokens2=tokens2,
        lambda_reg=lambda_reg,
        sigma_sqrt=sigma_sqrt,
        sigma_inv_sqrt=sigma_inv_sqrt,
        shared_basis=shared_basis,
        score_operators=score_operators,
        head_layout=head_layout,
        attention_kind="score-metric-self-bagged-gain",
        target_mode=target_mode,
        scale_candidates=scale_candidates or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        score_mode=score_mode,
        extra_stats={
            "shared_eigenvalues": projector_model.get("shared_eigenvalues", []),
            "shared_trace": float(projector_model.get("shared_trace", 0.0)),
            "delta_trace": float(projector_model.get("delta_trace", 0.0)),
            "power_iterations": int(num_power_iters),
            "power_seed": int(seed),
            "metric_eigenvalues": np.asarray(metric_eigenvalues, dtype=np.float64),
            "base_projector_scales": projector_model.get("selected_score_scales"),
            "base_projector_train_fit_loss": float(projector_model.get("train_fit_loss", np.nan)),
        },
    )


def fit_score_operator_on_projector_basis_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=24,
    seed=0,
    scale_candidates=None,
):
    projector_model = fit_score_power_bagged_gain_attention_from_token_pairs(
        tokens1,
        tokens2,
        lambda_reg=lambda_reg,
        total_rank=total_rank,
        num_heads=num_heads,
        target_mode=target_mode,
        num_power_iters=num_power_iters,
        seed=seed,
        scale_candidates=scale_candidates or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    )

    shared_basis = np.concatenate(projector_model["projection_heads"], axis=1)
    sigma_sqrt = projector_model["sigma_sqrt"]
    sigma_inv_sqrt = projector_model["sigma_inv_sqrt"]
    score_tokens1 = tokens1
    score_tokens2 = tokens2
    projected1 = _normalize_last_axis(score_tokens1 @ sigma_inv_sqrt @ shared_basis)
    projected2 = _normalize_last_axis(score_tokens2 @ sigma_inv_sqrt @ shared_basis)

    score_operators, objective_values = _score_operator_power_matrices(
        projected1,
        projected2,
        num_heads=max(1, int(num_heads)),
        num_iters=num_power_iters,
        seed=seed + 211,
    )
    head_layout = [
        {
            "operator_index": head_idx,
            "score_mode": "raw",
        }
        for head_idx in range(len(score_operators))
    ]
    return _fit_operator_attention_with_scale_search(
        tokens1=tokens1,
        tokens2=tokens2,
        lambda_reg=lambda_reg,
        sigma_sqrt=sigma_sqrt,
        sigma_inv_sqrt=sigma_inv_sqrt,
        shared_basis=shared_basis,
        score_operators=score_operators,
        head_layout=head_layout,
        attention_kind="score-operator-projector-basis-gain",
        target_mode=target_mode,
        scale_candidates=scale_candidates or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        score_mode="raw",
        extra_stats={
            "shared_eigenvalues": projector_model.get("shared_eigenvalues", []),
            "shared_trace": float(projector_model.get("shared_trace", 0.0)),
            "delta_trace": float(projector_model.get("delta_trace", 0.0)),
            "power_iterations": int(num_power_iters),
            "power_seed": int(seed),
            "operator_seed": int(seed + 211),
            "score_operator_objective_values": np.asarray(objective_values, dtype=np.float64),
            "base_projector_scales": projector_model.get("selected_score_scales"),
            "base_projector_train_fit_loss": float(projector_model.get("train_fit_loss", np.nan)),
        },
    )


def fit_score_blockpower_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=24,
    seed=0,
    scale_candidates=None,
):
    score_tokens1 = _center_tokens_within_sample(tokens1)
    score_tokens2 = _center_tokens_within_sample(tokens2)
    flat1 = _flatten_tokens(score_tokens1)
    flat2 = _flatten_tokens(score_tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    head_indices = _projection_head_indices(projection_rank, eigvecs.shape[1], num_heads)
    head_dims = [len(idxs) for idxs in head_indices]
    init_heads = [eigvecs[:, idxs] for idxs in head_indices]
    projection_heads, objective_values = _score_alignment_block_head_bases(
        score_tokens1,
        score_tokens2,
        sigma_inv_sqrt,
        head_dims=head_dims,
        num_iters=num_power_iters,
        seed=seed,
        init_heads=init_heads,
    )
    head_layout = _build_spectral_head_layout("score-self-power", len(projection_heads))
    return _fit_self_attention_with_scale_search(
        tokens1=tokens1,
        tokens2=tokens2,
        lambda_reg=lambda_reg,
        sigma_sqrt=sigma_sqrt,
        sigma_inv_sqrt=sigma_inv_sqrt,
        projection_heads=projection_heads,
        head_layout=head_layout,
        attention_kind="score-self-block-gain",
        target_mode=target_mode,
        scale_candidates=scale_candidates or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        score_mode="token-centered",
        extra_stats={
            "shared_eigenvalues": eigvals[: int(sum(head.shape[1] for head in projection_heads))],
            "shared_trace": float(np.trace(stats["shared"])),
            "delta_trace": float(np.trace(stats["delta"])),
            "power_iterations": int(num_power_iters),
            "power_seed": int(seed),
            "score_objective_values": objective_values,
        },
    )


def fit_score_blockpower_bagged_gain_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=24,
    seed=0,
    scale_candidates=None,
    num_bags=4,
    bag_fraction=0.7,
):
    score_tokens1 = _center_tokens_within_sample(tokens1)
    score_tokens2 = _center_tokens_within_sample(tokens2)
    flat1 = _flatten_tokens(score_tokens1)
    flat2 = _flatten_tokens(score_tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    head_indices = _projection_head_indices(projection_rank, eigvecs.shape[1], num_heads)
    head_dims = [len(idxs) for idxs in head_indices]
    init_heads = [eigvecs[:, idxs] for idxs in head_indices]

    num_samples = tokens1.shape[0]
    bag_size = max(max(head_dims), int(round(float(bag_fraction) * num_samples)))
    rng = np.random.default_rng(seed + 887)

    bagged_head_bases = [[] for _ in head_dims]
    weights = []
    objective_summaries = []
    for bag_idx in range(int(max(1, num_bags))):
        subset_idx = np.sort(rng.choice(num_samples, size=min(bag_size, num_samples), replace=False))
        head_bases, objective_values = _score_alignment_block_head_bases(
            score_tokens1[subset_idx],
            score_tokens2[subset_idx],
            sigma_inv_sqrt,
            head_dims=head_dims,
            num_iters=num_power_iters,
            seed=seed + bag_idx,
            init_heads=init_heads,
        )
        for head_idx, basis in enumerate(head_bases):
            bagged_head_bases[head_idx].append(basis)
        weight = max(float(np.mean(objective_values)), 1e-8)
        weights.append(weight)
        objective_summaries.append(weight)

    projection_heads = []
    for head_idx, head_dim in enumerate(head_dims):
        projection_heads.append(
            _average_projector_basis(
                bagged_head_bases[head_idx],
                rank=head_dim,
                weights=weights,
                prior_basis=init_heads[head_idx],
                prior_weight=0.25,
            )
        )

    head_layout = _build_spectral_head_layout("score-self-power", len(projection_heads))
    return _fit_self_attention_with_scale_search(
        tokens1=tokens1,
        tokens2=tokens2,
        lambda_reg=lambda_reg,
        sigma_sqrt=sigma_sqrt,
        sigma_inv_sqrt=sigma_inv_sqrt,
        projection_heads=projection_heads,
        head_layout=head_layout,
        attention_kind="score-self-block-bagged-gain",
        target_mode=target_mode,
        scale_candidates=scale_candidates or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        score_mode="token-centered",
        extra_stats={
            "shared_eigenvalues": eigvals[: int(sum(head.shape[1] for head in projection_heads))],
            "shared_trace": float(np.trace(stats["shared"])),
            "delta_trace": float(np.trace(stats["delta"])),
            "power_iterations": int(num_power_iters),
            "power_seed": int(seed),
            "bag_count": int(max(1, num_bags)),
            "bag_fraction": float(bag_fraction),
            "bag_seed": int(seed + 887),
            "bag_objective_means": np.asarray(objective_summaries, dtype=np.float64),
        },
    )


def fit_spectral_bt_context_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    center_values=False,
    head_weight_mode="uniform",
):
    flat1 = _flatten_tokens(tokens1)
    flat2 = _flatten_tokens(tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    head_indices = _projection_head_indices(projection_rank, eigvecs.shape[1], num_heads)
    projection_heads = [eigvecs[:, idxs] for idxs in head_indices]
    head_layout = _build_spectral_head_layout("spectral-self", len(projection_heads))

    if head_weight_mode == "uniform":
        head_weights = np.ones(len(projection_heads), dtype=np.float64)
    elif head_weight_mode == "spectral":
        head_weights = np.array([max(float(np.mean(np.maximum(eigvals[idxs], 0.0))), 1e-8) for idxs in head_indices], dtype=np.float64)
    else:
        raise ValueError(f"Unknown head_weight_mode: {head_weight_mode}")

    contexts1 = _spectral_head_contexts(
        tokens1,
        tokens1,
        tokens1,
        sigma_inv_sqrt,
        projection_heads,
        head_layout,
        center_values=center_values,
    )
    contexts2 = _spectral_head_contexts(
        tokens2,
        tokens2,
        tokens2,
        sigma_inv_sqrt,
        projection_heads,
        head_layout,
        center_values=center_values,
    )
    context1 = _aggregate_head_contexts(contexts1, head_weights=head_weights)
    context2 = _aggregate_head_contexts(contexts2, head_weights=head_weights)

    bt_model = cfbt.fit_layer(_flatten_tokens(context1), _flatten_tokens(context2), lambda_reg=lambda_reg)
    pred1 = (_flatten_tokens(context1) @ bt_model["transform_base"]).reshape(tokens1.shape)
    pred2 = (_flatten_tokens(context2) @ bt_model["transform_base"]).reshape(tokens2.shape)
    mix_scale = _fit_residual_scale(pred1, pred2, 0.5 * (tokens2 - tokens1), 0.5 * (tokens1 - tokens2))

    return {
        "attention_kind": (
            "spectral-bt-context-centered"
            if center_values and head_weight_mode == "uniform"
            else "spectral-bt-context-weighted"
            if head_weight_mode == "spectral"
            else "spectral-bt-context"
        ),
        "sigma_inv_sqrt": sigma_inv_sqrt,
        "sigma_sqrt": sigma_sqrt,
        "projection_heads": projection_heads,
        "head_layout": head_layout,
        "head_weights": head_weights,
        "center_values": center_values,
        "head_weight_mode": head_weight_mode,
        "transform_base": bt_model["transform_base"],
        "mix_scale": mix_scale,
        "residual_mode": True,
        "projection_rank": int(sum(head.shape[1] for head in projection_heads)),
        "num_heads": len(head_layout),
        "shared_eigenvalues": eigvals[: int(sum(head.shape[1] for head in projection_heads))],
        "shared_trace": float(np.trace(stats["shared"])),
        "delta_trace": float(np.trace(stats["delta"])),
        "bt_stats": {
            "transform_fro": bt_model["transform_fro"],
            "distance_to_identity": bt_model["distance_to_identity"],
            "max_whitened_delta": bt_model["max_whitened_delta"],
        },
        "parameter_count": int(
            sigma_inv_sqrt.size
            + sum(head.size for head in projection_heads)
            + bt_model["transform_base"].size
        ),
    }


def apply_spectral_bt_context_attention(tokens, model):
    contexts = _spectral_head_contexts(
        tokens,
        tokens,
        tokens,
        model["sigma_inv_sqrt"],
        model["projection_heads"],
        model["head_layout"],
        center_values=model.get("center_values", False),
    )
    context = _aggregate_head_contexts(contexts, head_weights=model.get("head_weights"))
    attended = (_flatten_tokens(context) @ model["transform_base"]).reshape(tokens.shape)
    return tokens + model.get("mix_scale", 1.0) * attended


def fit_cca_self_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    center_values=False,
):
    flat1 = _flatten_tokens(tokens1)
    flat2 = _flatten_tokens(tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    cca_model = cfbt.fit_paper_cca_layer(stats)

    transform_a = cca_model["transform_a"]
    transform_b = cca_model["transform_b"]
    rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    head_indices = _projection_head_indices(rank, transform_a.shape[1], num_heads)
    query_heads = [transform_a[:, idxs] for idxs in head_indices]
    key_heads = [transform_b[:, idxs] for idxs in head_indices]

    context1 = np.concatenate(
        _asymmetric_head_contexts(tokens1, tokens1, tokens1, query_heads, key_heads, center_values=center_values),
        axis=-1,
    )
    context2 = np.concatenate(
        _asymmetric_head_contexts(tokens2, tokens2, tokens2, query_heads, key_heads, center_values=center_values),
        axis=-1,
    )
    solved = _solve_context_output(
        context1=context1,
        context2=context2,
        input1=tokens1,
        input2=tokens2,
        lambda_reg=lambda_reg,
        target_mode=target_mode,
    )

    total_rank = int(sum(head.shape[1] for head in query_heads))
    return {
        "attention_kind": "cca-self-centered" if center_values else "cca-self",
        "query_heads": query_heads,
        "key_heads": key_heads,
        "center_values": center_values,
        "output_map": solved["output_map"],
        "mix_scale": solved["mix_scale"],
        "residual_mode": solved["residual_mode"],
        "target_mode": target_mode,
        "projection_rank": total_rank,
        "num_heads": len(query_heads),
        "canonical_correlations": cca_model["canonical_correlations"][:total_rank],
        "parameter_count": int(
            sum(head.size for head in query_heads)
            + sum(head.size for head in key_heads)
            + solved["output_map"].size
        ),
    }


def apply_cca_self_attention(tokens, model):
    context = np.concatenate(
        _asymmetric_head_contexts(
            tokens,
            tokens,
            tokens,
            model["query_heads"],
            model["key_heads"],
            center_values=model.get("center_values", False),
        ),
        axis=-1,
    )
    attended = (_flatten_tokens(context) @ model["output_map"]).reshape(tokens.shape)
    if model["residual_mode"]:
        return tokens + model.get("mix_scale", 1.0) * attended

    alpha = model.get("mix_scale", 0.0)
    return alpha * tokens + (1.0 - alpha) * attended


def _hybrid_content_landmark_context(tokens, sigma_inv_sqrt, projection_heads, head_layout, keys, temperature):
    self_context = _spectral_attention_context(tokens, sigma_inv_sqrt, projection_heads, head_layout, local_bias=None)
    landmark_weights, _ = _token_attention_weights(tokens, sigma_inv_sqrt, keys, temperature)
    return np.concatenate([self_context, landmark_weights], axis=-1)


def fit_spectral_landmark_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    num_landmarks=None,
    target_mode="mean",
):
    flat1 = _flatten_tokens(tokens1)
    flat2 = _flatten_tokens(tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    projection_heads = _split_projection_heads(eigvecs, projection_rank, num_heads)
    head_layout = _build_spectral_head_layout("spectral-self", len(projection_heads))

    landmark_count = min(num_landmarks or tokens1.shape[1], eigvecs.shape[1])
    keys = eigvecs[:, :landmark_count]
    temperature = 1.0 / np.sqrt(tokens1.shape[-1])
    context1 = _hybrid_content_landmark_context(tokens1, sigma_inv_sqrt, projection_heads, head_layout, keys, temperature)
    context2 = _hybrid_content_landmark_context(tokens2, sigma_inv_sqrt, projection_heads, head_layout, keys, temperature)
    solved = _solve_context_output(
        context1=context1,
        context2=context2,
        input1=tokens1,
        input2=tokens2,
        lambda_reg=lambda_reg,
        target_mode=target_mode,
    )

    total_projection_rank = int(sum(head.shape[1] for head in projection_heads))
    return {
        "attention_kind": "spectral-landmark" if target_mode != "bt-residual" else "spectral-landmark-bt",
        "sigma_inv_sqrt": sigma_inv_sqrt,
        "sigma_sqrt": sigma_sqrt,
        "projection_heads": projection_heads,
        "head_layout": head_layout,
        "keys": keys,
        "temperature": temperature,
        "output_map": solved["output_map"],
        "mix_scale": solved["mix_scale"],
        "residual_mode": solved["residual_mode"],
        "target_mode": target_mode,
        "projection_rank": total_projection_rank,
        "landmark_count": int(landmark_count),
        "num_heads": len(head_layout),
        "shared_eigenvalues": eigvals[: max(total_projection_rank, landmark_count)],
        "shared_trace": float(np.trace(stats["shared"])),
        "delta_trace": float(np.trace(stats["delta"])),
        "teacher_stats": solved["target_stats"],
        "parameter_count": int(
            sigma_inv_sqrt.size
            + sum(head.size for head in projection_heads)
            + keys.size
            + solved["output_map"].size
        ),
    }


def apply_spectral_landmark_attention(tokens, model):
    context = _hybrid_content_landmark_context(
        tokens,
        model["sigma_inv_sqrt"],
        model["projection_heads"],
        model["head_layout"],
        model["keys"],
        model["temperature"],
    )
    attended = (_flatten_tokens(context) @ model["output_map"]).reshape(tokens.shape)
    if model["residual_mode"]:
        return tokens + model.get("mix_scale", 1.0) * attended

    alpha = model.get("mix_scale", 0.0)
    return alpha * tokens + (1.0 - alpha) * attended


def fit_random_self_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    seed=0,
):
    flat1 = _flatten_tokens(tokens1)
    flat2 = _flatten_tokens(tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    projection_heads = _split_random_projection_heads(tokens1.shape[-1], projection_rank, num_heads, seed)
    head_layout = _build_spectral_head_layout("spectral-self", len(projection_heads))

    context1 = _spectral_attention_context(
        tokens1,
        sigma_inv_sqrt,
        projection_heads,
        head_layout,
        local_bias=None,
    )
    context2 = _spectral_attention_context(
        tokens2,
        sigma_inv_sqrt,
        projection_heads,
        head_layout,
        local_bias=None,
    )
    solved = _solve_context_output(
        context1=context1,
        context2=context2,
        input1=tokens1,
        input2=tokens2,
        lambda_reg=lambda_reg,
        target_mode=target_mode,
    )

    total_projection_rank = int(sum(head.shape[1] for head in projection_heads))
    return {
        "attention_kind": "random-self-ridge",
        "sigma_inv_sqrt": sigma_inv_sqrt,
        "sigma_sqrt": sigma_sqrt,
        "projection_heads": projection_heads,
        "head_layout": head_layout,
        "local_bias": None,
        "output_map": solved["output_map"],
        "mix_scale": solved["mix_scale"],
        "residual_mode": solved["residual_mode"],
        "target_mode": target_mode,
        "projection_rank": total_projection_rank,
        "num_heads": len(head_layout),
        "random_seed": int(seed),
        "shared_trace": float(np.trace(stats["shared"])),
        "delta_trace": float(np.trace(stats["delta"])),
        "teacher_stats": solved["target_stats"],
        "parameter_count": int(
            sigma_inv_sqrt.size
            + sum(head.size for head in projection_heads)
            + solved["output_map"].size
        ),
    }


def fit_random_untrained_self_attention_from_token_pairs(
    tokens1,
    tokens2,
    total_rank=None,
    num_heads=4,
    seed=0,
):
    flat1 = _flatten_tokens(tokens1)
    flat2 = _flatten_tokens(tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    projection_heads = _split_random_projection_heads(tokens1.shape[-1], projection_rank, num_heads, seed)
    head_layout = _build_spectral_head_layout("spectral-self", len(projection_heads))

    return {
        "attention_kind": "random-self-untrained",
        "sigma_inv_sqrt": sigma_inv_sqrt,
        "sigma_sqrt": sigma_sqrt,
        "projection_heads": projection_heads,
        "head_layout": head_layout,
        "local_bias": None,
        "mix_scale": 1.0,
        "residual_mode": True,
        "projection_rank": int(sum(head.shape[1] for head in projection_heads)),
        "num_heads": len(head_layout),
        "random_seed": int(seed),
        "shared_trace": float(np.trace(stats["shared"])),
        "delta_trace": float(np.trace(stats["delta"])),
        "parameter_count": int(
            sigma_inv_sqrt.size
            + sum(head.size for head in projection_heads)
        ),
    }


def fit_token_centered_spectral_self_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
):
    return _fit_spectral_attention_from_tokens(
        tokens1,
        tokens2,
        lambda_reg=lambda_reg,
        attention_kind="spectral-self-token-centered",
        total_rank=total_rank,
        num_heads=num_heads,
        target_mode=target_mode,
        score_mode="token-centered",
    )


def fit_token_stats_spectral_self_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
):
    score_tokens1 = _center_tokens_within_sample(tokens1)
    score_tokens2 = _center_tokens_within_sample(tokens2)
    flat1 = _flatten_tokens(score_tokens1)
    flat2 = _flatten_tokens(score_tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    head_indices = _projection_head_indices(projection_rank, eigvecs.shape[1], num_heads)
    projection_heads = [eigvecs[:, idxs] for idxs in head_indices]
    head_layout = _build_spectral_head_layout("spectral-self-token-stats", len(projection_heads))

    return _fit_self_attention_with_projections(
        tokens1=tokens1,
        tokens2=tokens2,
        lambda_reg=lambda_reg,
        sigma_sqrt=sigma_sqrt,
        sigma_inv_sqrt=sigma_inv_sqrt,
        projection_heads=projection_heads,
        head_layout=head_layout,
        attention_kind="spectral-self-token-stats",
        target_mode=target_mode,
        score_mode="raw",
        extra_stats={
            "shared_eigenvalues": eigvals[: int(sum(head.shape[1] for head in projection_heads))],
            "shared_trace": float(np.trace(stats["shared"])),
            "delta_trace": float(np.trace(stats["delta"])),
        },
    )


def fit_token_stats_scaled_self_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    scale_candidates=None,
):
    score_tokens1 = _center_tokens_within_sample(tokens1)
    score_tokens2 = _center_tokens_within_sample(tokens2)
    flat1 = _flatten_tokens(score_tokens1)
    flat2 = _flatten_tokens(score_tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    head_indices = _projection_head_indices(projection_rank, eigvecs.shape[1], num_heads)
    projection_heads = [eigvecs[:, idxs] for idxs in head_indices]
    head_layout = _build_spectral_head_layout("spectral-self-token-stats", len(projection_heads))
    return _fit_self_attention_with_scale_search(
        tokens1=tokens1,
        tokens2=tokens2,
        lambda_reg=lambda_reg,
        sigma_sqrt=sigma_sqrt,
        sigma_inv_sqrt=sigma_inv_sqrt,
        projection_heads=projection_heads,
        head_layout=head_layout,
        attention_kind="spectral-self-token-stats-gain",
        target_mode=target_mode,
        scale_candidates=scale_candidates or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        score_mode="raw",
        extra_stats={
            "shared_eigenvalues": eigvals[: int(sum(head.shape[1] for head in projection_heads))],
            "shared_trace": float(np.trace(stats["shared"])),
            "delta_trace": float(np.trace(stats["delta"])),
        },
    )


def fit_score_power_self_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=8,
    seed=0,
):
    return _fit_score_power_self_attention_from_token_pairs(
        tokens1,
        tokens2,
        lambda_reg=lambda_reg,
        total_rank=total_rank,
        num_heads=num_heads,
        target_mode=target_mode,
        num_power_iters=num_power_iters,
        seed=seed,
        score_mode="token-centered",
        attention_kind="score-self-power",
    )


def fit_score_power_raw_self_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=8,
    seed=0,
):
    return _fit_score_power_self_attention_from_token_pairs(
        tokens1,
        tokens2,
        lambda_reg=lambda_reg,
        total_rank=total_rank,
        num_heads=num_heads,
        target_mode=target_mode,
        num_power_iters=num_power_iters,
        seed=seed,
        score_mode="raw",
        attention_kind="score-self-power-raw",
    )


def fit_score_power_scaled_self_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=8,
    seed=0,
    scale_candidates=None,
):
    return _fit_score_power_self_attention_from_token_pairs(
        tokens1,
        tokens2,
        lambda_reg=lambda_reg,
        total_rank=total_rank,
        num_heads=num_heads,
        target_mode=target_mode,
        num_power_iters=num_power_iters,
        seed=seed,
        score_mode="token-centered",
        attention_kind="score-self-power-gain",
        scale_candidates=scale_candidates or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    )


def fit_score_power_per_head_scaled_self_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=8,
    seed=0,
    scale_candidates=None,
):
    score_tokens1 = _center_tokens_within_sample(tokens1)
    score_tokens2 = _center_tokens_within_sample(tokens2)
    flat1 = _flatten_tokens(score_tokens1)
    flat2 = _flatten_tokens(score_tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    basis, objective_values = _score_alignment_power_basis(
        score_tokens1,
        score_tokens2,
        sigma_inv_sqrt,
        rank=min(projection_rank, tokens1.shape[-1]),
        num_iters=num_power_iters,
        seed=seed,
        init_basis=eigvecs[:, : min(projection_rank, eigvecs.shape[1])],
    )
    head_indices = _projection_head_indices(basis.shape[1], basis.shape[1], num_heads)
    projection_heads = [basis[:, idxs] for idxs in head_indices]
    head_layout = []
    for head_idx in range(len(projection_heads)):
        head_layout.append(
            {
                "projection_index": head_idx,
                "bias_kind": "global",
                "scale_group": f"head_{head_idx}",
            }
        )

    return _fit_self_attention_with_scale_search(
        tokens1=tokens1,
        tokens2=tokens2,
        lambda_reg=lambda_reg,
        sigma_sqrt=sigma_sqrt,
        sigma_inv_sqrt=sigma_inv_sqrt,
        projection_heads=projection_heads,
        head_layout=head_layout,
        attention_kind="score-self-power-headgain",
        target_mode=target_mode,
        scale_candidates=scale_candidates or [0.5, 1.0, 2.0, 4.0],
        scale_group_key="scale_group",
        score_mode="token-centered",
        extra_stats={
            "shared_eigenvalues": eigvals[: basis.shape[1]],
            "shared_trace": float(np.trace(stats["shared"])),
            "delta_trace": float(np.trace(stats["delta"])),
            "power_iterations": int(num_power_iters),
            "power_seed": int(seed),
            "score_objective_values": objective_values,
        },
    )


def fit_score_power_deflated_scaled_self_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=8,
    seed=0,
    scale_candidates=None,
):
    score_tokens1 = _center_tokens_within_sample(tokens1)
    score_tokens2 = _center_tokens_within_sample(tokens2)
    flat1 = _flatten_tokens(score_tokens1)
    flat2 = _flatten_tokens(score_tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    basis, objective_values = _score_alignment_power_basis_deflated(
        score_tokens1,
        score_tokens2,
        sigma_inv_sqrt,
        rank=min(projection_rank, tokens1.shape[-1]),
        num_iters=num_power_iters,
        seed=seed,
        init_basis=eigvecs[:, : min(projection_rank, eigvecs.shape[1])],
    )
    head_indices = _projection_head_indices(basis.shape[1], basis.shape[1], num_heads)
    projection_heads = [basis[:, idxs] for idxs in head_indices]
    head_layout = _build_spectral_head_layout("score-self-power", len(projection_heads))

    return _fit_self_attention_with_scale_search(
        tokens1=tokens1,
        tokens2=tokens2,
        lambda_reg=lambda_reg,
        sigma_sqrt=sigma_sqrt,
        sigma_inv_sqrt=sigma_inv_sqrt,
        projection_heads=projection_heads,
        head_layout=head_layout,
        attention_kind="score-self-power-deflated-gain",
        target_mode=target_mode,
        scale_candidates=scale_candidates or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        score_mode="token-centered",
        extra_stats={
            "shared_eigenvalues": eigvals[: basis.shape[1]],
            "shared_trace": float(np.trace(stats["shared"])),
            "delta_trace": float(np.trace(stats["delta"])),
            "power_iterations": int(num_power_iters),
            "power_seed": int(seed),
            "score_objective_values": objective_values,
        },
    )


def fit_score_cosine_scaled_self_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=8,
    seed=0,
    scale_candidates=None,
):
    score_tokens1 = _center_tokens_within_sample(tokens1)
    score_tokens2 = _center_tokens_within_sample(tokens2)
    flat1 = _flatten_tokens(score_tokens1)
    flat2 = _flatten_tokens(score_tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    basis, objective_values = _score_alignment_cosine_basis(
        score_tokens1,
        score_tokens2,
        sigma_inv_sqrt,
        rank=min(projection_rank, tokens1.shape[-1]),
        num_iters=num_power_iters,
        seed=seed,
        init_basis=eigvecs[:, : min(projection_rank, eigvecs.shape[1])],
    )
    head_indices = _projection_head_indices(basis.shape[1], basis.shape[1], num_heads)
    projection_heads = [basis[:, idxs] for idxs in head_indices]
    head_layout = _build_spectral_head_layout("score-self-power", len(projection_heads))

    return _fit_self_attention_with_scale_search(
        tokens1=tokens1,
        tokens2=tokens2,
        lambda_reg=lambda_reg,
        sigma_sqrt=sigma_sqrt,
        sigma_inv_sqrt=sigma_inv_sqrt,
        projection_heads=projection_heads,
        head_layout=head_layout,
        attention_kind="score-self-cosine-gain",
        target_mode=target_mode,
        scale_candidates=scale_candidates or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        score_mode="token-centered",
        extra_stats={
            "shared_eigenvalues": eigvals[: basis.shape[1]],
            "shared_trace": float(np.trace(stats["shared"])),
            "delta_trace": float(np.trace(stats["delta"])),
            "power_iterations": int(num_power_iters),
            "power_seed": int(seed),
            "score_objective_values": objective_values,
        },
    )


def fit_score_power_multistart_gain_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=8,
    seed=0,
    scale_candidates=None,
):
    score_tokens1 = _center_tokens_within_sample(tokens1)
    score_tokens2 = _center_tokens_within_sample(tokens2)
    flat1 = _flatten_tokens(score_tokens1)
    flat2 = _flatten_tokens(score_tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    rank = min(projection_rank, tokens1.shape[-1])
    init_pool_rank = min(eigvecs.shape[1], max(rank, 4 * max(1, rank)))
    init_specs = [
        ("eig", eigvecs[:, :rank]),
        ("maxent", _maxent_stable_basis(eigvecs, eigvals, rank=rank, seed=seed)),
        ("random", _random_orthogonal_basis(tokens1.shape[-1], rank, seed)),
        (
            "pool-rot",
            eigvecs[:, :init_pool_rank] @ _random_orthogonal_basis(init_pool_rank, rank, seed + 17),
        ),
    ]

    best_model = None
    best_loss = None
    best_init_name = None
    for init_name, init_basis in init_specs:
        basis, objective_values = _score_alignment_power_basis(
            score_tokens1,
            score_tokens2,
            sigma_inv_sqrt,
            rank=rank,
            num_iters=num_power_iters,
            seed=seed,
            init_basis=init_basis,
        )
        head_indices = _projection_head_indices(basis.shape[1], basis.shape[1], num_heads)
        projection_heads = [basis[:, idxs] for idxs in head_indices]
        head_layout = _build_spectral_head_layout("score-self-power", len(projection_heads))
        model = _fit_self_attention_with_scale_search(
            tokens1=tokens1,
            tokens2=tokens2,
            lambda_reg=lambda_reg,
            sigma_sqrt=sigma_sqrt,
            sigma_inv_sqrt=sigma_inv_sqrt,
            projection_heads=projection_heads,
            head_layout=head_layout,
            attention_kind="score-self-power-multistart-gain",
            target_mode=target_mode,
            scale_candidates=scale_candidates or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            score_mode="token-centered",
            extra_stats={
                "shared_eigenvalues": eigvals[: basis.shape[1]],
                "shared_trace": float(np.trace(stats["shared"])),
                "delta_trace": float(np.trace(stats["delta"])),
                "power_iterations": int(num_power_iters),
                "power_seed": int(seed),
                "score_objective_values": objective_values,
                "init_name": init_name,
            },
        )
        loss = float(model.get("train_fit_loss", np.inf))
        if best_loss is None or loss < best_loss:
            best_model = model
            best_loss = loss
            best_init_name = init_name

    best_model["selected_init_name"] = best_init_name
    return best_model


def fit_score_power_holdout_gain_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=8,
    seed=0,
    scale_candidates=None,
    holdout_fraction=0.2,
):
    score_tokens1 = _center_tokens_within_sample(tokens1)
    score_tokens2 = _center_tokens_within_sample(tokens2)
    flat1 = _flatten_tokens(score_tokens1)
    flat2 = _flatten_tokens(score_tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    basis, objective_values = _score_alignment_power_basis(
        score_tokens1,
        score_tokens2,
        sigma_inv_sqrt,
        rank=min(projection_rank, tokens1.shape[-1]),
        num_iters=num_power_iters,
        seed=seed,
        init_basis=eigvecs[:, : min(projection_rank, eigvecs.shape[1])],
    )
    head_indices = _projection_head_indices(basis.shape[1], basis.shape[1], num_heads)
    projection_heads = [basis[:, idxs] for idxs in head_indices]
    base_head_layout = _build_spectral_head_layout("score-self-power", len(projection_heads))
    scale_grid = scale_candidates or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

    num_samples = tokens1.shape[0]
    holdout_size = max(1, int(round(holdout_fraction * num_samples)))
    rng = np.random.default_rng(seed + 409)
    perm = rng.permutation(num_samples)
    val_idx = np.sort(perm[:holdout_size])
    fit_idx = np.sort(perm[holdout_size:])
    if fit_idx.size == 0:
        fit_idx = val_idx

    best_scale = None
    best_val_loss = None
    for scale in scale_grid:
        scaled_head_layout = [{**head, "score_scale": float(scale)} for head in base_head_layout]
        fit_model, _ = _build_self_attention_model(
            tokens1=tokens1[fit_idx],
            tokens2=tokens2[fit_idx],
            lambda_reg=lambda_reg,
            sigma_sqrt=sigma_sqrt,
            sigma_inv_sqrt=sigma_inv_sqrt,
            projection_heads=projection_heads,
            head_layout=scaled_head_layout,
            attention_kind="score-self-power-holdout-gain",
            target_mode=target_mode,
            score_mode="token-centered",
            extra_stats=None,
        )
        val_context1 = _spectral_attention_context(
            tokens1[val_idx],
            sigma_inv_sqrt,
            projection_heads,
            scaled_head_layout,
            local_bias=None,
            score_tokens=_center_tokens_within_sample(tokens1[val_idx]),
        )
        val_context2 = _spectral_attention_context(
            tokens2[val_idx],
            sigma_inv_sqrt,
            projection_heads,
            scaled_head_layout,
            local_bias=None,
            score_tokens=_center_tokens_within_sample(tokens2[val_idx]),
        )
        solved = {
            "output_map": fit_model["output_map"],
            "mix_scale": fit_model["mix_scale"],
            "residual_mode": fit_model["residual_mode"],
        }
        val_loss = _attention_solution_loss(
            val_context1,
            val_context2,
            tokens1[val_idx],
            tokens2[val_idx],
            solved,
            lambda_reg,
            target_mode,
        )
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_scale = float(scale)

    final_head_layout = [{**head, "score_scale": best_scale} for head in base_head_layout]
    final_model, full_loss = _build_self_attention_model(
        tokens1=tokens1,
        tokens2=tokens2,
        lambda_reg=lambda_reg,
        sigma_sqrt=sigma_sqrt,
        sigma_inv_sqrt=sigma_inv_sqrt,
        projection_heads=projection_heads,
        head_layout=final_head_layout,
        attention_kind="score-self-power-holdout-gain",
        target_mode=target_mode,
        score_mode="token-centered",
        extra_stats={
            "shared_eigenvalues": eigvals[: basis.shape[1]],
            "shared_trace": float(np.trace(stats["shared"])),
            "delta_trace": float(np.trace(stats["delta"])),
            "power_iterations": int(num_power_iters),
            "power_seed": int(seed),
            "score_objective_values": objective_values,
            "holdout_fraction": float(holdout_fraction),
            "holdout_seed": int(seed + 409),
            "holdout_val_loss": float(best_val_loss),
        },
    )
    final_model["selected_score_scales"] = {"all": best_scale}
    final_model["train_fit_loss"] = float(full_loss)
    return final_model


def fit_score_power_bagged_gain_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=8,
    seed=0,
    scale_candidates=None,
    num_bags=4,
    bag_fraction=0.7,
):
    score_tokens1 = _center_tokens_within_sample(tokens1)
    score_tokens2 = _center_tokens_within_sample(tokens2)
    flat1 = _flatten_tokens(score_tokens1)
    flat2 = _flatten_tokens(score_tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    rank = min(projection_rank, tokens1.shape[-1])
    num_samples = tokens1.shape[0]
    bag_size = max(rank, int(round(float(bag_fraction) * num_samples)))
    rng = np.random.default_rng(seed + 613)

    bases = []
    weights = []
    objective_summaries = []
    for bag_idx in range(int(max(1, num_bags))):
        subset_idx = np.sort(rng.choice(num_samples, size=min(bag_size, num_samples), replace=False))
        basis, objective_values = _score_alignment_power_basis(
            score_tokens1[subset_idx],
            score_tokens2[subset_idx],
            sigma_inv_sqrt,
            rank=rank,
            num_iters=num_power_iters,
            seed=seed + bag_idx,
            init_basis=eigvecs[:, : min(rank, eigvecs.shape[1])],
        )
        bases.append(basis)
        weight = max(float(np.mean(objective_values)), 1e-8)
        weights.append(weight)
        objective_summaries.append(weight)

    basis = _average_projector_basis(
        bases,
        rank=rank,
        weights=weights,
        prior_basis=eigvecs[:, :rank],
        prior_weight=0.25,
    )
    head_indices = _projection_head_indices(basis.shape[1], basis.shape[1], num_heads)
    projection_heads = [basis[:, idxs] for idxs in head_indices]
    head_layout = _build_spectral_head_layout("score-self-power", len(projection_heads))

    model = _fit_self_attention_with_scale_search(
        tokens1=tokens1,
        tokens2=tokens2,
        lambda_reg=lambda_reg,
        sigma_sqrt=sigma_sqrt,
        sigma_inv_sqrt=sigma_inv_sqrt,
        projection_heads=projection_heads,
        head_layout=head_layout,
        attention_kind="score-self-power-bagged-gain",
        target_mode=target_mode,
        scale_candidates=scale_candidates or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        score_mode="token-centered",
        extra_stats={
            "shared_eigenvalues": eigvals[: basis.shape[1]],
            "shared_trace": float(np.trace(stats["shared"])),
            "delta_trace": float(np.trace(stats["delta"])),
            "power_iterations": int(num_power_iters),
            "power_seed": int(seed),
            "bag_count": int(max(1, num_bags)),
            "bag_fraction": float(bag_fraction),
            "bag_seed": int(seed + 613),
            "bag_objective_means": np.asarray(objective_summaries, dtype=np.float64),
        },
    )
    return model


def fit_score_power_aligned_bagged_gain_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=8,
    seed=0,
    scale_candidates=None,
    num_bags=4,
    bag_fraction=0.7,
):
    score_tokens1 = _center_tokens_within_sample(tokens1)
    score_tokens2 = _center_tokens_within_sample(tokens2)
    flat1 = _flatten_tokens(score_tokens1)
    flat2 = _flatten_tokens(score_tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    rank = min(projection_rank, tokens1.shape[-1])
    reference_basis, reference_objective_values = _score_alignment_power_basis(
        score_tokens1,
        score_tokens2,
        sigma_inv_sqrt,
        rank=rank,
        num_iters=num_power_iters,
        seed=seed,
        init_basis=eigvecs[:, : min(rank, eigvecs.shape[1])],
    )

    num_samples = tokens1.shape[0]
    bag_size = max(rank, int(round(float(bag_fraction) * num_samples)))
    rng = np.random.default_rng(seed + 613)

    bases = []
    weights = []
    objective_summaries = []
    for bag_idx in range(int(max(1, num_bags))):
        subset_idx = np.sort(rng.choice(num_samples, size=min(bag_size, num_samples), replace=False))
        basis, objective_values = _score_alignment_power_basis(
            score_tokens1[subset_idx],
            score_tokens2[subset_idx],
            sigma_inv_sqrt,
            rank=rank,
            num_iters=num_power_iters,
            seed=seed + bag_idx,
            init_basis=reference_basis,
        )
        bases.append(basis)
        weight = max(float(np.mean(objective_values)), 1e-8)
        weights.append(weight)
        objective_summaries.append(weight)

    basis, alignment_scores = _average_aligned_bases(
        bases,
        reference_basis=reference_basis,
        weights=weights,
        prior_weight=0.25,
    )
    head_indices = _projection_head_indices(basis.shape[1], basis.shape[1], num_heads)
    projection_heads = [basis[:, idxs] for idxs in head_indices]
    head_layout = _build_spectral_head_layout("score-self-power", len(projection_heads))

    model = _fit_self_attention_with_scale_search(
        tokens1=tokens1,
        tokens2=tokens2,
        lambda_reg=lambda_reg,
        sigma_sqrt=sigma_sqrt,
        sigma_inv_sqrt=sigma_inv_sqrt,
        projection_heads=projection_heads,
        head_layout=head_layout,
        attention_kind="score-self-power-aligned-bagged-gain",
        target_mode=target_mode,
        scale_candidates=scale_candidates or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        score_mode="token-centered",
        extra_stats={
            "shared_eigenvalues": eigvals[: basis.shape[1]],
            "shared_trace": float(np.trace(stats["shared"])),
            "delta_trace": float(np.trace(stats["delta"])),
            "power_iterations": int(num_power_iters),
            "power_seed": int(seed),
            "reference_objective_values": reference_objective_values,
            "bag_count": int(max(1, num_bags)),
            "bag_fraction": float(bag_fraction),
            "bag_seed": int(seed + 613),
            "bag_objective_means": np.asarray(objective_summaries, dtype=np.float64),
            "bag_alignment_scores": alignment_scores,
        },
    )
    return model


def fit_score_power_bagged_shrink_gain_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=8,
    seed=0,
    scale_candidates=None,
    num_bags=4,
    bag_fraction=0.7,
    prior_weight_candidates=None,
):
    score_tokens1 = _center_tokens_within_sample(tokens1)
    score_tokens2 = _center_tokens_within_sample(tokens2)
    flat1 = _flatten_tokens(score_tokens1)
    flat2 = _flatten_tokens(score_tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    rank = min(projection_rank, tokens1.shape[-1])
    num_samples = tokens1.shape[0]
    bag_size = max(rank, int(round(float(bag_fraction) * num_samples)))
    rng = np.random.default_rng(seed + 613)

    bases = []
    weights = []
    objective_summaries = []
    for bag_idx in range(int(max(1, num_bags))):
        subset_idx = np.sort(rng.choice(num_samples, size=min(bag_size, num_samples), replace=False))
        basis, objective_values = _score_alignment_power_basis(
            score_tokens1[subset_idx],
            score_tokens2[subset_idx],
            sigma_inv_sqrt,
            rank=rank,
            num_iters=num_power_iters,
            seed=seed + bag_idx,
            init_basis=eigvecs[:, : min(rank, eigvecs.shape[1])],
        )
        bases.append(basis)
        weight = max(float(np.mean(objective_values)), 1e-8)
        weights.append(weight)
        objective_summaries.append(weight)

    best_model = None
    best_loss = None
    best_prior_weight = None
    prior_grid = prior_weight_candidates or [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    prior_basis = eigvecs[:, :rank]
    for prior_weight in prior_grid:
        basis = _average_projector_basis(
            bases,
            rank=rank,
            weights=weights,
            prior_basis=prior_basis,
            prior_weight=float(prior_weight),
        )
        head_indices = _projection_head_indices(basis.shape[1], basis.shape[1], num_heads)
        projection_heads = [basis[:, idxs] for idxs in head_indices]
        head_layout = _build_spectral_head_layout("score-self-power", len(projection_heads))

        model = _fit_self_attention_with_scale_search(
            tokens1=tokens1,
            tokens2=tokens2,
            lambda_reg=lambda_reg,
            sigma_sqrt=sigma_sqrt,
            sigma_inv_sqrt=sigma_inv_sqrt,
            projection_heads=projection_heads,
            head_layout=head_layout,
            attention_kind="score-self-power-bagged-shrink-gain",
            target_mode=target_mode,
            scale_candidates=scale_candidates or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            score_mode="token-centered",
            extra_stats={
                "shared_eigenvalues": eigvals[: basis.shape[1]],
                "shared_trace": float(np.trace(stats["shared"])),
                "delta_trace": float(np.trace(stats["delta"])),
                "power_iterations": int(num_power_iters),
                "power_seed": int(seed),
                "bag_count": int(max(1, num_bags)),
                "bag_fraction": float(bag_fraction),
                "bag_seed": int(seed + 613),
                "bag_objective_means": np.asarray(objective_summaries, dtype=np.float64),
                "prior_weight": float(prior_weight),
            },
        )
        loss = float(model.get("train_fit_loss", np.inf))
        if best_loss is None or loss < best_loss:
            best_model = model
            best_loss = loss
            best_prior_weight = float(prior_weight)

    best_model["selected_prior_weight"] = best_prior_weight
    return best_model


def fit_score_power_bagged_consensus_gain_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=8,
    seed=0,
    scale_candidates=None,
    num_bags=4,
    bag_fraction=0.7,
):
    score_tokens1 = _center_tokens_within_sample(tokens1)
    score_tokens2 = _center_tokens_within_sample(tokens2)
    flat1 = _flatten_tokens(score_tokens1)
    flat2 = _flatten_tokens(score_tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    rank = min(projection_rank, tokens1.shape[-1])
    num_samples = tokens1.shape[0]
    bag_size = max(rank, int(round(float(bag_fraction) * num_samples)))
    rng = np.random.default_rng(seed + 613)

    bases = []
    weights = []
    objective_summaries = []
    for bag_idx in range(int(max(1, num_bags))):
        subset_idx = np.sort(rng.choice(num_samples, size=min(bag_size, num_samples), replace=False))
        basis, objective_values = _score_alignment_power_basis(
            score_tokens1[subset_idx],
            score_tokens2[subset_idx],
            sigma_inv_sqrt,
            rank=rank,
            num_iters=num_power_iters,
            seed=seed + bag_idx,
            init_basis=eigvecs[:, : min(rank, eigvecs.shape[1])],
        )
        bases.append(basis)
        weight = max(float(np.mean(objective_values)), 1e-8)
        weights.append(weight)
        objective_summaries.append(weight)

    prior_basis = eigvecs[:, :rank]
    initial_basis = _average_projector_basis(
        bases,
        rank=rank,
        weights=weights,
        prior_basis=prior_basis,
        prior_weight=0.25,
    )
    consensus_overlaps = [
        _normalized_projector_overlap(basis, initial_basis)
        for basis in bases
    ]
    consensus_weights = [
        max(float(weight) * float(overlap) ** 2, 1e-8)
        for weight, overlap in zip(weights, consensus_overlaps)
    ]
    basis = _average_projector_basis(
        bases,
        rank=rank,
        weights=consensus_weights,
        prior_basis=prior_basis,
        prior_weight=0.25,
    )
    head_indices = _projection_head_indices(basis.shape[1], basis.shape[1], num_heads)
    projection_heads = [basis[:, idxs] for idxs in head_indices]
    head_layout = _build_spectral_head_layout("score-self-power", len(projection_heads))

    model = _fit_self_attention_with_scale_search(
        tokens1=tokens1,
        tokens2=tokens2,
        lambda_reg=lambda_reg,
        sigma_sqrt=sigma_sqrt,
        sigma_inv_sqrt=sigma_inv_sqrt,
        projection_heads=projection_heads,
        head_layout=head_layout,
        attention_kind="score-self-power-bagged-consensus-gain",
        target_mode=target_mode,
        scale_candidates=scale_candidates or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        score_mode="token-centered",
        extra_stats={
            "shared_eigenvalues": eigvals[: basis.shape[1]],
            "shared_trace": float(np.trace(stats["shared"])),
            "delta_trace": float(np.trace(stats["delta"])),
            "power_iterations": int(num_power_iters),
            "power_seed": int(seed),
            "bag_count": int(max(1, num_bags)),
            "bag_fraction": float(bag_fraction),
            "bag_seed": int(seed + 613),
            "bag_objective_means": np.asarray(objective_summaries, dtype=np.float64),
            "bag_consensus_overlaps": np.asarray(consensus_overlaps, dtype=np.float64),
            "bag_consensus_weights": np.asarray(consensus_weights, dtype=np.float64),
        },
    )
    return model


def fit_token_maxent_self_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    seed=0,
):
    score_tokens1 = _center_tokens_within_sample(tokens1)
    score_tokens2 = _center_tokens_within_sample(tokens2)
    flat1 = _flatten_tokens(score_tokens1)
    flat2 = _flatten_tokens(score_tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    basis = _maxent_stable_basis(
        eigvecs,
        eigvals,
        rank=min(projection_rank, eigvecs.shape[1]),
        seed=seed,
    )
    head_indices = _projection_head_indices(basis.shape[1], basis.shape[1], num_heads)
    projection_heads = [basis[:, idxs] for idxs in head_indices]
    head_layout = _build_spectral_head_layout("spectral-self-token-stats", len(projection_heads))
    return _fit_self_attention_with_projections(
        tokens1=tokens1,
        tokens2=tokens2,
        lambda_reg=lambda_reg,
        sigma_sqrt=sigma_sqrt,
        sigma_inv_sqrt=sigma_inv_sqrt,
        projection_heads=projection_heads,
        head_layout=head_layout,
        attention_kind="token-self-maxent",
        target_mode=target_mode,
        score_mode="raw",
        extra_stats={
            "shared_eigenvalues": eigvals[: basis.shape[1]],
            "shared_trace": float(np.trace(stats["shared"])),
            "delta_trace": float(np.trace(stats["delta"])),
            "maxent_seed": int(seed),
            "maxent_pool_rank": int(min(eigvecs.shape[1], max(projection_rank, 4 * max(1, projection_rank)))),
        },
    )


def _fit_score_power_self_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=8,
    seed=0,
    score_mode="token-centered",
    attention_kind="score-self-power",
    scale_candidates=None,
):
    score_tokens1 = _center_tokens_within_sample(tokens1)
    score_tokens2 = _center_tokens_within_sample(tokens2)
    flat1 = _flatten_tokens(score_tokens1)
    flat2 = _flatten_tokens(score_tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    basis, objective_values = _score_alignment_power_basis(
        score_tokens1,
        score_tokens2,
        sigma_inv_sqrt,
        rank=min(projection_rank, tokens1.shape[-1]),
        num_iters=num_power_iters,
        seed=seed,
        init_basis=eigvecs[:, : min(projection_rank, eigvecs.shape[1])],
    )
    head_indices = _projection_head_indices(basis.shape[1], basis.shape[1], num_heads)
    projection_heads = [basis[:, idxs] for idxs in head_indices]
    head_layout = _build_spectral_head_layout("spectral-self", len(projection_heads))

    fit_fn = _fit_self_attention_with_projections if scale_candidates is None else _fit_self_attention_with_scale_search
    fit_kwargs = {
        "tokens1": tokens1,
        "tokens2": tokens2,
        "lambda_reg": lambda_reg,
        "sigma_sqrt": sigma_sqrt,
        "sigma_inv_sqrt": sigma_inv_sqrt,
        "projection_heads": projection_heads,
        "head_layout": head_layout,
        "attention_kind": attention_kind,
        "target_mode": target_mode,
        "score_mode": score_mode,
        "extra_stats": {
            "shared_eigenvalues": eigvals[: basis.shape[1]],
            "shared_trace": float(np.trace(stats["shared"])),
            "delta_trace": float(np.trace(stats["delta"])),
            "power_iterations": int(num_power_iters),
            "power_seed": int(seed),
            "score_objective_values": objective_values,
        },
    }
    if scale_candidates is not None:
        fit_kwargs["scale_candidates"] = scale_candidates
    return fit_fn(
        **fit_kwargs,
    )


def fit_mixed_self_objective_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=24,
    seed=0,
):
    return _fit_mixed_self_objective_attention_from_token_pairs(
        tokens1,
        tokens2,
        lambda_reg=lambda_reg,
        total_rank=total_rank,
        num_heads=num_heads,
        target_mode=target_mode,
        num_power_iters=num_power_iters,
        seed=seed,
        scale_candidates=None,
    )


def fit_mixed_self_objective_scaled_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=24,
    seed=0,
    scale_candidates=None,
):
    return _fit_mixed_self_objective_attention_from_token_pairs(
        tokens1,
        tokens2,
        lambda_reg=lambda_reg,
        total_rank=total_rank,
        num_heads=num_heads,
        target_mode=target_mode,
        num_power_iters=num_power_iters,
        seed=seed,
        scale_candidates=scale_candidates or [0.5, 1.0, 2.0, 4.0],
    )


def fit_mixed_token_random_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    seed=0,
):
    score_tokens1 = _center_tokens_within_sample(tokens1)
    score_tokens2 = _center_tokens_within_sample(tokens2)
    flat1 = _flatten_tokens(score_tokens1)
    flat2 = _flatten_tokens(score_tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    token_rank = max(1, projection_rank // 2)
    random_rank = max(1, projection_rank - token_rank)
    token_basis = eigvecs[:, :token_rank]
    random_basis = _random_orthogonal_basis(tokens1.shape[-1], random_rank, seed)

    token_head_count = max(1, num_heads // 2)
    random_head_count = max(1, num_heads - token_head_count)
    token_indices = _projection_head_indices(token_basis.shape[1], token_basis.shape[1], token_head_count)
    random_indices = _projection_head_indices(random_basis.shape[1], random_basis.shape[1], random_head_count)

    projection_heads = []
    head_layout = []
    for idxs in token_indices:
        projection_heads.append(token_basis[:, idxs])
        head_layout.append(
            {
                "projection_index": len(projection_heads) - 1,
                "bias_kind": "global",
                "score_mode": "raw",
                "objective_family": "token-stats",
            }
        )
    for idxs in random_indices:
        projection_heads.append(random_basis[:, idxs])
        head_layout.append(
            {
                "projection_index": len(projection_heads) - 1,
                "bias_kind": "global",
                "score_mode": "raw",
                "objective_family": "random",
            }
        )

    return _fit_self_attention_with_projections(
        tokens1=tokens1,
        tokens2=tokens2,
        lambda_reg=lambda_reg,
        sigma_sqrt=sigma_sqrt,
        sigma_inv_sqrt=sigma_inv_sqrt,
        projection_heads=projection_heads,
        head_layout=head_layout,
        attention_kind="mixed-token-random",
        target_mode=target_mode,
        score_mode="mixed",
        extra_stats={
            "shared_eigenvalues": eigvals[: min(int(sum(head.shape[1] for head in projection_heads)), eigvals.shape[0])],
            "shared_trace": float(np.trace(stats["shared"])),
            "delta_trace": float(np.trace(stats["delta"])),
            "random_seed": int(seed),
        },
    )


def fit_head_pool_gain_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=2,
    target_mode="mean",
    num_power_iters=24,
    seed=0,
    scale_candidates=None,
):
    score_tokens1 = _center_tokens_within_sample(tokens1)
    score_tokens2 = _center_tokens_within_sample(tokens2)
    flat1 = _flatten_tokens(score_tokens1)
    flat2 = _flatten_tokens(score_tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    head_width = max(1, projection_rank // max(1, num_heads))

    power_rank = max(projection_rank, 2 * head_width)
    power_basis, power_objective_values = _score_alignment_power_basis(
        score_tokens1,
        score_tokens2,
        sigma_inv_sqrt,
        rank=min(power_rank, tokens1.shape[-1]),
        num_iters=num_power_iters,
        seed=seed,
        init_basis=eigvecs[:, : min(power_rank, eigvecs.shape[1])],
    )
    maxent_basis = _maxent_stable_basis(
        eigvecs,
        eigvals,
        rank=head_width,
        seed=seed,
    )
    random_basis = _random_orthogonal_basis(tokens1.shape[-1], head_width, seed)

    candidate_heads = []

    def add_candidate(name, projection, score_mode):
        if projection.shape[1] == 0:
            return
        candidate_heads.append(
            {
                "name": name,
                "projection": projection,
                "score_mode": score_mode,
            }
        )

    add_candidate("token-top", eigvecs[:, :head_width], "raw")
    if eigvecs.shape[1] >= 2 * head_width:
        add_candidate("token-next", eigvecs[:, head_width : 2 * head_width], "raw")
    add_candidate("power-top", power_basis[:, :head_width], "token-centered")
    if power_basis.shape[1] >= 2 * head_width:
        add_candidate("power-next", power_basis[:, head_width : 2 * head_width], "token-centered")
    add_candidate("maxent", maxent_basis[:, :head_width], "raw")
    add_candidate("random", random_basis[:, :head_width], "raw")

    best_model = None
    best_loss = None
    best_combo = None
    for combo in combinations(candidate_heads, max(1, num_heads)):
        projection_heads = [head["projection"] for head in combo]
        head_layout = [
            {
                "projection_index": head_idx,
                "bias_kind": "global",
                "score_mode": head["score_mode"],
                "candidate_name": head["name"],
            }
            for head_idx, head in enumerate(combo)
        ]
        model = _fit_self_attention_with_scale_search(
            tokens1=tokens1,
            tokens2=tokens2,
            lambda_reg=lambda_reg,
            sigma_sqrt=sigma_sqrt,
            sigma_inv_sqrt=sigma_inv_sqrt,
            projection_heads=projection_heads,
            head_layout=head_layout,
            attention_kind="head-pool-gain",
            target_mode=target_mode,
            scale_candidates=scale_candidates or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
            score_mode="mixed",
            extra_stats={
                "shared_eigenvalues": eigvals[: min(projection_rank, eigvals.shape[0])],
                "shared_trace": float(np.trace(stats["shared"])),
                "delta_trace": float(np.trace(stats["delta"])),
                "power_iterations": int(num_power_iters),
                "power_seed": int(seed),
                "score_objective_values": power_objective_values,
            },
        )
        loss = float(model.get("train_fit_loss", np.inf))
        if best_loss is None or loss < best_loss:
            best_model = model
            best_loss = loss
            best_combo = [head["name"] for head in combo]

    best_model["selected_head_candidates"] = best_combo
    return best_model


def _fit_mixed_self_objective_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    num_power_iters=24,
    seed=0,
    scale_candidates=None,
):
    score_tokens1 = _center_tokens_within_sample(tokens1)
    score_tokens2 = _center_tokens_within_sample(tokens2)
    flat1 = _flatten_tokens(score_tokens1)
    flat2 = _flatten_tokens(score_tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    projection_rank = _default_projection_rank(tokens1.shape[-1], num_heads) if total_rank is None else int(total_rank)
    token_rank = max(1, projection_rank // 2)
    power_rank = max(1, projection_rank - token_rank)
    if token_rank + power_rank > tokens1.shape[-1]:
        power_rank = max(1, tokens1.shape[-1] - token_rank)

    token_basis = eigvecs[:, :token_rank]
    power_basis, power_objective_values = _score_alignment_power_basis(
        score_tokens1,
        score_tokens2,
        sigma_inv_sqrt,
        rank=power_rank,
        num_iters=num_power_iters,
        seed=seed,
        init_basis=eigvecs[:, token_rank : token_rank + power_rank],
    )

    token_head_count = max(1, num_heads // 2)
    power_head_count = max(1, num_heads - token_head_count)
    token_indices = _projection_head_indices(token_basis.shape[1], token_basis.shape[1], token_head_count)
    power_indices = _projection_head_indices(power_basis.shape[1], power_basis.shape[1], power_head_count)

    projection_heads = []
    head_layout = []
    for idxs in token_indices:
        projection_heads.append(token_basis[:, idxs])
        head_layout.append(
            {
                "projection_index": len(projection_heads) - 1,
                "bias_kind": "global",
                "score_mode": "raw",
                "objective_family": "token-stats",
            }
        )
    for idxs in power_indices:
        projection_heads.append(power_basis[:, idxs])
        head_layout.append(
            {
                "projection_index": len(projection_heads) - 1,
                "bias_kind": "global",
                "score_mode": "token-centered",
                "objective_family": "score-power",
            }
        )
    fit_fn = _fit_self_attention_with_projections if scale_candidates is None else _fit_self_attention_with_scale_search
    fit_kwargs = {
        "tokens1": tokens1,
        "tokens2": tokens2,
        "lambda_reg": lambda_reg,
        "sigma_sqrt": sigma_sqrt,
        "sigma_inv_sqrt": sigma_inv_sqrt,
        "projection_heads": projection_heads,
        "head_layout": head_layout,
        "attention_kind": "mixed-self-objective" if scale_candidates is None else "mixed-self-objective-gain",
        "target_mode": target_mode,
        "score_mode": "mixed",
        "extra_stats": {
            "shared_eigenvalues": eigvals[: min(int(sum(head.shape[1] for head in projection_heads)), eigvals.shape[0])],
            "shared_trace": float(np.trace(stats["shared"])),
            "delta_trace": float(np.trace(stats["delta"])),
            "power_iterations": int(num_power_iters),
            "power_seed": int(seed),
            "score_objective_values": power_objective_values,
        },
    }
    if scale_candidates is not None:
        fit_kwargs["scale_candidates"] = scale_candidates
        fit_kwargs["scale_group_key"] = "objective_family"
    return fit_fn(**fit_kwargs)


def fit_spectral_self_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    center_values=False,
    whiten_values=False,
):
    return _fit_spectral_attention_from_tokens(
        tokens1,
        tokens2,
        lambda_reg=lambda_reg,
        attention_kind="spectral-self",
        total_rank=total_rank,
        num_heads=num_heads,
        target_mode=target_mode,
        center_values=center_values,
        whiten_values=whiten_values,
    )


def fit_spectral_self_attention_interleaved_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    center_values=False,
):
    return _fit_spectral_attention_from_tokens(
        tokens1,
        tokens2,
        lambda_reg=lambda_reg,
        attention_kind="spectral-self",
        total_rank=total_rank,
        num_heads=num_heads,
        target_mode=target_mode,
        center_values=center_values,
        head_split_mode="interleaved",
    )


def fit_local_spectral_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    local_sigma=1.5,
    center_values=False,
):
    return _fit_spectral_attention_from_tokens(
        tokens1,
        tokens2,
        lambda_reg=lambda_reg,
        attention_kind="local-spectral",
        total_rank=total_rank,
        num_heads=num_heads,
        target_mode=target_mode,
        local_sigma=local_sigma,
        center_values=center_values,
    )


def fit_hybrid_spectral_attention_from_token_pairs(
    tokens1,
    tokens2,
    lambda_reg,
    total_rank=None,
    num_heads=4,
    target_mode="mean",
    local_sigma=1.5,
    center_values=False,
):
    return _fit_spectral_attention_from_tokens(
        tokens1,
        tokens2,
        lambda_reg=lambda_reg,
        attention_kind="hybrid-spectral" if target_mode != "bt-residual" else "hybrid-spectral-bt",
        total_rank=total_rank,
        num_heads=num_heads,
        target_mode=target_mode,
        local_sigma=local_sigma,
        center_values=center_values,
    )


def apply_spectral_self_attention(tokens, model):
    return _apply_spectral_attention(tokens, model)


def apply_local_spectral_attention(tokens, model):
    return _apply_spectral_attention(tokens, model)


def apply_hybrid_spectral_attention(tokens, model):
    return _apply_spectral_attention(tokens, model)


def apply_random_self_attention(tokens, model):
    if model["attention_kind"] == "random-self-ridge":
        return _apply_spectral_attention(tokens, model)

    contexts = _spectral_head_contexts(
        tokens,
        tokens,
        tokens,
        model["sigma_inv_sqrt"],
        model["projection_heads"],
        model["head_layout"],
    )
    context = _aggregate_head_contexts(contexts)
    return tokens + model.get("mix_scale", 1.0) * context


def _fit_landmark_attention_from_tokens(tokens1, tokens2, lambda_reg, num_landmarks=None, target_mode="residual"):
    flat1 = _flatten_tokens(tokens1)
    flat2 = _flatten_tokens(tokens2)
    stats = cfbt.compute_paired_stats(flat1, flat2)
    sigma_bar = stats["sigma_bar"]
    sigma_sqrt, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    token_count = tokens1.shape[1]
    landmark_count = min(
        num_landmarks or token_count,
        eigvecs.shape[1],
    )
    keys = eigvecs[:, :landmark_count]
    temperature = 1.0 / np.sqrt(tokens1.shape[2])

    weights1, _ = _token_attention_weights(tokens1, sigma_inv_sqrt, keys, temperature)
    weights2, _ = _token_attention_weights(tokens2, sigma_inv_sqrt, keys, temperature)
    flat_weights1 = _flatten_tokens(weights1)
    flat_weights2 = _flatten_tokens(weights2)

    if target_mode == "mean":
        target1 = 0.5 * (tokens1 + tokens2)
        target2 = target1
        residual_mode = False
    elif target_mode == "residual":
        target1 = 0.5 * (tokens2 - tokens1)
        target2 = 0.5 * (tokens1 - tokens2)
        residual_mode = True
    else:
        raise ValueError(f"Unknown attention target_mode: {target_mode}")

    targets = np.concatenate([_flatten_tokens(target1), _flatten_tokens(target2)], axis=0)
    design = np.concatenate([flat_weights1, flat_weights2], axis=0)
    gram = design.T @ design
    rhs = design.T @ targets
    values = np.linalg.solve(
        gram + lambda_reg * np.eye(gram.shape[0], dtype=np.float64),
        rhs,
    )

    return {
        "sigma_inv_sqrt": sigma_inv_sqrt,
        "sigma_sqrt": sigma_sqrt,
        "keys": keys,
        "values": values,
        "temperature": temperature,
        "target_mode": target_mode,
        "residual_mode": residual_mode,
        "landmark_count": landmark_count,
        "shared_eigenvalues": eigvals[:landmark_count],
        "shared_trace": float(np.trace(stats["shared"])),
        "delta_trace": float(np.trace(stats["delta"])),
    }


def fit_landmark_attention_from_token_pairs(tokens1, tokens2, lambda_reg, num_landmarks=None, target_mode="mean"):
    model = _fit_landmark_attention_from_tokens(
        tokens1,
        tokens2,
        lambda_reg=lambda_reg,
        num_landmarks=num_landmarks,
        target_mode=target_mode,
    )
    weights1, _ = _token_attention_weights(tokens1, model["sigma_inv_sqrt"], model["keys"], model["temperature"])
    weights2, _ = _token_attention_weights(tokens2, model["sigma_inv_sqrt"], model["keys"], model["temperature"])
    pred1 = (_flatten_tokens(weights1) @ model["values"]).reshape(tokens1.shape)
    pred2 = (_flatten_tokens(weights2) @ model["values"]).reshape(tokens2.shape)
    if model["residual_mode"]:
        model["mix_scale"] = _fit_residual_scale(pred1, pred2, 0.5 * (tokens2 - tokens1), 0.5 * (tokens1 - tokens2))
    else:
        target = 0.5 * (tokens1 + tokens2)
        model["mix_scale"] = _fit_mean_blend(tokens1, tokens2, pred1, pred2, target, target)
    model["parameter_count"] = int(model["sigma_inv_sqrt"].size + model["keys"].size + model["values"].size)
    return model


def fit_landmark_attention_from_pairs(H1, H2, lambda_reg, num_landmarks=None, target_mode="residual"):
    layout = infer_token_layout(H1.shape[1])
    tokens1 = vectors_to_tokens(H1, layout)
    tokens2 = vectors_to_tokens(H2, layout)
    model = _fit_landmark_attention_from_tokens(
        tokens1,
        tokens2,
        lambda_reg=lambda_reg,
        num_landmarks=num_landmarks,
        target_mode=target_mode,
    )
    weights1, _ = _token_attention_weights(tokens1, model["sigma_inv_sqrt"], model["keys"], model["temperature"])
    weights2, _ = _token_attention_weights(tokens2, model["sigma_inv_sqrt"], model["keys"], model["temperature"])
    pred1 = (_flatten_tokens(weights1) @ model["values"]).reshape(tokens1.shape)
    pred2 = (_flatten_tokens(weights2) @ model["values"]).reshape(tokens2.shape)
    if model["residual_mode"]:
        model["mix_scale"] = _fit_residual_scale(pred1, pred2, 0.5 * (tokens2 - tokens1), 0.5 * (tokens1 - tokens2))
    else:
        target = 0.5 * (tokens1 + tokens2)
        model["mix_scale"] = _fit_mean_blend(tokens1, tokens2, pred1, pred2, target, target)
    model["layout"] = layout
    model["parameter_count"] = int(model["sigma_inv_sqrt"].size + model["keys"].size + model["values"].size)
    return model


def fit_axial_landmark_attention_from_pairs(H1, H2, lambda_reg, num_landmarks=None, target_mode="residual"):
    layout = infer_token_layout(H1.shape[1])
    if layout["mode"] != "row-image":
        return fit_landmark_attention_from_pairs(
            H1,
            H2,
            lambda_reg=lambda_reg,
            num_landmarks=num_landmarks,
            target_mode=target_mode,
        )

    side = layout["side"]
    row_model = _fit_landmark_attention_from_tokens(
        _axis_tokens_from_vectors(H1, side, "row"),
        _axis_tokens_from_vectors(H2, side, "row"),
        lambda_reg=lambda_reg,
        num_landmarks=num_landmarks,
        target_mode=target_mode,
    )
    col_model = _fit_landmark_attention_from_tokens(
        _axis_tokens_from_vectors(H1, side, "col"),
        _axis_tokens_from_vectors(H2, side, "col"),
        lambda_reg=lambda_reg,
        num_landmarks=num_landmarks,
        target_mode=target_mode,
    )
    row_t1 = _axis_tokens_from_vectors(H1, side, "row")
    row_t2 = _axis_tokens_from_vectors(H2, side, "row")
    col_t1 = _axis_tokens_from_vectors(H1, side, "col")
    col_t2 = _axis_tokens_from_vectors(H2, side, "col")
    row_w1, _ = _token_attention_weights(row_t1, row_model["sigma_inv_sqrt"], row_model["keys"], row_model["temperature"])
    row_w2, _ = _token_attention_weights(row_t2, row_model["sigma_inv_sqrt"], row_model["keys"], row_model["temperature"])
    col_w1, _ = _token_attention_weights(col_t1, col_model["sigma_inv_sqrt"], col_model["keys"], col_model["temperature"])
    col_w2, _ = _token_attention_weights(col_t2, col_model["sigma_inv_sqrt"], col_model["keys"], col_model["temperature"])
    row_pred1 = (_flatten_tokens(row_w1) @ row_model["values"]).reshape(row_t1.shape)
    row_pred2 = (_flatten_tokens(row_w2) @ row_model["values"]).reshape(row_t2.shape)
    col_pred1 = (_flatten_tokens(col_w1) @ col_model["values"]).reshape(col_t1.shape)
    col_pred2 = (_flatten_tokens(col_w2) @ col_model["values"]).reshape(col_t2.shape)
    if row_model["residual_mode"]:
        row_model["mix_scale"] = _fit_residual_scale(row_pred1, row_pred2, 0.5 * (row_t2 - row_t1), 0.5 * (row_t1 - row_t2))
        col_model["mix_scale"] = _fit_residual_scale(col_pred1, col_pred2, 0.5 * (col_t2 - col_t1), 0.5 * (col_t1 - col_t2))
    else:
        row_target = 0.5 * (row_t1 + row_t2)
        col_target = 0.5 * (col_t1 + col_t2)
        row_model["mix_scale"] = _fit_mean_blend(row_t1, row_t2, row_pred1, row_pred2, row_target, row_target)
        col_model["mix_scale"] = _fit_mean_blend(col_t1, col_t2, col_pred1, col_pred2, col_target, col_target)
    return {
        "layout": layout,
        "axis_models": {"row": row_model, "col": col_model},
        "parameter_count": int(
            row_model["sigma_inv_sqrt"].size
            + row_model["keys"].size
            + row_model["values"].size
            + col_model["sigma_inv_sqrt"].size
            + col_model["keys"].size
            + col_model["values"].size
        ),
        "landmark_count": int(row_model["landmark_count"]),
        "shared_eigenvalues": 0.5 * (row_model["shared_eigenvalues"] + col_model["shared_eigenvalues"]),
        "shared_trace": 0.5 * (row_model["shared_trace"] + col_model["shared_trace"]),
        "delta_trace": 0.5 * (row_model["delta_trace"] + col_model["delta_trace"]),
    }


def _apply_token_attention(tokens, model):
    weights, _ = _token_attention_weights(tokens, model["sigma_inv_sqrt"], model["keys"], model["temperature"])
    attended = _flatten_tokens(weights) @ model["values"]
    attended = attended.reshape(tokens.shape[0], tokens.shape[1], tokens.shape[2])

    if model["residual_mode"]:
        combined = tokens + model.get("mix_scale", 1.0) * attended
    else:
        alpha = model.get("mix_scale", 0.0)
        combined = alpha * tokens + (1.0 - alpha) * attended
    return combined


def apply_token_attention(tokens, model):
    return _apply_token_attention(tokens, model)


def apply_landmark_attention(X, model, activation="relu"):
    layout = model["layout"]
    tokens = vectors_to_tokens(X, layout)
    combined = _apply_token_attention(tokens, model)
    combined = tokens_to_vectors(combined, layout)
    return cfbt.apply_activation(combined, activation)


def apply_axial_landmark_attention(X, model, activation="relu"):
    layout = model["layout"]
    if "axis_models" not in model:
        return apply_landmark_attention(X, model, activation=activation)

    side = layout["side"]
    row_tokens = _axis_tokens_from_vectors(X, side, "row")
    col_tokens = _axis_tokens_from_vectors(X, side, "col")
    row_out = _vectors_from_axis_tokens(_apply_token_attention(row_tokens, model["axis_models"]["row"]), side, "row")
    col_out = _vectors_from_axis_tokens(_apply_token_attention(col_tokens, model["axis_models"]["col"]), side, "col")
    combined = 0.5 * (row_out + col_out)
    return cfbt.apply_activation(combined, activation)


def fit_global_landmark_attention_from_pairs(H1, H2, lambda_reg, num_landmarks=64, target_mode="residual"):
    stats = cfbt.compute_paired_stats(H1, H2)
    sigma_bar = stats["sigma_bar"]
    _, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_matrix = sigma_inv_sqrt @ stats["shared"] @ sigma_inv_sqrt
    shared_matrix = _symmetrize(shared_matrix)
    eigvals, eigvecs = eigh(shared_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    landmark_count = min(num_landmarks, H1.shape[1], eigvecs.shape[1])
    keys = eigvecs[:, :landmark_count]
    temperature = 1.0 / np.sqrt(H1.shape[1])

    weights1, _ = _global_attention_weights(H1, sigma_inv_sqrt, keys, temperature)
    weights2, _ = _global_attention_weights(H2, sigma_inv_sqrt, keys, temperature)

    if target_mode == "mean":
        target1 = 0.5 * (H1 + H2)
        target2 = target1
        residual_mode = False
    elif target_mode == "residual":
        target1 = 0.5 * (H2 - H1)
        target2 = 0.5 * (H1 - H2)
        residual_mode = True
    else:
        raise ValueError(f"Unknown attention target_mode: {target_mode}")

    design = np.concatenate([weights1, weights2], axis=0)
    targets = np.concatenate([target1, target2], axis=0)
    gram = design.T @ design
    rhs = design.T @ targets
    values = np.linalg.solve(
        gram + lambda_reg * np.eye(gram.shape[0], dtype=np.float64),
        rhs,
    )

    pred1 = weights1 @ values
    pred2 = weights2 @ values
    if target_mode == "residual":
        mix_scale = _fit_residual_scale(pred1, pred2, target1, target2)
    else:
        mix_scale = _fit_mean_blend(H1, H2, pred1, pred2, target1, target2)

    return {
        "sigma_inv_sqrt": sigma_inv_sqrt,
        "keys": keys,
        "values": values,
        "temperature": temperature,
        "residual_mode": residual_mode,
        "target_mode": target_mode,
        "mix_scale": mix_scale,
        "landmark_count": landmark_count,
        "shared_eigenvalues": eigvals[:landmark_count],
        "shared_trace": float(np.trace(stats["shared"])),
        "delta_trace": float(np.trace(stats["delta"])),
        "parameter_count": int(sigma_inv_sqrt.size + keys.size + values.size),
    }


def apply_global_landmark_attention(X, model, activation="relu"):
    weights, _ = _global_attention_weights(X, model["sigma_inv_sqrt"], model["keys"], model["temperature"])
    attended = weights @ model["values"]
    if model["residual_mode"]:
        combined = X + model.get("mix_scale", 1.0) * attended
    else:
        alpha = model.get("mix_scale", 0.0)
        combined = alpha * X + (1.0 - alpha) * attended
    return cfbt.apply_activation(combined, activation)


def _orthogonal_class_codes(num_classes, dim, seed):
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((dim, num_classes)))
    return basis.T


def fit_supervised_prototype_attention(H, y, lambda_reg, seed):
    num_classes = int(np.max(y) + 1)
    sigma = cfbt.covariance(H - H.mean(axis=0, keepdims=True))
    _, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma, cfbt.REG_EPS)

    prototypes = np.zeros((num_classes, H.shape[1]), dtype=np.float64)
    for class_idx in range(num_classes):
        mask = y == class_idx
        prototypes[class_idx] = H[mask].mean(axis=0)

    whitened_proto = prototypes @ sigma_inv_sqrt
    proto_norm = np.maximum(np.linalg.norm(whitened_proto, axis=1, keepdims=True), 1e-8)
    whitened_proto = whitened_proto / proto_norm

    temperature = 1.0 / np.sqrt(H.shape[1])
    whitened_h = H @ sigma_inv_sqrt
    scores = temperature * (whitened_h @ whitened_proto.T)
    weights = softmax_rows(scores)

    class_codes = _orthogonal_class_codes(num_classes, H.shape[1], seed=seed)
    targets = class_codes[y]
    gram = weights.T @ weights
    rhs = weights.T @ targets
    values = np.linalg.solve(
        gram + lambda_reg * np.eye(gram.shape[0], dtype=np.float64),
        rhs,
    )
    pred = weights @ values
    alpha = _fit_mean_blend(H, H, pred, pred, targets, targets)

    return {
        "sigma_inv_sqrt": sigma_inv_sqrt,
        "prototypes": whitened_proto,
        "values": values,
        "temperature": temperature,
        "mix_scale": alpha,
        "num_classes": num_classes,
        "parameter_count": int(sigma_inv_sqrt.size + whitened_proto.size + values.size),
    }


def apply_supervised_prototype_attention(X, model, activation="relu"):
    whitened = X @ model["sigma_inv_sqrt"]
    scores = model["temperature"] * (whitened @ model["prototypes"].T)
    weights = softmax_rows(scores)
    attended = weights @ model["values"]
    alpha = model.get("mix_scale", 0.0)
    combined = alpha * X + (1.0 - alpha) * attended
    return cfbt.apply_activation(combined, activation)


def fit_memory_attention_from_pairs(H1, H2, lambda_reg, num_memories=256, target_mode="mean", seed=0):
    stats = cfbt.compute_paired_stats(H1, H2)
    sigma_bar = stats["sigma_bar"]
    _, sigma_inv_sqrt = cfbt.sqrt_and_inv_sqrt_psd(sigma_bar, cfbt.REG_EPS)

    shared_points = 0.5 * (H1 + H2)
    memory_count = min(num_memories, shared_points.shape[0])
    rng = np.random.default_rng(seed)
    memory_idx = rng.choice(shared_points.shape[0], size=memory_count, replace=False)
    memories = _normalized_features(shared_points[memory_idx] @ sigma_inv_sqrt)

    whitened_h1 = _normalized_features(H1 @ sigma_inv_sqrt)
    whitened_h2 = _normalized_features(H2 @ sigma_inv_sqrt)
    temperature = 1.0 / np.sqrt(H1.shape[1])
    weights1 = softmax_rows(temperature * (whitened_h1 @ memories.T))
    weights2 = softmax_rows(temperature * (whitened_h2 @ memories.T))

    if target_mode == "mean":
        target1 = 0.5 * (H1 + H2)
        target2 = target1
        residual_mode = False
    elif target_mode == "residual":
        target1 = 0.5 * (H2 - H1)
        target2 = 0.5 * (H1 - H2)
        residual_mode = True
    else:
        raise ValueError(f"Unknown attention target_mode: {target_mode}")

    design = np.concatenate([weights1, weights2], axis=0)
    targets = np.concatenate([target1, target2], axis=0)
    gram = design.T @ design
    rhs = design.T @ targets
    values = np.linalg.solve(
        gram + lambda_reg * np.eye(gram.shape[0], dtype=np.float64),
        rhs,
    )

    pred1 = weights1 @ values
    pred2 = weights2 @ values
    if residual_mode:
        mix_scale = _fit_residual_scale(pred1, pred2, target1, target2)
    else:
        mix_scale = _fit_mean_blend(H1, H2, pred1, pred2, target1, target2)

    return {
        "sigma_inv_sqrt": sigma_inv_sqrt,
        "memories": memories,
        "values": values,
        "temperature": temperature,
        "residual_mode": residual_mode,
        "mix_scale": mix_scale,
        "memory_count": memory_count,
        "shared_trace": float(np.trace(stats["shared"])),
        "delta_trace": float(np.trace(stats["delta"])),
        "parameter_count": int(sigma_inv_sqrt.size + memories.size + values.size),
    }


def apply_memory_attention(X, model, activation="relu"):
    whitened = _normalized_features(X @ model["sigma_inv_sqrt"])
    weights = softmax_rows(model["temperature"] * (whitened @ model["memories"].T))
    attended = weights @ model["values"]
    if model["residual_mode"]:
        combined = X + model.get("mix_scale", 1.0) * attended
    else:
        alpha = model.get("mix_scale", 0.0)
        combined = alpha * X + (1.0 - alpha) * attended
    return cfbt.apply_activation(combined, activation)
