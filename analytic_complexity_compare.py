import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

import greedy_fullwidth_whitened_cov as gwc
import spectral_gap_study as sgs


REPEATS = 5


def median_time(fn, repeats):
    result = None
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings)), result


def sym(matrix):
    return 0.5 * (matrix + matrix.T)


def fit_whitened_pca(stats, d):
    sigma = sym(stats["sigma"])
    evals, evecs = eigh(sigma)
    basis = evecs[:, -d:]
    top_evals = np.maximum(evals[-d:], gwc.REG_EPS)
    w = (basis / np.sqrt(top_evals)).T
    return {
        "W": w,
        "basis": basis,
        "evals": evals,
    }


def pca_stats(X):
    return {
        "sigma": gwc.covariance(X),
    }


def pca_gap(evals, d):
    if d >= len(evals):
        return 0.0
    return float(evals[-d] - evals[-d - 1])


def bottom_gap(evals, d):
    if d >= len(evals):
        return 0.0
    return float(evals[d] - evals[d - 1])


def condition_number(matrix):
    evals = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    evals = np.maximum(evals, 1e-12)
    return float(np.max(evals) / np.min(evals))


def analyze_pca(stats, d):
    sigma = sym(stats["sigma"])
    evals = np.linalg.eigvalsh(sigma)
    return {
        "solver": "symmetric eigendecomposition",
        "objective_type": "top-eigenspace of covariance",
        "convergence": "global optimum; numerical stability governed by the top-d eigengap",
        "eigengap": pca_gap(evals, d),
        "condition_number": condition_number(sigma + gwc.REG_EPS * np.eye(sigma.shape[0], dtype=np.float64)),
        "time_complexity": "O(n p^2 + p^3)",
        "complexity_note": "one covariance build and one dense symmetric eigendecomposition",
    }


def analyze_whitened_pca(stats, d):
    sigma = sym(stats["sigma"])
    evals = np.linalg.eigvalsh(sigma)
    return {
        "solver": "symmetric eigendecomposition plus eigenvalue rescaling",
        "objective_type": "top-eigenspace of covariance with whitening on the selected modes",
        "convergence": "same eigenproblem as PCA; whitening adds conditioning sensitivity on small selected eigenvalues",
        "eigengap": pca_gap(evals, d),
        "condition_number": condition_number(sigma + gwc.REG_EPS * np.eye(sigma.shape[0], dtype=np.float64)),
        "time_complexity": "O(n p^2 + p^3)",
        "complexity_note": "same eigensolve as PCA plus diagonal rescaling on the selected modes",
    }


def analyze_hard_whitened(stats, d):
    sigma_bar = sym(stats["sigma_bar"])
    delta = sym(stats["delta"])
    evals_sigma, evecs_sigma = eigh(sigma_bar)
    evals_sigma = np.maximum(evals_sigma, gwc.REG_EPS)
    sigma_inv_sqrt = (evecs_sigma / np.sqrt(evals_sigma)) @ evecs_sigma.T
    n_matrix = 0.5 * (sigma_inv_sqrt @ delta @ sigma_inv_sqrt + sigma_inv_sqrt @ delta @ sigma_inv_sqrt.T)
    evals_n = np.linalg.eigvalsh(n_matrix)
    return {
        "solver": "whitening eigendecomposition plus bottom-eigenspace of the normalized disagreement matrix",
        "objective_type": "hard-whitened invariance",
        "convergence": "global optimum of the spectral surrogate, but sensitive to sigma_bar conditioning and degenerate when delta approaches zero",
        "eigengap": bottom_gap(evals_n, d),
        "condition_number": float(np.max(evals_sigma) / np.min(evals_sigma)),
        "time_complexity": "O(K n p^2 + p^3)",
        "complexity_note": "pair statistics plus one whitening eigendecomposition and one eigensolve of the normalized disagreement matrix",
    }


def analyze_auto_fisher(stats, d):
    sigma = sym(stats["sigma"])
    delta = sym(stats["delta"])
    p = sigma.shape[0]
    floor = float(np.trace(delta) / p + 1e-6 * np.trace(sigma) / p)
    denom = delta + floor * np.eye(p, dtype=np.float64)
    evals, _ = eigh(sigma, denom)
    return {
        "solver": "generalized symmetric eigendecomposition",
        "objective_type": "total-vs-within-orbit generalized Rayleigh quotient",
        "convergence": "global optimum once the denominator is regularized; numerical sensitivity set by the generalized eigengap and denominator conditioning",
        "eigengap": pca_gap(evals, d),
        "condition_number": condition_number(denom),
        "time_complexity": "O(K n p^2 + p^3)",
        "floor": floor,
        "complexity_note": "pair statistics plus one dense generalized symmetric eigendecomposition",
    }


def analyze_one_parameter_layer(stats, lambda_reg):
    sigma_bar = sym(stats["sigma_bar"])
    delta = sym(stats["delta"])
    _, sigma_inv_sqrt = gwc.sqrt_and_inv_sqrt_psd(sigma_bar, gwc.REG_EPS)
    m_matrix = sym(sigma_inv_sqrt @ delta @ sigma_inv_sqrt)
    lhs = m_matrix + lambda_reg * np.eye(m_matrix.shape[0], dtype=np.float64)
    eigvals_m = np.linalg.eigvalsh(m_matrix)
    gains = lambda_reg / (np.maximum(eigvals_m, 0.0) + lambda_reg)
    return {
        "solver": "whitening eigendecomposition and full-space spectral filter",
        "objective_type": "one-parameter whitened quadratic surrogate tr(G^T M G) + lambda ||G-I||_F^2",
        "convergence": "unique global minimizer for any lambda > 0; no iterative optimization, only dense linear algebra",
        "condition_number": condition_number(lhs),
        "gain_min": float(np.min(gains)),
        "gain_max": float(np.max(gains)),
        "time_complexity": "O(n p^2 + p^3)",
        "complexity_note": "same asymptotic order as PCA, but with two eigendecompositions and several dense matrix multiplications, so the cubic constant is much larger",
        "whitened_delta_min": float(np.min(eigvals_m)),
        "whitened_delta_max": float(np.max(eigvals_m)),
        "lambda": lambda_reg,
    }


def pca_solver_core(stats, d):
    return sgs.fit_pca(stats, d)


def whitened_pca_solver_core(stats, d):
    return fit_whitened_pca(stats, d)


def hard_whitened_solver_core(stats, d):
    return sgs.fit_hard_whitened_invariance(stats, d)


def auto_fisher_solver_core(stats, d):
    return sgs.fit_auto_fisher(stats, d)


def one_parameter_solver_core(stats, lambda_reg):
    return gwc.fit_whitened_cov_layer(stats, lambda_reg=lambda_reg)


def pca_breakdown(X, d, repeats):
    covariance_time, sigma = median_time(lambda: gwc.covariance(X), repeats)
    eig_time, eig_result = median_time(lambda: eigh(sym(sigma)), repeats)
    evals, evecs = eig_result
    project_time, _ = median_time(lambda: evecs[:, -d:], repeats)
    return {
        "covariance": covariance_time,
        "eigendecomposition": eig_time,
        "projection_extract": project_time,
        "total_measured": covariance_time + eig_time + project_time,
    }


def whitened_pca_breakdown(X, d, repeats):
    covariance_time, sigma = median_time(lambda: gwc.covariance(X), repeats)
    eig_time, eig_result = median_time(lambda: eigh(sym(sigma)), repeats)
    evals, evecs = eig_result
    basis = evecs[:, -d:]
    top_evals = np.maximum(evals[-d:], gwc.REG_EPS)
    rescale_time, _ = median_time(lambda: (basis / np.sqrt(top_evals)).T, repeats)
    return {
        "covariance": covariance_time,
        "eigendecomposition": eig_time,
        "whitening_rescale": rescale_time,
        "total_measured": covariance_time + eig_time + rescale_time,
    }


def hard_whitened_breakdown(base, family, d, repeats):
    stats_time, stats = median_time(lambda: sgs.compute_pair_statistics(base, family), repeats)
    sigma_bar = sym(stats["sigma_bar"])
    delta = sym(stats["delta"])
    sigma_eigh_time, sigma_eigh = median_time(lambda: eigh(sigma_bar), repeats)
    evals_sigma, evecs_sigma = sigma_eigh
    evals_sigma = np.maximum(evals_sigma, gwc.REG_EPS)
    inv_sqrt_time, sigma_inv_sqrt = median_time(
        lambda: (evecs_sigma / np.sqrt(evals_sigma)) @ evecs_sigma.T,
        repeats,
    )
    n_form_time, n_matrix = median_time(lambda: sym(sigma_inv_sqrt @ delta @ sigma_inv_sqrt), repeats)
    n_eigh_time, _ = median_time(lambda: eigh(n_matrix), repeats)
    return {
        "pair_statistics": stats_time,
        "sigma_bar_eigendecomposition": sigma_eigh_time,
        "sigma_bar_inverse_sqrt": inv_sqrt_time,
        "form_n_matrix": n_form_time,
        "n_eigendecomposition": n_eigh_time,
        "total_measured": stats_time + sigma_eigh_time + inv_sqrt_time + n_form_time + n_eigh_time,
    }


def auto_fisher_breakdown(base, family, d, repeats):
    stats_time, stats = median_time(lambda: sgs.compute_pair_statistics(base, family), repeats)
    sigma = sym(stats["sigma"])
    delta = sym(stats["delta"])
    p = sigma.shape[0]
    floor = float(np.trace(delta) / p + 1e-6 * np.trace(sigma) / p)
    denom_form_time, denom = median_time(lambda: delta + floor * np.eye(p, dtype=np.float64), repeats)
    gen_eigh_time, _ = median_time(lambda: eigh(sigma, denom), repeats)
    return {
        "pair_statistics": stats_time,
        "denominator_build": denom_form_time,
        "generalized_eigendecomposition": gen_eigh_time,
        "total_measured": stats_time + denom_form_time + gen_eigh_time,
        "floor": floor,
    }


def one_parameter_breakdown(view1, view2, lambda_reg, repeats):
    stats_time, stats = median_time(lambda: gwc.compute_paired_stats(view1, view2), repeats)
    sigma_bar = sym(stats["sigma_bar"])
    delta = sym(stats["delta"])
    sigma_eigh_time, sigma_eigh = median_time(lambda: eigh(sigma_bar), repeats)
    evals_sigma, evecs_sigma = sigma_eigh
    evals_sigma = np.maximum(evals_sigma, gwc.REG_EPS)
    sqrt_inv_time, sqrt_inv = median_time(
        lambda: (
            (evecs_sigma * np.sqrt(evals_sigma)) @ evecs_sigma.T,
            (evecs_sigma / np.sqrt(evals_sigma)) @ evecs_sigma.T,
        ),
        repeats,
    )
    sigma_sqrt, sigma_inv_sqrt = sqrt_inv
    m_form_time, m_matrix = median_time(lambda: sym(sigma_inv_sqrt @ delta @ sigma_inv_sqrt), repeats)
    m_eigh_time, m_eigh = median_time(lambda: eigh(m_matrix), repeats)
    eigvals_m, eigvecs_m = m_eigh
    reconstruct_time, _ = median_time(
        lambda: (
            sigma_sqrt
            @ ((eigvecs_m * (lambda_reg / (np.maximum(eigvals_m, 0.0) + lambda_reg))) @ eigvecs_m.T)
            @ sigma_inv_sqrt
        ),
        repeats,
    )
    return {
        "pair_statistics": stats_time,
        "sigma_bar_eigendecomposition": sigma_eigh_time,
        "sigma_bar_sqrt_and_inverse_sqrt": sqrt_inv_time,
        "form_m_matrix": m_form_time,
        "m_eigendecomposition": m_eigh_time,
        "reconstruct_transform": reconstruct_time,
        "total_measured": stats_time + sigma_eigh_time + sqrt_inv_time + m_form_time + m_eigh_time + reconstruct_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Complexity and conditioning comparison for analytic spectral solvers.")
    parser.add_argument("--suite", default="single-translation")
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--lambda", dest="lambda_reg", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    if args.lambda_reg <= 0.0:
        raise ValueError("--lambda must be strictly positive.")

    xtr, _, _, _ = gwc.load_mnist_numpy()

    view_build_time, family_views = median_time(lambda: sgs.make_suite_views(xtr, args.suite), args.repeats)
    pair_build_time, pair_views = median_time(lambda: gwc.sample_pair_views(xtr, args.suite, seed=gwc.SEED + 13), args.repeats)
    view1_tr, view2_tr = pair_views

    pca_moment_time, stats_pca = median_time(lambda: pca_stats(xtr), args.repeats)
    pair_moment_time, stats_pair = median_time(lambda: sgs.compute_pair_statistics(xtr, family_views), args.repeats)
    dnn_moment_time, stats_dnn = median_time(lambda: gwc.compute_paired_stats(view1_tr, view2_tr), args.repeats)

    pca_solver_time, _ = median_time(lambda: pca_solver_core(stats_pca, args.d), args.repeats)
    wpca_solver_time, _ = median_time(lambda: whitened_pca_solver_core(stats_pca, args.d), args.repeats)
    hard_solver_time, _ = median_time(lambda: hard_whitened_solver_core(stats_pair, args.d), args.repeats)
    fisher_solver_time, _ = median_time(lambda: auto_fisher_solver_core(stats_pair, args.d), args.repeats)
    dnn_solver_time, _ = median_time(lambda: one_parameter_solver_core(stats_dnn, lambda_reg=args.lambda_reg), args.repeats)

    result = {
        "suite": args.suite,
        "n": int(xtr.shape[0]),
        "p": int(xtr.shape[1]),
        "d": args.d,
        "lambda": args.lambda_reg,
        "timings_seconds": {
            "pair_family_build": view_build_time,
            "pair_sampling_build": pair_build_time,
            "pca_moments": pca_moment_time,
            "pair_moments": pair_moment_time,
            "dnn_pair_moments": dnn_moment_time,
            "pca_solver": pca_solver_time,
            "whitened_pca_solver": wpca_solver_time,
            "hard_whitened_solver": hard_solver_time,
            "auto_fisher_solver": fisher_solver_time,
            "one_parameter_dnn_solver": dnn_solver_time,
        },
        "breakdown_seconds": {
            "pca": pca_breakdown(xtr, args.d, args.repeats),
            "whitened_pca": whitened_pca_breakdown(xtr, args.d, args.repeats),
            "hard_whitened_invariance": hard_whitened_breakdown(xtr, family_views, args.d, args.repeats),
            "auto_fisher": auto_fisher_breakdown(xtr, family_views, args.d, args.repeats),
            "one_parameter_whitened_dnn_layer": one_parameter_breakdown(
                view1_tr, view2_tr, lambda_reg=args.lambda_reg, repeats=args.repeats
            ),
        },
        "methods": {
            "pca": analyze_pca(stats_pca, args.d),
            "whitened_pca": analyze_whitened_pca(stats_pca, args.d),
            "hard_whitened_invariance": analyze_hard_whitened(stats_pair, args.d),
            "auto_fisher": analyze_auto_fisher(stats_pair, args.d),
            "one_parameter_whitened_dnn_layer": analyze_one_parameter_layer(stats_dnn, lambda_reg=args.lambda_reg),
        },
        "notes": [
            "Timing separates moment construction from the analytic solve itself.",
            "The one-parameter DNN layer is compared at a single layer; a depth-L network scales roughly linearly in L.",
            "All methods are exact dense linear-algebra solvers; there is no iterative training loop in this benchmark.",
            "The asymptotic order of the one-parameter layer now matches PCA and auto-Fisher up to constants; the remaining gap is a large dense-linear-algebra constant factor.",
        ],
    }

    print(json.dumps(result, indent=2))
    if args.save_json is not None:
        args.save_json.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
