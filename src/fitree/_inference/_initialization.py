import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
import pymc as pm
import pytensor.tensor as pt

from fitree import VectorizedTrees
from fitree._inference._likelihood import jlogp_one_node, update_params


def recoverable_entries(
    trees: VectorizedTrees,
    nr_observed_threshold: int = 2,
) -> np.ndarray:
    n_mutations = trees.genotypes.shape[1]
    diag_indices = np.diag_indices(n_mutations)
    triu_indices = np.triu_indices(n_mutations, k=1)
    all_indices = (
        np.concatenate([diag_indices[0], triu_indices[0]]),
        np.concatenate([diag_indices[1], triu_indices[1]]),
    )
    nr_observed = trees.observed.sum(axis=0)
    to_keep = []

    for i, j in zip(*all_indices):
        if i != j:
            indices = jnp.where(
                jnp.all(trees.genotypes[:, [i, j]], axis=1)
                # * (trees.genotypes.sum(axis=1) == 2)
            )[0]
            if len(indices) > 0:
                if np.max(nr_observed[indices]) > nr_observed_threshold:
                    to_keep.append((i, j))
        else:
            to_keep.append((i, j))

    return np.array(to_keep)


@jax.jit
def logp_f_ij(f, i, j, trees, F_mat, idx, eps=1e-64):
    F_mat = F_mat.at[i, j].set(f)
    trees = update_params(trees, F_mat)

    logp = jlogp_one_node(trees, idx, eps)

    return -logp


def optim_f_ij(i, j, trees, F_mat, indices, eps=1e-64):
    @jax.jit
    def logp_f(f):
        def scan_fun(carry, idx):
            carry += logp_f_ij(f, i, j, trees, F_mat, idx, eps)

            return carry, None

        lp, _ = jax.lax.scan(scan_fun, jnp.zeros(1), indices)

        return lp

    @jax.jit
    def logp_f_vec(f_vec):
        l_vec = jax.vmap(logp_f)(f_vec)

        return l_vec

    res = minimize(
        lambda f_vec: np.array(logp_f_vec(f_vec)),
        0.0,
        method="COBYQA",
        bounds=[(-1, 1)],
    )

    return res.x[0]


def greedy_init_fmat(
    trees: VectorizedTrees,
    eps: float = 1e-64,
    nr_observed_threshold: int = 10,
) -> np.ndarray:
    """Greedy initialization of the fitness matrix F_mat."""

    n_mutations = trees.genotypes.shape[1]
    F_mat = np.zeros((n_mutations, n_mutations))
    nr_observed = trees.observed.sum(axis=0)
    nr_mut_present = trees.genotypes.sum(axis=1)

    entries = recoverable_entries(trees, nr_observed_threshold)
    for i, j in entries:
        indices = jnp.where(
            jnp.all(trees.genotypes[:, [i, j]], axis=1)
            * (nr_mut_present == np.where(i == j, 1, 2))
        )[0]
        if len(indices) == 0:
            continue
        if i == j:
            F_mat[i, j] = optim_f_ij(i, j, trees, F_mat, indices, eps)
        if np.max(nr_observed[indices]) > nr_observed_threshold:
            F_mat[i, j] = optim_f_ij(i, j, trees, F_mat, indices, eps)

    return F_mat


def init_prior_rhs(
    N_patients: int,
    trees: VectorizedTrees,
    eps: float = 1e-64,
    min_occurrences: int = 0,
    halft_dof: int = 5,
    local_scale: float = 0.2,
    s2: float = 0.04,
    n_tune: int = 1000,
    n_samples: int = 1000,
) -> dict:
    F_mat_init = greedy_init_fmat(
        trees, eps=eps, nr_observed_threshold=np.round(N_patients * 0.1)
    )

    geno_idx = np.where(trees.observed.sum(axis=0) >= min_occurrences)[0]
    mut_idx = np.where(trees.genotypes[geno_idx, :].sum(axis=0) > 0)[0]
    nr_eff_mut = len(mut_idx)
    nr_entries = nr_eff_mut * (nr_eff_mut + 1) // 2

    submat_idx = np.ix_(mut_idx, mut_idx)
    fitness_matrix = np.zeros_like(F_mat_init)
    fitness_matrix[submat_idx] = F_mat_init[submat_idx]
    sub_triu_indices = np.triu_indices(len(mut_idx), k=0)
    betas_init = fitness_matrix[submat_idx][sub_triu_indices]

    p0 = np.round(np.sqrt(5 * (nr_eff_mut**2 + nr_eff_mut) / 2 * (2 * 0.95 - 1)))
    D = nr_eff_mut * (nr_eff_mut + 1) / 2
    tau0 = p0 / (D - p0) / np.sqrt(N_patients)

    with pm.Model():
        lambdas = pm.HalfStudentT(
            "lambdas_raw", halft_dof, local_scale, shape=nr_entries
        )
        c2 = pm.InverseGamma("c2", halft_dof, halft_dof * s2)  # type: ignore
        tau = pm.HalfStudentT("tau", halft_dof, tau0)

        lambdas_ = pm.Deterministic(
            "lambdas_tilde",
            lambdas * pt.sqrt(c2 / (c2 + tau**2 * lambdas**2)),  # type: ignore
        )

        z = pm.Normal("z", 0.0, 1.0, shape=nr_entries)
        pm.Normal("betas", z * tau * lambdas_, 0.0001, observed=betas_init)
        trace = pm.sample(draws=n_samples, tune=n_tune, chains=1)

    lambdas_raw_init = trace.posterior["lambdas_raw"].values[0][-1, :]  # type: ignore
    tau_init = trace.posterior["tau"].values[0][-1]  # type: ignore
    z_init = trace.posterior["z"].values[0][-1, :]  # type: ignore
    c2_init = trace.posterior["c2"].values[0][-1]  # type: ignore

    init_values = {
        "z": z_init,
        "tau": tau_init,
        "c2": c2_init,
        "lambdas_raw": lambdas_raw_init,
    }

    return init_values


def init_prior_normal(
    N_patients: int,
    trees: VectorizedTrees,
    eps: float = 1e-64,
    min_occurrences: int = 0,
) -> dict:
    F_mat_init = greedy_init_fmat(
        trees, eps=eps, nr_observed_threshold=np.round(N_patients * 0.1)
    )

    geno_idx = np.where(trees.observed.sum(axis=0) >= min_occurrences)[0]
    mut_idx = np.where(trees.genotypes[geno_idx, :].sum(axis=0) > 0)[0]

    diag_entries = np.diag(F_mat_init)

    submat_idx = np.ix_(mut_idx, mut_idx)
    fitness_matrix = np.zeros_like(F_mat_init)
    fitness_matrix[submat_idx] = F_mat_init[submat_idx]
    sub_triu_indices = np.triu_indices(len(mut_idx), k=1)
    offdiag_entries = fitness_matrix[submat_idx][sub_triu_indices]

    init_values = {
        "fitness_matrix": fitness_matrix,
        "diag_entries": diag_entries,
        "offdiag_entries": offdiag_entries,
    }

    return init_values
