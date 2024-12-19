import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

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
        if i != j and F_mat[i, j] == 0:
            F_mat[i, j] = -F_mat[i, i] - F_mat[j, j]

    triu_indices = np.triu_indices(n_mutations, k=1)
    triu_pairs = np.column_stack(triu_indices)
    for pair in triu_pairs:
        if not any((pair == x).all() for x in entries):
            i = pair[0]
            j = pair[1]
            F_mat[i, j] = -F_mat[i, i] - F_mat[j, j]

    return F_mat


def init_rhs_prior(
    n_mutations: int,
    N_patients: int,
    trees: VectorizedTrees,
    eps: float = 1e-64,
    nr_observed_threshold: int = 10,
    halft_dof: int = 5,
    s2: float = 0.04,
) -> dict:
    F_mat_init = greedy_init_fmat(
        trees, eps=eps, nr_observed_threshold=nr_observed_threshold
    )

    indices = np.triu_indices(n_mutations, k=0)
    betas_init = F_mat_init[indices]
    z_init = (betas_init - np.mean(betas_init)) / np.std(betas_init)

    p0 = np.round(np.sqrt(5 * (n_mutations**2 + n_mutations) / 2 * (2 * 0.95 - 1)))
    D = n_mutations * (n_mutations + 1) / 2
    tau0 = p0 / (D - p0) / np.sqrt(N_patients)

    tau_init = tau0
    c2_init = halft_dof * s2 / (halft_dof - 1)
    lambdas_tilde_init = betas_init / (z_init * tau_init)

    lambdas_init = lambdas_tilde_init / np.sqrt(
        c2_init / (c2_init + tau_init**2 * lambdas_tilde_init**2)
    )

    init_values = {
        "fitness_matrix": F_mat_init,
        "betas": betas_init,
        "z": z_init,
        "tau": tau_init,
        "c2": c2_init,
        "lambdas_raw": lambdas_init,
    }

    return init_values
