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
        indices = jnp.where(
            jnp.all(trees.genotypes[:, [i, j]], axis=1)
            # * (trees.genotypes.sum(axis=1) == 2)
        )[0]
        if len(indices) > 0:
            if np.max(nr_observed[indices]) > nr_observed_threshold:
                to_keep.append((i, j))

    return np.array(to_keep)


@jax.jit
def logp_f_ij(f, i, j, trees, F_mat, idx, eps=1e-64):
    F_mat = F_mat.at[i, j].set(f)
    trees = update_params(trees, F_mat)

    logp = jlogp_one_node(trees, idx, eps)

    return -logp


def optim_f_ij(i, j, trees, F_mat, indices, eps=1e-64):
    # f_trials = jnp.linspace(-1.0, 1.0, 100)

    # def scan_fun(carry, idx):
    #     carry += jax.vmap(lambda f: logp_f_ij(f, i, j, trees, F_mat, idx, eps))(
    #         f_trials
    #     )

    #     return carry, None

    # p_vec, _ = jax.lax.scan(scan_fun, jnp.zeros_like(f_trials), indices)

    # return f_trials[jnp.argmin(p_vec)]

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

    # for i in range(n_mutations):
    #     indices = jnp.where(
    #         jnp.all(trees.genotypes[:, [i, i]], axis=1) * (nr_mut_present == 1)
    #     )[0]
    #     # if len(indices) > 0:
    #     #     if np.sum(nr_observed[indices]) > nr_observed_threshold:
    #     #         F_mat[i, i] = optim_f_ij(i, i, trees, F_mat, indices, eps)
    #     F_mat[i, i] = optim_f_ij(i, i, trees, F_mat, indices, eps)

    # for i in range(n_mutations):
    #     for j in range(i + 1, n_mutations):
    #         indices = jnp.where(
    #             jnp.all(trees.genotypes[:, [i, j]], axis=1) * (nr_mut_present == 2)
    #         )[0]
    #         if len(indices) > 0:
    #             if np.sum(nr_observed[indices]) > nr_observed_threshold:
    #                 F_mat[i, j] = optim_f_ij(i, j, trees, F_mat, indices, eps)

    entries = recoverable_entries(trees, nr_observed_threshold)
    for i, j in entries:
        indices = jnp.where(
            jnp.all(trees.genotypes[:, [i, j]], axis=1)
            * (nr_mut_present == np.where(i == j, 1, 2))
        )[0]
        if len(indices) > 0:
            if np.max(nr_observed[indices]) > nr_observed_threshold:
                F_mat[i, j] = optim_f_ij(i, j, trees, F_mat, indices, eps)

    return F_mat


# Op = pt.Op  # type: ignore

# class LogLikelihoodOneNode(Op):
#     itypes = [pt.dscalar]  # the fitness value f
#     otypes = [pt.dscalar]  # the log-likelihood

#     def __init__(
#         self,
#         trees: VectorizedTrees,
#         F_mat: np.ndarray,
#         indices: np.ndarray,
#         i: int,
#         j: int,
#         eps: float = 1e-64,
#     ):
#         self.trees = trees
#         self.F_mat = F_mat
#         self.indices = indices
#         self.i = i
#         self.j = j
#         self.eps = eps

#     def perform(self, node, inputs, outputs):  # type: ignore
#         (f,) = inputs

#         logp = 0.0

#         for idx in self.indices:
#             self.F_mat[self.i, self.j] = f
#             self.trees = update_params(self.trees, self.F_mat)
#             logp += jlogp_one_node(self.trees, idx, self.eps)

#         outputs[0][0] = np.array(logp, dtype=np.float64)


# def sample_f_ij(i, j, trees, F_mat, indices, eps=1e-64):

#     ll_one_node = LogLikelihoodOneNode(trees, F_mat, indices, i, j, eps)

#     with pm.Model():

#         horseshoe_tau = pm.HalfCauchy("tau", 1.0)
#         horseshoe_lambda = pm.HalfCauchy("lambda", 1.0)
#         horseshoe_z = pm.Normal("z", 0.0, 1.0)

#         f = pm.Deterministic("f", horseshoe_z * horseshoe_tau * horseshoe_lambda)
#         pm.Potential("logp_f_ij", ll_one_node(f)) # type: ignore

#         trace = pm.sample(draws=200, tune=200, chains=1)

#     return trace.posterior["f"].mean() # type: ignore

# def sample_init_fmat(
#     trees: VectorizedTrees,
#     eps: float = 1e-64,
#     nr_observed_threshold: int = 10,
# ) -> np.ndarray:
#     """Greedy initialization of the fitness matrix F_mat."""

#     n_mutations = trees.genotypes.shape[1]
#     F_mat = np.zeros((n_mutations, n_mutations))
#     nr_observed = trees.observed.sum(axis=0)
#     nr_mut_present = trees.genotypes.sum(axis=1)

#     for i in range(n_mutations):
#         indices = jnp.where(
#             jnp.all(trees.genotypes[:, [i, i]], axis=1)
#             * (nr_mut_present == 1)
#         )[0]
#         if len(indices) > 0:
#             if np.sum(nr_observed[indices]) > nr_observed_threshold:
#                 F_mat[i, i] = sample_f_ij(i, i, trees, F_mat, indices, eps)

#     for i in range(n_mutations):
#         for j in range(i + 1, n_mutations):
#             indices = jnp.where(
#                 jnp.all(trees.genotypes[:, [i, j]], axis=1)
#                 * (nr_mut_present == 2)
#             )[0]
#             if len(indices) > 0:
#                 if np.sum(nr_observed[indices]) > nr_observed_threshold:
#                     F_mat[i, j] = sample_f_ij(i, j, trees, F_mat, indices, eps)

#     return F_mat
