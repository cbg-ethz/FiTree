import jax
import jax.numpy as jnp
import numpy as np

from fitree import VectorizedTrees
from fitree._inference._likelihood import jlogp_one_node, update_params


@jax.jit
def logp_f_ij(f, i, j, trees, F_mat, idx, eps=1e-64):
    F_mat = F_mat.at[i, j].set(f)
    trees = update_params(trees, F_mat)

    logp = jlogp_one_node(trees, idx, eps)

    return -logp


@jax.jit
def optim_f_ij(i, j, trees, F_mat, indices, eps=1e-64):
    f_trials = jnp.linspace(-0.5, 0.5, 100)

    def scan_fun(carry, idx):
        carry += jax.vmap(lambda f: logp_f_ij(f, i, j, trees, F_mat, idx, eps))(
            f_trials
        )

        return carry, None

    p_vec, _ = jax.lax.scan(scan_fun, jnp.zeros_like(f_trials), indices)

    return f_trials[jnp.argmin(p_vec)]


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

    for i in range(n_mutations):
        indices = jnp.where(
            jnp.all(trees.genotypes[:, [i, i]], axis=1) * (nr_mut_present == 1)
        )[0]
        # if len(indices) > 0:
        #     if np.sum(nr_observed[indices]) > nr_observed_threshold:
        #         F_mat[i, i] = optim_f_ij(i, i, trees, F_mat, indices, eps)
        F_mat[i, i] = optim_f_ij(i, i, trees, F_mat, indices, eps)

    for i in range(n_mutations):
        for j in range(i + 1, n_mutations):
            indices = jnp.where(
                jnp.all(trees.genotypes[:, [i, j]], axis=1) * (nr_mut_present == 2)
            )[0]
            if len(indices) > 0:
                if np.sum(nr_observed[indices]) > nr_observed_threshold:
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
