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
def optim_f_ij(i, j, trees, F_mat, idx, eps=1e-64):
    f_trials = jnp.linspace(-1.0, 1.0, 100)

    p_vec = jax.vmap(lambda f: logp_f_ij(f, i, j, trees, F_mat, idx, eps))(f_trials)

    return f_trials[jnp.argmin(p_vec)]


def greedy_init_fmat(
    trees: VectorizedTrees,
    eps: float = 1e-64,
) -> np.ndarray:
    """Greedy initialization of the fitness matrix F_mat."""

    n_mutations = trees.genotypes.shape[1]
    F_mat = np.zeros((n_mutations, n_mutations))

    for i in range(n_mutations):
        idx = jnp.where(jnp.all(trees.genotypes[:, [i, i]], axis=1))[0][0]
        F_mat[i, i] = optim_f_ij(i, i, trees, F_mat, idx, eps)

    for i in range(n_mutations):
        for j in range(i + 1, n_mutations):
            idx = jnp.where(jnp.all(trees.genotypes[:, [i, j]], axis=1))[0][0]
            F_mat[i, j] = optim_f_ij(i, j, trees, F_mat, idx, eps)

    return F_mat
