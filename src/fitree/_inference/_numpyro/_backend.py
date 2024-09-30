import jax
import jax.numpy as jnp

from fitree._trees import TumorTreeCohort
from fitree._inference._likelihood import unnormalized_joint_logp, update_params
from fitree._inference._wrapper import wrap_trees, VectorizedTrees


def prepare_trees(trees: TumorTreeCohort) -> VectorizedTrees:
    vec_trees, _ = wrap_trees(trees)
    return vec_trees


@jax.jit
def logp(trees: VectorizedTrees, F_mat: jnp.ndarray) -> jnp.ndarray:
    trees = update_params(trees, F_mat)
    jlogp = unnormalized_joint_logp(trees)
    jlogp = jnp.min(jnp.array([jlogp, 0.0]))
    jlogp = jnp.where(jnp.isnan(jlogp), -jnp.inf, jlogp)
    return jlogp


# def build_fitree_model(
# 	trees: TumorTreeCohort,
# 	prior: str = "normal",
# ):
# 	"""This function builds the FiTree model in NumPyro.

# 	Args:
# 		trees (TumorTreeCohort): A cohort of tumor trees.
# 		prior (str, optional): Prior distribution. Takes values in
# 			["normal", "spike_and_slab", "regularized_horseshoe"]. Defaults to "normal".
# 	"""

#   assert prior in [
# 		"normal",
# 	    "spike_and_slab",
# 	    "regularized_horseshoe"
# 	], "Invalid prior"

# 	vec_trees = prepare_trees(trees)
# 	n_mutations = trees.n_mutations

# 	if prior == "normal":
# 		F_mat = sample_normal(n_mutations)
# 	elif prior == "spike_and_slab":
# 		F_mat = sample_spike_and_slab(n_mutations)
# 	elif prior == "regularized_horseshoe":
# 		F_mat = sample_regularized_horseshoe(n_mutations)

# 	deterministic("F_mat", F_mat)
# 	factor("logp", logp(vec_trees, F_mat))
