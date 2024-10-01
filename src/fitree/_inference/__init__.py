"""This subpackage implements the inference scheme of the FiTree model.
"""

from ._wrapper import VectorizedTrees, wrap_trees
from ._pymc import (
    FiTreeJointLikelihood,
    prior_only_diagonal,
    prior_normal,
    prior_horseshoe,
    prior_regularized_horseshoe,
    prior_spike_and_slab_marginalized,
)


__all__ = [
    "FiTreeJointLikelihood",
    "VectorizedTrees",
    "wrap_trees",
    "prior_only_diagonal",
    "prior_normal",
    "prior_horseshoe",
    "prior_regularized_horseshoe",
    "prior_spike_and_slab_marginalized",
]
