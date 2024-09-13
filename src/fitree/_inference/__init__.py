"""This subpackage implements the inference scheme of the FiTree model.
"""

from ._backend import FiTreeJointLikelihood
from ._wrapper import VectorizedTrees, wrap_trees, update_params
from ._prior import (
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
    "update_params",
    "prior_only_diagonal",
    "prior_normal",
    "prior_horseshoe",
    "prior_regularized_horseshoe",
    "prior_spike_and_slab_marginalized",
]
