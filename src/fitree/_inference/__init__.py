"""This subpackage implements the inference scheme of the FiTree model.
"""

from ._likelihood import jlogp, update_params, compute_normalizing_constant
from ._pymc import (
    FiTreeJointLikelihood,
    prior_only_diagonal,
    prior_normal,
    prior_horseshoe,
    prior_regularized_horseshoe,
    prior_spike_and_slab_marginalized,
)


__all__ = [
    "jlogp",
    "update_params",
    "compute_normalizing_constant",
    "FiTreeJointLikelihood",
    "prior_only_diagonal",
    "prior_normal",
    "prior_horseshoe",
    "prior_regularized_horseshoe",
    "prior_spike_and_slab_marginalized",
]
