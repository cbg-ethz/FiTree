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
    prior_horseshoe_with_mask,
    prior_normal_with_mask,
)
from ._initialization import recoverable_entries, greedy_init_fmat


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
    "prior_horseshoe_with_mask",
    "prior_normal_with_mask",
    "recoverable_entries",
    "greedy_init_fmat",
]
