"""This subpackage implements the inference scheme of the FiTree model.
"""

from ._likelihood import jlogp, update_params, compute_normalizing_constant
from ._initialization import recoverable_entries, greedy_init_fmat, init_rhs_prior
from ._backend import FiTreeJointLikelihood
from ._prior import prior_fitree


__all__ = [
    "jlogp",
    "update_params",
    "compute_normalizing_constant",
    "FiTreeJointLikelihood",
    "prior_fitree",
    "recoverable_entries",
    "greedy_init_fmat",
    "init_rhs_prior",
]
