"""This subpackage implements the inference scheme of the FiTree model.
"""

from ._likelihood import jlogp, update_params, compute_normalizing_constant
from ._initialization import (
    recoverable_entries,
    greedy_init_fmat,
    init_prior_rhs,
    init_prior_normal,
)
from ._backend import FiTreeJointLikelihood
from ._prior import prior_fitree, prior_fitree_mixed


__all__ = [
    "jlogp",
    "update_params",
    "compute_normalizing_constant",
    "FiTreeJointLikelihood",
    "prior_fitree",
    "recoverable_entries",
    "greedy_init_fmat",
    "init_prior_rhs",
    "init_prior_normal",
    "prior_fitree_mixed",
]
