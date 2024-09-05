"""This subpackage implements the inference scheme of the FiTree model.
"""

from ._backend import FiTreeJointLikelihood
from ._wrapper import VectorizedTrees, wrap_trees, update_params


__all__ = [
    "FiTreeJointLikelihood",
    "VectorizedTrees",
    "wrap_trees",
    "update_params",
]
