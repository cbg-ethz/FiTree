import pytensor.tensor as pt
import numpy as np

from fitree._inference._likelihood import jlogp
from fitree._trees._wrapper import wrap_trees
from fitree._trees import TumorTreeCohort

Op = pt.Op


class FiTreeJointLikelihood(Op):
    itypes = [
        pt.dmatrix
    ]  # the fitness matrix F_mat of shape (n_mutations, n_mutations)
    otypes = [pt.dscalar]  # the joing log-likelihood

    def __init__(self, trees: TumorTreeCohort, eps: float = 1e-16, tau: float = 1e-2):
        self.vectorized_trees, _ = wrap_trees(trees)
        self.eps = eps
        self.tau = tau

    def perform(self, node, inputs, outputs):
        (F_mat,) = inputs
        joint_likelihood = jlogp(self.vectorized_trees, F_mat, self.eps, self.tau)

        if np.isnan(joint_likelihood):
            joint_likelihood = -np.inf
        outputs[0][0] = np.array(joint_likelihood, dtype=np.float64)
