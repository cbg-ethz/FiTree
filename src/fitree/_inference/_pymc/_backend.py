import pytensor.tensor as pt
import numpy as np

from fitree._inference._likelihood import jlogp
from fitree._trees._wrapper import wrap_trees
from fitree._trees import TumorTreeCohort

Op = pt.Op  # type: ignore


class FiTreeJointLikelihood(Op):
    itypes = [
        pt.dmatrix,  # fitness matrix F_mat of shape (n_mutations, n_mutations)
        pt.dscalar,  # C_s: scaling factor of the tumor size at sampling time
        pt.lscalar,  # nr_neg_samples: number of negative samples
    ]
    otypes = [pt.dscalar]  # the joing log-likelihood

    def __init__(
        self,
        trees: TumorTreeCohort,
        eps: float = 1e-64,
        tau: float = 1e-2,
        augment_max_level: int | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.vectorized_trees, _ = wrap_trees(trees, augment_max_level)
        self.eps = eps
        self.tau = tau

        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng

    def perform(self, node, inputs, outputs):  # type: ignore
        (
            F_mat,
            C_s,
            nr_neg_samples,
        ) = inputs
        self.vectorized_trees = self.vectorized_trees._replace(C_s=C_s)
        joint_likelihood = jlogp(
            self.vectorized_trees, F_mat, nr_neg_samples, self.eps, self.tau
        )

        if np.isnan(joint_likelihood):
            joint_likelihood = -np.inf
        outputs[0][0] = np.array(joint_likelihood, dtype=np.float64)
