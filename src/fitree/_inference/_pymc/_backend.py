import pytensor.tensor as pt
import numpy as np
import jax

from fitree._inference._likelihood import jlogp, jlogp_sampled
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
        augment_max_level: int | None = 2,
        pseudo_count: float = 0,
        eps: float = 1e-64,
        tau: float = 1e-2,
        seed: int = 0,
        jlogp_version: str = "expected",
    ):
        self.vectorized_trees, _ = wrap_trees(trees, augment_max_level, pseudo_count)
        self.eps = eps
        self.tau = tau

        self.key = jax.random.PRNGKey(seed)

        self.jlogp_version = jlogp_version

    def perform(self, node, inputs, outputs):  # type: ignore
        (
            F_mat,
            C_s,
            nr_neg_samples,
        ) = inputs

        if self.jlogp_version == "sampled":
            joint_likelihood, self.key = jlogp_sampled(
                trees=self.vectorized_trees,
                F_mat=F_mat,
                C_s=C_s,
                nr_neg_samples=nr_neg_samples,
                key=self.key,
                eps=self.eps,
                tau=self.tau,
            )
        else:
            joint_likelihood = jlogp(
                trees=self.vectorized_trees,
                F_mat=F_mat,
                C_s=C_s,
                nr_neg_samples=nr_neg_samples,
                eps=self.eps,
                tau=self.tau,
            )

        if np.isnan(joint_likelihood):
            joint_likelihood = -np.inf
        outputs[0][0] = np.array(joint_likelihood, dtype=np.float64)
