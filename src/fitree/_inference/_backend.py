import pytensor.tensor as pt
import numpy as np

from ._likelihood import unnormalized_joint_logp
from ._wrapper import wrap_trees, update_params
from fitree._trees import TumorTreeCohort
from fitree._mtbp import _mcdf_sampling

Op = pt.Op


class FiTreeJointLikelihood(Op):
    itypes = [
        pt.dmatrix
    ]  # the fitness matrix F_mat of shape (n_mutations, n_mutations)
    otypes = [pt.dscalar]  # the joing log-likelihood

    def __init__(self, trees: TumorTreeCohort, eps: float = 1e-16, tau: float = 1e-2):
        self.vectorized_trees, self.union_tree = wrap_trees(trees)
        self.mu_vec = trees.mu_vec
        self.common_beta = trees.common_beta
        self.N_patients = trees.N_patients
        self.t_max = trees.t_max
        self.C_sampling = trees.C_sampling
        self.C_0 = trees.C_0
        self.eps = eps
        self.tau = tau

    def perform(self, node, inputs, outputs):
        (F_mat,) = inputs
        self.vectorized_trees, self.union_tree = update_params(
            vec_trees=self.vectorized_trees,
            union_tree=self.union_tree,
            F_mat=F_mat,
            mu_vec=self.mu_vec,
            common_beta=self.common_beta,
        )
        unnormalized_ll = unnormalized_joint_logp(self.vectorized_trees, self.eps)
        joint_likelihood = unnormalized_ll - self.N_patients * np.log(
            float(
                _mcdf_sampling(
                    self.union_tree, self.t_max, self.C_sampling, self.C_0, self.tau
                ).real
            )
        )

        if np.isnan(joint_likelihood):
            joint_likelihood = -np.inf
        outputs[0][0] = np.array(joint_likelihood, dtype=np.float64)
