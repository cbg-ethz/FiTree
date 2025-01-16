import pytensor.tensor as pt
import numpy as np

from fitree._inference._likelihood import (
    jlogp,
)
from fitree._trees._wrapper import wrap_trees
from fitree._trees import TumorTreeCohort

Op = pt.Op  # type: ignore


class FiTreeJointLikelihood(Op):
    itypes = [
        pt.dmatrix,  # fitness matrix F_mat of shape (n_mutations, n_mutations)
    ]
    otypes = [pt.dscalar]  # the joing log-likelihood

    def __init__(
        self,
        trees: TumorTreeCohort,
        augment_max_level: int | None = 2,
        C_s: float | None = None,
        conditioning: bool = True,
        lifetime_risk_mean: float | None = None,
        lifetime_risk_std: float | None = None,
        eps: float = 1e-64,
        tau: float = 1e-2,
    ):
        self.vectorized_trees, _ = wrap_trees(trees, augment_max_level)
        self.N_patients = trees.N_patients
        self.eps = eps
        self.tau = tau

        if C_s is None:
            self.C_s = trees.compute_mean_std_tumor_size()[0]
        else:
            self.C_s = C_s

        if not conditioning:
            # error if lifetime_risk_mean or lifetime_risk_std is None
            if lifetime_risk_mean is None or lifetime_risk_std is None:
                raise ValueError(
                    "lifetime_risk_mean and lifetime_risk_std must be provided"
                )

        self.conditioning = conditioning
        self.lifetime_risk_mean = lifetime_risk_mean
        self.lifetime_risk_std = lifetime_risk_std

    def perform(self, node, inputs, outputs):  # type: ignore
        (F_mat,) = inputs

        joint_likelihood, lifetime_risk = jlogp(
            trees=self.vectorized_trees,
            F_mat=F_mat,
            C_s=self.C_s,
            eps=self.eps,
            tau=self.tau,
        )

        if self.conditioning:
            joint_likelihood -= self.N_patients * np.log(lifetime_risk + self.eps)
        else:
            # compute the probability of the lifetime risk using normal log-likelihood
            joint_likelihood += (
                -0.5
                * (lifetime_risk - self.lifetime_risk_mean) ** 2
                / self.lifetime_risk_std**2  # pyright: ignore
                - 0.5 * np.log(2 * np.pi)
                - np.log(self.lifetime_risk_std)  # pyright: ignore
            )

        if np.isnan(joint_likelihood):
            joint_likelihood = -np.inf

        if joint_likelihood > 0:
            joint_likelihood = 0.0
        outputs[0][0] = np.array(joint_likelihood, dtype=np.float64)
