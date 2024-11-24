import pytensor.tensor as pt
import numpy as np

from fitree._inference._likelihood import (
    jlogp,
    _pt,
)
from fitree._trees._wrapper import wrap_trees, VectorizedTrees
from fitree._trees import TumorTreeCohort

Op = pt.Op  # type: ignore


def sample_cell_numbers(
    trees: VectorizedTrees, rng: np.random.Generator
) -> VectorizedTrees:
    # Estimate observed cell numbers using sequenced cell numbers
    frequencies = trees.seq_cell_number / trees.seq_cell_number.mean(axis=1).reshape(
        -1, 1
    )
    cell_number = np.array(frequencies * trees.tumor_size.reshape(-1, 1))

    t = trees.sampling_time
    C_0 = trees.C_0
    beta = trees.beta

    # Update cell numbers sequentially
    for i in trees.node_id:
        pa_i = trees.parent_id[i]
        delta_i = trees.delta[i]
        r_i = trees.r[i]
        lam_i = trees.lam[i]
        nu_i = trees.nu[i]
        alpha_i = trees.alpha[i]
        rho_i = trees.rho[i]
        observed = trees.observed[:, i].astype(np.float64)
        cell_number_i = cell_number[:, i]
        cell_number_i *= observed

        if pa_i == -1:
            cell_number_i += (1.0 - observed) * rng.negative_binomial(
                n=C_0 * rho_i, p=_pt(alpha_i, beta, lam_i, t)
            )
        else:
            delta_pa_i = trees.delta[pa_i]
            r_pa_i = trees.r[pa_i]

            if lam_i < delta_pa_i:
                h_mean = 1.0 / (delta_pa_i - lam_i)
            elif lam_i == delta_pa_i:
                h_mean = 1.0 / r_pa_i
            else:
                # h_mean = gamma(r_pa_i) / np.power(lam_i - delta_pa_i, r_pa_i)
                h_mean = 0.0

            cell_number_i += (1.0 - observed) * (
                np.power(t, r_i - r_pa_i)
                * np.exp((delta_i - delta_pa_i) * t)
                * nu_i
                * cell_number[:, pa_i]
                * h_mean
            )

            cell_number[:, i] = cell_number_i

    # Renormalize the cell numbers
    cell_number = cell_number / cell_number.sum(axis=1).reshape(-1, 1)
    cell_number *= trees.tumor_size.reshape(-1, 1)
    cell_number = np.round(cell_number)

    trees = trees._replace(cell_number=cell_number)

    return trees


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
        # self.vectorized_trees = update_params(self.vectorized_trees, F_mat)
        # self.vectorized_trees = sample_cell_numbers(self.vectorized_trees, self.rng)
        # self.vectorized_trees = estimate_cell_numbers(self.vectorized_trees)
        # joint_likelihood = jlogp(self.vectorized_trees, F_mat, self.eps, self.tau)

        # joint_likelihood += (
        #     _log_pt(self.vectorized_trees, self.eps, self.tau) * nr_neg_samples
        # )

        joint_likelihood = jlogp(
            self.vectorized_trees, F_mat, nr_neg_samples, self.eps, self.tau
        )

        if np.isnan(joint_likelihood):
            joint_likelihood = -np.inf
        outputs[0][0] = np.array(joint_likelihood, dtype=np.float64)
