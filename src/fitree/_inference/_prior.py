import pymc as pm
import numpy as np
import pytensor.tensor as pt
from typing import Optional

from fitree import TumorTreeCohort, wrap_trees


# def construct_fmat(n, entries, mut_idx):
#     mat = pt.zeros((n, n))

#     sub_n = len(mut_idx)
#     idx = 0

#     for i in range(sub_n):
#         for j in range(i, sub_n):
#             mat = pt.set_subtensor(
#                 mat[mut_idx[i], mut_idx[j]], entries[idx]  # pyright: ignore
#             )
#             idx += 1

#     return mat


# def prior_normal_fmat(
#     n_mutations: int,
#     mut_idx: np.ndarray,
#     mean: float = 0.0,
#     sigma: float = 1.0,
# ) -> pm.Model:
#     nr_eff_mut = len(mut_idx)
#     nr_entries = nr_eff_mut * (nr_eff_mut + 1) // 2

#     with pm.Model() as model:
#         entries = pm.Normal("entries", mu=mean, sigma=sigma, shape=nr_entries)
#         pm.Deterministic(
#             "fitness_matrix",
#             construct_fmat(n_mutations, entries, mut_idx),
#         )

#     return model


# def prior_horseshoe_fmat(
#     n_mutations: int,
#     mut_idx: np.ndarray,
#     tau_scale: float = 1.0,
#     lambda_scale: float = 1.0,
# ) -> pm.Model:
#     nr_eff_mut = len(mut_idx)
#     nr_entries = nr_eff_mut * (nr_eff_mut + 1) // 2

#     with pm.Model() as model:
#         tau_var = pm.HalfCauchy("tau", tau_scale)
#         lambdas = pm.HalfCauchy("lambdas", lambda_scale, shape=nr_entries)
#         # Reparametrization trick for efficiency
#         z = pm.Normal("_latent", 0.0, 1.0, shape=nr_entries)
#         entries = z * tau_var * lambdas
#         # Construct the theta matrix
#         pm.Deterministic(
#             "fitness_matrix",
#             construct_fmat(n_mutations, entries, mut_idx),
#         )

#     return model


# def prior_regularized_horseshoe_fmat(
#     n_mutations: int,
#     mut_idx: np.ndarray,
#     halft_dof: int = 5,
#     local_scale: float = 0.2,
#     s2: float = 0.04,
#     tau0: Optional[float] = None,
# ) -> pm.Model:
#     nr_eff_mut = len(mut_idx)
#     nr_entries = nr_eff_mut * (nr_eff_mut + 1) // 2

#     with pm.Model() as model:
#         lambdas = pm.HalfStudentT(
#             "lambdas_raw", halft_dof, local_scale, shape=nr_entries
#         )
#         c2 = pm.InverseGamma("c2", halft_dof, halft_dof * s2)  # type: ignore
#         tau_scale = s2 if tau0 is None else tau0
#         tau = pm.HalfStudentT("tau", halft_dof, tau_scale)

#         lambdas_ = pm.Deterministic(
#             "lambdas_tilde",
#             lambdas * pt.sqrt(c2 / (c2 + tau**2 * lambdas**2)),  # type: ignore
#         )

#         # Reparametrization trick for efficiency
#         z = pm.Normal("z", 0.0, 1.0, shape=nr_entries)
#         betas = z * tau * lambdas_

#         # Construct the theta matrix
#         pm.Deterministic(
#             "fitness_matrix",
#             construct_fmat(n_mutations, betas, mut_idx),
#         )

#     return model


# def prior_spike_and_slab_fmat(
#     n_mutations: int,
#     mut_idx: np.ndarray,
#     sparsity_a: float = 3.0,
#     sparsity_b: float = 1.0,
#     spike_scale: float = 0.001,
#     slab_scale: float = 10.0,
# ) -> pm.Model:
#     nr_eff_mut = len(mut_idx)
#     nr_entries = nr_eff_mut * (nr_eff_mut + 1) // 2

#     with pm.Model() as model:
#         gamma = pm.Beta("sparsity", sparsity_a, sparsity_b)
#         sigmas = pm.HalfNormal(
#             "sigmas", pt.stack([spike_scale, slab_scale])  # pyright: ignore
#         )
#         entries = pm.NormalMixture(
#             "entries",
#             mu=0.0,
#             w=pt.stack([gamma, 1.0 - gamma]),  # type: ignore
#             sigma=sigmas,
#             shape=nr_entries,
#         )

#         pm.Deterministic(
#             "fitness_matrix",
#             construct_fmat(n_mutations, entries, mut_idx),
#         )

#     return model


# def prior_fitree(
#     trees: TumorTreeCohort,
#     fmat_prior_mean: float = 0.0,
#     fmat_prior_sigma: float = 1.0,
#     tau_scale: float = 1.0,
#     lambda_scale: float = 1.0,
#     sparsity_a: float = 3.0,
#     sparsity_b: float = 1.0,
#     spike_scale: float = 0.001,
#     slab_scale: float = 10.0,
#     halft_dof: int = 5,
#     local_scale: float = 0.2,
#     s2: float = 0.04,
#     tau0: Optional[float] = None,
#     fmat_prior_type: str = "normal",
#     min_occurrences: int = 0,
#     augment_max_level: int = 2,
# ) -> pm.Model:
#     # mean_tumor_size, std_tumor_size = trees.compute_mean_std_tumor_size()
#     # lnorm_mu = np.log(mean_tumor_size) - 0.5 * np.log(
#     #     1 + std_tumor_size**2 / mean_tumor_size**2
#     # )
#     # lnorm_sigma = np.sqrt(np.log(1 + std_tumor_size**2 / mean_tumor_size**2))
#     # lnorm_tau = 1 / lnorm_sigma**2

#     # lnorm_mu = pt.as_tensor(lnorm_mu)
#     # lnorm_tau = pt.as_tensor(lnorm_tau)
#     # pt.as_tensor(trees.lifetime_risk)
#     # pt.as_tensor(trees.N_patients)

#     vec_trees, _ = wrap_trees(trees, augment_max_level=augment_max_level)
#     geno_idx = np.where(vec_trees.observed.sum(axis=0) >= min_occurrences)[0]
#     mut_idx = np.where(vec_trees.genotypes[geno_idx, :].sum(axis=0) > 0)[0]

#     if fmat_prior_type == "normal":
#         model = prior_normal_fmat(
#             n_mutations=trees.n_mutations,
#             mut_idx=mut_idx,
#             mean=fmat_prior_mean,
#             sigma=fmat_prior_sigma,
#         )
#     elif fmat_prior_type == "horseshoe":
#         model = prior_horseshoe_fmat(
#             n_mutations=trees.n_mutations,
#             mut_idx=mut_idx,
#             tau_scale=tau_scale,
#             lambda_scale=lambda_scale,
#         )
#     elif fmat_prior_type == "spike_and_slab":
#         model = prior_spike_and_slab_fmat(
#             n_mutations=trees.n_mutations,
#             mut_idx=mut_idx,
#             sparsity_a=sparsity_a,
#             sparsity_b=sparsity_b,
#             spike_scale=spike_scale,
#             slab_scale=slab_scale,
#         )
#     elif fmat_prior_type == "regularized_horseshoe":
#         model = prior_regularized_horseshoe_fmat(
#             n_mutations=trees.n_mutations,
#             mut_idx=mut_idx,
#             halft_dof=halft_dof,
#             local_scale=local_scale,
#             s2=s2,
#             tau0=tau0,
#         )
#     else:
#         raise ValueError(f"Unknown fmat_prior_type: {fmat_prior_type}")

#     # with model:
#     #     # Log-normal prior on the tumor size scaling factor C_sampling
#     #     # pm.Lognormal("C_sampling", mu=lnorm_mu, tau=lnorm_tau)
#     #     pm.Deterministic("C_sampling", pt.as_tensor(mean_tumor_size))

#     #     # Negative binomial prior on the number of negative samples
#     #     # if trees.lifetime_risk == 1.0:
#     #     #     pm.Deterministic("nr_neg_samples", pt.as_tensor(0, dtype=pt.lscalar))
#     #     # else:
#     #     #     pm.NegativeBinomial("nr_neg_samples", n=nr_successes, p=lifetime_risk)

#     #     nr_neg_samples = int(
#     #         trees.N_patients * (1 - trees.lifetime_risk) / trees.lifetime_risk
#     #     )
#     #     pm.Deterministic(
#     #         "nr_neg_samples", pt.as_tensor(nr_neg_samples, dtype=pt.lscalar)
#     #     )

#     return model


def construct_fmat_masked(n, diag, offdiag, mut_idx):
    mat = pt.zeros((n, n))

    # Set the diagonal elements
    diag_indices = pt.arange(n), pt.arange(n)
    mat = pt.set_subtensor(mat[diag_indices], diag)  # pyright: ignore

    # Set the upper-triangular off-diagonal elements
    idx = 0
    for i in range(len(mut_idx)):
        for j in range(i + 1, len(mut_idx)):
            mat = pt.set_subtensor(
                mat[mut_idx[i], mut_idx[j]], offdiag[idx]  # pyright: ignore
            )
            idx += 1

    return mat


def prior_fitree(
    trees: TumorTreeCohort,
    diag_mean: float = 0.0,
    diag_sigma: float = 0.1,
    offdiag_mean: float = 0.0,
    offdiag_sigma: float = 0.1,
    min_occurrences: int = 0,
    augment_max_level: int = 2,
) -> pm.Model:
    nr_mutations = trees.n_mutations

    vec_trees, _ = wrap_trees(trees, augment_max_level=augment_max_level)
    geno_idx = np.where(vec_trees.observed.sum(axis=0) >= min_occurrences)[0]
    mut_idx = np.where(vec_trees.genotypes[geno_idx, :].sum(axis=0) > 0)[0]

    nr_eff_mut = len(mut_idx)
    nr_diag = nr_mutations
    nr_offdiag = nr_eff_mut * (nr_eff_mut - 1) // 2

    with pm.Model() as model:
        # use normal prior for the diagonal (no mask)
        diag_entries = pm.Normal(
            "diag_entries", mu=diag_mean, sigma=diag_sigma, shape=nr_diag
        )

        # use normal prior for the off-diagonal (with mask)
        offdiag_entries = pm.Normal(
            "offdiag_entries", mu=offdiag_mean, sigma=offdiag_sigma, shape=nr_offdiag
        )

        pm.Deterministic(
            "fitness_matrix",
            construct_fmat_masked(nr_mutations, diag_entries, offdiag_entries, mut_idx),
        )

    return model


def construct_fmat_mixed(n, diag, offdiag):
    # Create a square matrix of size n filled with zeros
    mat = pt.zeros((n, n))

    # Set the diagonal elements
    diag_indices = pt.arange(n), pt.arange(n)
    mat = pt.set_subtensor(
        mat[diag_indices], diag  # pyright: ignore
    )  # Set the diagonal values

    # Set the upper-triangular off-diagonal elements
    upper_triangular_indices = pt.triu_indices(n, k=1)
    mat = pt.set_subtensor(
        mat[upper_triangular_indices], offdiag  # pyright: ignore
    )  # Set the upper-triangular values

    return mat


def prior_fitree_mixed(
    trees: TumorTreeCohort,
    diag_mean: float = 0.0,
    diag_sigma: float = 0.1,
    halft_dof: int = 5,
    local_scale: float = 0.2,
    s2: float = 0.04,
    tau0: Optional[float] = None,
    min_occurrences: int = 0,
    augment_max_level: int = 2,
) -> pm.Model:
    nr_mutations = trees.n_mutations

    vec_trees, _ = wrap_trees(trees, augment_max_level=augment_max_level)
    geno_idx = np.where(vec_trees.observed.sum(axis=0) >= min_occurrences)[0]
    mut_idx = np.where(vec_trees.genotypes[geno_idx, :].sum(axis=0) > 0)[0]

    nr_eff_mut = len(mut_idx)
    nr_diag = nr_mutations
    nr_offdiag = nr_eff_mut * (nr_eff_mut - 1) // 2

    with pm.Model() as model:
        # use normal prior for the diagonal
        diag_entries = pm.Normal(
            "diag_entries", mu=diag_mean, sigma=diag_sigma, shape=nr_diag
        )

        # use regularized horseshoe prior for the off-diagonal
        lambdas = pm.HalfStudentT(
            "lambdas_raw", halft_dof, local_scale, shape=nr_offdiag
        )
        c2 = pm.InverseGamma("c2", halft_dof, halft_dof * s2)  # type: ignore
        tau_scale = s2 if tau0 is None else tau0
        tau = pm.HalfStudentT("tau", halft_dof, tau_scale)
        lambdas_ = lambdas * pt.sqrt(c2 / (c2 + tau**2 * lambdas**2))
        z = pm.Normal("z", 0.0, 1.0, shape=nr_offdiag)
        offdiag_entries = z * tau * lambdas_

        pm.Deterministic(
            "fitness_matrix",
            construct_fmat_masked(nr_mutations, diag_entries, offdiag_entries, mut_idx),
        )

    return model
