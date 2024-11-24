import pymc as pm
import numpy as np
import pytensor.tensor as pt
from typing import Optional

from fitree._trees import TumorTreeCohort


def construct_fmat(n, entries):
    # Create a square matrix of size n filled with zeros
    mat = pt.zeros((n, n))

    # Set the upper-triangular off-diagonal elements
    upper_triangular_indices = pt.triu_indices(n, k=0)
    mat = pt.set_subtensor(
        mat[upper_triangular_indices], entries  # pyright: ignore
    )  # Set the upper-triangular values

    return mat


def prior_normal_fmat(
    n_mutations: int,
    mean: float = 0.0,
    sigma: float = 1.0,
) -> pm.Model:
    nr_entries = n_mutations * (n_mutations + 1) // 2

    with pm.Model() as model:
        entries = pm.Normal("entries", mu=mean, sigma=sigma, shape=nr_entries)
        pm.Deterministic(
            "fitness_matrix",
            construct_fmat(n_mutations, entries),
        )

    return model


def prior_horseshoe_fmat(
    n_mutations: int,
    tau_scale: float = 1.0,
    lambda_scale: float = 1.0,
) -> pm.Model:
    nr_entries = n_mutations * (n_mutations + 1) // 2

    with pm.Model() as model:
        tau_var = pm.HalfCauchy("tau", tau_scale)
        lambdas = pm.HalfCauchy("lambdas", lambda_scale, shape=nr_entries)
        # Reparametrization trick for efficiency
        z = pm.Normal("_latent", 0.0, 1.0, shape=nr_entries)
        entries = z * tau_var * lambdas
        # Construct the theta matrix
        pm.Deterministic(
            "fitness_matrix",
            construct_fmat(n_mutations, entries),
        )

    return model


def prior_regularized_horseshoe_fmat(
    n_mutations: int,
    sparsity_sigma: float = 0.3,
    c2: Optional[float] = None,
    tau: Optional[float] = None,
    lambdas_dof: int = 5,
) -> pm.Model:
    nr_entries = n_mutations * (n_mutations + 1) // 2

    with pm.Model() as model:
        tau_var = pm.HalfStudentT(
            "tau", 2, sparsity_sigma, observed=tau
        )  # type: ignore
        lambdas = pm.HalfStudentT("lambdas_raw", lambdas_dof, shape=nr_entries)
        c2 = pm.InverseGamma("c2", 1, 1, observed=c2)  # type: ignore

        lambdas_ = pm.Deterministic(
            "lambdas_tilde",
            lambdas * pt.sqrt(c2 / (c2 + tau_var**2 * lambdas**2)),  # type: ignore
        )

        # Reparametrization trick for efficiency
        z = pm.Normal("z", 0.0, 1.0, shape=nr_entries)
        betas = pm.Deterministic("betas", z * tau_var * lambdas_)

        # Construct the theta matrix
        pm.Deterministic(
            "fitness_matrix",
            construct_fmat(n_mutations, betas),
        )

    return model


def prior_spike_and_slab_fmat(
    n_mutations: int,
    sparsity_a: float = 3.0,
    sparsity_b: float = 1.0,
    spike_scale: float = 0.001,
    slab_scale: float = 10.0,
) -> pm.Model:
    nr_entries = n_mutations * (n_mutations + 1) // 2

    with pm.Model() as model:
        gamma = pm.Beta("sparsity", sparsity_a, sparsity_b)
        sigmas = pm.HalfNormal(
            "sigmas", pt.stack([spike_scale, slab_scale])  # pyright: ignore
        )
        entries = pm.NormalMixture(
            "entries",
            mu=0.0,
            w=pt.stack([gamma, 1.0 - gamma]),  # type: ignore
            sigma=sigmas,
            shape=nr_entries,
        )

        pm.Deterministic(
            "fitness_matrix",
            construct_fmat(n_mutations, entries),
        )

    return model


def prior_fitree(
    trees: TumorTreeCohort,
    fmat_prior_mean: float = 0.0,
    fmat_prior_sigma: float = 1.0,
    tau_scale: float = 1.0,
    lambda_scale: float = 1.0,
    sparsity_a: float = 3.0,
    sparsity_b: float = 1.0,
    spike_scale: float = 0.001,
    slab_scale: float = 10.0,
    sparsity_sigma: float = 0.3,
    c2: Optional[float] = None,
    tau: Optional[float] = None,
    fmat_prior_type: str = "normal",
) -> pm.Model:
    mean_tumor_size, std_tumor_size = trees.compute_mean_std_tumor_size()
    lnorm_mu = np.log(mean_tumor_size) - 0.5 * np.log(
        1 + std_tumor_size**2 / mean_tumor_size**2
    )
    lnorm_sigma = np.sqrt(np.log(1 + std_tumor_size**2 / mean_tumor_size**2))
    lnorm_tau = 1 / lnorm_sigma**2

    lnorm_mu = pt.as_tensor(lnorm_mu)
    lnorm_tau = pt.as_tensor(lnorm_tau)
    pt.as_tensor(trees.lifetime_risk)
    pt.as_tensor(trees.N_patients)

    if fmat_prior_type == "normal":
        model = prior_normal_fmat(
            n_mutations=trees.n_mutations,
            mean=fmat_prior_mean,
            sigma=fmat_prior_sigma,
        )
    elif fmat_prior_type == "horseshoe":
        model = prior_horseshoe_fmat(
            n_mutations=trees.n_mutations,
            tau_scale=tau_scale,
            lambda_scale=lambda_scale,
        )
    elif fmat_prior_type == "spike_and_slab":
        model = prior_spike_and_slab_fmat(
            n_mutations=trees.n_mutations,
            sparsity_a=sparsity_a,
            sparsity_b=sparsity_b,
            spike_scale=spike_scale,
            slab_scale=slab_scale,
        )
    elif fmat_prior_type == "regularized_horseshoe":
        model = prior_regularized_horseshoe_fmat(
            n_mutations=trees.n_mutations,
            sparsity_sigma=sparsity_sigma,
            c2=c2,
            tau=tau,
        )
    else:
        raise ValueError(f"Unknown fmat_prior_type: {fmat_prior_type}")

    with model:
        # Log-normal prior on the tumor size scaling factor C_sampling
        pm.Lognormal("C_sampling", mu=lnorm_mu, tau=lnorm_tau)

        # Negative binomial prior on the number of negative samples
        # if trees.lifetime_risk == 1.0:
        #     pm.Deterministic("nr_neg_samples", pt.as_tensor(0, dtype=pt.lscalar))
        # else:
        #     pm.NegativeBinomial("nr_neg_samples", n=nr_successes, p=lifetime_risk)

        nr_neg_samples = int(
            trees.N_patients * (1 - trees.lifetime_risk) / trees.lifetime_risk
        )
        pm.Deterministic(
            "nr_neg_samples", pt.as_tensor(nr_neg_samples, dtype=pt.lscalar)
        )

    return model
