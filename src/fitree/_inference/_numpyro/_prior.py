import jax.numpy as jnp
from numpyro import sample
from numpyro.distributions import FoldedDistribution, StudentT, Normal, InverseGamma


def sample_normal(n_mutations):
    upper_tri_idx = jnp.triu_indices(n_mutations, k=0)
    values = sample("values", Normal(0.0, 1.0).expand(upper_tri_idx[0].shape))

    F_mat = jnp.zeros((n_mutations, n_mutations))
    F_mat = F_mat.at[upper_tri_idx].set(values)

    return F_mat


def sample_spike_and_slab(n_mutations):
    pass


def HalfStudentT(df, scale=1.0):
    return FoldedDistribution(StudentT(df, loc=0.0, scale=scale))


def sample_regularized_horseshoe(
    n_mutations,
    sparsity_sigma: float = 0.3,
    lambdas_dof: int = 5,
):
    upper_tri_idx = jnp.triu_indices(n_mutations, k=0)

    tau = sample("tau", HalfStudentT(df=2, scale=sparsity_sigma))

    lambdas = sample(
        "lambdas_raw", HalfStudentT(df=lambdas_dof).expand(upper_tri_idx[0].shape)
    )

    c2 = sample("c2", InverseGamma(1.0, 1.0))

    lambdas_ = lambdas * jnp.sqrt(c2 / (c2 + tau**2 * lambdas**2))

    z = sample("z", Normal(0.0, 1.0).expand(upper_tri_idx[0].shape))

    values = z * tau * lambdas_

    F_mat = jnp.zeros((n_mutations, n_mutations))
    F_mat = F_mat.at[upper_tri_idx].set(values)

    return F_mat
