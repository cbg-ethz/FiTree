import pymc as pm
import numpy as np
import pytensor.tensor as pt
from typing import Optional

from fitree._trees import TumorTreeCohort


def construct_square_matrix(n: int, diag, offdiag):
    """Constructs a square matrix from the diagonal and upper-triangular
    off-diagonal elements.

    Args:
        n: size of the matrix
        diag: vector of shape (n,) containing the diagonal elements
        offdiag: vector of shape (n*(n-1)//2,) containing the upper-triangular
        off-diagonal elements

    Returns:
        matrix of shape (n, n)
    """
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


def _offdiag_size(n: int) -> int:
    """Computes the number of off-diagonal elements in a square matrix of size n."""
    return n * (n - 1) // 2


def prior_only_diagonal(
    n_mutations: int,
    mean: float = 0.0,
    sigma: float = 1.0,
) -> pm.Model:
    """Constructs a PyMC model in which the theta
    matrix contains only diagonal entries."""

    with pm.Model() as model:  # type: ignore
        diag = pm.Normal(
            "diagonal",
            mu=mean,
            sigma=sigma,
            size=(n_mutations,),
        )
        mask = pt.eye(n_mutations)
        pm.Deterministic("fitness_matrix", mask * diag)  # type: ignore
    return model


def prior_normal(
    n_mutations: int,
    mean: float = 0.0,
    sigma: float = 1.0,
    mean_offdiag: Optional[float] = None,
    sigma_offdiag: Optional[float] = None,
) -> pm.Model:
    """Constructs PyMC model in which each entry is sampled
    from multivariate normal distribution.
    Author: Pawel Czyz
    Source: https://github.com/cbg-ethz/pMHN/

    Args:
        mean: prior mean of the diagonal entries
        sigma: prior standard deviation of the diagonal entries
        mean_offdiag: prior mean of the off-diagonal entries, defaults to `mean`
        sigma_offdiag: prior standard deviation of the off-diagonal entries,
            defaults to `sigma`

    Note:
        This model is unlikely to result in sparse solutions
        and for very weak priors (e.g., very large sigma) the solution
        may be very multimodal.
    """
    if mean_offdiag is None:
        mean_offdiag = mean
    if sigma_offdiag is None:
        sigma_offdiag = sigma

    with pm.Model() as model:  # type: ignore
        diag = pm.Normal("diagonal", mean, sigma, shape=n_mutations)
        offdiag = pm.Normal(
            "offdiag", mean_offdiag, sigma_offdiag, shape=_offdiag_size(n_mutations)
        )
        pm.Deterministic(
            "fitness_matrix",
            construct_square_matrix(n_mutations, diag=diag, offdiag=offdiag),
        )
    return model


def prior_horseshoe(
    n_mutations: int,
    tau_scale: float = 1.0,
    lambda_scale: float = 1.0,
) -> pm.Model:
    """Constructs PyMC model with horseshoe prior on the off-diagonal terms.
    Author: Pawel Czyz
    Source: https://github.com/cbg-ethz/pMHN/

    For full description of this prior, see
    C.M. Caralho et al., _Handling Sparsity via the Horseshoe_, AISTATS 2009.

    Args:
        n_mutations: number of mutations
        diag_mean: prior mean of the diagonal rates
        diag_sigma: prior standard deviation of the diagonal rates


    Returns:
        PyMC model. Use `model.theta` to
           access the (log-)mutual hazard network variable,
           which has shape (n_mutations, n_mutations)
    """
    with pm.Model() as model:  # type: ignore
        tau_var = pm.HalfCauchy("tau", tau_scale)
        lambdas_offdiag = pm.HalfCauchy(
            "lambdas_offdiag", lambda_scale, shape=_offdiag_size(n_mutations)
        )

        # Reparametrization trick for efficiency
        z = pm.Normal("_latent", 0.0, 1.0, shape=_offdiag_size(n_mutations))
        offdiag = z * tau_var * lambdas_offdiag

        # Construct diagonal terms explicitly
        lambdas_diag = pm.HalfCauchy("lambdas_diag", lambda_scale, shape=n_mutations)
        z_diag = pm.Normal("_latent_diag", 0.0, 1.0, shape=n_mutations)
        diag = z_diag * tau_var * lambdas_diag

        # Construct the theta matrix
        pm.Deterministic(
            "fitness_matrix",
            construct_square_matrix(n_mutations, diag=diag, offdiag=offdiag),
        )

    return model


def prior_regularized_horseshoe(
    n_mutations: int,
    sparsity_sigma: float = 0.3,
    c2: Optional[float] = None,
    tau: Optional[float] = None,
    lambdas_dof: int = 5,
) -> pm.Model:
    """Constructs PyMC model for regularized horseshoe prior.
    To access the (log-)mutual hazard network parameters, use the `theta` variable.
    Author: Pawel Czyz
    Source: https://github.com/cbg-ethz/pMHN/

    Args:
        n_mutations: number of mutations
        diag_mean: prior mean of the diagonal rates
        sigma: prior standard deviation of the diagonal rates
        sparsity_sigma: sparsity parameter, controls the prior on `tau`.
          Ignored if `tau` is provided.
        tau: if provided, will be used as the value of `tau` in the model

    Returns:
        PyMC model. Use `model.theta` to
           access the (log-)mutual hazard network variable,
           which has shape (n_mutations, n_mutations)

    Example:
        ```python
        model = prior_regularized_horseshoe(n_mutations=10)
        with model:
            theta = model.theta
            pm.Potential("potential", some_function_of(theta))
        ```
    """
    if sparsity_sigma <= 0:
        raise ValueError("sparsity_sigma must be positive")

    if c2 is not None:
        if c2 <= 0:
            raise ValueError("c2 must be positive")

    # Below we ignore the type of some variables because Pyright
    # is not fully compatible with PyMC type annotations.
    with pm.Model() as model:  # type: ignore
        tau_var = pm.HalfStudentT(
            "tau", 2, sparsity_sigma, observed=tau
        )  # type: ignore
        lambdas = pm.HalfStudentT(
            "lambdas_raw", lambdas_dof, shape=_offdiag_size(n_mutations)
        )
        c2 = pm.InverseGamma("c2", 1, 1, observed=c2)  # type: ignore

        lambdas_ = pm.Deterministic(
            "lambdas_tilde",
            lambdas * pt.sqrt(c2 / (c2 + tau_var**2 * lambdas**2)),  # type: ignore
        )

        # Reparametrization trick for efficiency
        z = pm.Normal("z", 0.0, 1.0, shape=_offdiag_size(n_mutations))
        betas = pm.Deterministic("betas", z * tau_var * lambdas_)

        lambdas_diag = pm.HalfStudentT("lambdas_diag", lambdas_dof, shape=n_mutations)
        lambdas_diag_ = pm.Deterministic(
            "lambdas_diag_tilde",
            lambdas_diag * pt.sqrt(c2 / (c2 + tau_var**2 * lambdas_diag**2)),
        )
        z_diag = pm.Normal("z_diag", 0.0, 1.0, shape=n_mutations)
        diag = pm.Deterministic("diag", z_diag * tau_var * lambdas_diag_)

        # Construct the theta matrix
        pm.Deterministic(
            "fitness_matrix",
            construct_square_matrix(n_mutations, diag=diag, offdiag=betas),
        )

    return model


def prior_spike_and_slab_marginalized(
    n_mutations: int,
    sparsity_a: float = 3.0,
    sparsity_b: float = 1.0,
    spike_scale: float = 0.001,
    slab_scale: float = 10.0,
) -> pm.Model:
    """Construct a spike-and-slab mixture prior for the off-diagonal entries.
    Author: Pawel Czyz
    Source: https://github.com/cbg-ethz/pMHN/

    See the spike-and-slab mixture prior in this
    [post](https://betanalpha.github.io/assets/case_studies/modeling_sparsity.html#221_Discrete_Mixture_Models).

    Args:
        n_mutations: number of mutations
        diag_mean: mean of the normal prior on the diagonal rates
        diag_sigma: standard deviation of the normal prior on the diagonal rates
        sparsity_a: shape parameter of the Beta distribution controling sparsity
        sparsity_b: shape parameter of the Beta distribution controling sparsity

    Note:
        By default we set `sparsity` prior Beta(3, 1) for
        $E[\\gamma] \\approx 0.75$, which
        should result in 75% of the off-diagonal entries being close to zero.
    """
    with pm.Model() as model:
        gamma_offdiag = pm.Beta("sparsity", sparsity_a, sparsity_b)
        gamma_diag = pm.Beta("sparsity_diag", sparsity_a, sparsity_b)
        sigmas = pm.HalfNormal(
            "offdiag_sigmas", pt.stack([spike_scale, slab_scale])  # pyright: ignore
        )
        offdiag_entries = pm.NormalMixture(
            "offdiag_entries",
            mu=0.0,
            w=pt.stack([gamma_offdiag, 1.0 - gamma_offdiag]),  # type: ignore
            sigma=sigmas,
            shape=_offdiag_size(n_mutations),
        )

        # Now sample diagonal rates
        diag = pm.NormalMixture(
            "diagonal",
            mu=0.0,
            w=pt.stack([gamma_diag, 1.0 - gamma_diag]),  # type: ignore
            sigma=sigmas,
            size=(n_mutations,),
        )

        pm.Deterministic(
            "fitness_matrix",
            construct_square_matrix(n_mutations, diag=diag, offdiag=offdiag_entries),
        )

    return model


def construct_square_matrix_with_mask(n: int, indices, entries):
    # Create a square matrix of size n filled with zeros
    mat = pt.zeros((n, n))

    # Convert indices to tensor
    indices = pt.as_tensor_variable(indices)

    # Fill the matrix with the provided entries at the specified locations
    mat = pt.set_subtensor(
        mat[indices[:, 0], indices[:, 1]], entries  # pyright: ignore
    )

    # Get upper triangular indices (excluding diagonal)
    upper_triangular_indices = pt.triu_indices(n, k=1)

    # Stack the row and column indices to create pairs
    upper_tri_pairs = pt.stack(
        [upper_triangular_indices[0], upper_triangular_indices[1]], axis=1
    )

    # Compare each upper triangular pair with the provided indices
    # Keep the ones that are not in the provided indices
    is_in_indices = pt.any(
        pt.all(
            pt.eq(upper_tri_pairs[:, None, :], indices[None, :, :]),  # pyright: ignore
            axis=-1,
        ),
        axis=1,
    )

    # Extract the indices that are not in the provided indices
    missing_indices = upper_tri_pairs[~is_in_indices]  # pyright: ignore

    # Set these missing indices to the negative sum of the diagonal elements
    mat = pt.set_subtensor(
        mat[missing_indices[:, 0], missing_indices[:, 1]],  # pyright: ignore
        -(
            mat[missing_indices[:, 0], missing_indices[:, 0]]  # pyright: ignore
            + mat[missing_indices[:, 1], missing_indices[:, 1]]  # pyright: ignore
        ),
    )

    return mat


def prior_horseshoe_with_mask(
    n_mutations: int,
    indices: np.ndarray,
    tau_scale: float = 1.0,
    lambda_scale: float = 1.0,
) -> pm.Model:
    nr_entries = indices.shape[0]

    with pm.Model() as model:  # type: ignore
        tau_var = pm.HalfCauchy("tau", tau_scale)
        lambdas = pm.HalfCauchy("lambdas", lambda_scale, shape=nr_entries)
        # Reparametrization trick for efficiency
        z = pm.Normal("_latent", 0.0, 1.0, shape=nr_entries)
        entries = z * tau_var * lambdas
        # Construct the theta matrix
        pm.Deterministic(
            "fitness_matrix",
            construct_square_matrix_with_mask(n_mutations, indices, entries),
        )

    return model


def prior_normal_with_mask(
    n_mutations: int,
    indices: np.ndarray,
    mean: float = 0.0,
    sigma: float = 1.0,
) -> pm.Model:
    nr_entries = indices.shape[0]

    with pm.Model() as model:  # type: ignore
        entries = pm.Normal("entries", mu=mean, sigma=sigma, shape=nr_entries)
        pm.Deterministic(
            "fitness_matrix",
            construct_square_matrix_with_mask(n_mutations, indices, entries),
        )
    return model


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


def prior_fitree(
    trees: TumorTreeCohort,
    fmat_prior_mean: float = 0.0,
    fmat_prior_sigma: float = 1.0,
) -> pm.Model:
    mean_tumor_size, std_tumor_size = trees.compute_mean_std_tumor_size()
    lnorm_mu = np.log(mean_tumor_size) - 0.5 * np.log(
        1 + std_tumor_size**2 / mean_tumor_size**2
    )
    lnorm_sigma = np.sqrt(np.log(1 + std_tumor_size**2 / mean_tumor_size**2))
    lnorm_tau = 1 / lnorm_sigma**2

    lnorm_mu = pt.as_tensor(lnorm_mu)
    lnorm_tau = pt.as_tensor(lnorm_tau)
    lifetime_risk = pt.as_tensor(trees.lifetime_risk)
    nr_successes = pt.as_tensor(trees.N_patients)

    model = prior_normal_fmat(
        n_mutations=trees.n_mutations,
        mean=fmat_prior_mean,
        sigma=fmat_prior_sigma,
    )

    with model:
        # Log-normal prior on the tumor size scaling factor C_sampling
        pm.Lognormal("C_sampling", mu=lnorm_mu, tau=lnorm_tau)

        # Negative binomial prior on the number of negative samples
        if trees.lifetime_risk == 1.0:
            pm.Deterministic("nr_neg_samples", pt.as_tensor(0, dtype=pt.lscalar))
        else:
            pm.NegativeBinomial("nr_neg_samples", n=nr_successes, p=lifetime_risk)

    return model
