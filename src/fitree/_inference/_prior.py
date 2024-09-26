import pymc as pm
import pytensor.tensor as pt
from typing import Optional


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
    mat = pt.set_subtensor(mat[diag_indices], diag)  # Set the diagonal values

    # Set the upper-triangular off-diagonal elements
    upper_triangular_indices = pt.triu_indices(n, k=1)
    mat = pt.set_subtensor(
        mat[upper_triangular_indices], offdiag
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
    diag_mean: float = 0,
    diag_sigma: float = 10.0,
    tau: Optional[float] = None,
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
        tau_var = pm.HalfCauchy("tau", 1, observed=tau)
        lambdas = pm.HalfCauchy("lambdas", 1, shape=_offdiag_size(n_mutations))

        # Reparametrization trick for efficiency
        z = pm.Normal("_latent", 0.0, 1.0, shape=_offdiag_size(n_mutations))
        offdiag = z * tau_var * lambdas

        # Construct diagonal terms explicitly
        diag = pm.Normal("diagonal", diag_mean, diag_sigma, shape=n_mutations)

        # Construct the theta matrix
        pm.Deterministic(
            "fitness_matrix",
            construct_square_matrix(n_mutations, diag=diag, offdiag=offdiag),
        )

    return model


def prior_regularized_horseshoe(
    n_mutations: int,
    diag_mean: float = 0,
    diag_sigma: float = 10.0,
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
    if diag_sigma <= 0:
        raise ValueError("diag_sigma must be positive")
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

        # Now sample diagonal rates
        diag = pm.Normal(
            "diagonal",
            mu=diag_mean,
            sigma=diag_sigma,
            size=(n_mutations,),
        )
        # Construct the theta matrix
        pm.Deterministic(
            "fitness_matrix",
            construct_square_matrix(n_mutations, diag=diag, offdiag=betas),
        )

    return model


def prior_offdiagonal_laplace(
    n_mutations: int,
    penalty: Optional[float] = 1.0,
    scale: Optional[float] = None,
    diag_mean: float = 0.0,
    diag_sigma: float = 20.0,
) -> pm.Model:
    """Prior modelling off-diagonal entries of the theta matrix
    using the Laplace distribution.
    Author: Pawel Czyz
    Source: https://github.com/cbg-ethz/pMHN/

    Args:
        n_mutations: number of mutations
        penalty: L1 penalty to be applied to the off-diagonal entries
        scale: Laplace prior scale, equal to `1/penalty`.
        diag_mean: mean of the normal prior on the diagonal rates
        diag_sigma: standard deviation of the normal prior on the diagonal rates.
          Use large values to provide very weak regularization
          on the diagonal rates.

    Note:
      - Laplace distribution is not the best way of enforcing sparsity
        (i.e., Bayesian Lasso). We recommend horseshoe prior instead.
        However, this distribution is suitable for a point estimation,
        with MAP corresponding (approximately) to the Lasso solution.
      - Exactly one of `penalty` and `scale` must be provided.
    """
    if penalty is None and scale is None:
        raise ValueError("Either penalty or scale must be provided")
    if penalty is not None and scale is not None:
        raise ValueError("Only one of penalty and scale must be provided")
    if scale is None and penalty is not None:
        if penalty <= 0:
            raise ValueError("penalty must be positive")
        scale = 1.0 / penalty

    assert scale is not None
    if scale <= 0:
        raise ValueError("Scale must be positive.")

    with pm.Model() as model:  # type: ignore
        diag = pm.Normal(
            "diagonal",
            mu=diag_mean,
            sigma=diag_sigma,
            size=(n_mutations,),
        )
        laplaces = pm.Laplace(
            "laplace", mu=0.0, b=scale, shape=_offdiag_size(n_mutations)
        )
        pm.Deterministic(
            "fitness_matrix",
            construct_square_matrix(n_mutations, diag=diag, offdiag=laplaces),
        )

    return model


def prior_spike_and_slab_marginalized(
    n_mutations: int,
    diag_mean: float = 0.0,
    diag_sigma: float = 10.0,
    sparsity_a: float = 3.0,
    sparsity_b: float = 1.0,
    spike_scale: float = 0.1,
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
        gamma = pm.Beta("sparsity", sparsity_a, sparsity_b)
        offdiag_sigmas = pm.HalfNormal(
            "offdiag_sigmas", pt.stack([spike_scale, slab_scale])  # pyright: ignore
        )
        offdiag_entries = pm.NormalMixture(
            "offdiag_entries",
            mu=0,
            w=pt.stack([gamma, 1.0 - gamma]),  # type: ignore
            sigma=offdiag_sigmas,
            shape=_offdiag_size(n_mutations),
        )

        # Now sample diagonal rates
        diag = pm.Normal(
            "diagonal",
            mu=diag_mean,
            sigma=diag_sigma,
            size=(n_mutations,),
        )

        pm.Deterministic(
            "fitness_matrix",
            construct_square_matrix(n_mutations, diag=diag, offdiag=offdiag_entries),
        )

    return model
