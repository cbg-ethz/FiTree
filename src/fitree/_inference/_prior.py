import pymc as pm
import pytensor.tensor as pt

# Names of the variables in the PyMC model
_BASELINE_RATES: str = "baseline_rates"
_F_MAT: str = "fitness_matrix"


def construct_square_matrix(n: int, diagonal, offdiag):
    """Constructs a square matrix from the diagonal and off-diagonal elements.
    This function is taken from the pMHN package source code
    (https://github.com/cbg-ethz/pMHN).
    Author: Pawel Czyz

    Args:
        n: size of the matrix
        diagonal: vector of shape (n,) containing the diagonal elements
        offdiag: vector of shape (n*(n-1),) containing the off-diagonal elements

    Returns:
        matrix of shape (n, n)
    """
    # Create a square matrix of size n filled with zeros
    mat = pt.zeros((n, n))

    # Set the off-diagonal elements
    off_diag_indices = pt.nonzero(~pt.eye(n, dtype="bool"))  # type: ignore
    mat = pt.set_subtensor(mat[off_diag_indices], offdiag)  # type: ignore

    # Set the diagonal elements
    diag_indices = pt.arange(n), pt.arange(n)
    mat = pt.set_subtensor(mat[diag_indices], diagonal)  # type: ignore

    return mat


def prior_only_baseline_rates(
    n_mutations: int,
    mean: float = 0.0,
    sigma: float = 1.0,
) -> pm.Model:
    """Constructs a PyMC model in which the theta
    matrix contains only diagonal entries."""

    with pm.Model() as model:  # type: ignore
        baselines = pm.Normal(
            _BASELINE_RATES,
            mu=mean,
            sigma=sigma,
            size=(n_mutations,),
        )
        mask = pt.eye(n_mutations)
        pm.Deterministic(_F_MAT, pt.exp(mask * baselines))  # type: ignore
    return model
