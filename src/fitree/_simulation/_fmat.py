import numpy as np


def construct_matrix(n: int, diag: np.ndarray, offdiag: np.ndarray) -> np.ndarray:
    assert diag.shape == (n,)
    assert offdiag.shape == (n * (n - 1) // 2,)

    # Create a square matrix of size n filled with zeros
    mat = np.zeros((n, n))

    # Set the diagonal elements
    np.fill_diagonal(mat, diag)

    # Set the upper-triangular off-diagonal elements
    upper_triangular_indices = np.triu_indices(n, k=1)
    mat[upper_triangular_indices] = offdiag

    return mat


def sample_spike_and_slab(
    rng,
    n_mutations: int,
    diag_mean: float = 0.0,
    diag_sigma: float = 1.0,
    offdiag_effect: float = 1.0,
    p_offdiag: float = 0.2,
) -> np.ndarray:
    """Samples a matrix using diagonal terms from a normal
    distribution and offdiagonal terms sampled from spike and slab
    distribution.
    Author: Pawel Czyz
    Source: https://github.com/cbg-ethz/pMHN/

    Args:
        rng: NumPy random number generator.
        n_mutations: number of mutations.
        diag_mean: mean of the normal distribution used to sample
            diagonal terms.
        diag_scale: standard deviation of the normal distribution
            used to sample diagonal terms.
        offdiag_effect: the standard deviation of the slab used
            to sample non-zero offdiagonal terms
        p_offdiag: the probability of sampling a non-zero offdiagonal
            term.
    """
    assert n_mutations > 0, "n_mutations should be positive."
    assert 0.0 <= p_offdiag <= 1.0, "p_offdiag should be between 0 and 1."

    diag = rng.normal(loc=diag_mean, scale=diag_sigma, size=n_mutations)
    diag = np.sort(diag)[::-1]  # Sort from highest baseline effects to the smallest
    offdiag = rng.normal(
        loc=0.0, scale=offdiag_effect, size=n_mutations * (n_mutations - 1) // 2
    )
    offdiag = np.where(rng.uniform(size=offdiag.shape) < p_offdiag, offdiag, 0.0)
    return construct_matrix(n=n_mutations, diag=diag, offdiag=offdiag)
