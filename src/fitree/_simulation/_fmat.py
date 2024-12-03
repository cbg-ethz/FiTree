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


def generate_fmat(
    rng,
    n_mutations: int,
    mean: float = 0.0,
    sigma: float = 0.2,
    p_diag: float = 0.7,
    p_offdiag: float = 0.3,
    # positive_ratio: float = 0.5,
) -> np.ndarray:
    # Sample the entries of the fitness matrix
    # from spike-and-slab distributions

    assert n_mutations > 0, "n_mutations should be positive."
    assert 0.0 <= p_diag <= 1.0, "p_diag should be between 0 and 1."
    assert 0.0 <= p_offdiag <= 1.0, "p_offdiag should be between 0 and 1."

    diag = rng.normal(loc=mean, scale=sigma, size=n_mutations)
    nonzero_mask = rng.uniform(size=diag.shape) < p_diag
    diag = np.where(nonzero_mask, diag, 0.0)

    # Sort from highest baseline effects to the smallest
    diag = np.sort(diag)[::-1]
    diag = np.round(diag, 2)

    offdiag = rng.normal(
        loc=0.0, scale=sigma, size=n_mutations * (n_mutations - 1) // 2
    )
    offdiag = np.where(rng.uniform(size=offdiag.shape) < p_offdiag, offdiag, 0.0)
    offdiag = np.round(offdiag, 2)

    return construct_matrix(n=n_mutations, diag=diag, offdiag=offdiag)
