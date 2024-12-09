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
    mean: float = 0.05,
    sigma: float = 0.1,
    p_diag: float = 0.7,
    p_offdiag: float = 0.3,
    positive_ratio: float = 0.5,
) -> np.ndarray:
    # Sample the entries of the fitness matrix
    # from spike-and-slab distributions

    # Determine the parameters of the log-normal distribution
    lnorm_var = np.log1p(sigma**2 / mean**2)
    lnorm_mean = np.log(mean) - 0.5 * lnorm_var
    lnorm_std = np.sqrt(lnorm_var)

    # Sample the diagonal elements
    diag = rng.lognormal(mean=lnorm_mean, sigma=lnorm_std, size=n_mutations)
    nonzero_mask = rng.uniform(size=diag.shape) < p_diag
    diag = np.where(nonzero_mask, diag, 0.0)

    nonzero_idx = np.where(nonzero_mask)[0]
    nr_nonzero = nonzero_idx.size
    neg_idx = rng.choice(
        nonzero_idx, size=int(nr_nonzero * (1 - positive_ratio)), replace=False
    )
    diag[neg_idx] = -diag[neg_idx]

    # Sort from highest diagonal effects to the smallest
    diag = np.sort(diag)[::-1]
    diag = np.round(diag, 2)

    # Sample the off-diagonal elements
    offdiag = rng.lognormal(
        mean=lnorm_mean, sigma=lnorm_std, size=n_mutations * (n_mutations - 1) // 2
    )
    nonzero_mask = rng.uniform(size=offdiag.shape) < p_offdiag
    offdiag = np.where(nonzero_mask, offdiag, 0.0)

    nonzero_idx = np.where(nonzero_mask)[0]
    nr_nonzero = nonzero_idx.size
    nonzero_idx = np.where(nonzero_mask)[0]

    # sample half of the off-diagonal elements as negative
    neg_idx = rng.choice(nonzero_idx, size=int(nr_nonzero * 0.5), replace=False)
    offdiag[neg_idx] = -offdiag[neg_idx]

    offdiag = np.round(offdiag, 2)

    return construct_matrix(n=n_mutations, diag=diag, offdiag=offdiag)
