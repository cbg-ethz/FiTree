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
    mean: float = 0.0,
    sigma: float = 0.2,
    p_diag: float = 0.7,
    p_offdiag: float = 0.3,
) -> np.ndarray:
    assert n_mutations > 0, "n_mutations should be positive."
    assert 0.0 <= p_diag <= 1.0, "p_diag should be between 0 and 1."
    assert 0.0 <= p_offdiag <= 1.0, "p_offdiag should be between 0 and 1."

    diag = rng.normal(loc=mean, scale=sigma, size=n_mutations)
    diag = np.where(rng.uniform(size=diag.shape) < p_diag, diag, 0.0)
    diag = np.sort(diag)[::-1]  # Sort from highest baseline effects to the smallest
    offdiag = rng.normal(
        loc=0.0, scale=sigma, size=n_mutations * (n_mutations - 1) // 2
    )
    offdiag = np.where(rng.uniform(size=offdiag.shape) < p_offdiag, offdiag, 0.0)

    return construct_matrix(n=n_mutations, diag=diag, offdiag=offdiag)


def generate_fmat(
    rng,
    n_mutations: int,
    base_mean: float = 0.0,
    base_sigma: float = 0.5,
    base_sparsity: float = 0.7,
    epis_mean: float = 0.0,
    epis_sigma: float = 1.0,
    epis_sparsity: float = 0.5,
    positive_ratio: float = 0.5,
) -> np.ndarray:
    # Sample the baseline fitness effects using spike-and-slab
    diag = rng.normal(loc=base_mean, scale=base_sigma, size=n_mutations)
    diag = np.where(rng.uniform(size=diag.shape) < base_sparsity, diag, 0.0)

    # Assign positive fitness effects to a fraction of the mutations
    diag = np.where(diag > 0, -diag, diag)
    nonzero_idx = np.where(diag != 0)[0]
    pos_idx = rng.choice(
        nonzero_idx, size=int(positive_ratio * len(nonzero_idx)), replace=False
    )
    diag[pos_idx] = -diag[pos_idx]

    # Sample the epistatic fitness effects using spike-and-slab
    epis = rng.normal(
        loc=epis_mean, scale=epis_sigma, size=n_mutations * (n_mutations - 1) // 2
    )
    epis = np.where(rng.uniform(size=epis.shape) < epis_sparsity, epis, 0.0)

    # Assemble the fitness matrix
    F_mat = construct_matrix(n=n_mutations, diag=diag, offdiag=epis)

    # Compute the off-diagonal elements
    # F[i, j] -= F[i, i] + F[j, j] for all upper-triangular elements
    for i in range(n_mutations):
        for j in range(i + 1, n_mutations):
            F_mat[i, j] -= F_mat[i, i] + F_mat[j, j]

    return F_mat
