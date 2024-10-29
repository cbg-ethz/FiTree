import os
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

from fitree import VectorizedTrees


def prepare_SCIFIL_input(vec_trees: VectorizedTrees, output_dir: str):
    """Prepare two inputs for SCIFIL:
    1. A matrix representing the tree structure.
    2. A cell count matrix.

    MATLAB commmand to run SCIFIL:
    matlab -nodisplay -nodesktop -r "folder='{output_dir}';SCIFIL_matrix;exit"
    """

    # Create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create the matrix representing the tree structure
    tree_matrix = np.zeros((vec_trees.n_nodes + 1, vec_trees.n_nodes + 1))
    for i in range(vec_trees.n_nodes):
        tree_matrix[vec_trees.parent_id[i] + 1, i + 1] = 1

    # Create the cell count vector
    cell_count = np.zeros((vec_trees.N_trees, vec_trees.n_nodes + 1))
    cell_count[:, 1:] = vec_trees.seq_cell_number

    # Save the matrix and vector
    np.savetxt(f"{output_dir}/tree_matrix.txt", tree_matrix, fmt="%d")
    np.savetxt(f"{output_dir}/cell_count.txt", cell_count, fmt="%d")


def compute_diffusion_fitness_subclone(vec_trees: VectorizedTrees, eps: float = 1e-16):
    """This function estimates the fitness of each subclone in the tree
    using the diffusion approximation method described in Watson et al. (2020).
    Specifically, we perform maximum likelihood estimation of the fitness
    using the normalized version of density given in Eq.1 of the paper:

    rho(log_vaf) = theta * exp(-exp(log_vaf) / phi)
    """

    population_size = np.sum(vec_trees.seq_cell_number, axis=1)
    vaf = vec_trees.seq_cell_number / (2 * population_size[:, None])
    tau = 1 / vec_trees.beta
    tree_ids = np.arange(vec_trees.N_trees)  # pyright: ignore

    def rho(log_vaf, theta, phi):
        return theta * np.exp(-np.exp(log_vaf) / phi)

    def get_normalizing_constant(theta, phi):
        l_vec = np.log(np.linspace(eps, 0.5, 1000))
        rho_vec = rho(l_vec, theta, phi)

        return np.trapz(rho_vec, l_vec)

    def logp_s_i(log_vaf, theta, phi, i):
        theta_i = theta[i]
        phi_i = phi[i]
        l_i = log_vaf[i]

        return np.log(rho(l_i, theta_i, phi_i)) - np.log(
            get_normalizing_constant(theta_i, phi_i)
        )

    def logp_s(idx, s):
        mu = vec_trees.nu[idx]
        theta = 2 * population_size * mu * tau
        time_vec = vec_trees.sampling_time
        n_tilde = np.expm1(s * time_vec) / (s * tau + eps)
        phi = n_tilde / (2 * population_size)
        f = vaf[:, idx]
        log_vaf = np.where(f > 0, np.log(f), np.log(eps))

        logp = 0.0
        for i in tree_ids:
            logp += logp_s_i(log_vaf, theta, phi, i)

        return -logp

    # Initialize the fitness vector
    s_vec = np.zeros(vec_trees.n_nodes)

    # Optimization for each node using minimize_scalar
    for idx in range(vec_trees.n_nodes):
        res = minimize(lambda s: logp_s(idx, s), 0.1, method="COBYQA")
        s_vec[idx] = res.x[0]

    return s_vec


def compute_diffusion_fitness_mutation(vec_trees: VectorizedTrees, eps: float = 1e-16):
    """This function estimates the fitness of each mutation in the tree
    using the diffusion approximation method described in Watson et al. (2020).
    Specifically, we perform maximum likelihood estimation of the fitness
    using the normalized version of density given in Eq.1 of the paper:

    rho(log_vaf) = theta * exp(-exp(log_vaf) / phi)
    """

    population_size = np.sum(vec_trees.seq_cell_number, axis=1)
    mutations = np.where(np.sum(vec_trees.genotypes, axis=0) > 0)[0]
    mutations.shape[0]

    vaf = vec_trees.seq_cell_number / (2 * population_size[:, None])
    tau = 1 / vec_trees.beta
    tree_ids = np.arange(vec_trees.N_trees)  # pyright: ignore

    def rho(log_vaf, theta, phi):
        return theta * np.exp(-np.exp(log_vaf) / (phi + eps))

    def get_normalizing_constant(theta, phi):
        l_vec = np.log(np.linspace(eps, 0.5, 1000))
        rho_vec = rho(l_vec, theta, phi)

        return np.trapz(rho_vec, l_vec)

    def logp_s_i(log_vaf, theta, phi, i):
        theta_i = theta[i]
        phi_i = phi[i]
        l_i = log_vaf[i]

        return np.log(rho(l_i, theta_i, phi_i)) - np.log(
            get_normalizing_constant(theta_i, phi_i)
        )

    def logp_s(idx, s):
        clones_w_mut = np.where(vec_trees.genotypes[:, idx] > 0)[0]
        mu = np.max(vec_trees.nu[clones_w_mut])
        theta = 2 * population_size * mu * tau
        time_vec = vec_trees.sampling_time
        n_tilde = np.expm1(s * time_vec) / (s * tau + eps)
        phi = n_tilde / (2 * population_size)
        f = np.sum(vaf[:, clones_w_mut], axis=1)
        log_vaf = np.where(f > 0, np.log(f), np.log(eps))

        logp = 0.0
        for i in tree_ids:
            logp += logp_s_i(log_vaf, theta, phi, i)

        return -logp

    # Initialize the fitness vector
    s_vec = np.zeros(vec_trees.genotypes.shape[1])

    # Optimization for each node using minimize_scalar
    for idx in mutations:
        res = minimize(lambda s: logp_s(idx, s), 0.1, method="COBYQA")
        s_vec[idx] = res.x[0]

    s_vec = np.sum(vec_trees.genotypes * s_vec, axis=1)

    return np.array(s_vec)


def weighted_spearman(x, y, w=None):
    if w is None:
        w = np.ones_like(x)
    x_rank = rankdata(x)
    y_rank = rankdata(y)
    return np.corrcoef(x_rank * w, y_rank * w)[0, 1]


def get_available_simulations(n_mutations, N_trees):
    sims = []
    os.chdir("/Users/luox/Documents/Projects/FiTree/workflows")
    for i in range(100):
        required_files = [
            f"results/muts{n_mutations}_trees{N_trees}/sim{i}/fitree_posterior.nc",
            f"results/muts{n_mutations}_trees{N_trees}/sim{i}/diffusion_subclone_fitness.txt",
            f"results/muts{n_mutations}_trees{N_trees}/sim{i}/diffusion_mutation_fitness.txt",
            f"results/muts{n_mutations}_trees{N_trees}/sim{i}/SCIFIL_result.txt",
            f"data/muts{n_mutations}_trees{N_trees}/sim{i}/vectorized_trees.npz",
            f"data/muts{n_mutations}_trees{N_trees}/sim{i}/fitness_matrix.npz",
        ]
        if all(os.path.exists(f) for f in required_files):
            sims.append(i)
    return sims
