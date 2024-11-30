import os
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

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


def prepare_fitclone_input(
    vec_trees: VectorizedTrees, data_dir: str, results_dir: str, exe_dir: str
):
    """This function prepares the input for fitclone.

    The return files are:
    1. A folder with the tree structure in tsv format (time, K, X)
        - each tree has only two time points (0, sampling_time)
        - We consider all nodes in the union tree.
        - X is the frequency of the node in the tree.
    2. A yaml file with the parameters.
    3. A python script to run fitclone.
    4. A mapping file to map the node ids in individual trees to the union tree.

    Note: need to save everything to the fitclone directory.
    """

    # Create directory if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Create the mapping file
    mapping_str = "tree_id node_id union_node_id\n"

    # Process the cell numbers to get the frequencies
    cell_numbers = vec_trees.cell_number * vec_trees.observed
    # add first column for the root with size C_0
    cell_numbers = np.hstack(
        (np.ones((vec_trees.N_trees, 1)) * vec_trees.C_0, cell_numbers)
    )
    # divide by row sums
    frequencies = cell_numbers / cell_numbers.sum(axis=1)[:, None]
    initial_freq = np.zeros(vec_trees.n_nodes + 1)
    initial_freq[0] = 1.0

    # For each tree in vec_trees, prepare a separate tsv file
    for tree_id in range(vec_trees.N_trees):
        # Create a subdirectory for the tree
        if not os.path.exists(f"{data_dir}/tree_{tree_id}"):
            os.makedirs(f"{data_dir}/tree_{tree_id}")

        current_node_id = 0

        # Create the tsv file
        with open(f"{data_dir}/tree_{tree_id}/tree.tsv", "w") as f:
            # header: "time"	"K"	"X" (with quotes around each word)
            f.write('"time"\t"K"\t"X"\n')
            # write the root
            f.write(f"0\t0\t{initial_freq[0]:.6f}\n")
            f.write(
                f"{vec_trees.sampling_time[tree_id]}\t0"
                + f"\t{frequencies[tree_id, 0]:.6f}\n"
            )
            # write the leaves
            for i in range(vec_trees.n_nodes):
                if vec_trees.observed[tree_id, i]:
                    current_node_id += 1
                    f.write(f"0\t{current_node_id}\t{initial_freq[i+1]:.6f}\n")
                    f.write(
                        f"{vec_trees.sampling_time[tree_id]}\t"
                        + f"{current_node_id}\t{frequencies[tree_id, i+1]:.6f}\n"
                    )
                    mapping_str += f"{tree_id} {current_node_id} {i}\n"

        # Create the yaml file
        with open(f"{data_dir}/tree_{tree_id}/params.yaml", "w") as f:
            f.write("K: 1\n")
            f.write(f"K_prime: {current_node_id + 1}\n")
            f.write("MCMC_in_Gibbs_nIter: 20\n")
            f.write("disable_ancestor_bridge: true\n")
            f.write("Ne: 500\n")
            f.write("bridge_n_cores: 1\n")
            f.write("do_predict: 0\n")
            f.write("gp_epsilon: 0.005\n")
            f.write("obs_num: 2\n")
            f.write("gp_n_opt_restarts: 20\n")
            f.write("h: 0.1\n")
            f.write("infer_epsilon: 0.025\n")
            f.write("infer_epsilon_tolerance: 0\n")
            f.write("inference_n_iter: 200\n")
            f.write(f"learn_time: {vec_trees.sampling_time[tree_id]}\n")
            f.write("lower_s: -5\n")
            f.write("upper_s: 5\n")
            f.write("n_cores: 1\n")
            f.write("pf_n_particles: 100\n")
            f.write("pf_n_theta: 50\n")
            f.write("pgas_n_particles: 500\n")
            sigma_vec = np.ones(current_node_id + 1) * 0.025
            sigma_str = ", ".join(map(str, sigma_vec))
            f.write(f"proposal_step_sigma: [{sigma_str}]\n")
            f.write("seed: 0\n")
            f.write("original_data: " + f"{data_dir}/tree_{tree_id}/tree.tsv\n")
            f.write("out_path: " + f"{results_dir}/tree_{tree_id}\n")

    # Write the mapping file
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(f"{results_dir}/mapping.txt", "w") as f:
        f.write(mapping_str)

    # Define the script as a multiline string
    script_content = f"""#!/usr/bin/env python3
import os
import argparse
from multiprocessing import Pool
import yaml

# Change to the working directory for revive_conditional.py
os.chdir('{exe_dir}')
exec(open('revive_conditional.py').read())

def process_tree(tree_id):
    print(f"Processing tree ID: {{tree_id}}")
    param_file = f"{data_dir}/tree_{{tree_id}}/params.yaml"
    if not os.path.exists(param_file):
        print(f"Parameter file not found: {{param_file}}")
        return
    CondExp().run_with_config_file(param_file)
    print(f"Finished processing tree ID: {{tree_id}}")

def main():
    parser = argparse.ArgumentParser(description="Run CondExp with specified tree IDs.")
    parser.add_argument('--start', type=int, required=True, help="Start of tree ID range (inclusive).")
    parser.add_argument('--end', type=int, required=True, help="End of tree ID range (inclusive).")
    parser.add_argument('--workers', type=int, default=4, help="Number of parallel workers to use.")
    args = parser.parse_args()

    # Generate the list of tree IDs
    tree_ids = range(args.start, args.end + 1)

    # Use multiprocessing to process the tree IDs in parallel
    with Pool(args.workers) as pool:
        pool.map(process_tree, tree_ids)

if __name__ == "__main__":
    main()
"""  # noqa: E501

    # Write the script to the file
    with open(f"{data_dir}/run_fitclone.py", "w") as f:
        f.write(script_content)


def weighted_spearman(x, y, w=None):
    if w is None:
        w = np.ones_like(x, dtype=bool)
    return spearmanr(x[w], y[w])[0]


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
