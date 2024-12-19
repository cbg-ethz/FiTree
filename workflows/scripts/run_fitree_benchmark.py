######### Imports #########
import fitree
import numpy as np
import argparse
import pymc as pm
import os

# Initialize the ArgumentParser
parser = argparse.ArgumentParser(description="Parse values for fitree.")

# Add arguments for n_mutations and N_trees
parser.add_argument(
    "--n_mutations", type=int, required=True, help="Number of mutations"
)

parser.add_argument("--N_trees", type=int, required=True, help="Number of trees")

parser.add_argument("-i", type=int, required=True, help="Simulation index")

parser.add_argument("--ncores", type=int, required=True, help="Number of cores")

parser.add_argument("--workdir", type=str, required=True, help="Working directory")

# Parse the command-line arguments
args = parser.parse_args()

# Assign the parsed values to variables
n_mutations = args.n_mutations
N_trees = args.N_trees
i = args.i
ncores = args.ncores
workdir = args.workdir


if __name__ == "__main__":
    # Change the working directory
    os.chdir(workdir)

    # Example: Print the values to verify
    print(f"Number of mutations: {n_mutations}")
    print(f"Number of trees: {N_trees}")
    print(f"Simulation index: {i}")

    cohort = fitree.load_cohort_from_json(
        f"data/muts{n_mutations}_trees{N_trees}/sim{i}/cohort.json"
    )
    F_mat = np.load(f"data/muts{n_mutations}_trees{N_trees}/sim{i}/fitness_matrix.npz")[
        "F_mat"
    ]
    vec_trees = fitree.load_vectorized_trees_npz(
        f"data/muts{n_mutations}_trees{N_trees}/sim{i}/vectorized_trees.npz"
    )

    fitree_joint_likelihood = fitree.FiTreeJointLikelihood(cohort, augment_max_level=2)
    vec_trees = fitree.update_params(vec_trees, F_mat)
    cohort.lifetime_risk = float(fitree.compute_normalizing_constant(vec_trees))

    F_mat_init = fitree.greedy_init_fmat(vec_trees)
    np.savez(
        f"results/muts{n_mutations}_trees{N_trees}/sim{i}/F_mat_init.npz",
        F_mat=F_mat_init,
    )

    p0 = np.round(np.sqrt(5 * (n_mutations**2 + n_mutations) / 2 * (2 * 0.95 - 1)))
    D = n_mutations * (n_mutations + 1) / 2
    N = cohort.N_patients
    tau0 = p0 / (D - p0) / np.sqrt(N)
    model = fitree.prior_fitree(
        cohort,
        fmat_prior_type="regularized_horseshoe",
        tau0=tau0,
        local_scale=0.1,
        s2=0.02,
    )

    with model:
        pm.Potential(
            "joint_likelihood",
            fitree_joint_likelihood(
                model.fitness_matrix,  # pyright: ignore
                model.C_sampling,  # pyright: ignore
                model.nr_neg_samples,  # pyright: ignore
            ),  # pyright: ignore
        )
        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=ncores,
            cores=ncores,
            mp_ctx="spawn",
            blas_cores=None,
            return_inferencedata=True,
        )

    trace.to_netcdf(
        f"results/muts{n_mutations}_trees{N_trees}/sim{i}/fitree_posterior.nc"
    )
