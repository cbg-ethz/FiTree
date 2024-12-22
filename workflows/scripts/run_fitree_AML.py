######### Imports #########
import argparse
import fitree
import pymc as pm
import jax
import os
import numpy as np

jax.config.update("jax_enable_x64", True)

# Initialize the ArgumentParser
parser = argparse.ArgumentParser(description="Parse values for fitree.")

parser.add_argument("--n_tune", type=int, required=True, help="Number of tuning steps")

parser.add_argument("--n_samples", type=int, required=True, help="Number of samples")

parser.add_argument("--n_chains", type=int, required=True, help="Number of chains")

parser.add_argument(
    "--local_scale", type=float, required=True, help="Regularized horseshoe local scale"
)

parser.add_argument("--s2", type=float, required=True, help="Regularized horseshoe s2")

parser.add_argument("--workdir", type=str, required=True, help="Working directory")

parser.add_argument("--seed", type=int, required=False, help="Random seed")

parser.add_argument("--mask", type=bool, required=False, help="Mask", default=False)


# Parse the command-line arguments
args = parser.parse_args()

# Assign the parsed values to variables
n_tune = args.n_tune
n_samples = args.n_samples
n_chains = args.n_chains
local_scale = args.local_scale
s2 = args.s2
workdir = args.workdir
seed = args.seed
mask = args.mask

if __name__ == "__main__":
    os.environ["NUMEXPR_MAX_THREADS"] = f"{n_chains}"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    os.chdir(workdir)

    cohort = fitree.load_cohort_from_json(
        "/cluster/home/luox/FiTree/analysis/sampling/AML_cohort_Morita_2020.json"
    )

    fitree_joint_likelihood = fitree.FiTreeJointLikelihood(cohort, augment_max_level=2)

    D = 31 * 32 / 2
    p0 = np.sqrt(5 * D * (2 * 0.95 - 1))
    N = cohort.N_patients
    tau0 = p0 / (D - p0) / np.sqrt(N)
    model = fitree.prior_fitree(
        cohort,
        fmat_prior_type="regularized_horseshoe",
        tau0=tau0,
        local_scale=local_scale,
        s2=s2,
    )

    AML_vec_trees, _ = fitree.wrap_trees(cohort, augment_max_level=2)

    with model:
        pm.Potential(
            "joint_likelihood",
            fitree_joint_likelihood(
                model.fitness_matrix,  # pyright: ignore
                model.C_sampling,  # pyright: ignore
                model.nr_neg_samples,  # pyright: ignore
            ),  # pyright: ignore
        )
        idata = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            cores=n_chains,
            blas_cores=None,
            mp_ctx="spawn",
            return_inferencedata=True,
            discard_tuned_samples=False,
        )

    # Save the InferenceData object
    if not mask:
        idata.to_netcdf(
            f"results/AML_no_mask_tune{n_tune}_draw{n_samples}_RHS{local_scale}_{s2}.nc"
        )
    else:
        idata.to_netcdf(
            f"results/AML_mask_tune{n_tune}_draw{n_samples}_RHS{local_scale}_{s2}.nc"
        )
