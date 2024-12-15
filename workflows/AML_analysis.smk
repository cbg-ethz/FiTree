######### Imports #########
import fitree
import pymc as pm
import jax
import arviz as az
import os
import numpy as np
import pytensor

jax.config.update("jax_enable_x64", True)


# set export NUMEXPR_MAX_THREADS=128
os.environ["NUMEXPR_MAX_THREADS"] = "128"

######### Setup #########

N_TUNE: list[int] = [200, 500]
N_DRAW: list[int] = [500, 1000]
N_CHAINS: int = 16
RHS_local_scale: list[float] = [0.1, 0.2]
RHS_s2: list[float] = [0.01, 0.02, 0.04]


######### Workflow #########


rule all:
    input:
        expand(
            "results/AML_with_init_tune{N_TUNE}_draw{N_DRAW}_RHS{RHS_local_scale}_{RHS_s2}_chain{chain_id}.nc",
            N_TUNE=N_TUNE,
            N_DRAW=N_DRAW,
            RHS_local_scale=RHS_local_scale,
            RHS_s2=RHS_s2,
            chain_id=range(N_CHAINS),
        ),
        expand(
            "results/AML_without_init_tune{N_TUNE}_draw{N_DRAW}_RHS{RHS_local_scale}_{RHS_s2}_chain{chain_id}.nc",
            N_TUNE=N_TUNE,
            N_DRAW=N_DRAW,
            RHS_local_scale=RHS_local_scale,
            RHS_s2=RHS_s2,
            chain_id=range(N_CHAINS),
        ),


rule run_fitree_with_init:
    input:
        "/cluster/home/luox/FiTree/analysis/sampling/AML_cohort_Morita_2020.json",
    output:
        "results/AML_with_init_tune{N_TUNE}_draw{N_DRAW}_RHS{RHS_local_scale}_{RHS_s2}_chain{chain_id}.nc",
    threads: 4
    resources:
        runtime=10000,
        tasks=1,
        nodes=1,
    run:
        N_draw = int(wildcards.N_DRAW)
        N_tune = int(wildcards.N_TUNE)
        chain_id = int(wildcards.chain_id)
        RHS_local_scale = float(wildcards.RHS_local_scale)
        RHS_s2 = float(wildcards.RHS_s2)

        pytensor.config.compiledir = (
            f"/tmp/fitree_with_init_{N_draw}_{N_tune}_{RHS_local_scale}_{RHS_s2}"
        )

        cohort = fitree.load_cohort_from_json(input[0])

        fitree_joint_likelihood = fitree.FiTreeJointLikelihood(
            cohort, augment_max_level=2
        )

        D = 31 * 32 / 2
        p0 = np.sqrt(5 * D * (2 * 0.95 - 1))
        N = cohort.N_patients
        tau0 = p0 / (D - p0) / np.sqrt(N)
        model = fitree.prior_fitree(
            cohort,
            fmat_prior_type="regularized_horseshoe",
            tau0=tau0,
            local_scale=RHS_local_scale,
            s2=RHS_s2,
        )

        AML_vec_trees, _ = fitree.wrap_trees(cohort, augment_max_level=2)

        F_mat_init = fitree.greedy_init_fmat(AML_vec_trees)
        C_sampling_init, _ = cohort.compute_mean_std_tumor_size()
        nr_neg_samples_init = (
            cohort.N_patients / cohort.lifetime_risk * (1 - cohort.lifetime_risk)
        )

        init_vals = {
            "fitness_matrix": F_mat_init,
            "C_sampling": C_sampling_init,
            "nr_neg_samples": nr_neg_samples_init,
        }

        with model:
            pm.Potential(
                "joint_likelihood",
                fitree_joint_likelihood(
                    model.fitness_matrix, model.C_sampling, model.nr_neg_samples
                ),  # pyright: ignore
            )
            idata = pm.sample(
                draws=N_draw,
                tune=N_tune,
                initvals=init_vals,
                chains=1,
                return_inferencedata=True,
                discard_tuned_samples=False,
            )

            idata.to_netcdf(output[0])


rule run_fitree_without_init:
    input:
        "/cluster/home/luox/FiTree/analysis/sampling/AML_cohort_Morita_2020.json",
    output:
        "results/AML_without_init_tune{N_TUNE}_draw{N_DRAW}_RHS{RHS_local_scale}_{RHS_s2}_chain{chain_id}.nc",
    threads: 4
    resources:
        runtime=10000,
        tasks=1,
        nodes=1,
    run:
        N_draw = int(wildcards.N_DRAW)
        N_tune = int(wildcards.N_TUNE)
        chain_id = int(wildcards.chain_id)
        RHS_local_scale = float(wildcards.RHS_local_scale)
        RHS_s2 = float(wildcards.RHS_s2)

        pytensor.config.compiledir = (
            f"/tmp/fitree_without_init_{N_draw}_{N_tune}_{RHS_local_scale}_{RHS_s2}"
        )

        cohort = fitree.load_cohort_from_json(input[0])

        fitree_joint_likelihood = fitree.FiTreeJointLikelihood(
            cohort, augment_max_level=2
        )

        D = 31 * 32 / 2
        p0 = np.sqrt(5 * D * (2 * 0.95 - 1))
        N = cohort.N_patients
        tau0 = p0 / (D - p0) / np.sqrt(N)
        model = fitree.prior_fitree(
            cohort,
            fmat_prior_type="regularized_horseshoe",
            tau0=tau0,
            local_scale=RHS_local_scale,
            s2=RHS_s2,
        )

        AML_vec_trees, _ = fitree.wrap_trees(cohort, augment_max_level=2)

        with model:
            pm.Potential(
                "joint_likelihood",
                fitree_joint_likelihood(
                    model.fitness_matrix, model.C_sampling, model.nr_neg_samples
                ),  # pyright: ignore
            )
            idata = pm.sample(
                draws=N_draw,
                tune=N_tune,
                chains=1,
                return_inferencedata=True,
                discard_tuned_samples=False,
            )

            idata.to_netcdf(output[0])
