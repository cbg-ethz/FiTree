######### Setup #########
# Parameters
N_TUNE: list[int] = [500]
N_DRAW: list[int] = [500, 1000]
LIFETIME_RISK_STD_FACTOR: list[float] = [1e-6, 1e-7, 1e-8]
N_CHAINS: int = 24


######### Workflow #########
rule all:
    input:
        expand(
            "results/AML_with_mask_mixed_tune{n_tune}_draw{n_draw}_std{std}/chain_{chain}.nc",
            n_tune=N_TUNE,
            n_draw=N_DRAW,
            std=LIFETIME_RISK_STD_FACTOR,
            chain=range(N_CHAINS),
        ),


rule run_fitree_with_mask_mixed:
    input:
        "/cluster/home/luox/FiTree/analysis/sampling/AML_cohort_Morita_2020.json",
    output:
        "results/AML_with_mask_mixed_tune{n_tune}_draw{n_draw}_std{std}/chain_{chain}.nc",
    threads: 1
    resources:
        runtime=10080,
        tasks=1,
        nodes=1,
    run:
        import pytensor

        pytensor.config.compiledir = (
            "/cluster/work/bewi/members/xgluo/pytensor_tmp"
            + f"/AML/with_mask_mixed_{wildcards.n_tune}_{wildcards.n_draw}_{wildcards.chain}_{wildcards.std}"
        )

        import fitree
        import pymc as pm
        import arviz as az
        import os
        import numpy as np

        n_tune = int(wildcards.n_tune)
        n_draw = int(wildcards.n_draw)
        chain = int(wildcards.chain)
        lifetime_risk_std = float(wildcards.std)

        seed = abs(hash(f"{n_tune}_{n_draw}_{chain}_{lifetime_risk_std}")) % 2**32
        cohort = fitree.load_cohort_from_json(input[0])
        vec_trees, _ = fitree.wrap_trees(cohort, augment_max_level=2)

        fitree_joint_likelihood = fitree.FiTreeJointLikelihood(
            cohort,
            augment_max_level=2,
            conditioning=False,
            lifetime_risk_mean=0.004762,
            lifetime_risk_std=lifetime_risk_std,
        )

        n_mutations = 15
        p0 = np.round(
            np.sqrt(5 * (n_mutations**2 - n_mutations) / 2 * (2 * 0.95 - 1))
        )
        D = n_mutations * (n_mutations - 1) / 2
        N = cohort.N_patients
        tau0 = p0 / (D - p0) / np.sqrt(N)

        model = fitree.prior_fitree_mixed(
            cohort,
            diag_sigma=0.05,
            local_scale=0.3,
            s2=0.09,
            tau0=tau0,
            min_occurrences=3,
        )

        with model:
            pm.Potential(
                "joint_likelihood",
                fitree_joint_likelihood(
                    model.fitness_matrix,  # pyright: ignore
                ),  # pyright: ignore
            )
            trace = pm.sample(
                draws=n_draw,
                tune=n_tune,
                chains=1,
                return_inferencedata=True,
                random_seed=seed,
            )

        trace.to_netcdf(output[0])

        if os.path.exists(pytensor.config.compiledir):
            import shutil

            shutil.rmtree(pytensor.config.compiledir)
