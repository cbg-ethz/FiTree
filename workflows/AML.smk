######### Setup #########
# Parameters
N_TUNE: list[int] = [500]
N_DRAW: list[int] = [2000]
LIFETIME_RISK_STD_FACTOR: list[float] = [1e-8]
N_CHAINS: int = 24
N_SAMPLES: int = 100


######### Workflow #########
rule all:
    input:
        expand(
            "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/chain_{chain}.nc",
            n_tune=N_TUNE,
            n_draw=N_DRAW,
            std=LIFETIME_RISK_STD_FACTOR,
            chain=range(N_CHAINS),
        ),
        expand(
            "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/sampled.npz",
            n_tune=N_TUNE,
            n_draw=N_DRAW,
            std=LIFETIME_RISK_STD_FACTOR,
        ),
        expand(
            "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/trees/sample_{sample_id}.json",
            n_tune=N_TUNE,
            n_draw=N_DRAW,
            std=LIFETIME_RISK_STD_FACTOR,
            sample_id=range(N_SAMPLES),
        ),


rule run_fitree_masked_normal:
    input:
        "/cluster/home/luox/FiTree/analysis/sampling/AML_cohort_Morita_2020.json",
    output:
        "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/chain_{chain}.nc",
    threads: 1
    resources:
        runtime=10080,
        tasks=1,
        nodes=1,
    run:
        import pytensor

        pytensor.config.compiledir = (
            "/cluster/work/bewi/members/xgluo/pytensor_tmp"
            + f"/AML/with_mask_normal_{wildcards.n_tune}_{wildcards.n_draw}_{wildcards.chain}_{wildcards.std}"
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

        fitree_joint_likelihood = fitree.FiTreeJointLikelihood(
            cohort,
            augment_max_level=1,
            conditioning=False,
            lifetime_risk_mean=0.004762,
            lifetime_risk_std=lifetime_risk_std,
        )

        model = fitree.prior_fitree(
            cohort,
            diag_sigma=0.1,
            offdiag_sigma=0.1,
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


rule sample_chain_normal:
    input:
        lambda wildcards: expand(
            "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/chain_{chain}.nc",
            n_tune=wildcards.n_tune,
            n_draw=wildcards.n_draw,
            std=wildcards.std,
            chain=range(N_CHAINS),
        ),
    output:
        "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/sampled.npz",
    threads: 1
    resources:
        runtime=10080,
        tasks=1,
        nodes=1,
        mem_mb_per_cpu=2048,
    params:
        n_samples=N_SAMPLES,
    run:
        import numpy as np
        import arviz as az

        seed = (
            abs(hash(f"{wildcards.n_tune}_{wildcards.n_draw}_{wildcards.std}"))
            % 2**32
        )
        rng = np.random.default_rng(seed)

        chains = [az.from_netcdf(f) for f in input]
        combined = az.concat(chains, dim="chain")
        combined = combined.sel(draw=slice(500, None))
        combined = combined.posterior.stack(sample=("chain", "draw"))
        sampled = combined.isel(
            sample=rng.choice(
                combined.sample.size, size=params.n_samples, replace=False
            )
        )

        np.savez(output[0], F_mat_samples=sampled.fitness_matrix.values)


rule generate_trees_normal:
    input:
        "/cluster/home/luox/FiTree/analysis/sampling/AML_cohort_Morita_2020.json",
        "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/sampled.npz",
    output:
        "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/trees/sample_{sample_id}.json",
    threads: 100
    resources:
        runtime=1440,
        tasks=1,
        nodes=1,
    run:
        import fitree
        import numpy as np

        seed = (
            abs(
                hash(
                    f"{wildcards.n_tune}_{wildcards.n_draw}_{wildcards.std}_{wildcards.sample_id}"
                )
            )
            % 2**32
        )

        cohort = fitree.load_cohort_from_json(input[0])
        F_mat_samples = np.load(input[1])["F_mat_samples"]

        N_trees = 200
        sample_id = int(wildcards.sample_id)

        F_mat = F_mat_samples[:, :, sample_id]

        rng = np.random.default_rng(seed)

        simulated_cohort = fitree.generate_trees(
            rng=rng,
            n_mutations=cohort.n_mutations,
            N_trees=N_trees,
            mu_vec=cohort.mu_vec,
            F_mat=F_mat,
            common_beta=cohort.common_beta,
            C_0=cohort.C_0,
            C_sampling=cohort.C_sampling,
            C_seq=1e6,
            t_max=cohort.t_max,
            tau=1e-2,
            return_time=True,
            parallel=True,
        )
        fitree.save_cohort_to_json(simulated_cohort, output[0])
