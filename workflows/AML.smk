from scripts.benchmarking_helpers import (
    prepare_SCIFIL_input,
    compute_diffusion_fitness_subclone,
    compute_diffusion_fitness_mutation,
    prepare_fitclone_input,
)

######### Setup #########
# Parameters
N_TUNE: list[int] = [500]
N_DRAW: list[int] = [2000]
LIFETIME_RISK_STD_FACTOR: list[float] = [1e-8]
N_CHAINS: int = 24
N_SAMPLES: int = 100

AML_cohort_file: str = "/cluster/home/luox/FiTree/analysis/sampling/AML_cohort_Morita_2020.json"  # replace with the path to the input file
scifil_exe_dir: str = (
    "/cluster/home/luox/SCIFIL"  # replace with the path to the SCIFIL executable
)
fitclone_exe_dir: str = "/cluster/home/luox/fitclone/fitclone"  # replace with the path to the fitclone executable


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
        expand(
            "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/SCIFIL_result.txt",
            n_tune=N_TUNE,
            n_draw=N_DRAW,
            std=LIFETIME_RISK_STD_FACTOR,
        ),
        expand(
            [
                "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/diffusion_subclone_fitness.txt",
                "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/diffusion_mutation_fitness.txt",
            ],
            n_tune=N_TUNE,
            n_draw=N_DRAW,
            std=LIFETIME_RISK_STD_FACTOR,
        ),
        expand(
            "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/fitclone_fitness.txt",
            n_tune=N_TUNE,
            n_draw=N_DRAW,
            std=LIFETIME_RISK_STD_FACTOR,
        ),


rule run_fitree_masked_normal:
    input:
        AML_cohort_file,
    output:
        "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/chain_{chain}.nc",
    threads: 1
    resources:
        runtime=10080,
        tasks=1,
        nodes=1,
    run:
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
        AML_cohort_file,
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


rule prepare_SCIFIL_input:
    input:
        AML_cohort_file,
    output:
        "data/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/tree_matrix.txt",
        "data/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/cell_count.txt",
    threads: 1
    resources:
        runtime=60,
        tasks=1,
        nodes=1,
    run:
        import fitree

        cohort = fitree.load_cohort_from_json(input[0])
        vec_trees, _ = fitree.wrap_trees(cohort, augment_max_level=1)

        scifil_output_dir = os.path.dirname(output[0])
        os.makedirs(scifil_output_dir, exist_ok=True)

        prepare_SCIFIL_input(vec_trees, scifil_output_dir)


rule run_SCIFIL:
    input:
        "data/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/SCIFIL_input/tree_matrix.txt",
        "data/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/SCIFIL_input/cell_count.txt",
    output:
        "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/SCIFIL_result.txt",
    threads: 1
    resources:
        runtime=10080,
        tasks=1,
        nodes=1,
        mem_mb_per_cpu=2048,
    params:
        scifil_dir=scifil_exe_dir,
        input_dir=os.path.abspath(
            "data/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/SCIFIL_input"
        ),
        output=os.path.abspath(
            "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/SCIFIL_result.txt"
        ),
    shell:
        """
        cd {params.scifil_dir} && \
        matlab -nodisplay -nodesktop -r "folder='{params.input_dir}';output='{params.output}';SCIFIL_matrix;exit"
        """


rule run_diffusion:
    input:
        AML_cohort_file,
    output:
        "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/diffusion_subclone_fitness.txt",
        "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/diffusion_mutation_fitness.txt",
    threads: 1
    resources:
        runtime=60,
        tasks=1,
        nodes=1,
    run:
        import fitree
        import numpy as np

        cohort = fitree.load_cohort_from_json(input[0])
        vec_trees, _ = fitree.wrap_trees(cohort, augment_max_level=1)

        # compute subclone fitness
        fitness = compute_diffusion_fitness_subclone(vec_trees)
        np.savetxt(output[0], fitness)

        # compute mutation fitness
        fitness = compute_diffusion_fitness_mutation(vec_trees)
        np.savetxt(output[1], fitness)


rule prepare_fitclone_input:
    input:
        AML_cohort_file,
    output:
        "data/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/fitclone_input.done",
    threads: 1
    resources:
        runtime=60,
        tasks=1,
        nodes=1,
    params:
        fitclone_data_dir=os.path.abspath(
            "data/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/fitclone_input"
        ),
        fitclone_results_dir=os.path.abspath(
            "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/fitclone_results"
        ),
        fitclone_exe_dir=fitclone_exe_dir,
    run:
        import fitree

        cohort = fitree.load_cohort_from_json(input[0])
        vec_trees, _ = fitree.wrap_trees(cohort, augment_max_level=1)

        os.makedirs(params.fitclone_data_dir, exist_ok=True)

        prepare_fitclone_input(
            vec_trees,
            params.fitclone_data_dir,
            params.fitclone_results_dir,
            params.fitclone_exe_dir,
        )

        with open(output[0], "w") as f:
            f.write("")


rule run_fitclone:
    input:
        "data/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/fitclone_input.done",
    output:
        "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/fitclone_results/.done",
    threads: 124
    resources:
        runtime=10080,
        tasks=1,
        nodes=1,
    params:
        fitclone_input_dir=os.path.abspath(
            "data/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/fitclone_input"
        ),
        fitclone_output_dir=os.path.abspath(
            "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/fitclone_results"
        ),
    shell:
        """
        cd {params.fitclone_output_dir} && \
        rm -rf tree* && \
        cd {params.fitclone_input_dir} && \
        chmod a+x run_fitclone.py && \
        export NUMEXPR_MAX_THREADS=123 && \
        export OPENBLAS_NUM_THREADS=1 && \
        ./run_fitclone.py --start 0 --end 122 --workers 123 && \
        touch {params.fitclone_output_dir}/.done
        """


rule process_fitclone_output:
    input:
        "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/fitclone_results/.done",
    output:
        "results/AML_with_mask_normal_tune{n_tune}_draw{n_draw}_std{std}/fitclone_fitness.txt",
    threads: 1
    resources:
        runtime=60,
        tasks=1,
        nodes=1,
    run:
        import pandas as pd
        import gzip
        import os
        from glob import glob

        # Paths
        base_dir = os.path.dirname(input[0])
        mapping_file = os.path.join(base_dir, "mapping.txt")
        output_file = output[0]

        # Read mapping.txt
        print(f"Reading mapping file: {mapping_file}")
        mapping = pd.read_csv(
            mapping_file, sep=" ", names=["tree_id", "node_id", "union_node_id"]
        )

        # Prepare output data
        output_data = []

        # Process each tree directory
        for tree_id in mapping["tree_id"].unique():
            # Use glob to dynamically find the correct directory
            tree_dirs = glob(os.path.join(base_dir, f"tree_{tree_id}_*"))
            if not tree_dirs:
                print(f"No directory found for tree ID {tree_id}. Skipping.")
                continue

                # Use the first matching directory
            tree_dir = tree_dirs[0]
            infer_theta_file = os.path.join(tree_dir, "infer_theta.tsv.gz")

            # Check if the infer_theta.tsv.gz file exists
            if not os.path.exists(infer_theta_file):
                print(
                    f"File not found: {infer_theta_file}. Skipping tree ID {tree_id}."
                )
                continue

            print(
                f"Processing tree ID {tree_id}, directory: {tree_dir}, file: {infer_theta_file}"
            )

            # Unzip and read infer_theta.tsv.gz
            with gzip.open(infer_theta_file, "rt") as f:
                theta_data = pd.read_csv(f, sep="\t", header=None)

                # Ignore the first column (column 0)
            theta_data = theta_data.iloc[:, 1:]

            # Select the last 100 rows
            theta_last_100 = theta_data.iloc[-100:]

            # Compute medians
            medians = theta_last_100.median(axis=0).values

            # Map medians to node IDs using mapping
            tree_mapping = mapping[mapping["tree_id"] == tree_id]
            for _, row in tree_mapping.iterrows():
                node_id = int(row["node_id"])
                union_node_id = row["union_node_id"]

                # Get the corresponding median value
                # Adjust index for columns, skipping column 0 in theta_data
                if node_id - 1 < len(medians):
                    median_value = medians[node_id - 1]  # Node IDs are 1-indexed
                else:
                    print(
                        f"Node ID {node_id} out of range for tree ID {tree_id}. Skipping."
                    )
                    continue

                    # Append to output data
                output_data.append([tree_id, node_id, union_node_id, median_value])

                # Save the output
        output_df = pd.DataFrame(
            output_data, columns=["tree_id", "node_id", "union_node_id", "median_value"]
        )
        output_df.to_csv(output_file, sep="\t", index=False)
        print(f"Mapped medians saved to {output_file}.")
