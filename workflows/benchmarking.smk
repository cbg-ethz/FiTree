######### Imports #########
import os

from scripts.benchmarking_helpers import (
    prepare_SCIFIL_input,
    prepare_fitclone_input,
    compute_diffusion_fitness_subclone,
    compute_diffusion_fitness_mutation,
    weighted_spearman,
)


######### Simulation setup #########

# cluster setup
N_MUTATIONS: list[int] = [5, 10, 15]
N_TREES: list[int] = [200, 500]
N_SIMULATIONS: int = 100
N_CHAINS: int = 12

fitclone_exe_dir: str = "/cluster/home/luox/fitclone/fitclone"
scifil_exe_dir: str = "/cluster/home/luox/SCIFIL"


######### Workflow #########


rule all:
    input:
        expand(
            "data/muts{n_mutations}_trees{N_trees}/sim{i}/fitness_matrix.npz",
            n_mutations=N_MUTATIONS,
            N_trees=N_TREES,
            i=range(N_SIMULATIONS),
        ),
        expand(
            "data/muts{n_mutations}_trees{N_trees}/sim{i}/cohort.json",
            n_mutations=N_MUTATIONS,
            N_trees=N_TREES,
            i=range(N_SIMULATIONS),
        ),
        expand(
            "data/muts{n_mutations}_trees{N_trees}/sim{i}/vectorized_trees.npz",
            n_mutations=N_MUTATIONS,
            N_trees=N_TREES,
            i=range(N_SIMULATIONS),
        ),
        expand(
            "data/muts{n_mutations}_trees{N_trees}/sim{i}/SCIFIL_input/tree_matrix.txt",
            n_mutations=N_MUTATIONS,
            N_trees=N_TREES,
            i=range(N_SIMULATIONS),
        ),
        expand(
            "data/muts{n_mutations}_trees{N_trees}/sim{i}/SCIFIL_input/cell_count.txt",
            n_mutations=N_MUTATIONS,
            N_trees=N_TREES,
            i=range(N_SIMULATIONS),
        ),
        expand(
            "data/muts5_trees{N_trees}/sim{i}/fitclone_input/.done",
            N_trees=N_TREES,
            i=range(N_SIMULATIONS),
        ),
        expand(
            "results/muts{n_mutations}_trees{N_trees}/sim{i}/SCIFIL_result.txt",
            n_mutations=N_MUTATIONS,
            N_trees=N_TREES,
            i=range(N_SIMULATIONS),
        ),
        expand(
            "results/muts{n_mutations}_trees{N_trees}/sim{i}/diffusion_subclone_fitness.txt",
            n_mutations=N_MUTATIONS,
            N_trees=N_TREES,
            i=range(N_SIMULATIONS),
        ),
        expand(
            "results/muts{n_mutations}_trees{N_trees}/sim{i}/diffusion_mutation_fitness.txt",
            n_mutations=N_MUTATIONS,
            N_trees=N_TREES,
            i=range(N_SIMULATIONS),
        ),
        expand(
            "results/muts5_trees{N_trees}/sim{i}/fitclone_results/.done",
            N_trees=N_TREES,
            i=range(N_SIMULATIONS),
        ),
        expand(
            "results/muts5_trees{N_trees}/sim{i}/fitclone_fitness.txt",
            N_trees=N_TREES,
            i=range(N_SIMULATIONS),
        ),
        expand(
            "results/muts{n_mutations}_trees{N_trees}/sim{i}/fitree_posterior/chain{chain}_masked_mixed.nc",
            n_mutations=N_MUTATIONS,
            N_trees=N_TREES,
            i=range(N_SIMULATIONS),
            chain=range(N_CHAINS),
        ),
        expand(
            "results/muts{n_mutations}_trees{N_trees}/sim{i}/fitree_posterior_masked_mixed.nc",
            n_mutations=N_MUTATIONS,
            N_trees=N_TREES,
            i=range(N_SIMULATIONS),
        ),
        expand(
            "results/muts{n_mutations}_trees{N_trees}/sc_summary.txt",
            n_mutations=N_MUTATIONS,
            N_trees=N_TREES,
        ),
        expand(
            "results/muts{n_mutations}_trees{N_trees}/fitness_summary.txt",
            n_mutations=N_MUTATIONS,
            N_trees=N_TREES,
        ),


rule generate_data:
    output:
        "data/muts{n_mutations}_trees500/sim{i}/fitness_matrix.npz",
        "data/muts{n_mutations}_trees500/sim{i}/cohort.json",
        "data/muts{n_mutations}_trees500/sim{i}/vectorized_trees.npz",
    threads: 100
    resources:
        runtime=1440,
        tasks=1,
        nodes=1,
        mem_mb_per_cpu=500,
    run:
        import numpy as np
        import fitree

        N_trees = 500
        n_mutations = int(wildcards.n_mutations)
        i = int(wildcards.i)
        rng = np.random.default_rng(2024 * i + i)

        max_fitness = -np.inf

        # ensure the diagonal of F_mat has at least one value > 0.12
        # such that the runtime of the simulation is not too long
        while max_fitness < 0.12:
            F_mat = fitree.generate_fmat(
                rng=rng,
                n_mutations=n_mutations,
                mean_diag=0.12,
                sigma_diag=0.03,
                mean_offdiag=0.0,
                sigma_offdiag=0.5,
            )
            max_fitness = np.max(np.diag(F_mat))
        np.savez(output[0], F_mat=F_mat)

        mu_vec = np.ones(n_mutations) * 3e-7
        cohort = fitree.generate_trees(
            rng=rng,
            n_mutations=n_mutations,
            N_trees=N_trees,
            mu_vec=mu_vec,
            F_mat=F_mat,
            common_beta=1.0,
            C_0=1e5,
            C_seq=1e4,
            C_sampling=1e8,
            tau=1e-2,
            t_max=100,
            return_time=True,
            parallel=True,
        )
        fitree.save_cohort_to_json(cohort, output[1])

        vec_trees, _ = fitree.wrap_trees(cohort, augment_max_level=2)
        fitree.save_vectorized_trees_npz(vec_trees, output[2])


rule subsample_trees:
    input:
        "data/muts{n_mutations}_trees500/sim{i}/fitness_matrix.npz",
        "data/muts{n_mutations}_trees500/sim{i}/cohort.json",
    output:
        "data/muts{n_mutations}_trees200/sim{i}/fitness_matrix.npz",
        "data/muts{n_mutations}_trees200/sim{i}/cohort.json",
        "data/muts{n_mutations}_trees200/sim{i}/vectorized_trees.npz",
    threads: 1
    resources:
        runtime=60,
        tasks=1,
        nodes=1,
    run:
        import numpy as np
        import fitree

        n_mutations = int(wildcards.n_mutations)
        i = int(wildcards.i)

        # Load data
        F_mat = np.load(input[0])["F_mat"]
        cohort = fitree.load_cohort_from_json(input[1])

        # Subsample trees
        rng = np.random.default_rng(2024 * i + i)
        patient_indices = rng.choice(cohort.N_patients, size=200, replace=False)
        new_cohort = fitree.TumorTreeCohort(
            name=cohort.name,
            trees=[cohort.trees[i] for i in patient_indices],
            n_mutations=cohort.n_mutations,
            N_trees=200,
            N_patients=200,
            mu_vec=cohort.mu_vec,
            common_beta=cohort.common_beta,
            C_0=cohort.C_0,
            C_seq=cohort.C_seq,
            C_sampling=cohort.C_sampling,
            t_max=cohort.t_max,
            mutation_labels=cohort.mutation_labels,
            tree_labels=[cohort.tree_labels[i] for i in patient_indices],
            patient_labels=[cohort.patient_labels[i] for i in patient_indices],
        )
        vec_trees, _ = fitree.wrap_trees(new_cohort, augment_max_level=2)

        # Save data
        np.savez(output[0], F_mat=F_mat)
        fitree.save_cohort_to_json(new_cohort, output[1])
        fitree.save_vectorized_trees_npz(vec_trees, output[2])


rule prepare_SCIFIL_input:
    input:
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/vectorized_trees.npz",
    output:
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/SCIFIL_input/tree_matrix.txt",
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/SCIFIL_input/cell_count.txt",
    threads: 1
    resources:
        runtime=60,
        tasks=1,
        nodes=1,
    run:
        import fitree

        vec_trees = fitree.load_vectorized_trees_npz(input[0])

        scifil_output_dir = os.path.dirname(output[0])
        os.makedirs(scifil_output_dir, exist_ok=True)

        prepare_SCIFIL_input(vec_trees, scifil_output_dir)


rule prepare_fitclone_input:
    input:
        "data/muts5_trees{N_trees}/sim{i}/vectorized_trees.npz",
    output:
        "data/muts5_trees{N_trees}/sim{i}/fitclone_input/.done",
    threads: 1
    resources:
        runtime=60,
        tasks=1,
        nodes=1,
    params:
        fitclone_data_dir=os.path.abspath(
            "data/muts5_trees{N_trees}/sim{i}/fitclone_input"
        ),
        fitclone_results_dir=os.path.abspath(
            "results/muts5_trees{N_trees}/sim{i}/fitclone_results"
        ),
        fitclone_exe_dir=fitclone_exe_dir,
    run:
        import fitree

        vec_trees = fitree.load_vectorized_trees_npz(input[0])

        os.makedirs(params.fitclone_data_dir, exist_ok=True)

        prepare_fitclone_input(
            vec_trees,
            params.fitclone_data_dir,
            params.fitclone_results_dir,
            params.fitclone_exe_dir,
        )

        with open(output[0], "w") as f:
            f.write("")


rule run_SCIFIL:
    input:
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/SCIFIL_input/tree_matrix.txt",
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/SCIFIL_input/cell_count.txt",
    output:
        "results/muts{n_mutations}_trees{N_trees}/sim{i}/SCIFIL_result.txt",
    params:
        scifil_dir=scifil_exe_dir,
        input_dir=os.path.abspath(
            "data/muts{n_mutations}_trees{N_trees}/sim{i}/SCIFIL_input"
        ),
        output=os.path.abspath(
            "results/muts{n_mutations}_trees{N_trees}/sim{i}/SCIFIL_result.txt"
        ),
    threads: 1
    resources:
        runtime=2880,
        tasks=1,
        nodes=1,
        mem_mb_per_cpu=2048,
    shell:
        """
        cd {params.scifil_dir} && \
        matlab -nodisplay -nodesktop -r "folder='{params.input_dir}';output='{params.output}';SCIFIL_matrix;exit"
        """


rule run_diffusion:
    input:
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/vectorized_trees.npz",
    output:
        "results/muts{n_mutations}_trees{N_trees}/sim{i}/diffusion_subclone_fitness.txt",
        "results/muts{n_mutations}_trees{N_trees}/sim{i}/diffusion_mutation_fitness.txt",
    threads: 1
    resources:
        runtime=60,
        tasks=1,
        nodes=1,
    run:
        import fitree
        import numpy as np

        vec_trees = fitree.load_vectorized_trees_npz(input[0])

        # compute subclone fitness
        fitness = compute_diffusion_fitness_subclone(vec_trees)
        np.savetxt(output[0], fitness)

        # compute mutation fitness
        fitness = compute_diffusion_fitness_mutation(vec_trees)
        np.savetxt(output[1], fitness)


rule run_fitclone:
    input:
        "data/muts5_trees{N_trees}/sim{i}/fitclone_input/.done",
    output:
        "results/muts5_trees{N_trees}/sim{i}/fitclone_results/.done",
    threads: 100
    resources:
        runtime=240,
        tasks=1,
        nodes=1,
    params:
        fitclone_input_dir=os.path.abspath(
            "data/muts5_trees{N_trees}/sim{i}/fitclone_input"
        ),
        fitclone_output_dir=os.path.abspath(
            "results/muts5_trees{N_trees}/sim{i}/fitclone_results"
        ),
    shell:
        """
        cd {params.fitclone_output_dir} && \
        rm -rf tree* && \
        cd {params.fitclone_input_dir} && \
        chmod a+x run_fitclone.py && \
        export NUMEXPR_MAX_THREADS=100 && \
        export OPENBLAS_NUM_THREADS=1 && \
        ./run_fitclone.py --start 0 --end 499 --workers 100 && \
        touch {params.fitclone_output_dir}/.done
        """


rule process_fitclone_output:
    input:
        "results/muts5_trees{N_trees}/sim{i}/fitclone_results/.done",
    output:
        "results/muts5_trees{N_trees}/sim{i}/fitclone_fitness.txt",
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

        # remove the directory
        import shutil

        shutil.rmtree(base_dir)


rule run_fitree:
    input:
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/cohort.json",
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/fitness_matrix.npz",
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/vectorized_trees.npz",
    output:
        "results/muts{n_mutations}_trees{N_trees}/sim{i}/fitree_posterior/chain{chain}_masked_mixed.nc",
    threads: 1
    resources:
        runtime=2880,
        tasks=1,
        nodes=1,
        mem_mb_per_cpu=2048,
    run:
        import pytensor

        pytensor.config.compiledir = (
            "/cluster/work/bewi/members/xgluo/pytensor_tmp"
            + f"/muts{wildcards.n_mutations}_trees{wildcards.N_trees}_sim{wildcards.i}_chain{wildcards.chain}_masked_mixed"
        )

        import pymc as pm
        import numpy as np
        import fitree
        import arviz as az


        n_mutations = int(wildcards.n_mutations)
        i = int(wildcards.i)
        chain = int(wildcards.chain)
        seed = n_mutations * 10**6 + i * 10**3 + chain

        cohort = fitree.load_cohort_from_json(input[0])
        F_mat = np.load(input[1])["F_mat"]
        vec_trees = fitree.load_vectorized_trees_npz(input[2])

        vec_trees = fitree.update_params(vec_trees, np.diag(np.diag(F_mat)))
        cohort.lifetime_risk = float(fitree.compute_normalizing_constant(vec_trees))

        fitree_joint_likelihood = fitree.FiTreeJointLikelihood(
            cohort,
            augment_max_level=2,
            conditioning=False,
            lifetime_risk_mean=cohort.lifetime_risk,
            lifetime_risk_std=0.001,
        )

        p0 = np.round(
            np.sqrt(5 * (n_mutations**2 - n_mutations) / 2 * (2 * 0.95 - 1))
        )
        D = n_mutations * (n_mutations - 1) / 2
        N = cohort.N_patients
        tau0 = p0 / (D - p0) / np.sqrt(N)

        model = fitree.prior_fitree_mixed(
            cohort,
            diag_sigma=0.1,
            tau0=tau0,
            local_scale=0.1,
            s2=0.02,
            min_occurrences=5,
        )

        with model:
            pm.Potential(
                "joint_likelihood",
                fitree_joint_likelihood(
                    model.fitness_matrix,  # pyright: ignore
                ),  # pyright: ignore
            )
            trace = pm.sample(
                draws=1000,
                tune=1000,
                chains=1,
                return_inferencedata=True,
                random_seed=seed,
            )

        trace.to_netcdf(output[0])

        # remove compiledir
        if os.path.exists(pytensor.config.compiledir):
            import shutil

            shutil.rmtree(pytensor.config.compiledir)


rule combine_fitree_chains:
    input:
        lambda wildcards: expand(
            "results/muts{n_mutations}_trees{N_trees}/sim{i}/fitree_posterior/chain{chain}_masked_mixed.nc",
            n_mutations=wildcards.n_mutations,
            N_trees=wildcards.N_trees,
            i=wildcards.i,
            chain=range(N_CHAINS),
        ),
    output:
        "results/muts{n_mutations}_trees{N_trees}/sim{i}/fitree_posterior_masked_mixed.nc",
    threads: 1
    resources:
        runtime=60,
        tasks=1,
        nodes=1,
    run:
        import arviz as az

        trace = az.concat(
            [az.from_netcdf(f) for f in input],
            dim="chain",
        )
        trace.to_netcdf(output[0])


rule evaluate_sc:
    output:
        summary="results/muts{n_mutations}_trees{N_trees}/sc_summary.txt",
    threads: 1
    resources:
        runtime=60,
        tasks=1,
        nodes=1,
        mem_mb_per_cpu=100,
    run:
        import arviz as az
        import numpy as np
        import fitree
        import pandas as pd
        from scipy.stats import spearmanr

        n_mutations = int(wildcards.n_mutations)
        N_trees = int(wildcards.N_trees)

        # Initialize the output files
        with open(output.summary, "w") as f:
            f.write("sim method observed value n N\n")

            for i in range(N_SIMULATIONS):
                # Load data for simulation i
                vec_trees = fitree.load_vectorized_trees_npz(
                    f"data/muts{n_mutations}_trees{N_trees}/sim{i}/vectorized_trees.npz"
                )
                F_mat = np.load(
                    f"data/muts{n_mutations}_trees{N_trees}/sim{i}/fitness_matrix.npz"
                )["F_mat"]

                # Determine observed and recoverable mutations
                observed = vec_trees.observed.sum(axis=0) > 0
                recoverable = (observed + (vec_trees.genotypes.sum(axis=1) <= 2)) > 0
                observed_unique_idx = np.unique(
                    vec_trees.genotypes[observed, :], axis=0, return_index=True
                )[1]
                recoverable_unique_idx = np.unique(
                    vec_trees.genotypes[recoverable, :], axis=0, return_index=True
                )[1]

                # Calculate true fitness
                vec_trees = fitree.update_params(vec_trees, F_mat)
                true_fitness = np.log(vec_trees.alpha) - np.log(vec_trees.beta)

                # Calculate frequency
                frequency = np.array(
                    np.mean(
                        vec_trees.cell_number
                        / np.sum(vec_trees.cell_number, axis=1)[:, None],
                        axis=0,
                    )
                )
                freq_observed_corr = spearmanr(
                    frequency[observed][observed_unique_idx],
                    true_fitness[observed][observed_unique_idx],
                )[0]
                freq_recoverable_corr = spearmanr(
                    frequency[recoverable][recoverable_unique_idx],
                    true_fitness[recoverable][recoverable_unique_idx],
                )[0]

                f.write(
                    f"{i} freq observed {freq_observed_corr} {n_mutations} {N_trees}\n"
                )
                f.write(
                    f"{i} freq recoverable {freq_recoverable_corr} {n_mutations} {N_trees}\n"
                )

                # Evaluate FiTree
                try:
                    trace = az.from_netcdf(
                        f"results/muts{wildcards.n_mutations}_trees{wildcards.N_trees}/sim{i}/fitree_posterior_masked_mixed.nc"
                    )

                    # F_mat_posterior = trace.posterior["fitness_matrix"].values
                    # F_mat_posterior = F_mat_posterior.reshape(-1, n_mutations, n_mutations)

                    # fitree_obs_score = np.zeros(F_mat_posterior.shape[0])
                    # fitree_rec_score = np.zeros(F_mat_posterior.shape[0])
                    # for j, F_mat_j in enumerate(F_mat_posterior):
                    #     vec_trees = fitree.update_params(vec_trees, F_mat_j)
                    #     fitree_fitness = np.log(vec_trees.alpha) - np.log(vec_trees.beta)
                    #     fitree_obs_score[j] = weighted_spearman(
                    #         true_fitness, fitree_fitness, observed
                    #     )
                    #     fitree_rec_score[j] = weighted_spearman(
                    #         true_fitness, fitree_fitness, recoverable
                    #     )

                    # fitree_obs_summary = az.summary(fitree_obs_score, stat_focus="median", hdi_prob=0.5)
                    # fitree_obs = fitree_obs_summary["median"].values[0].astype(float)
                    # fitree_obs_lower = fitree_obs_summary["eti_25%"].values[0].astype(float)
                    # fitree_obs_upper = fitree_obs_summary["eti_75%"].values[0].astype(float)
                    # fitree_rec_summary = az.summary(fitree_rec_score, stat_focus="median", hdi_prob=0.5)
                    # fitree_rec = fitree_rec_summary["median"].values[0].astype(float)
                    # fitree_rec_lower = fitree_rec_summary["eti_25%"].values[0].astype(float)
                    # fitree_rec_upper = fitree_rec_summary["eti_75%"].values[0].astype(float)

                    fitree_posterior = trace.posterior["fitness_matrix"].values
                    inferred_F_mat = np.median(fitree_posterior, axis=(0, 1))

                    vec_trees_inferred = fitree.update_params(vec_trees, inferred_F_mat)
                    fitree_fitness = np.log(vec_trees_inferred.alpha) - np.log(
                        vec_trees_inferred.beta
                    )
                    fitree_observed_corr = spearmanr(
                        true_fitness[observed][observed_unique_idx],
                        fitree_fitness[observed][observed_unique_idx],
                    )[0]
                    fitree_recoverable_corr = spearmanr(
                        true_fitness[recoverable][recoverable_unique_idx],
                        fitree_fitness[recoverable][recoverable_unique_idx],
                    )[0]
                except FileNotFoundError:
                    fitree_observed_corr = np.nan
                    fitree_recoverable_corr = np.nan

                    # fitree_obs = np.nan
                    # fitree_obs_lower = np.nan
                    # fitree_obs_upper = np.nan
                    # fitree_rec = np.nan
                    # fitree_rec_lower = np.nan
                    # fitree_rec_upper = np.nan

                f.write(
                    f"{i} fitree observed {fitree_observed_corr} {n_mutations} {N_trees}\n"
                )
                f.write(
                    f"{i} fitree recoverable {fitree_recoverable_corr} {n_mutations} {N_trees}\n"
                )

                # Evaluate diffusion
                try:
                    subclone_fitness = np.loadtxt(
                        f"results/muts{wildcards.n_mutations}_trees{wildcards.N_trees}/sim{i}/diffusion_subclone_fitness.txt"
                    )
                    diffusion_subclone_observed_corr = spearmanr(
                        true_fitness[observed][observed_unique_idx],
                        subclone_fitness[observed][observed_unique_idx],
                    )[0]
                    diffusion_subclone_recoverable_corr = spearmanr(
                        true_fitness[recoverable][recoverable_unique_idx],
                        subclone_fitness[recoverable][recoverable_unique_idx],
                    )[0]

                except FileNotFoundError:
                    diffusion_subclone_observed_corr = np.nan
                    diffusion_subclone_recoverable_corr = np.nan

                try:
                    mutation_fitness = np.loadtxt(
                        f"results/muts{wildcards.n_mutations}_trees{wildcards.N_trees}/sim{i}/diffusion_mutation_fitness.txt"
                    )
                    diffusion_mutation_observed_corr = spearmanr(
                        true_fitness[observed][observed_unique_idx],
                        mutation_fitness[observed][observed_unique_idx],
                    )[0]
                    diffusion_mutation_recoverable_corr = spearmanr(
                        true_fitness[recoverable][recoverable_unique_idx],
                        mutation_fitness[recoverable][recoverable_unique_idx],
                    )[0]

                except FileNotFoundError:
                    diffusion_mutation_observed_corr = np.nan
                    diffusion_mutation_recoverable_corr = np.nan

                f.write(
                    f"{i} Diffusion_subclone observed {diffusion_subclone_observed_corr} {n_mutations} {N_trees}\n"
                )
                f.write(
                    f"{i} Diffusion_subclone recoverable {diffusion_subclone_recoverable_corr} {n_mutations} {N_trees}\n"
                )
                f.write(
                    f"{i} Diffusion_mutation observed {diffusion_mutation_observed_corr} {n_mutations} {N_trees}\n"
                )
                f.write(
                    f"{i} Diffusion_mutation recoverable {diffusion_mutation_recoverable_corr} {n_mutations} {N_trees}\n"
                )

                # Evaluate SCIFIL
                try:
                    scifil_res = np.loadtxt(
                        f"results/muts{wildcards.n_mutations}_trees{wildcards.N_trees}/sim{i}/SCIFIL_result.txt"
                    )
                    scifil_res = scifil_res[:, 1:]  # Adjust indexing if necessary
                    scifil_fitness = np.mean(scifil_res, axis=0)
                    scifil_fitness[np.isnan(scifil_fitness)] = 0

                    scifil_observed_corr = spearmanr(
                        true_fitness[observed][observed_unique_idx],
                        scifil_fitness[observed][observed_unique_idx],
                    )[0]
                    scifil_recoverable_corr = spearmanr(
                        true_fitness[recoverable][recoverable_unique_idx],
                        scifil_fitness[recoverable][recoverable_unique_idx],
                    )[0]
                except FileNotFoundError:
                    scifil_observed_corr = np.nan
                    scifil_recoverable_corr = np.nan

                f.write(
                    f"{i} scifil observed {scifil_observed_corr} {n_mutations} {N_trees}\n"
                )
                f.write(
                    f"{i} scifil recoverable {scifil_recoverable_corr} {n_mutations} {N_trees}\n"
                )

                # Evaluate FitClone
                try:
                    fitclone_res = pd.read_csv(
                        f"results/muts{wildcards.n_mutations}_trees{wildcards.N_trees}/sim{i}/fitclone_fitness.txt",
                        sep="\t",
                    )
                    fitclone_res = (
                        fitclone_res.groupby("union_node_id")["median_value"]
                        .mean()
                        .reset_index()
                    )
                    fitclone_fitness = np.ones_like(frequency) * -np.inf
                    fitclone_fitness[fitclone_res["union_node_id"]] = fitclone_res[
                        "median_value"
                    ]

                    fitclone_observed_corr = spearmanr(
                        true_fitness[observed][observed_unique_idx],
                        fitclone_fitness[observed][observed_unique_idx],
                    )[0]
                    fitclone_recoverable_corr = spearmanr(
                        true_fitness[recoverable][recoverable_unique_idx],
                        fitclone_fitness[recoverable][recoverable_unique_idx],
                    )[0]
                except FileNotFoundError:
                    fitclone_observed_corr = np.nan
                    fitclone_recoverable_corr = np.nan

                f.write(
                    f"{i} fitclone observed {fitclone_observed_corr} {n_mutations} {N_trees}\n"
                )
                f.write(
                    f"{i} fitclone recoverable {fitclone_recoverable_corr} {n_mutations} {N_trees}\n"
                )


rule evaluate_fitness:
    output:
        summary="results/muts{n_mutations}_trees{N_trees}/fitness_summary.txt",
    threads: 1
    resources:
        runtime=60,
        tasks=1,
        nodes=1,
        mem_mb_per_cpu=100,
    run:
        import arviz as az
        import numpy as np
        import fitree
        import pandas as pd

        n_mutations = int(wildcards.n_mutations)
        N_trees = int(wildcards.N_trees)

        with open(output.summary, "w") as f:
            f.write("sim method metric observed value n N\n")

            for i in range(N_SIMULATIONS):
                # Load data for simulation i
                vec_trees = fitree.load_vectorized_trees_npz(
                    f"data/muts{n_mutations}_trees{N_trees}/sim{i}/vectorized_trees.npz"
                )
                F_mat = np.load(
                    f"data/muts{n_mutations}_trees{N_trees}/sim{i}/fitness_matrix.npz"
                )["F_mat"]

                # Determine observed and recoverable mutations
                observed = vec_trees.observed.sum(axis=0) > 0
                recoverable = (observed + (vec_trees.genotypes.sum(axis=1) <= 2)) > 0
                observed_unique_idx = np.unique(
                    vec_trees.genotypes[observed, :], axis=0, return_index=True
                )[1]
                recoverable_unique_idx = np.unique(
                    vec_trees.genotypes[recoverable, :], axis=0, return_index=True
                )[1]

                # Calculate true fitness
                vec_trees = fitree.update_params(vec_trees, F_mat)
                true_fitness = np.log(vec_trees.alpha) - np.log(vec_trees.beta)

                # Evaluate FiTree
                try:
                    trace = az.from_netcdf(
                        f"results/muts{wildcards.n_mutations}_trees{wildcards.N_trees}/sim{i}/fitree_posterior_masked_mixed.nc"
                    )

                    fitree_posterior = trace.posterior["fitness_matrix"].values
                    inferred_F_mat = np.median(fitree_posterior, axis=(0, 1))

                    vec_trees_inferred = fitree.update_params(vec_trees, inferred_F_mat)
                    fitree_fitness = np.log(vec_trees_inferred.alpha) - np.log(
                        vec_trees_inferred.beta
                    )
                    fitree_mae_obs = np.mean(
                        np.abs(
                            true_fitness[observed][observed_unique_idx]
                            - fitree_fitness[observed][observed_unique_idx]
                        )
                    )
                    fitree_mae_rec = np.mean(
                        np.abs(
                            true_fitness[recoverable][recoverable_unique_idx]
                            - fitree_fitness[recoverable][recoverable_unique_idx]
                        )
                    )
                    fitree_sign_obs = np.mean(
                        np.sign(
                            np.round(true_fitness[observed][observed_unique_idx], 2)
                        )
                        == np.sign(fitree_fitness[observed][observed_unique_idx])
                    )
                    fitree_sign_rec = np.mean(
                        np.sign(
                            np.round(
                                true_fitness[recoverable][recoverable_unique_idx], 2
                            )
                        )
                        == np.sign(fitree_fitness[recoverable][recoverable_unique_idx])
                    )

                except FileNotFoundError:
                    fitree_mae_obs = np.nan
                    fitree_mae_rec = np.nan
                    fitree_sign_obs = np.nan
                    fitree_sign_rec = np.nan

                f.write(
                    f"{i} FiTree mae observed {fitree_mae_obs} {n_mutations} {N_trees}\n"
                )
                f.write(
                    f"{i} FiTree mae recoverable {fitree_mae_rec} {n_mutations} {N_trees}\n"
                )
                f.write(
                    f"{i} FiTree sign observed {fitree_sign_obs} {n_mutations} {N_trees}\n"
                )
                f.write(
                    f"{i} FiTree sign recoverable {fitree_sign_rec} {n_mutations} {N_trees}\n"
                )

                try:
                    mutation_fitness = np.loadtxt(
                        f"results/muts{wildcards.n_mutations}_trees{wildcards.N_trees}/sim{i}/diffusion_mutation_fitness.txt"
                    )
                    diffusion_mae_obs = np.mean(
                        np.abs(
                            true_fitness[observed][observed_unique_idx]
                            - mutation_fitness[observed][observed_unique_idx]
                        )
                    )
                    diffusion_mae_rec = np.mean(
                        np.abs(
                            true_fitness[recoverable][recoverable_unique_idx]
                            - mutation_fitness[recoverable][recoverable_unique_idx]
                        )
                    )

                    diffusion_sign_obs = np.mean(
                        np.sign(
                            np.round(true_fitness[observed][observed_unique_idx], 2)
                        )
                        == np.sign(mutation_fitness[observed][observed_unique_idx])
                    )
                    diffusion_sign_rec = np.mean(
                        np.sign(
                            np.round(
                                true_fitness[recoverable][recoverable_unique_idx], 2
                            )
                        )
                        == np.sign(
                            mutation_fitness[recoverable][recoverable_unique_idx]
                        )
                    )
                except FileNotFoundError:
                    diffusion_mae_obs = np.nan
                    diffusion_mae_rec = np.nan
                    diffusion_sign_obs = np.nan
                    diffusion_sign_rec = np.nan

                f.write(
                    f"{i} Diffusion mae observed {diffusion_mae_obs} {n_mutations} {N_trees}\n"
                )
                f.write(
                    f"{i} Diffusion mae recoverable {diffusion_mae_rec} {n_mutations} {N_trees}\n"
                )
                f.write(
                    f"{i} Diffusion sign observed {diffusion_sign_obs} {n_mutations} {N_trees}\n"
                )
                f.write(
                    f"{i} Diffusion sign recoverable {diffusion_sign_rec} {n_mutations} {N_trees}\n"
                )
