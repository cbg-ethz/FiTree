######### Imports #########
import fitree
import os
import numpy as np
import jax.numpy as jnp
import jax
import pymc as pm
from scipy.optimize import minimize
import arviz as az
import gzip
import pandas as pd
from glob import glob
import pytensor


from scripts.benchmarking_helpers import (
    prepare_SCIFIL_input,
    prepare_fitclone_input,
    compute_diffusion_fitness_subclone,
    compute_diffusion_fitness_mutation,
    weighted_spearman,
    get_available_simulations,
)

from fitree import VectorizedTrees


######### Simulation setup #########

# cluster setup
N_MUTATIONS: list[int] = [5, 10, 15]
N_TREES: list[int] = [200, 500]
N_SIMULATIONS: int = 100
fitclone_exe_dir: str = "/cluster/home/luox/fitclone/fitclone"
scifil_exe_dir: str = "/cluster/home/luox/SCIFIL"
script_dir: str = "/cluster/home/luox/FiTree/workflows/scripts"
working_dir: str = os.getcwd()


# # local setup
# N_MUTATIONS: list[int] = [5]
# N_TREES: list[int] = [4]
# N_SIMULATIONS: int = 2

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
            n_mutations=[5, 10],
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
        # expand(
        #     "results/muts{n_mutations}_trees{N_trees}/sim{i}/fitree_posterior.nc",
        #     n_mutations=N_MUTATIONS,
        #     N_trees=N_TREES,
        #     i=range(N_SIMULATIONS),
        # ),
        # expand(
        #     "results/muts{n_mutations}_trees{N_trees}/evaluations_observed.txt",
        #     n_mutations=N_MUTATIONS,
        #     N_trees=N_TREES,
        # ),
        # expand(
        #     "results/muts{n_mutations}_trees{N_trees}/evaluations_recoverable.txt",
        #     n_mutations=N_MUTATIONS,
        #     N_trees=N_TREES,
        # ),


rule generate_data:
    output:
        "data/muts{n_mutations}_trees500/sim{i}/fitness_matrix.npz",
        "data/muts{n_mutations}_trees500/sim{i}/cohort.json",
        "data/muts{n_mutations}_trees500/sim{i}/vectorized_trees.npz",
    threads: 100
    resources:
        runtime=480,
        tasks=1,
        nodes=1,
        mem_mb_per_cpu=500,
    run:
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
    threads: 4
    resources:
        runtime=480,
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
        cd {params.fitclone_input_dir} && \
        chmod a+x run_fitclone.py && \
        export NUMEXPR_MAX_THREADS=100 && \
        export OPENBLAS_NUM_THREADS=1 && \
        ./run_fitclone.py --start 0 --end 499 --workers 100 ; \
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


rule run_fitree:
    input:
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/cohort.json",
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/fitness_matrix.npz",
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/vectorized_trees.npz",
    output:
        "results/muts{n_mutations}_trees{N_trees}/sim{i}/fitree_posterior.nc",
        "results/muts{n_mutations}_trees{N_trees}/sim{i}/F_mat_init.npz",
    threads: 12
    resources:
        runtime=960,
        tasks=1,
        nodes=1,
    params:
        ncores=12,
    shell:
        """
        cd {script_dir} && \
        python run_fitree_benchmark.py \
            --n_mutations {wildcards.n_mutations} \
            --N_trees {wildcards.N_trees} \
            -i {wildcards.i} \
            --workdir {working_dir} \
            --ncores {params.ncores}
        """


rule evaluate:
    output:
        observed="results/muts{n_mutations}_trees{N_trees}/evaluations_observed.txt",
        recoverable="results/muts{n_mutations}_trees{N_trees}/evaluations_recoverable.txt",
    threads: 1
    resources:
        runtime=60,
        tasks=1,
        nodes=1,
        mem_mb_per_cpu=100,
    run:
        # Extract the list of available simulations
        available_sims = get_available_simulations(
            wildcards.n_mutations, wildcards.N_trees
        )

        # Initialize the output files
        with open(output.observed, "w") as f_obs, open(
            output.recoverable, "w"
        ) as f_rec:
            f_obs.write(
                "sim "
                + "FiTree "
                + "Diffusion_subclone "
                + "Diffusion_mutation "
                + "SCIFIL "
                + "freq"
                + "\n"
            )
            f_rec.write(
                "sim "
                + "FiTree "
                + "Diffusion_subclone "
                + "Diffusion_mutation "
                + "SCIFIL "
                + "freq"
                + "\n"
            )

            # Iterate over the available simulations
            for idx, i in enumerate(available_sims):
                # Load data for simulation i
                trace = az.from_netcdf(
                    f"results/muts{wildcards.n_mutations}_trees{wildcards.N_trees}/sim{i}/fitree_posterior.nc"
                )
                subclone_fitness = np.loadtxt(
                    f"results/muts{wildcards.n_mutations}_trees{wildcards.N_trees}/sim{i}/diffusion_subclone_fitness.txt"
                )
                mutation_fitness = np.loadtxt(
                    f"results/muts{wildcards.n_mutations}_trees{wildcards.N_trees}/sim{i}/diffusion_mutation_fitness.txt"
                )
                scifil_res = np.loadtxt(
                    f"results/muts{wildcards.n_mutations}_trees{wildcards.N_trees}/sim{i}/SCIFIL_result.txt"
                )
                vec_trees = fitree.load_vectorized_trees_npz(
                    f"data/muts{wildcards.n_mutations}_trees{wildcards.N_trees}/sim{i}/vectorized_trees.npz"
                )
                F_mat = np.load(
                    f"data/muts{wildcards.n_mutations}_trees{wildcards.N_trees}/sim{i}/fitness_matrix.npz"
                )["F_mat"]

                # Determine observed and recoverable mutations
                observed = vec_trees.observed.sum(axis=0) > 0
                recoverable = (observed + (vec_trees.genotypes.sum(axis=1) <= 2)) > 0

                # Calculate frequency
                frequency = np.array(
                    np.mean(
                        vec_trees.cell_number
                        / np.sum(vec_trees.cell_number, axis=1)[:, None],
                        axis=0,
                    )
                )

                # Calculate true fitness
                vec_trees = fitree.update_params(vec_trees, F_mat)
                true_fitness = np.log(vec_trees.alpha) - np.log(vec_trees.beta)

                # Evaluate FiTree
                fitree_posterior = trace.posterior["fitness_matrix"].values
                inferred_F_mat = np.median(fitree_posterior, axis=(0, 1))
                vec_trees_inferred = fitree.update_params(vec_trees, inferred_F_mat)
                fitree_fitness = np.log(vec_trees_inferred.alpha) - np.log(
                    vec_trees_inferred.beta
                )

                # Evaluate SCIFIL
                scifil_res = scifil_res[:, 1:]  # Adjust indexing if necessary
                scifil_fitness = np.mean(scifil_res, axis=0)
                scifil_fitness[np.isnan(scifil_fitness)] = 0

                # Compute correlations
                fitree_observed_corr = weighted_spearman(
                    true_fitness, fitree_fitness, observed
                )
                fitree_recoverable_corr = weighted_spearman(
                    true_fitness, fitree_fitness, recoverable
                )

                diffusion_subclone_observed_corr = weighted_spearman(
                    true_fitness, subclone_fitness, observed
                )
                diffusion_subclone_recoverable_corr = weighted_spearman(
                    true_fitness, subclone_fitness, recoverable
                )

                diffusion_mutation_observed_corr = weighted_spearman(
                    true_fitness, mutation_fitness, observed
                )
                diffusion_mutation_recoverable_corr = weighted_spearman(
                    true_fitness, mutation_fitness, recoverable
                )

                scifil_observed_corr = weighted_spearman(
                    true_fitness, scifil_fitness, observed
                )
                scifil_recoverable_corr = weighted_spearman(
                    true_fitness, scifil_fitness, recoverable
                )

                freq_observed_corr = weighted_spearman(
                    true_fitness, frequency, observed
                )
                freq_recoverable_corr = weighted_spearman(
                    true_fitness, frequency, recoverable
                )

                # Write results to output files
                f_obs.write(
                    f"{i} {fitree_observed_corr} {diffusion_subclone_observed_corr} {diffusion_mutation_observed_corr} {scifil_observed_corr} {freq_observed_corr}\n"
                )
                f_rec.write(
                    f"{i} {fitree_recoverable_corr} {diffusion_subclone_recoverable_corr} {diffusion_mutation_recoverable_corr} {scifil_recoverable_corr} {freq_recoverable_corr}\n"
                )
