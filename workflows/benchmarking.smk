######### Imports #########
import fitree
import os
import numpy as np
import jax.numpy as jnp
import jax
import pymc as pm
from scipy.optimize import minimize
import arviz as az
from scripts.benchmarking_helpers import (
    prepare_SCIFIL_input,
    compute_diffusion_fitness_subclone,
    compute_diffusion_fitness_mutation,
    weighted_spearman,
    get_available_simulations,
)

from fitree import VectorizedTrees


######### Simulation setup #########

# cluster setup
N_MUTATIONS: list[int] = [5, 10, 15, 20]
N_TREES: list[int] = [500]
N_SIMULATIONS: int = 100
NCORES: int = 100

# # local setup
# N_MUTATIONS: list[int] = [5]
# N_TREES: list[int] = [4]
# N_SIMULATIONS: int = 2
# NCORES: int = 4

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
            "results/muts{n_mutations}_trees{N_trees}/sim{i}/fitree_posterior.nc",
            n_mutations=N_MUTATIONS,
            N_trees=N_TREES,
            i=range(N_SIMULATIONS),
        ),
        expand(
            "results/muts{n_mutations}_trees{N_trees}/evaluations_observed.txt",
            n_mutations=N_MUTATIONS,
            N_trees=N_TREES,
        ),
        expand(
            "results/muts{n_mutations}_trees{N_trees}/evaluations_recoverable.txt",
            n_mutations=N_MUTATIONS,
            N_trees=N_TREES,
        ),


rule generate_data:
    output:
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/fitness_matrix.npz",
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/cohort.json",
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/vectorized_trees.npz",
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/SCIFIL_input/tree_matrix.txt",
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/SCIFIL_input/cell_count.txt",
    threads: NCORES
    resources:
        runtime=240,
        tasks=1,
        nodes=1,
    run:
        N_trees = int(wildcards.N_trees)
        n_mutations = int(wildcards.n_mutations)
        i = int(wildcards.i)
        rng = np.random.default_rng(2024 * i)


        F_mat = fitree.generate_fmat(
            rng=rng,
            n_mutations=n_mutations,
            base_sigma=0.3,
            epis_sigma=1.0,
            base_sparsity=0.1,
            positive_ratio=0.7,
        )
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
            use_joblib=True,
            n_jobs=NCORES,
        )
        fitree.save_cohort_to_json(cohort, output[1])

        vec_trees, _ = fitree.wrap_trees(cohort, augment_max_level=2, pseudo_count=1e-4)
        fitree.save_vectorized_trees_npz(vec_trees, output[2])

        scifil_dir = os.path.dirname(output[3])
        os.makedirs(scifil_dir, exist_ok=True)

        prepare_SCIFIL_input(vec_trees, scifil_dir)


rule run_SCIFIL:
    input:
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/SCIFIL_input/tree_matrix.txt",
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/SCIFIL_input/cell_count.txt",
    output:
        "results/muts{n_mutations}_trees{N_trees}/sim{i}/SCIFIL_result.txt",
    params:
        # scifil_dir = "/Users/luox/Documents/Projects/SCIFIL",
        scifil_dir="/cluster/home/luox/SCIFIL",
        input_dir=os.path.abspath(
            "data/muts{n_mutations}_trees{N_trees}/sim{i}/SCIFIL_input"
        ),
        output=os.path.abspath(
            "results/muts{n_mutations}_trees{N_trees}/sim{i}/SCIFIL_result.txt"
        ),
    threads: 4
    resources:
        runtime=60,
        tasks=1,
        nodes=1,
        mem_mb_per_cpu=4096,
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


rule run_fitree:
    input:
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/cohort.json",
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/fitness_matrix.npz",
        "data/muts{n_mutations}_trees{N_trees}/sim{i}/vectorized_trees.npz",
    output:
        "results/muts{n_mutations}_trees{N_trees}/sim{i}/fitree_posterior.nc",
    threads: 4
    resources:
        runtime=240,
        tasks=1,
        nodes=1,
    run:
        cohort = fitree.load_cohort_from_json(input[0])
        F_mat = np.load(input[1])["F_mat"]
        vec_trees = fitree.load_vectorized_trees_npz(input[2])

        fitree_joint_likelihood = fitree.FiTreeJointLikelihood(
            cohort, augment_max_level=2, pseudo_count=1e-4
        )
        vec_trees = fitree.update_params(vec_trees, F_mat)
        cohort.lifetime_risk = float(fitree.compute_normalizing_constant(vec_trees))

        model = fitree.prior_fitree(cohort)
        F_mat_init = fitree.greedy_init_fmat(vec_trees)
        C_sampling_init = float(vec_trees.cell_number.sum(axis=1).mean())
        nr_neg_samples_init = (
            cohort.N_patients / cohort.lifetime_risk * (1 - cohort.lifetime_risk)
        )

        outputs = [[0.0]]
        fitree_joint_likelihood.perform(
            None,
            (
                F_mat_init,
                C_sampling_init,
                nr_neg_samples_init,
            ),
            outputs,
        )

        with model:
            start_vals = {
                "fitness_matrix": F_mat_init,
                "C_sampling": C_sampling_init,
                "nr_neg_samples": nr_neg_samples_init,
            }
            pm.Potential(
                "joint_likelihood",
                fitree_joint_likelihood(
                    model.fitness_matrix, model.C_sampling, model.nr_neg_samples
                ),
            )
            trace = pm.sample(draws=500, tune=500, chains=4, initvals=start_vals)

        trace.to_netcdf(output[0])


rule evaluate:
    output:
        observed="results/muts{n_mutations}_trees{N_trees}/evaluations_observed.txt",
        recoverable="results/muts{n_mutations}_trees{N_trees}/evaluations_recoverable.txt",
    threads: 1
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
                "FiTree "
                + "Diffusion_subclone "
                + "Diffusion_mutation "
                + "SCIFIL "
                + "freq"
                + "\n"
            )
            f_rec.write(
                "FiTree "
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
                recoverable = (observed + vec_trees.genotypes.sum(axis=1) <= 2) > 0

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
                    f"{fitree_observed_corr} {diffusion_subclone_observed_corr} {diffusion_mutation_observed_corr} {scifil_observed_corr} {freq_observed_corr}\n"
                )
                f_rec.write(
                    f"{fitree_recoverable_corr} {diffusion_subclone_recoverable_corr} {diffusion_mutation_recoverable_corr} {scifil_recoverable_corr} {freq_recoverable_corr}\n"
                )
