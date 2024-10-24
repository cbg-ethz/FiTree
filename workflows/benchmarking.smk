######### Imports #########
import fitree
import os
import numpy as np
import jax.numpy as jnp
import jax
import pymc as pm
from scipy.optimize import minimize
from scripts.benchmarking_helpers import (
    prepare_SCIFIL_input,
    compute_diffusion_fitness_subclone,
    compute_diffusion_fitness_mutation,
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
    threads: 1
    resources:
        runtime=60,
        tasks=1,
        nodes=1,
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
            trace = pm.sample(draws=500, tune=500, chains=4, initvals=start_vals)

        trace.to_netcdf(output[0])
