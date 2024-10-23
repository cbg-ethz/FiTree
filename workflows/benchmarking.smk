######### Imports #########
import fitree
import os
import numpy as np
import jax.numpy as jnp
import jax
from scipy.optimize import minimize
from scripts.benchmarking_helpers import (
    prepare_SCIFIL_input,
    compute_diffusion_fitness_subclone,
    compute_diffusion_fitness_mutation,
)

from fitree import VectorizedTrees


######### Simulation setup #########

N_MUTATIONS: list[int] = [5]
N_TREES: list[int] = [4]
N_SIMULATIONS: int = 2
SEED: int = 2024
NCORES: int = 4

######### Workflow #########


rule all:
    input:
        expand(
            "data/muts{n_mutations}_trees{N_trees}/sim{i}_fitness_matrix.npz",
            n_mutations=N_MUTATIONS,
            N_trees=N_TREES,
            i=range(N_SIMULATIONS),
        ),
        expand(
            "data/muts{n_mutations}_trees{N_trees}/sim{i}_cohort.json",
            n_mutations=N_MUTATIONS,
            N_trees=N_TREES,
            i=range(N_SIMULATIONS),
        ),
        expand(
            "data/muts{n_mutations}_trees{N_trees}/sim{i}_vectorized_trees.npz",
            n_mutations=N_MUTATIONS,
            N_trees=N_TREES,
            i=range(N_SIMULATIONS),
        ),
        expand(
            "data/muts{n_mutations}_trees{N_trees}/SCIFIL_input/sim{i}/tree_matrix.txt",
            n_mutations=N_MUTATIONS,
            N_trees=N_TREES,
            i=range(N_SIMULATIONS),
        ),
        expand(
            "data/muts{n_mutations}_trees{N_trees}/SCIFIL_input/sim{i}/cell_count.txt",
            n_mutations=N_MUTATIONS,
            N_trees=N_TREES,
            i=range(N_SIMULATIONS),
        ),


rule generate_data:
    output:
        "data/muts{n_mutations}_trees{N_trees}/sim{i}_fitness_matrix.npz",
        "data/muts{n_mutations}_trees{N_trees}/sim{i}_cohort.json",
        "data/muts{n_mutations}_trees{N_trees}/sim{i}_vectorized_trees.npz",
        "data/muts{n_mutations}_trees{N_trees}/SCIFIL_input/sim{i}/tree_matrix.txt",
        "data/muts{n_mutations}_trees{N_trees}/SCIFIL_input/sim{i}/cell_count.txt",
    run:
        N_trees = int(wildcards.N_trees)
        n_mutations = int(wildcards.n_mutations)
        i = int(wildcards.i)
        rng = np.random.default_rng(SEED)


        F_mat = fitree.generate_fmat(
            rng=rng, n_mutations=n_mutations, base_sigma=0.3, epis_sigma=1.0
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


# rule run_SCIFIL:
#     input:
#         "data/muts{n_mutations}_trees{N_trees}/SCIFIL_input/sim{i}/tree_matrix.txt",
#         "data/muts{n_mutations}_trees{N_trees}/SCIFIL_input/sim{i}/cell_count.txt",
#     output:
#         "results/muts{n_mutations}_trees{N_trees}/SCIFIL/sim{i}_result.txt",
