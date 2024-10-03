import jax
from typing import NamedTuple
import numpy as np
from anytree import PreOrderIter

from fitree._trees import TumorTreeCohort, TumorTree, Subclone


class VectorizedTrees(NamedTuple):
    # All trees are stored in array format for vectorized computation in JAX

    cell_number: jax.Array | np.ndarray  # (N_trees, n_nodes)
    observed: jax.Array | np.ndarray  # (N_trees, n_nodes)
    sampling_time: jax.Array | np.ndarray  # (N_trees,)
    weight: jax.Array | np.ndarray  # (N_trees,)

    node_id: jax.Array | np.ndarray  # (n_nodes,)
    parent_id: jax.Array | np.ndarray  # (n_nodes,)
    alpha: jax.Array | np.ndarray  # (n_nodes,)
    nu: jax.Array | np.ndarray  # (n_nodes,)
    lam: jax.Array | np.ndarray  # (n_nodes,)
    rho: jax.Array | np.ndarray  # (n_nodes,)
    phi: jax.Array | np.ndarray  # (n_nodes,)
    delta: jax.Array | np.ndarray  # (n_nodes,)
    r: jax.Array | np.ndarray  # (n_nodes,)
    gamma: jax.Array | np.ndarray  # (n_nodes,)
    genotypes: jax.Array | np.ndarray  # (n_nodes, n_mutations)

    N_trees: jax.Array | np.ndarray  # scalar: number of observed trees
    N_patients: jax.Array | np.ndarray  # scalar: number of patients
    n_nodes: jax.Array | np.ndarray  # scalar: number of union nodes (w/o root)
    beta: jax.Array | np.ndarray  # scalar: common death rate
    C_s: jax.Array | np.ndarray  # scalar: sampling scale
    C_0: jax.Array | np.ndarray  # scalar: root size
    C_min: jax.Array | np.ndarray  # scalar: minimum detectable size
    t_max: jax.Array | np.ndarray  # scalar: maximum sampling time


def get_possible_mutations(node: Subclone, n_mutations: int) -> set[int]:
    """This function returns a set of all mutations of the given node,
    as well as its parent and children.
    """
    mutations = set(node.get_genotype())
    for child in node.children:
        mutations.update(child.mutation_ids)

    return set(range(n_mutations)).difference(mutations)


def get_augmented_tree(
    tree: Subclone,
    n_mutations: int,
    mu_vec: np.ndarray,
    F_mat: np.ndarray,
    common_beta: float = 0.8,
    rule: str = "parallel",
    k_repeat: int = 0,
    k_multiple: int = 1,
) -> Subclone:
    if rule == "parallel":
        all_nodes = list(PreOrderIter(tree))
        for node in all_nodes:
            possible_mutations = get_possible_mutations(node, n_mutations)
            for j in possible_mutations:
                Subclone(
                    node_id=tree.size,
                    mutation_ids=[j],
                    cell_number=0,
                    parent=node,
                )
    else:
        raise NotImplementedError
        # TODO: implement other rules

    return tree


def wrap_trees(trees: TumorTreeCohort) -> tuple[VectorizedTrees, TumorTree]:
    """This function takes a TumorTreeCohort object as input
    and returns a VectorizedTrees object.
    """

    # 1. Create the union tree
    union_root = Subclone(node_id=0, mutation_ids=[], cell_number=trees.C_0)

    node_dict = {union_root.node_path: union_root}

    for tree in trees.trees:
        root = tree.root

        node_iter = PreOrderIter(root)
        for node in node_iter:
            union_node = node_dict[node.node_path]
            for child in node.children:
                child_path = child.node_path
                if child_path not in node_dict:
                    new_node = Subclone(
                        node_id=union_root.size,
                        mutation_ids=child.mutation_ids,
                        cell_number=child.cell_number,  # this number is not used
                        parent=union_node,
                    )
                    node_dict[child_path] = new_node

    # Augment the union tree
    F_mat = np.zeros((trees.n_mutations, trees.n_mutations))
    mu_vec = trees.mu_vec
    union_root = get_augmented_tree(
        tree=union_root,
        n_mutations=trees.n_mutations,
        mu_vec=mu_vec,
        F_mat=F_mat,
        common_beta=trees.common_beta,
        rule="parallel",
    )

    union_tree = TumorTree(patient_id=-1, tree_id=-1, root=union_root)

    # 2. Create the vectorized trees
    N_trees = trees.N_trees
    n_nodes = union_root.size - 1
    cell_number = np.ones((N_trees, n_nodes)) * trees.C_min
    observed = np.zeros((N_trees, n_nodes), dtype=bool)
    sampling_time = np.zeros(N_trees)
    weight = np.zeros(N_trees)

    for i, tree in enumerate(trees.trees):
        root = tree.root
        sampling_time[i] = tree.sampling_time
        weight[i] = tree.weight

        node_iter = PreOrderIter(root)
        next(node_iter)  # skip the root
        for node in node_iter:
            idx = node_dict[node.node_path].node_id - 1
            if node.cell_number > trees.C_min:
                observed[i, idx] = True
                cell_number[i, idx] = node.cell_number

    node_id = np.arange(n_nodes)
    parent_id = np.zeros(n_nodes, dtype=np.int32)
    genotypes = np.zeros((n_nodes, trees.n_mutations), dtype=bool)
    nu_vec = np.zeros(n_nodes, dtype=np.float64)
    node_iter = PreOrderIter(union_root)
    next(node_iter)  # skip the root
    for node in node_iter:
        idx = node.node_id - 1
        parent_id[idx] = node.parent.node_id - 1
        genotypes[idx, list(node.genotype)] = True
        nu_vec[idx] = np.prod(
            mu_vec[list(set(node.genotype) - set(node.parent.genotype))]
        )

    vec_trees = VectorizedTrees(
        cell_number=cell_number,
        observed=observed,
        sampling_time=sampling_time,
        weight=weight,
        node_id=node_id,
        parent_id=parent_id,
        alpha=np.zeros(n_nodes, dtype=np.float64),
        nu=nu_vec,
        lam=np.zeros(n_nodes, dtype=np.float64),
        rho=np.zeros(n_nodes, dtype=np.float64),
        phi=np.zeros(n_nodes, dtype=np.float64),
        delta=np.zeros(n_nodes, dtype=np.float64),
        r=np.zeros(n_nodes, dtype=np.float64),
        gamma=np.zeros(n_nodes, dtype=np.float64),
        genotypes=genotypes,
        N_trees=N_trees,
        N_patients=trees.N_patients,
        n_nodes=n_nodes,
        beta=trees.common_beta,
        C_s=trees.C_sampling,
        C_0=trees.C_0,
        C_min=trees.C_min,
        t_max=trees.t_max,
    )

    # 3. Initialize the growth parameters of the trees
    vec_trees, union_tree = initialize_params(
        vec_trees, union_tree, F_mat, mu_vec, trees.common_beta
    )

    return vec_trees, union_tree


def initialize_params(
    vec_trees: VectorizedTrees,
    union_tree: TumorTree,
    F_mat: np.ndarray,
    mu_vec: np.ndarray,
    common_beta: float,
) -> tuple[VectorizedTrees, TumorTree]:
    """This function updates the growth parameters of the trees
    based on the given fitness matrix F_mat
    """

    node_iter = PreOrderIter(union_tree.root)
    next(node_iter)  # skip the root
    for node in node_iter:
        node.get_growth_params(mu_vec=mu_vec, F_mat=F_mat, common_beta=common_beta)
        idx = node.node_id - 1
        vec_trees.alpha[idx] = node.growth_params["alpha"]
        vec_trees.nu[idx] = node.growth_params["nu"]
        vec_trees.lam[idx] = node.growth_params["lam"]
        vec_trees.rho[idx] = node.growth_params["rho"]
        vec_trees.phi[idx] = node.growth_params["phi"]
        vec_trees.delta[idx] = node.growth_params["delta"]
        vec_trees.r[idx] = node.growth_params["r"]
        vec_trees.gamma[idx] = node.growth_params["gamma"]

    return vec_trees, union_tree
