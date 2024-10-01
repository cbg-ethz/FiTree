import jax
from typing import NamedTuple
import numpy as np
from anytree import PreOrderIter
import copy

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
    ch_mat: jax.Array | np.ndarray  # (n_nodes + 1, n_nodes)

    N_trees: jax.Array | np.ndarray  # scalar: number of observed trees
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
    mutations = node.get_genotype()
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
                new_node = Subclone(
                    node_id=tree.size,
                    mutation_ids=[j],
                    cell_number=0,
                    parent=node,
                )
                new_node.get_growth_params(
                    mu_vec=mu_vec, F_mat=F_mat, common_beta=common_beta
                )
    else:
        raise NotImplementedError
        # TODO: implement other rules

    return tree


def wrap_trees(cohort: TumorTreeCohort) -> tuple[VectorizedTrees, TumorTree]:
    """This function takes a TumorTreeCohort object as input
    and returns a VectorizedTrees object.
    """

    trees = copy.deepcopy(cohort)  # avoid modifying the original trees

    # 0. Expand all trees
    F_mat = np.zeros((trees.n_mutations, trees.n_mutations))
    mu_vec = trees.mu_vec
    for tree in trees.trees:
        tree.root = get_augmented_tree(
            tree=tree.root,
            n_mutations=trees.n_mutations,
            mu_vec=mu_vec,
            F_mat=F_mat,
            common_beta=trees.common_beta,
            rule="parallel",
        )

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
    ch_mat = np.zeros((n_nodes + 1, n_nodes), dtype=bool)
    nu_vec = np.zeros(n_nodes, dtype=np.float64)
    node_iter = PreOrderIter(union_root)
    next(node_iter)  # skip the root
    for node in node_iter:
        idx = node.node_id - 1
        parent_id[idx] = node.parent.node_id - 1
        genotypes[idx, list(node.genotype)] = True
        ch_mat[node.parent.node_id, idx] = True
        nu_vec[idx] = np.prod(mu_vec[list(node.genotype - node.parent.genotype)])

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
        ch_mat=ch_mat,
        N_trees=N_trees,
        n_nodes=n_nodes,
        beta=trees.common_beta,
        C_s=trees.C_sampling,
        C_0=trees.C_0,
        C_min=trees.C_min,
        t_max=trees.t_max,
    )

    return vec_trees, union_tree
