import jax
from typing import NamedTuple
import numpy as np
from anytree import PreOrderIter

from fitree._trees import TumorTreeCohort, TumorTree, Subclone
from fitree._simulation._utils import _expand_tree


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

    N_trees: jax.Array | np.ndarray  # scalar: number of observed trees
    n_nodes: jax.Array | np.ndarray  # scalar: number of union nodes (w/o root)
    beta: jax.Array | np.ndarray  # scalar: common death rate
    C_s: jax.Array | np.ndarray  # scalar: sampling scale
    C_0: jax.Array | np.ndarray  # scalar: root size
    C_min: jax.Array | np.ndarray  # scalar: minimum detectable size


def wrap_trees(trees: TumorTreeCohort) -> tuple[VectorizedTrees, TumorTree]:
    """This function takes a TumorTreeCohort object as input
    and returns a VectorizedTrees object and a union TumorTree.
    The VectorizedTrees is for the computation of the unnormalized likelihood,
    and the union TumorTree is for the normalizing constant.
    """

    # 0. Expand all trees
    F_mat = np.ones((trees.n_mutations, trees.n_mutations))
    mu_vec = trees.mu_vec
    for tree in trees.trees:
        tree.root = _expand_tree(
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
    cell_number = np.zeros((N_trees, n_nodes))
    observed = np.zeros((N_trees, n_nodes))
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
            cell_number[i, idx] = node.cell_number
            if node.cell_number > trees.C_min:
                observed[i, idx] = 1
            else:
                cell_number[i, idx] = trees.C_min

    node_id = np.arange(n_nodes)
    parent_id = np.zeros(n_nodes, dtype=np.int32)
    node_iter = PreOrderIter(union_root)
    next(node_iter)  # skip the root
    for node in node_iter:
        idx = node.node_id - 1
        parent_id[idx] = node.parent.node_id - 1

    vec_trees = VectorizedTrees(
        cell_number=cell_number,
        observed=observed,
        sampling_time=sampling_time,
        weight=weight,
        node_id=node_id,
        parent_id=parent_id,
        alpha=np.zeros(n_nodes),
        nu=np.zeros(n_nodes),
        lam=np.zeros(n_nodes),
        rho=np.zeros(n_nodes),
        phi=np.zeros(n_nodes),
        delta=np.zeros(n_nodes),
        r=np.zeros(n_nodes),
        gamma=np.zeros(n_nodes),
        N_trees=N_trees,
        n_nodes=n_nodes,
        beta=trees.common_beta,
        C_s=trees.C_sampling,
        C_0=trees.C_0,
        C_min=trees.C_min,
    )

    # 3. Update the growth parameters of the trees
    vec_trees, union_tree = update_params(
        vec_trees, union_tree, F_mat, mu_vec, trees.common_beta
    )

    return vec_trees, union_tree


def update_params(
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
