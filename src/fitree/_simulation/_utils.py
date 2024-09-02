import numpy as np

from anytree import PreOrderIter

from fitree._trees import Subclone


def _truncate_tree(tree: Subclone, C_min: int | float | np.ndarray = 1e3) -> Subclone:
    """
    Recursively truncate the non-detected leaves,
    assign zeros to non-detected internal nodes,
    and re-assign node ids.

    Parameters
    ----------
    tree : Subclone
            The tree to be truncated.
    C_min : int | float | np.ndarray, optional
            The minimum detectable number of cells. Defaults to 1e3.
    """

    # recursively truncate the non-detected leaves
    check = False
    while not check:
        check = True
        for node in tree.leaves:
            if node.cell_number < C_min:
                node.parent = None
                del node
                check = False

    # assign zeros to non-detected internal nodes
    for node in PreOrderIter(tree):
        if node.cell_number < C_min:
            node.cell_number = 0

    # re-assign node ids
    node_id_counter = 0
    for node in PreOrderIter(tree):
        node.node_id = node_id_counter
        node_id_counter += 1

    return tree


def _expand_tree(
    tree: Subclone,
    n_mutations: int,
    mu_vec: np.ndarray,
    F_mat: np.ndarray,
    common_beta: float = 0.8,
    rule: str = "parallel",
    k_repeat: int = 0,
    k_multiple: int = 1,
) -> Subclone:
    """
    Expand the tree based on the tree expansion rule.

    Parameters
    ----------
    tree : Subclone
            The tree to be expanded.
    n_mutations : int
            The number of mutations to be considered.
    mu_vec : np.ndarray
            The n-by-1 mutation rate vector.
    F_mat : np.ndarray
            The n-by-n fitness matrix.
    common_beta : float, optional
            The common death rate. Defaults to 0.8 (average 40 weeks).
    rule : str, optional
            The type of the tree generation. Defaults to "parallel".
            All options:
            1. "ISA": Infinite Sites Assumption.
            2. "parallel": parallel mutations are allowed, but no repeated mutations
                    along the same branch or duplicated siblings.
            3. "repeat": repeated mutations are allowed, but up to k_repeat times.
            4. "multiple": multiple mutations in the same subclone are allowed,
                    but up to k_multiple times.
    k_repeat : int, optional
            The maximum number of repeated mutations allowed. Defaults to 0.
    k_multiple : int, optional
            The maximum number of multiple mutations allowed. Defaults to 1.
    """

    if rule == "parallel":
        for node in tree.leaves:
            if node.cell_number > 0:
                possible_mutations = set(range(n_mutations)).difference(
                    node.get_genotype()
                )
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
