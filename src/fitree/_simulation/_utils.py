import numpy as np

from anytree import PreOrderIter

from fitree._trees import Subclone


def _sample_cells(rng: np.random.Generator, tree: Subclone, C_seq: int) -> Subclone:
    """
    Sample C_seq cells from the tree using the multivariate hypergeometric distribution.
    i.e. sampling without replacement.

    Parameters
    ----------
    rng : np.random.Generator
        The random number generator.
    tree : Subclone
        The tree to be sampled.
    C_seq : int
        The number of cells to be sampled.

    Returns
    -------
    Subclone
        The sampled tree.
    """

    # Get a vector of cell numbers of all nodes in the tree
    node_iter = PreOrderIter(tree)
    next(node_iter)  # skip the root
    node_iter = list(node_iter)
    cell_numbers = np.array([node.cell_number for node in node_iter])
    nonzero_idx = np.where(cell_numbers > 0)[0].astype(int)
    nodes_to_sample = [node_iter[idx] for idx in nonzero_idx]

    total_cell_number = np.sum(cell_numbers)
    C_seq = np.min([C_seq, total_cell_number])
    if total_cell_number > 1e9:
        # scale both C_seq and cell_numbers to bypass issues with large numbers
        # in multivariate_hypergeometric
        ndigits = int(np.log10(total_cell_number))
        C_seq_factor = 10 ** (ndigits - 8)
        C_seq = int(C_seq / C_seq_factor)
        cell_numbers = (cell_numbers / C_seq_factor).astype(int)
    else:
        C_seq_factor = 1

    seq_cell_numbers = rng.multivariate_hypergeometric(
        cell_numbers[nonzero_idx], int(C_seq)
    )

    # Assign the sampled cell numbers to the tree
    for idx, node in enumerate(nodes_to_sample):
        node.seq_cell_number = seq_cell_numbers[idx] * C_seq_factor

    return tree


def _truncate_tree(tree: Subclone) -> Subclone:
    """
    Recursively truncate leaves with zero sequenced cell numbers.
    and re-assign node ids.

    Parameters
    ----------
    tree : Subclone
            The tree to be truncated.
    """

    # recursively truncate the non-detected leaves
    check = False
    while not check:
        check = True
        for node in tree.leaves:
            if node.seq_cell_number == 0:
                node.parent = None
                del node
                check = False

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
                        seq_cell_number=0,
                        parent=node,
                    )
                    new_node.get_growth_params(
                        mu_vec=mu_vec, F_mat=F_mat, common_beta=common_beta
                    )
    else:
        raise NotImplementedError
        # TODO: implement other rules

    return tree
