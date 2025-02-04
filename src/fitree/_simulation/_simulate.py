import numpy as np
from multiprocessing import Pool

from anytree import PreOrderIter
from typing import Tuple, Any

from fitree._trees import Subclone, TumorTree, TumorTreeCohort
from ._utils import _expand_tree, _sample_cells, _truncate_tree


def _generate_one_tree(
    rng: np.random.Generator,
    n_mutations: int,
    mu_vec: np.ndarray | Any,
    F_mat: np.ndarray | Any,
    common_beta: float | Any = 1.0,
    C_0: int | Any = 1e5,
    C_seq: int | Any = 1e4,
    C_sampling: int | Any = 1e9,
    tau: float | Any = 1e-3,
    t_max: float | Any = 100,
    rule: str | Any = "parallel",
    k_repeat: int | Any = 0,
    k_multiple: int | Any = 1,
) -> Tuple[Subclone, float] | Any:
    """
    Generate one tree with the given number of mutations and the given
    mutation rate vector and fitness matrix.

    Parameters
    ----------
    rng : np.random.Generator
                        The random number generator.
    n_mutations : int
            The number of mutations to be considered.
    mu_vec : np.ndarray
            The n-by-1 mutation rate vector.
    F_mat : np.ndarray
            The n-by-n fitness matrix.
    common_beta : float, optional
            The common death rate. Defaults to 0.8 (average 40 weeks).
    C_0 : int | float | np.ndarray, optional
            The static wild-type population size. Defaults to 1e5.
    C_seq : int | np.ndarray, optional
            Number of cells to sequence. Defaults to 1e4.
    C_sampling : int | float | np.ndarray, optional
            The number of cells to sample. Defaults to 1e9.
    rule : str, optional
            The type of the tree generation. Defaults to "parallel".
            All options:
            1. "ISA": Infinite Sites Assumption.
            2. "parallel": parallel mutations are allowed, but no repeated mutations
                    along the same branch or duplicated siblings.
            3. "repeat": repeated mutations are allowed, but up to k_repeat times.
            4. "multiple": multiple mutations in the same subclone are allowed,
                    but up to k_multiple times.
    tau: float, optional
            The step size of the tau-leaping algorithm. Defaults to 1e-3.
    t_max : float, optional
            The maximum time to generate the tree. Defaults to 100.

    Returns
    -------
    Subclone
            The generated tree.
    | (Subclone, float)
            Generated tree and the time.
    """

    """ Initialization phase """
    t = 0
    root = Subclone(node_id=0, mutation_ids=[], seq_cell_number=C_0)
    sampling = 0

    """ Gillespie loop """
    while t < t_max and sampling == 0:
        # Dynamically expand the tree based on the tree expansion rule
        root = _expand_tree(
            tree=root,
            n_mutations=n_mutations,
            mu_vec=mu_vec,
            F_mat=F_mat,
            common_beta=common_beta,
            rule=rule,
            k_repeat=k_repeat,
            k_multiple=k_multiple,
        )

        """ Model definition """
        # For each node except the root,
        # we have the following reactions:
        # 1. Mutation: X_pa_i -> X_pa_i + X_i
        # with rate X_i.growth_params["nu"]
        # 2. Birth: X_i -> X_i + X_i
        # with rate X_i.growth_params["alpha"]
        # 3. Death: X_i -> 0
        # with rate X_i.growth_params["beta"]
        # The sampling reaction is 0 -> S with rate equal to the sum of all
        # subclone cell numbers divided by C_sampling.
        """""" """""" """""" """"""

        C_all = 0
        tree_iter = PreOrderIter(root)
        next(tree_iter)  # skip the root
        for node in tree_iter:
            if node.parent.cell_number > 0 or node.cell_number > 0:
                C_all += node.cell_number

                # Calculate propensities
                a1 = node.growth_params["nu"] * node.parent.cell_number
                a2 = node.growth_params["alpha"] * node.cell_number
                a3 = node.growth_params["beta"] * node.cell_number

                # Calculate number of reactions to occur in time step tau
                r1 = rng.poisson(a1 * tau)
                r2 = rng.poisson(a2 * tau)
                r3 = rng.poisson(a3 * tau)

                # Update molecule counts and ensure non-negativity
                node.cell_number += r1 + r2 - r3
                node.cell_number = np.max([node.cell_number, 0])

        # Repeat the above for the sampling reaction
        a_sampling = C_all / C_sampling
        sampling += rng.poisson(a_sampling * tau)

        # Update time
        t += tau

    # Sample cells from the tree
    root = _sample_cells(rng=rng, tree=root, C_seq=C_seq)

    # Recursively truncate the non-detected leaves
    root = _truncate_tree(root)

    return root, t


def _generate_valid_tree(
    rng: np.random.Generator,
    i: int,
    n_mutations: int,
    mu_vec: np.ndarray | Any,
    F_mat: np.ndarray | Any,
    common_beta: float | Any = 1.0,
    C_0: int | Any = 1e5,
    C_seq: int | Any = 1e4,
    C_sampling: int | Any = 1e9,
    tau: float | Any = 1e-3,
    t_max: float | Any = 100,
    rule: str | Any = "parallel",
    k_repeat: int | Any = 0,
    k_multiple: int | Any = 1,
    return_time: bool | Any = True,
) -> Tuple[Subclone, float] | Any:
    # Generate a tree and ensure that the sampling event occurs
    # before the maximum time t_max
    while True:
        root, t = _generate_one_tree(
            rng=rng,
            n_mutations=n_mutations,
            mu_vec=mu_vec,
            F_mat=F_mat,
            common_beta=common_beta,
            C_0=C_0,
            C_seq=C_seq,
            C_sampling=C_sampling,
            tau=tau,
            t_max=t_max,
            rule=rule,
            k_repeat=k_repeat,
            k_multiple=k_multiple,
        )
        if t < t_max:
            break

    tumor_size = 0
    node_iter = PreOrderIter(root)
    next(node_iter)  # Skip the root node
    for node in node_iter:
        tumor_size += node.cell_number

    if return_time:
        tree = TumorTree(
            patient_id=i, tree_id=i, root=root, sampling_time=t, tumor_size=tumor_size
        )
    else:
        tree = TumorTree(patient_id=i, tree_id=i, root=root, tumor_size=tumor_size)

    return tree


def generate_trees(
    rng: np.random.Generator,
    n_mutations: int,
    N_trees: int,
    mu_vec: np.ndarray | Any,
    F_mat: np.ndarray | Any,
    common_beta: float | Any = 1.0,
    C_0: int | Any = 1e5,
    C_seq: int | Any = 1e4,
    C_sampling: int | Any = 1e9,
    tau: float | Any = 1e-3,
    t_max: float | Any = 100,
    rule: str | Any = "parallel",
    k_repeat: int | Any = 0,
    k_multiple: int | Any = 1,
    return_time: bool | Any = True,
    parallel: bool | Any = False,
    n_jobs: int | Any = -1,
) -> TumorTreeCohort:
    """
    Generate a list of trees with the given number of mutations and the given
    mutation rate vector and fitness matrix.

    Parameters
    ----------
    rng : np.random.Generator
                        The random number generator.
    n_mutations : int
            The number of mutations to be considered.
    N_trees : int
            The number of trees to generate.
    mu_vec : np.ndarray
            The n-by-1 mutation rate vector.
    F_mat : np.ndarray
            The n-by-n fitness matrix.
    common_beta : float, optional
            The common death rate. Defaults to 0.8 (average 40 weeks).
    C_0 : int | float | np.ndarray, optional
            The static wild-type population size. Defaults to 1e5.
    C_seq : int | np.ndarray, optional
            Number of cells to sequence. Defaults to 1e4.
    C_sampling : int | float | np.ndarray, optional
            The number of cells to sample. Defaults to 1e9.
    rule : str, optional
            The type of the tree generation. Defaults to "parallel".
            All options:
            1. "ISA": Infinite Sites Assumption.
            2. "parallel": parallel mutations are allowed, but no repeated mutations
                    along the same branch or duplicated siblings.
            3. "repeat": repeated mutations are allowed, but up to k_repeat times.
            4. "multiple": multiple mutations in the same subclone are allowed,
                    but up to k_multiple times.
    tau: float, optional
            The step size of the tau-leaping algorithm. Defaults to 1e-3.
    t_max : float, optional
            The maximum time to generate the tree. Defaults to 100.
    return_time : bool, optional
            Whether to return the sampling time. Defaults to True.

    Returns
    -------
    list[Subclone]
            The generated trees.
    | list[(Subclone, float)]
            Generated trees and the times.
    """

    seeds = rng.integers(0, 2**32 - 1, size=N_trees)

    if parallel:
        n_jobs = n_jobs if n_jobs > 0 else None  # None means using all available cores
        with Pool(processes=n_jobs) as pool:
            trees = pool.starmap(
                _generate_valid_tree,
                [
                    (
                        np.random.default_rng(seeds[i]),
                        i,
                        n_mutations,
                        mu_vec,
                        F_mat,
                        common_beta,
                        C_0,
                        C_seq,
                        C_sampling,
                        tau,
                        t_max,
                        rule,
                        k_repeat,
                        k_multiple,
                        return_time,
                    )
                    for i in range(N_trees)
                ],
            )
    else:
        trees = [
            _generate_valid_tree(
                rng=np.random.default_rng(seeds[i]),
                i=i,
                n_mutations=n_mutations,
                mu_vec=mu_vec,
                F_mat=F_mat,
                common_beta=common_beta,
                C_0=C_0,
                C_seq=C_seq,
                C_sampling=C_sampling,
                tau=tau,
                t_max=t_max,
                rule=rule,
                k_repeat=k_repeat,
                k_multiple=k_multiple,
                return_time=return_time,
            )
            for i in range(N_trees)
        ]

    cohort = TumorTreeCohort(
        name="simulated",
        trees=trees,
        n_mutations=n_mutations,
        N_trees=N_trees,
        N_patients=N_trees,
        mu_vec=mu_vec,
        common_beta=common_beta,
        C_0=C_0,
        C_seq=C_seq,
        C_sampling=C_sampling,
        t_max=t_max,
        mutation_labels=["M" + str(i) for i in range(n_mutations)],
        tree_labels=["T" + str(i) for i in range(N_trees)],
        patient_labels=["P" + str(i) for i in range(N_trees)],
    )

    return cohort
