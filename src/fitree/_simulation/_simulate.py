import numpy as np

from anytree import PreOrderIter
from typing import Union, Tuple

from fitree._trees import Subclone
from ._utils import _truncate_tree, _expand_tree


# def _generate_one_tree(
# 	n_mutations: int,
# 	mu_vec: np.ndarray,
# 	F: np.ndarray,
# 	common_beta: float = 0.8,
# 	C_0: int | float | np.ndarray = 1e5,
# 	C_min: int | float | np.ndarray = 1e3,
# 	C_sampling: int | float | np.ndarray = 1e9,
# 	tau_eps: float = 0.03,
# 	T_max: float = 100,
# 	rule: str = "parallel",
# 	k_repeat: int = 0,
# 	k_multiple: int = 1,
# 	return_time: bool = False
# ) -> Union[Subclone, Tuple[Subclone, float]]:
# 	"""
# 	Generate one tree with the given number of mutations and the given
# 	mutation rate vector and fitness matrix.

# 	Parameters
# 	----------
# 	n_mutations : int
# 		The number of mutations to be considered.
# 	mu_vec : np.ndarray
# 		The n-by-1 mutation rate vector.
# 	F : np.ndarray
# 		The n-by-n fitness matrix.
# 	common_beta : float, optional
# 		The common death rate. Defaults to 0.8 (average 40 weeks).
# 	C_0 : int | float | np.ndarray, optional
# 		The static wild-type population size. Defaults to 1e5.
# 	C_min : int | float | np.ndarray, optional
# 		The minimum detectable number of cells. Defaults to 1e3.
# 	C_sampling : int | float | np.ndarray, optional
# 		The number of cells to sample. Defaults to 1e9.
# 	rule : str, optional
# 		The type of the tree generation. Defaults to "parallel".
# 		All options:
# 		1. "ISA": Infinite Sites Assumption.
# 		2. "parallel": parallel mutations are allowed, but no repeated mutations
# 			along the same branch or duplicated siblings.
# 		3. "repeat": repeated mutations are allowed, but up to k_repeat times.
# 		4. "multiple": multiple mutations in the same subclone are allowed,
# 			but up to k_multiple times.
# 	tau_eps : float, optional
# 		The threshold for the time interval. Defaults to 0.03.
# 	T_max : float, optional
# 		The maximum time to generate the tree. Defaults to 100.
# 	return_time : bool, optional
# 		Whether to return the time. Defaults to False.

# 	Returns
# 	-------
# 	Subclone
# 		The generated tree.
# 	| (Subclone, float)
# 		Generated tree and the time.
# 	"""

# 	""" Initialization phase """
# 	t = 0
# 	root = Subclone(node_id=0, mutation_ids=[], cell_number=C_0)
# 	sampling = 0

# 	""" Gillespie loop """
# 	while t < T_max and sampling == 0:
# 		""" Dynamically expand the tree based on the tree expansion rule """
# 		root = _expand_tree(
# 			tree=root,
# 			n_mutations=n_mutations,
# 			mu_vec=mu_vec,
# 			F=F,
# 			common_beta=common_beta,
# 			rule=rule,
# 			k_repeat=k_repeat,
# 			k_multiple=k_multiple
# 		)

# 		""" Model definition """
# 		# For each node except the root,
# 		# we have the following reactions:
# 		# 1. Mutation: X_pa_i -> X_pa_i + X_i
# 		# with rate X_i.growth_params["nu"]
# 		# 2. Birth: X_i -> X_i + X_i
# 		# with rate X_i.growth_params["alpha"]
# 		# 3. Death: X_i -> 0
# 		# with rate X_i.growth_params["beta"]
# 		# The sampling reaction is 0 -> S with rate equal to the sum of all
# 		# subclone cell numbers divided by C_sampling.

# 		""" Calculate propensity functions and step size tau """
# 		""" Step size selection based on Cao et al. (2006) """
# 		nr_reactions = 3 * (root.size - 1)
# 		a_vec = np.zeros(nr_reactions)
# 		tau = np.inf
# 		C_all = 0

# 		tree_iter = PreOrderIter(root)
# 		next(tree_iter) # skip the root
# 		for node in tree_iter:
# 			if node.parent.cell_number > 0 or node.cell_number > 0:
# 				C_all += node.cell_number
# 				idx = 3 * (node.node_id - 1)
# 				a1 = node.growth_params["nu"] * node.parent.cell_number
# 				a2 = node.growth_params["alpha"] * node.cell_number
# 				a3 = node.growth_params["beta"] * node.cell_number
# 				a_vec[idx] = a1
# 				a_vec[idx + 1] = a2
# 				a_vec[idx + 2] = a3
# 				tau_g = np.max([a1, a2, a3])
# 				tau_mu = a1 + a2 - a3
# 				tau_sigma = a1 + a2 + a3
# 				if tau_g == 0:
# 					tau_max = 1
# 				else:
# 					tau_max = np.max([tau_eps * node.cell_number / tau_g, 1])
# 				if tau_mu == 0:
# 					tau_min = np.power(tau_max, 2) / tau_sigma
# 				else:
# 					tau_min = np.min([
# 						tau_max / np.abs(tau_mu),
# 						np.power(tau_max, 2) / tau_sigma
# 					])
# 				tau = np.min([tau, tau_min])
# 		a_sampling = C_all / C_sampling
# 		if a_sampling > 0:
# 			tau = np.min([tau, 1 / a_sampling])
# 		tau = np.max([tau, 1e-4])

# 		""" Calculate the number of reactions to occur in time step tau """
# 		""" Update cell numbers """
# 		""" Ensure cell numbers don't go negative """
# 		""" Update sampling event """
# 		tree_iter = PreOrderIter(root)
# 		next(tree_iter) # skip the root
# 		for node in tree_iter:
# 			if node.parent.cell_number > 0 or node.cell_number > 0:
# 				idx = 3 * (node.node_id - 1)
# 				r1 = np.random.poisson(a_vec[idx] * tau)
# 				r2 = np.random.poisson(a_vec[idx + 1] * tau)
# 				r3 = np.random.poisson(a_vec[idx + 2] * tau)
# 				node.cell_number += r1 + r2 - r3
# 				node.cell_number = np.max([node.cell_number, 0])
# 		sampling += np.random.poisson(a_sampling * tau)

# 		""" Update time """
# 		t += tau

# 	""" Recursively truncate the non-detected leaves """
# 	root = _truncate_tree(root, C_min=C_min)

# 	if return_time:
# 		return (root, t)

# 	return root


def _generate_one_tree(
    n_mutations: int,
    mu_vec: np.ndarray,
    F: np.ndarray,
    common_beta: float = 0.8,
    C_0: int | float | np.ndarray = 1e5,
    C_min: int | float | np.ndarray = 1e3,
    C_sampling: int | float | np.ndarray = 1e9,
    tau: float = 1e-3,
    T_max: float = 100,
    rule: str = "parallel",
    k_repeat: int = 0,
    k_multiple: int = 1,
    return_time: bool = False,
) -> Union[Subclone, Tuple[Subclone, float]]:
    """
    Generate one tree with the given number of mutations and the given
    mutation rate vector and fitness matrix.

    Parameters
    ----------
    n_mutations : int
            The number of mutations to be considered.
    mu_vec : np.ndarray
            The n-by-1 mutation rate vector.
    F : np.ndarray
            The n-by-n fitness matrix.
    common_beta : float, optional
            The common death rate. Defaults to 0.8 (average 40 weeks).
    C_0 : int | float | np.ndarray, optional
            The static wild-type population size. Defaults to 1e5.
    C_min : int | float | np.ndarray, optional
            The minimum detectable number of cells. Defaults to 1e3.
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
    T_max : float, optional
            The maximum time to generate the tree. Defaults to 100.
    return_time : bool, optional
            Whether to return the time. Defaults to False.

    Returns
    -------
    Subclone
            The generated tree.
    | (Subclone, float)
            Generated tree and the time.
    """

    """ Initialization phase """
    t = 0
    root = Subclone(node_id=0, mutation_ids=[], cell_number=C_0)
    sampling = 0

    """ Gillespie loop """
    while t < T_max and sampling == 0:
        # Dynamically expand the tree based on the tree expansion rule
        root = _expand_tree(
            tree=root,
            n_mutations=n_mutations,
            mu_vec=mu_vec,
            F=F,
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
                r1 = np.random.poisson(a1 * tau)
                r2 = np.random.poisson(a2 * tau)
                r3 = np.random.poisson(a3 * tau)

                # Update molecule counts and ensure non-negativity
                node.cell_number += r1 + r2 - r3
                node.cell_number = np.max([node.cell_number, 0])

                # # Expand the tree if necessary
                # if node.is_leaf and node.cell_number > 0:
                # 	possible_mutations = set(range(n_mutations)).difference(
                # 		node.get_genotype()
                # 	)
                # 	for j in possible_mutations:
                # 		new_node = Subclone(
                # 			node_id=node.node_id + 1,
                # 			mutation_ids=[j],
                # 			cell_number=0,
                # 			parent=node
                # 		)
                # 		new_node.get_growth_params(
                # 			mu_vec=mu_vec,
                # 			F=F,
                # 			common_beta=common_beta
                # 		)

        # Repeat the above for the sampling reaction
        a_sampling = C_all / C_sampling
        sampling += np.random.poisson(a_sampling * tau)

        # Update time
        t += tau

    # Recursively truncate the non-detected leaves
    root = _truncate_tree(root, C_min=C_min)

    if return_time:
        return (root, t)

    return root


def generate_trees(
    n_mutations: int,
    N_trees: int,
    mu_vec: np.ndarray,
    F: np.ndarray,
    common_beta: float = 0.8,
    C_0: int | float | np.ndarray = 1e5,
    C_min: int | float | np.ndarray = 1e3,
    C_sampling: int | float | np.ndarray = 1e9,
    tau: float = 1e-3,
    T_max: float = 100,
    rule: str = "parallel",
    k_repeat: int = 0,
    k_multiple: int = 1,
    return_time: bool = False,
) -> Union[list[Subclone], list[Tuple[Subclone, float]]]:
    """
    Generate a list of trees with the given number of mutations and the given
    mutation rate vector and fitness matrix.

    Parameters
    ----------
    n_mutations : int
            The number of mutations to be considered.
    N_trees : int
            The number of trees to generate.
    mu_vec : np.ndarray
            The n-by-1 mutation rate vector.
    F : np.ndarray
            The n-by-n fitness matrix.
    common_beta : float, optional
            The common death rate. Defaults to 0.8 (average 40 weeks).
    C_0 : int | float | np.ndarray, optional
            The static wild-type population size. Defaults to 1e5.
    C_min : int | float | np.ndarray, optional
            The minimum detectable number of cells. Defaults to 1e3.
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
    T_max : float, optional
            The maximum time to generate the tree. Defaults to 100.
    return_time : bool, optional
            Whether to return the time. Defaults to False.

    Returns
    -------
    list[Subclone]
            The generated trees.
    | list[(Subclone, float)]
            Generated trees and the times.
    """

    trees = []
    for i in range(N_trees):
        trees.append(
            _generate_one_tree(
                n_mutations=n_mutations,
                mu_vec=mu_vec,
                F=F,
                common_beta=common_beta,
                C_0=C_0,
                C_min=C_min,
                C_sampling=C_sampling,
                tau=tau,
                T_max=T_max,
                rule=rule,
                k_repeat=k_repeat,
                k_multiple=k_multiple,
                return_time=return_time,
            )
        )

    return trees
