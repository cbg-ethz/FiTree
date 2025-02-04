from __future__ import annotations

from typing import Iterable, Any
from anytree import NodeMixin

import numpy as np


class SubcloneBase:
    def __init__(self) -> None:
        pass


class Subclone(SubcloneBase, NodeMixin):
    def __init__(
        self,
        node_id: int,
        mutation_ids: Iterable[int],
        seq_cell_number: int,
        cell_number: int | None = None,
        parent: Subclone | None = None,
        children: Iterable[Subclone] | None = None,
        genotype: list[int] | None = None,
        growth_params: dict | None = None,
        node_path: str | None = None,
    ) -> None:
        """A subclone in the tree

        Args:
            node_id (int): node id
            mutation_ids (Iterable[int]): mutation_ids in the subclone
            cell_number (int): number of cells attached
            seq_cell_number (int, optional): number of cells in the
                sequencing data. Defaults to None.
            parent (Subclone, optional): parent subclone. Defaults to None.
            children (Iterable[Subclone], optional): children subclones.
                Defaults to None.
        """

        super().__init__()
        self.node_id = node_id
        self.mutation_ids = mutation_ids
        self.seq_cell_number = seq_cell_number

        if cell_number is None:
            self.cell_number = seq_cell_number
        else:
            self.cell_number = cell_number

        self.parent = parent
        if children:
            self.children = children

        if genotype is None:
            self.genotype = self.get_genotype()
        else:
            self.genotype = genotype

        self.growth_params = growth_params
        self.node_path = node_path

        if self.is_root:
            self.growth_params = {
                "nu": 0,
                "alpha": 0,
                "beta": 0,
                "lam": 0,
                "delta": 0,
                "r": 1,
                "rho": 0,
                "phi": 0,
                "gamma": 0,
            }
            self.node_path = str([])
        else:
            self.node_path = (
                self.parent.node_path  # pyright: ignore
                + "->"
                + str(np.sort(self.mutation_ids))  # pyright: ignore
            )

    def get_genotype(self) -> list[int]:
        genotype = set()
        for node in self.path:
            genotype.update(node.mutation_ids)  # pyright: ignore
        return list(genotype)

    def update_mutation_ids(self, mutation_ids: Iterable[int]) -> None:
        self.mutation_ids = mutation_ids
        self.genotype = self.get_genotype()
        for child in self.children:
            child.genotype = child.get_genotype()

    def get_growth_params(
        self,
        mu_vec: np.ndarray,
        F_mat: np.ndarray,
        common_beta: float,
        return_dict: bool = False,
    ) -> dict | Any:
        """get growth parameters for the subclone

        Args:
            mu_vec: mutation rate vector
            F_mat: fitness matrix
            common_beta: common death rate
            return_dict: whether to return a dict or not

        Returns: None or
            growth_params: dict with growth parameters
            {
                "nu": mutation rate,
                "alpha": birth rate,
                "beta": death rate,
                "lam": net growth rate,
                "delta": running-max net growth rate,
                "r": number of times achieving the running-max net growth rate,
                "rho": shape parameter of the subclonal
                    population size distribution (nu / alpha),
                "phi": scale parameter of the subclonal population size distribution,
                "gamma": growth ratio
            }
        """

        if self.is_root:
            # mutations in the root are considered as germline mutations
            # and thus do not change the mutation rate and the fitness
            growth_params = {
                "nu": 0,
                "alpha": common_beta,
                "beta": common_beta,
                "lam": 0,
                "delta": 0,
                "r": 1,
                "rho": 0,
                "phi": common_beta,
                "gamma": 0,
            }

        else:
            gen_list = list(self.genotype)
            delta_pa = self.parent.growth_params["delta"]
            r_pa = self.parent.growth_params["r"]

            # mutation rate
            nu = np.prod(mu_vec[list(set(self.genotype) - set(self.parent.genotype))])

            # birth rate
            # compute the sum of fitness effects of a genotype
            # based on the fitness matrix F_mat.
            genotype_bool = np.zeros(F_mat.shape[0], dtype=bool)
            genotype_bool[gen_list] = True
            mask = genotype_bool[:, None] & genotype_bool[None, :]
            upper_tri_mask = np.triu(np.ones_like(F_mat, dtype=bool), k=1)
            combined_mask = mask & upper_tri_mask
            coef = np.sum(np.where(combined_mask, F_mat, 0.0))
            coef += np.max(np.where(genotype_bool, np.diag(F_mat), 0.0))
            log_alpha = np.log(common_beta) + coef
            alpha = np.exp(log_alpha)

            # death rate
            beta = common_beta

            # net growth rate
            lam = alpha - beta

            # running-max net growth rate
            delta = np.max([delta_pa, lam])  # pyright: ignore

            # number of times achieving the running-max net growth rate
            if lam > delta_pa:
                r = 1
            elif lam == delta_pa:
                r = r_pa + 1
            else:
                r = r_pa

            # subclonal population size distribution shape
            rho = nu / alpha

            # subclonal population size distribution scale
            if lam < 0:
                phi = -beta / lam
            elif lam == 0:
                phi = alpha
            else:
                phi = alpha / lam

            # growth ratio
            if delta == 0:
                gamma = 0
            else:
                gamma = delta_pa / delta

            growth_params = {
                "nu": nu,
                "alpha": alpha,
                "beta": beta,
                "lam": lam,
                "delta": delta,
                "r": r,
                "rho": rho,
                "phi": phi,
                "gamma": gamma,
            }

        self.growth_params = growth_params

        if return_dict:
            return growth_params

    def get_mrca(self) -> Subclone | Any:
        """Get the most recent common ancestor of the subclone

        Returns:
            Subclone: most recent common ancestor
        """

        if self.is_root:
            return self

        # recursively find the ancestor of the subclone
        # which has the root as its parent
        mrca = self
        while not mrca.parent.is_root:
            mrca = mrca.parent

        return mrca
