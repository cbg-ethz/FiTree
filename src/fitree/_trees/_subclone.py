from __future__ import annotations

from typing import Iterable
from anytree import NodeMixin

import numpy as np


class SubcloneBase:
    def __init__(self) -> None:
        pass


class Subclone(SubcloneBase, NodeMixin):
    def __init__(
        self,
        node_id: int,
        mutations: Iterable[int],
        cell_number: int,
        parent: Subclone | None = None,
        children: Iterable[Subclone] | None = None,
    ) -> None:
        """A subclone in the tree

        Args:
            node_id (int): node id
            mutations (Iterable[int]): mutations in the subclone
            cell_number (int): number of cells attached
            parent (Subclone, optional): parent subclone. Defaults to None.
            children (Iterable[Subclone], optional): children subclones.
                Defaults to None.
        """

        super().__init__()
        self.node_id = node_id
        self.mutations = mutations
        self.cell_number = cell_number
        self.parent = parent
        if children:
            self.children = children

        self.genotype = self.get_genotype()

    def get_genotype(self) -> set:
        genotype = set()
        for node in self.path:
            genotype.update(node.mutations)  # pyright: ignore
        return genotype

    def update_mutations(self, mutations: Iterable[int]) -> None:
        self.mutations = mutations
        self.genotype = self.get_genotype()
        for child in self.children:
            child.genotype = child.get_genotype()

    def get_growth_params(
        self,
        mu_vec: np.ndarray,
        F: np.ndarray,
        common_beta: float,
        return_dict: bool = False,
    ) -> dict | None:
        """get growth parameters for the subclone

        Args:
            mu_vec: mutation rate vector
            F: fitness matrix
            common_beta: common death rate
            return_dict: whether to return a dict or not

        Returns: None or
            growth_params: dict with growth parameters
            {
                "nu": mutation rate,
                "alpha": birth rate,
                "beta": death rate,
                "lambda": net growth rate,
                "delta": running-max net growth rate,
                "r": number of times achieving the running-max net growth rate,
                "rho": shape parameter of the subclonal
                    population size distribution (nu / alpha),
                "phi": scale parameter of the subclonal population size distribution,
                "gamma": growth ratio
            }
        """

        if self.is_root:
            self.nu = 0
            self.alpha = common_beta
            self.beta = common_beta
            self.lam = 0
            self.delta = 0
            self.r = 1
            self.rho = 0
            self.phi = self.alpha
            self.gamma = 0

        else:
            gen_list = list(self.genotype)

            # mutation rate
            self.nu = np.prod(mu_vec[list(self.genotype - self.parent.genotype)])

            # birth rate
            coef = 1
            for i in range(len(gen_list)):
                for j in range(i, len(gen_list)):
                    coef *= F[gen_list[i], gen_list[j]]
            self.alpha = common_beta * coef

            # death rate
            self.beta = common_beta

            # net growth rate
            self.lam = self.alpha - self.beta

            # running-max net growth rate
            self.delta = max(self.parent.delta, self.lam)  # pyright: ignore

            # number of times achieving the running-max net growth rate
            if self.lam > self.parent.delta:
                self.r = 1
            elif self.lam == self.parent.delta:
                self.r = self.parent.r + 1
            else:
                self.r = self.parent.r

            # subclonal population size distribution shape
            self.rho = self.nu / self.alpha

            # subclonal population size distribution scale
            if self.lam < 0:
                self.phi = -self.beta / self.lam
            elif self.lam == 0:
                self.phi = self.alpha
            else:
                self.phi = self.alpha / self.lam

            # growth ratio
            if self.delta == 0:
                self.gamma = 0
            else:
                self.gamma = self.parent.delta / self.delta

        if return_dict:
            growth_params = {
                "nu": self.nu,
                "alpha": self.alpha,
                "beta": self.beta,
                "lambda": self.lam,
                "delta": self.delta,
                "r": self.r,
                "rho": self.rho,
                "phi": self.phi,
                "gamma": self.gamma,
            }

            return growth_params
