from typing import Any
import numpy as np

from ._tumor import TumorTree


class TumorTreeCohort:
    def __init__(
        self,
        name: str,
        trees: list[TumorTree] | Any = None,
        n_mutations: int | Any = 0,
        N_trees: int | Any = 0,
        N_patients: int | Any = 0,
        mu_vec: np.ndarray | Any = None,
        common_beta: float | Any = None,
        C_0: int | float | Any = None,
        C_seq: int | float | Any = None,
        C_sampling: int | float | Any = None,
        t_max: float | Any = None,
        mutation_labels: list | Any = None,
        tree_labels: list | Any = None,
        patient_labels: list | Any = None,
        lifetime_risk: float | Any = None,
    ) -> None:
        self.name = name
        self.trees = trees
        self.n_mutations = n_mutations
        self.N_trees = N_trees
        self.N_patients = N_patients
        self.mu_vec = mu_vec
        self.common_beta = common_beta
        self.C_0 = C_0
        self.C_seq = C_seq
        self.C_sampling = C_sampling

        if len(trees) > 0:
            self.get_t_max()
        else:
            self.t_max = t_max

        self.mutation_labels = mutation_labels
        self.tree_labels = tree_labels
        self.patient_labels = patient_labels

        if lifetime_risk is not None:
            # check if lifetime_risk falls within [0, 1]
            if lifetime_risk < 0 or lifetime_risk > 1:
                raise ValueError("lifetime_risk must be in [0, 1]")

        self.lifetime_risk = lifetime_risk

        self._check_trees()

    def _check_trees(self) -> None:
        # check if the mutations in the trees all have labels

        mutation_ids_in_trees = set()
        for tree in self.trees:
            mutation_ids_in_trees.update(tree.get_mutation_ids())

        # check index error
        for mutation_id in mutation_ids_in_trees:
            try:
                self.mutation_labels[mutation_id]
            except IndexError:
                raise IndexError(
                    f"mutation_labels does not have label for mutation {mutation_id}"
                )

        if len(self.trees) != self.N_trees:
            raise ValueError("trees must have length N_trees")

        if len(self.mutation_labels) != self.n_mutations:
            raise ValueError("mutation_labels must have length n_mutations")

        if len(self.tree_labels) != self.N_trees:
            raise ValueError("tree_labels must have length N_trees")

        if len(self.patient_labels) != self.N_patients:
            raise ValueError("patient_labels must have length N_patients")

        # TODO: implement other checks

    def get_t_max(self) -> None:
        self.t_max = 0.0

        for tree in self.trees:
            if tree.sampling_time > self.t_max:  # pyright: ignore
                self.t_max = tree.sampling_time

    def get_observed_mutations(self) -> set[int]:
        observed_mutations = set()
        for tree in self.trees:
            observed_mutations.update(tree.get_mutation_ids())

        return observed_mutations

    def compute_mean_std_tumor_size(self) -> tuple[float, float]:
        tumor_sizes = np.array([tree.tumor_size for tree in self.trees])
        return tumor_sizes.mean(), tumor_sizes.std()
