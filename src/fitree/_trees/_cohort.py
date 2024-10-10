from typing import Any
import numpy as np

from ._tumor import TumorTree


class TumorTreeCohort:
    def __init__(
        self,
        name: str,
        trees: list[TumorTree] | Any = None,
        n_mutations: int = 0,
        N_trees: int = 0,
        N_patients: int = 0,
        mu_vec: np.ndarray | None = None,
        common_beta: float | None = None,
        C_0: int | float | None = None,
        C_seq: int | float | None = None,
        C_sampling: int | float | None = None,
        t_max: float | None = None,
        mutation_labels: list | Any = None,
        tree_labels: list | Any = None,
        patient_labels: list | Any = None,
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
