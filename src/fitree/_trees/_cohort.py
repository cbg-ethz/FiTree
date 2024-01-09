from typing import Any

from ._tumor import TumorTree


class TumorTreeCohort:
    def __init__(
        self,
        name: str,
        trees: list[TumorTree] | Any = None,
        n_mutations: int = 0,
        N_trees: int = 0,
        N_patients: int = 0,
        mutation_labels: dict | Any = None,
        tree_labels: dict | Any = None,
        patient_labels: dict | Any = None,
    ) -> None:
        self.name = name

        if trees is not None:
            self.n_mutations = n_mutations
            self.N_trees = N_trees
            self.N_patients = N_patients

            if len(mutation_labels) != n_mutations:
                raise ValueError("mutation_labels must have length n_mutations")
            self.mutation_labels = mutation_labels

            if len(tree_labels) != N_trees:
                raise ValueError("tree_labels must have length N_trees")
            self.tree_labels = tree_labels

            if len(trees) != N_trees:
                raise ValueError("trees must have length N_trees")
            self.trees = trees

            if len(patient_labels) != N_patients:
                raise ValueError("patient_labels must have length N_patients")
            self.patient_labels = patient_labels

            self._check_trees()

        else:
            self.trees = []
            self.n_mutations = 0
            self.N_trees = 0
            self.N_patients = 0
            self.mutation_labels = {}
            self.tree_labels = {}
            self.patient_labels = {}

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

    def add_tree(
        self,
        tree: TumorTree,
        patient_label: str | None = None,
        tree_label: str | None = None,
    ) -> None:
        self.trees.append(tree)
        mutation_ids_in_tree = tree.get_mutation_ids()

        # check if the mutation_ids in the tree are already in the cohort
        # if not, add them
        for mutation_id in mutation_ids_in_tree:
            if mutation_id not in self.mutation_labels:
                self.mutation_labels[mutation_id] = f"M_{mutation_id}"
                self.n_mutations += 1

        # check if the patient label is already in the cohort
        # if not, add it
        if tree.patient_id not in self.patient_labels:
            if patient_label is None:
                patient_label = f"P_{tree.patient_id}"
            self.patient_labels[tree.patient_id] = patient_label
            self.N_patients += 1
        else:
            if patient_label is not None:
                if patient_label != self.patient_labels[tree.patient_id]:
                    raise ValueError(
                        f"patient_label for patient {tree.patient_id} \
                            does not match the existing label"
                    )

        # check if the tree label is already in the cohort
        # if not, add it
        if tree.tree_id not in self.tree_labels:
            if tree_label is None:
                tree_label = f"T_{tree.tree_id}"
            self.tree_labels[tree.tree_id] = tree_label
            self.N_trees += 1
        else:
            if tree_label is not None:
                if tree_label != self.tree_labels[tree.tree_id]:
                    raise ValueError(
                        f"tree_label for tree {tree.tree_id} \
                            does not match the existing label"
                    )
