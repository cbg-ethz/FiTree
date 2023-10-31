from anytree import PreOrderIter

from ._subclone import Subclone


class TumorTree:
    def __init__(
        self,
        patient_id: int,
        tree_id: int,
        tree: Subclone,
        weight: float = 1.0,
        sampling_time: float | None = None,
    ) -> None:
        """A tumor tree

        Args:
                patient_id (int): patient id
                tree_id (int): tree id
                tree (Subclone): root subclone
                weight (float, optional): weight of the tree. Defaults to 1.0.
                sampling_time (float, optional): sampling time of the tree.
                        Defaults to None.
        """
        self.patient_id = patient_id
        self.tree_id = tree_id
        self.tree = tree
        self.weight = weight
        self.sampling_time = sampling_time

    def get_mutation_ids(self) -> set:
        all_mutation_ids = set()
        for node in PreOrderIter(self.tree):
            all_mutation_ids.update(node.mutation_ids)
        return all_mutation_ids
