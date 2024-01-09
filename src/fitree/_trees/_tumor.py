from anytree import PreOrderIter

from ._subclone import Subclone


class TumorTree:
    def __init__(
        self,
        patient_id: int,
        tree_id: int,
        root: Subclone,
        weight: float = 1.0,
        sampling_time: float | None = None,
    ) -> None:
        """A tumor tree

        Args:
                patient_id (int): patient id
                tree_id (int): tree id
                root (Subclone): root subclone
                weight (float, optional): weight of the tree. Defaults to 1.0.
                sampling_time (float, optional): sampling time of the tree.
                        Defaults to None.
        """

        if not root.is_root:
            raise ValueError("The tree given is not a root node!")

        self.patient_id = patient_id
        self.tree_id = tree_id
        self.root = root
        self.weight = weight
        self.sampling_time = sampling_time

    def get_mutation_ids(self) -> set:
        all_mutation_ids = set()
        for node in PreOrderIter(self.root):
            all_mutation_ids.update(node.mutation_ids)
        return all_mutation_ids
