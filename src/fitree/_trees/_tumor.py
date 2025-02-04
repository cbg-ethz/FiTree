import numpy as np
from anytree import PreOrderIter, RenderTree

from ._subclone import Subclone


class TumorTree:
    def __init__(
        self,
        patient_id: int,
        tree_id: int,
        root: Subclone,
        tumor_size: float,
        weight: float = 1.0,
        sampling_time: float | None = None,
    ) -> None:
        """A tumor tree

        Args:
                patient_id (int): patient id
                tree_id (int): tree id
                root (Subclone): root subclone
                tumor_size (float): total number of tumor cells
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
        self.tumor_size = tumor_size

        self.assign_cells()

    def get_mutation_ids(self) -> set:
        all_mutation_ids = set()
        for node in PreOrderIter(self.root):
            all_mutation_ids.update(node.mutation_ids)
        return all_mutation_ids

    def __str__(self) -> str:
        if self.sampling_time is not None:
            tree_str = f"Tumor tree {self.tree_id} of patient \
                {self.patient_id} at time {self.sampling_time}\n"
        else:
            tree_str = f"Tumor tree {self.tree_id} of patient {self.patient_id}\n"

        tree_str += f" - Total number of tumor cells: {self.tumor_size:.4E}\n"

        tree_str += RenderTree(self.root).by_attr(
            lambda node: f"(Node ID: {node.node_id}) \n"  # pyright: ignore
            + f" - Mutations: {node.mutation_ids} \n"
            + f" - Cell number: {node.cell_number:.4E} \n"
            + f" - Sequence cell number: {node.seq_cell_number:.4E} \n"
        )

        return tree_str

    def assign_cells(self) -> None:
        # Compute the relative frequency of each subclone except the root
        node_iter = PreOrderIter(self.root)
        next(node_iter)  # skip the root
        seq_cells = np.array([node.seq_cell_number for node in node_iter])
        freq = seq_cells / np.sum(seq_cells)
        cells = self.tumor_size * freq

        # Assign the cells to each subclone
        node_iter = PreOrderIter(self.root)
        next(node_iter)
        for node, cell_number in zip(node_iter, cells):
            node.cell_number = cell_number
