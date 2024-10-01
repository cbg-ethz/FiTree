import json
import numpy as np
from anytree.exporter import DictExporter
from anytree.importer import DictImporter

from ._subclone import Subclone
from ._tumor import TumorTree
from ._cohort import TumorTreeCohort


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)  # Convert np.int64 to Python int
        elif isinstance(obj, np.floating):
            return float(obj)  # Convert np.float64 to Python float
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert np.array to list
        else:
            return super().default(obj)  # Default behavior for other types


def save_cohort(cohort: TumorTreeCohort, path: str) -> None:
    """Save a TumorTreeCohort object to a JSON file."""

    exporter = DictExporter()

    def serialize_tree(tree: TumorTree) -> dict:
        """Helper function to serialize a TumorTree object."""
        root_dict = exporter.export(tree.root)

        return {
            "patient_id": tree.patient_id,
            "tree_id": tree.tree_id,
            "weight": tree.weight,
            "sampling_time": tree.sampling_time,
            "tree": root_dict,
        }

    serialized_trees = [serialize_tree(tree) for tree in cohort.trees]

    serialized_cohort = {
        "name": cohort.name,
        "n_mutations": cohort.n_mutations,
        "N_trees": cohort.N_trees,
        "N_patients": cohort.N_patients,
        "mu_vec": cohort.mu_vec.tolist(),
        "common_beta": cohort.common_beta,
        "C_0": cohort.C_0,
        "C_min": cohort.C_min,
        "C_sampling": cohort.C_sampling,
        "t_max": cohort.t_max,
        "mutation_labels": cohort.mutation_labels,
        "tree_labels": cohort.tree_labels,
        "patient_labels": cohort.patient_labels,
        "trees": serialized_trees,
    }

    with open(path, "w") as f:
        json.dump(serialized_cohort, f, indent=2, cls=NumpyEncoder)


def load_cohort_from_json(path: str) -> TumorTreeCohort:
    """Load a TumorTreeCohort object from a JSON file."""

    # Initialize the Subclone importer
    importer = DictImporter(nodecls=Subclone)

    # Helper function to reconstruct TumorTree objects from JSON
    def reconstruct_tree(tree_data):
        """Reconstructs a TumorTree object from JSON data."""
        # Use DictImporter to recreate the Subclone tree
        root_node = importer.import_(tree_data["tree"])
        tumor_tree = TumorTree(
            patient_id=tree_data["patient_id"],
            tree_id=tree_data["tree_id"],
            root=root_node,
            weight=tree_data["weight"],
            sampling_time=tree_data["sampling_time"],
        )
        return tumor_tree

    # Load the JSON data from the file
    with open(path, "r") as f:
        data = json.load(f)

    # Reconstruct the TumorTreeCohort object
    trees = [reconstruct_tree(tree_data) for tree_data in data["trees"]]

    cohort = TumorTreeCohort(
        name=data["name"],
        trees=trees,
        n_mutations=data["n_mutations"],
        N_trees=data["N_trees"],
        N_patients=data["N_patients"],
        mu_vec=np.array(data["mu_vec"]),
        common_beta=data["common_beta"],
        C_0=data["C_0"],
        C_min=data["C_min"],
        C_sampling=data["C_sampling"],
        t_max=data["t_max"],
        mutation_labels=data["mutation_labels"],
        tree_labels=data["tree_labels"],
        patient_labels=data["patient_labels"],
    )

    return cohort
