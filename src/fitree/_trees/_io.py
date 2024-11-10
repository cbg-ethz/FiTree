import json
import numpy as np
from anytree.exporter import DictExporter
from anytree.importer import DictImporter

from ._subclone import Subclone
from ._tumor import TumorTree
from ._cohort import TumorTreeCohort
from ._wrapper import VectorizedTrees


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):  # pyright: ignore
        if isinstance(obj, np.integer):
            return int(obj)  # Convert np.int64 to Python int
        elif isinstance(obj, np.floating):
            return float(obj)  # Convert np.float64 to Python float
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert np.array to list
        else:
            return super().default(obj)  # Default behavior for other types


def save_cohort_to_json(cohort: TumorTreeCohort, path: str) -> None:
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
            "tumor_size": tree.tumor_size,
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
        "C_seq": cohort.C_seq,
        "C_sampling": cohort.C_sampling,
        "t_max": cohort.t_max,
        "mutation_labels": cohort.mutation_labels,
        "tree_labels": cohort.tree_labels,
        "patient_labels": cohort.patient_labels,
        "lifetime_risk": cohort.lifetime_risk,
        "trees": serialized_trees,
    }

    with open(path, "w") as f:
        json.dump(serialized_cohort, f, cls=NumpyEncoder)


def load_cohort_from_json(path: str) -> TumorTreeCohort:
    """Load a TumorTreeCohort object from a JSON file."""

    # Initialize the Subclone importer
    importer = DictImporter(nodecls=Subclone)  # pyright: ignore

    # Helper function to reconstruct TumorTree objects from JSON
    def reconstruct_tree(tree_data):
        """Reconstructs a TumorTree object from JSON data."""
        # Use DictImporter to recreate the Subclone tree
        root_node = importer.import_(tree_data["tree"])
        tumor_tree = TumorTree(
            patient_id=tree_data["patient_id"],
            tree_id=tree_data["tree_id"],
            root=root_node,  # pyright: ignore
            weight=tree_data["weight"],
            sampling_time=tree_data["sampling_time"],
            tumor_size=tree_data["tumor_size"],
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
        C_seq=data["C_seq"],
        C_sampling=data["C_sampling"],
        t_max=data["t_max"],
        mutation_labels=data["mutation_labels"],
        tree_labels=data["tree_labels"],
        patient_labels=data["patient_labels"],
        lifetime_risk=data["lifetime_risk"],
    )

    return cohort


def save_vectorized_trees_npz(vectorized_trees: VectorizedTrees, path: str):
    """Save VectorizedTrees NamedTuple to a compressed .npz file."""
    np.savez_compressed(
        path,
        cell_number=vectorized_trees.cell_number,
        seq_cell_number=vectorized_trees.seq_cell_number,
        observed=vectorized_trees.observed,
        sampling_time=vectorized_trees.sampling_time,
        weight=vectorized_trees.weight,
        tumor_size=vectorized_trees.tumor_size,
        node_id=vectorized_trees.node_id,
        parent_id=vectorized_trees.parent_id,
        alpha=vectorized_trees.alpha,
        nu=vectorized_trees.nu,
        lam=vectorized_trees.lam,
        rho=vectorized_trees.rho,
        phi=vectorized_trees.phi,
        delta=vectorized_trees.delta,
        r=vectorized_trees.r,
        gamma=vectorized_trees.gamma,
        genotypes=vectorized_trees.genotypes,
        N_trees=vectorized_trees.N_trees,
        N_patients=vectorized_trees.N_patients,
        n_nodes=vectorized_trees.n_nodes,
        beta=vectorized_trees.beta,
        C_s=vectorized_trees.C_s,
        C_0=vectorized_trees.C_0,
        t_max=vectorized_trees.t_max,
    )


def load_vectorized_trees_npz(path: str) -> VectorizedTrees:
    """Load a VectorizedTrees NamedTuple from an .npz file."""
    data = np.load(path)
    return VectorizedTrees(
        cell_number=data["cell_number"],
        seq_cell_number=data["seq_cell_number"],
        observed=data["observed"],
        sampling_time=data["sampling_time"],
        weight=data["weight"],
        tumor_size=data["tumor_size"],
        node_id=data["node_id"],
        parent_id=data["parent_id"],
        alpha=data["alpha"],
        nu=data["nu"],
        lam=data["lam"],
        rho=data["rho"],
        phi=data["phi"],
        delta=data["delta"],
        r=data["r"],
        gamma=data["gamma"],
        genotypes=data["genotypes"],
        N_trees=data["N_trees"],
        N_patients=data["N_patients"],
        n_nodes=data["n_nodes"],
        beta=data["beta"],
        C_s=data["C_s"],
        C_0=data["C_0"],
        t_max=data["t_max"],
    )
