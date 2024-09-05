import numpy as np
from anytree import PreOrderIter
from anytree.search import findall_by_attr

from fitree._trees import Subclone, TumorTree, TumorTreeCohort
from fitree._inference import wrap_trees, update_params

root = Subclone(node_id=0, mutation_ids=[], cell_number=100)
v1 = Subclone(node_id=1, mutation_ids=[0], cell_number=50, parent=root)
v2 = Subclone(node_id=2, mutation_ids=[2], cell_number=50, parent=root)
v3 = Subclone(node_id=3, mutation_ids=[1, 4], cell_number=25, parent=v1)
v4 = Subclone(node_id=4, mutation_ids=[3], cell_number=25, parent=v2)
v5 = Subclone(node_id=5, mutation_ids=[2], cell_number=25, parent=v2)

tree1 = TumorTree(patient_id=1, tree_id=1, root=root, weight=1.0, sampling_time=5.0)

root2 = Subclone(node_id=0, mutation_ids=[], cell_number=100)
v21 = Subclone(node_id=1, mutation_ids=[0], cell_number=50, parent=root2)
v22 = Subclone(node_id=2, mutation_ids=[2], cell_number=50, parent=root2)
v23 = Subclone(node_id=3, mutation_ids=[1, 4], cell_number=25, parent=v21)
v24 = Subclone(node_id=4, mutation_ids=[2], cell_number=25, parent=v21)
v25 = Subclone(node_id=5, mutation_ids=[3], cell_number=25, parent=v22)
v26 = Subclone(node_id=6, mutation_ids=[6], cell_number=25, parent=root2)
v27 = Subclone(node_id=7, mutation_ids=[4, 5], cell_number=25, parent=v22)

tree2 = TumorTree(patient_id=2, tree_id=2, root=root2, weight=1.0, sampling_time=10.0)

cohort = TumorTreeCohort(
	name="test_cohort",
	trees=[tree1, tree2],
	n_mutations=7,
	N_trees=2,
	N_patients=2,
	mu_vec=np.ones(7) * 1e-5,
	common_beta=1.0,
	C_0=1e5,
	C_min=1e3,
	C_sampling=1e8,
	t_max=100.0,
	mutation_labels={i: "M" + str(i) for i in range(7)},
	tree_labels={i: "T" + str(i) for i in range(2)},
	patient_labels={i: "P" + str(i) for i in range(2)},
)

vec_trees, union_tree = wrap_trees(cohort)

union_node_paths = [node.node_path for node in PreOrderIter(union_tree.root)]

def test_wrap_trees():

	assert union_tree.root.size == 9
	assert vec_trees.cell_number.shape == (2, 8)
	assert vec_trees.sampling_time[0] == 5.0
	assert vec_trees.sampling_time[1] == 10.0
	assert vec_trees.weight[0] == 1.0
	assert vec_trees.weight[1] == 1.0
	assert vec_trees.beta == 1.0
	assert vec_trees.C_s == 1e8
	assert vec_trees.C_0 == 1e5
	assert vec_trees.n_nodes == 8
	assert vec_trees.N_trees == 2

	for node in PreOrderIter(tree1.root):

		assert node.node_path in union_node_paths

	for node in PreOrderIter(tree2.root):

		assert node.node_path in union_node_paths
		
	for i in range(vec_trees.n_nodes):

		node_id = vec_trees.node_id[i] + 1

		union_node = findall_by_attr(union_tree.root, value=node_id, name="node_id")

		assert len(union_node) == 1

		union_node = union_node[0]

		parent_id = vec_trees.parent_id[i] + 1

		if parent_id != 0:

			parent_node = findall_by_attr(union_tree.root, value=parent_id, name="node_id")

			assert len(parent_node) == 1

			parent_node = parent_node[0]

			assert union_node.parent.node_id == parent_node.node_id
