import numpy as np
from anytree import PreOrderIter
from anytree.search import findall_by_attr

from fitree._trees import Subclone, TumorTree, TumorTreeCohort
from fitree._trees._wrapper import wrap_trees, get_augmented_tree


def test_get_augmented_tree():

	root = Subclone(node_id=0, mutation_ids=[], seq_cell_number=100)
	v1 = Subclone(node_id=1, mutation_ids=[0], seq_cell_number=50, parent=root)
	v2 = Subclone(node_id=2, mutation_ids=[2], seq_cell_number=50, parent=root)

	F_mat = np.zeros((3, 3))
	mu_vec = np.ones(3) * 1e-5
	common_beta = 1.0

	for node in PreOrderIter(root):
		node.get_growth_params(mu_vec=mu_vec, F_mat=F_mat, common_beta=common_beta)

	observed_mutations = {0, 2}
	augmented_tree = get_augmented_tree(
		tree=root,
		mutation_set=observed_mutations,
		mu_vec=mu_vec,
		F_mat=F_mat,
		common_beta=common_beta,
		rule="parallel",
	)

	assert augmented_tree.size == 5

	root = Subclone(node_id=0, mutation_ids=[], seq_cell_number=100)
	v1 = Subclone(node_id=1, mutation_ids=[0], seq_cell_number=50, parent=root)
	v2 = Subclone(node_id=2, mutation_ids=[2], seq_cell_number=50, parent=v1)

	F_mat = np.zeros((3, 3))
	mu_vec = np.ones(3) * 1e-5
	common_beta = 1.0

	for node in PreOrderIter(root):
		node.get_growth_params(mu_vec=mu_vec, F_mat=F_mat, common_beta=common_beta)

	observed_mutations = {0, 2}
	augmented_tree = get_augmented_tree(
		tree=root,
		mutation_set=observed_mutations,
		mu_vec=mu_vec,
		F_mat=F_mat,
		common_beta=common_beta,
		rule="parallel",
	)

	assert augmented_tree.size == 4




def test_wrap_trees():

	root = Subclone(node_id=0, mutation_ids=[], seq_cell_number=100)
	v1 = Subclone(node_id=1, mutation_ids=[0], seq_cell_number=50, parent=root)
	v2 = Subclone(node_id=2, mutation_ids=[2], seq_cell_number=50, parent=root)
	v3 = Subclone(node_id=3, mutation_ids=[1, 4], seq_cell_number=25, parent=v1)
	v4 = Subclone(node_id=4, mutation_ids=[3], seq_cell_number=25, parent=v2)
	v5 = Subclone(node_id=5, mutation_ids=[2], seq_cell_number=25, parent=v2)

	tree1 = TumorTree(patient_id=1, tree_id=1, root=root, weight=1.0, sampling_time=5.0, tumor_size=275)

	root2 = Subclone(node_id=0, mutation_ids=[], seq_cell_number=100)
	v21 = Subclone(node_id=1, mutation_ids=[0], seq_cell_number=50, parent=root2)
	v22 = Subclone(node_id=2, mutation_ids=[2], seq_cell_number=50, parent=root2)
	v23 = Subclone(node_id=3, mutation_ids=[1, 4], seq_cell_number=25, parent=v21)
	v24 = Subclone(node_id=4, mutation_ids=[2], seq_cell_number=25, parent=v21)
	v25 = Subclone(node_id=5, mutation_ids=[3], seq_cell_number=25, parent=v22)
	v26 = Subclone(node_id=6, mutation_ids=[6], seq_cell_number=25, parent=root2)
	v27 = Subclone(node_id=7, mutation_ids=[4, 5], seq_cell_number=25, parent=v22)

	tree2 = TumorTree(patient_id=2, tree_id=2, root=root2, weight=1.0, sampling_time=10.0, tumor_size=325)

	cohort = TumorTreeCohort(
		name="test_cohort",
		trees=[tree1, tree2],
		n_mutations=7,
		N_trees=2,
		N_patients=2,
		mu_vec=np.ones(7) * 1e-5,
		common_beta=1.0,
		C_0=1e5,
		C_seq=1e3,
		C_sampling=1e8,
		t_max=100.0,
		mutation_labels={i: "M" + str(i) for i in range(7)},
		tree_labels={i: "T" + str(i) for i in range(2)},
		patient_labels={i: "P" + str(i) for i in range(2)},
	)

	F_mat = np.zeros((cohort.n_mutations, cohort.n_mutations))
	mu_vec = cohort.mu_vec
	common_beta = cohort.common_beta

	for tree in cohort.trees:
		for node in PreOrderIter(tree.root):
			node.get_growth_params(mu_vec=mu_vec, F_mat=F_mat, common_beta=common_beta)

	vec_trees, union_tree = wrap_trees(cohort)

	union_node_paths = [node.node_path for node in PreOrderIter(union_tree.root)]

	assert vec_trees.sampling_time[0] == 5.0
	assert vec_trees.sampling_time[1] == 10.0
	assert vec_trees.weight[0] == 1.0
	assert vec_trees.weight[1] == 1.0
	assert vec_trees.beta == 1.0
	assert vec_trees.C_s == 1e8
	assert vec_trees.C_0 == 1e5
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
