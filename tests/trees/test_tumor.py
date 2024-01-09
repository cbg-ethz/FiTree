import pytest

from fitree._trees import Subclone, TumorTree

def test_tumor_1():

	root = Subclone(node_id=0, mutation_ids=[0, 1], cell_number=100)
	v1 = Subclone(node_id=1, mutation_ids=[2], cell_number=50, parent=root)
	v2 = Subclone(node_id=2, mutation_ids=[3], cell_number=50, parent=v1)

	tree = TumorTree(patient_id=0, tree_id=0, root=root)

	assert tree.patient_id == 0
	assert tree.tree_id == 0
	assert tree.root == root
	assert tree.weight == 1.0
	assert tree.sampling_time is None

	assert tree.get_mutation_ids() == {0, 1, 2, 3}

def test_tumor_2():

	root = Subclone(node_id=0, mutation_ids=[0, 1], cell_number=100)
	v1 = Subclone(node_id=1, mutation_ids=[2], cell_number=50, parent=root)
	v2 = Subclone(node_id=2, mutation_ids=[3], cell_number=50, parent=v1)

	tree = TumorTree(patient_id=0, tree_id=0, root=root, weight=0.5, sampling_time=0.5)

	assert tree.patient_id == 0
	assert tree.tree_id == 0
	assert tree.root == root
	assert tree.weight == 0.5
	assert tree.sampling_time == 0.5

	assert tree.get_mutation_ids() == {0, 1, 2, 3}

def test_tumor_3():

	root = Subclone(node_id=0, mutation_ids=[0, 1], cell_number=100)
	v1 = Subclone(node_id=1, mutation_ids=[2], cell_number=50, parent=root)
	v2 = Subclone(node_id=2, mutation_ids=[3], cell_number=50, parent=v1)

	with pytest.raises(ValueError):
		tree = TumorTree(patient_id=0, tree_id=0, root=v1)
		tree = TumorTree(patient_id=0, tree_id=0, root=v2)

