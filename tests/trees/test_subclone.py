import numpy as np

from fitree._trees import Subclone


def test_genotype():
    root = Subclone(node_id=0, mutation_ids=[0, 1], seq_cell_number=100)
    v1 = Subclone(node_id=1, mutation_ids=[2], seq_cell_number=50, parent=root)
    v2 = Subclone(node_id=2, mutation_ids=[3], seq_cell_number=50, parent=root)
    v3 = Subclone(node_id=3, mutation_ids=[2, 5], seq_cell_number=25, parent=v1)
    v4 = Subclone(node_id=4, mutation_ids=[4], seq_cell_number=25, parent=v2)
    v5 = Subclone(node_id=5, mutation_ids=[5], seq_cell_number=25, parent=v2)

    assert root.genotype == [0, 1]
    assert v1.genotype == [0, 1, 2]
    assert v2.genotype == [0, 1, 3]
    assert v3.genotype == [0, 1, 2, 5]
    assert v4.genotype == [0, 1, 3, 4]
    assert v5.genotype == [0, 1, 3, 5]

    v2.update_mutation_ids([3, 4])
    assert v2.genotype == [0, 1, 3, 4]
    assert v4.genotype == [0, 1, 3, 4]
    assert v5.genotype == [0, 1, 3, 4, 5]


def test_growth_params():
    root = Subclone(node_id=0, mutation_ids=[0, 1], seq_cell_number=100)
    v1 = Subclone(node_id=1, mutation_ids=[2], seq_cell_number=50, parent=root)
    v2 = Subclone(node_id=2, mutation_ids=[3], seq_cell_number=50, parent=root)
    v3 = Subclone(node_id=3, mutation_ids=[2, 4, 5], seq_cell_number=25, parent=v1)

    mu_vec = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    F_mat = np.ones((6, 6))
    F_mat[2, 2] = 1.1
    F_mat[4, 4] = 0.9
    F_mat[0, 2] = 1.2
    F_mat[2, 4] = 0.5
    F_mat[4, 5] = 0.3
    F_mat = np.log(F_mat)
    common_beta = 1

    gpar = root.get_growth_params(mu_vec, F_mat, common_beta, True)
    assert gpar["nu"] == 0
    assert gpar["alpha"] == 1
    assert gpar["beta"] == 1
    assert gpar["lam"] == 0
    assert gpar["delta"] == 0
    assert gpar["r"] == 1
    assert gpar["rho"] == 0
    assert gpar["phi"] == 1
    assert gpar["gamma"] == 0

    gpar = v1.get_growth_params(mu_vec, F_mat, common_beta, True)
    assert gpar["nu"] == 0.3
    assert gpar["alpha"] == 1.2 * 1.1
    assert gpar["beta"] == 1
    assert gpar["lam"] == 1.2 * 1.1 - 1
    assert gpar["delta"] == gpar["lam"]
    assert gpar["r"] == 1
    assert gpar["rho"] == 0.3 / (1.2 * 1.1)
    assert gpar["phi"] == (1.2 * 1.1) / gpar["lam"]
    assert gpar["gamma"] == 0

    gpar = v2.get_growth_params(mu_vec, F_mat, common_beta, True)
    assert gpar["nu"] == 0.4
    assert gpar["alpha"] == 1
    assert gpar["beta"] == 1
    assert gpar["lam"] == 0
    assert gpar["delta"] == 0
    assert gpar["r"] == 2
    assert gpar["rho"] == 0.4
    assert gpar["phi"] == 1
    assert gpar["gamma"] == 0

    gpar = v3.get_growth_params(mu_vec, F_mat, common_beta, True)
    assert gpar["nu"] == 0.5 * 0.6
    assert np.isclose(gpar["alpha"], 1.2 * 0.5 * 0.3 * 1.1)
    assert gpar["beta"] == common_beta
    assert np.isclose(gpar["lam"], 1.2 * 0.5 * 0.3 * 1.1 - 1)
    assert gpar["delta"] == 1.2 * 1.1 - 1
    assert gpar["r"] == 1
    assert np.isclose(gpar["rho"], (0.5 * 0.6) / (1.2 * 0.5 * 0.3 * 1.1))
    assert np.isclose(gpar["phi"], -1.0 / (1.2 * 0.5 * 0.3 * 1.1 - 1))
    assert gpar["gamma"] == 1.0


def test_mrca():
	root = Subclone(node_id=0, mutation_ids=[0, 1], seq_cell_number=100)
	v1 = Subclone(node_id=1, mutation_ids=[2], seq_cell_number=50, parent=root)
	v2 = Subclone(node_id=2, mutation_ids=[3], seq_cell_number=50, parent=root)
	v3 = Subclone(node_id=3, mutation_ids=[2, 5], seq_cell_number=25, parent=v1)
	v4 = Subclone(node_id=4, mutation_ids=[4], seq_cell_number=25, parent=v2)
	v5 = Subclone(node_id=5, mutation_ids=[5], seq_cell_number=25, parent=v2)

	assert root.get_mrca() == root
	assert v1.get_mrca() == v1
	assert v2.get_mrca() == v2
	assert v3.get_mrca() == v1
	assert v4.get_mrca() == v2
	assert v5.get_mrca() == v2