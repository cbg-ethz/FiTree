import numpy as np

from fitree._trees import Subclone


def test_genotype():
    root = Subclone(node_id=0, mutation_ids=[0, 1], cell_number=100)
    v1 = Subclone(node_id=1, mutation_ids=[2], cell_number=50, parent=root)
    v2 = Subclone(node_id=2, mutation_ids=[3], cell_number=50, parent=root)
    v3 = Subclone(node_id=3, mutation_ids=[2, 5], cell_number=25, parent=v1)
    v4 = Subclone(node_id=4, mutation_ids=[4], cell_number=25, parent=v2)
    v5 = Subclone(node_id=5, mutation_ids=[5], cell_number=25, parent=v2)

    assert root.genotype == {0, 1}
    assert v1.genotype == {0, 1, 2}
    assert v2.genotype == {0, 1, 3}
    assert v3.genotype == {0, 1, 2, 5}
    assert v4.genotype == {0, 1, 3, 4}
    assert v5.genotype == {0, 1, 3, 5}

    v2.update_mutation_ids([3, 4])
    assert v2.genotype == {0, 1, 3, 4}
    assert v4.genotype == {0, 1, 3, 4}
    assert v5.genotype == {0, 1, 3, 4, 5}


def test_growth_params():
    root = Subclone(node_id=0, mutation_ids=[0, 1], cell_number=100)
    v1 = Subclone(node_id=1, mutation_ids=[2], cell_number=50, parent=root)
    v2 = Subclone(node_id=2, mutation_ids=[3], cell_number=50, parent=root)
    v3 = Subclone(node_id=3, mutation_ids=[2, 4, 5], cell_number=25, parent=v1)

    mu_vec = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    F = np.ones((6, 6))
    F[0, 2] = 1.2
    F[2, 4] = 0.5
    F[4, 5] = 0.3
    common_beta = 1

    root.get_growth_params(mu_vec, F, common_beta)
    assert root.nu == 0
    assert root.alpha == common_beta
    assert root.beta == common_beta
    assert root.lam == 0
    assert root.delta == 0
    assert root.r == 1
    assert root.rho == 0
    assert root.phi == root.alpha
    assert root.gamma == 0

    v1.get_growth_params(mu_vec, F, common_beta)
    assert v1.nu == 0.3
    assert v1.alpha == 1.2
    assert v1.beta == common_beta
    assert v1.lam == v1.alpha - v1.beta
    assert v1.delta == v1.lam
    assert v1.r == 1
    assert v1.rho == v1.nu / v1.alpha
    assert v1.phi == v1.alpha / v1.lam
    assert v1.gamma == 0

    v2.get_growth_params(mu_vec, F, common_beta)
    assert v2.nu == 0.4
    assert v2.alpha == 1
    assert v2.beta == common_beta
    assert v2.lam == v2.alpha - v2.beta
    assert v2.delta == root.lam
    assert v2.r == 2
    assert v2.rho == v2.nu / v2.alpha
    assert v2.phi == v2.alpha
    assert v2.gamma == 0

    v3.get_growth_params(mu_vec, F, common_beta)
    assert v3.nu == 0.5 * 0.6
    assert v3.alpha == 1.2 * 0.5 * 0.3
    assert v3.beta == common_beta
    assert v3.lam == v3.alpha - v3.beta
    assert v3.delta == v1.lam
    assert v3.r == 1
    assert v3.rho == v3.nu / v3.alpha
    assert v3.phi == -v3.beta / v3.lam
    assert v3.gamma == v1.delta / v3.delta
