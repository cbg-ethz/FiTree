from mpmath import *
mp.dps = 100
mp.pretty = True

from fitree._trees import Subclone
from fitree._mtbp import _mcdf_sampling, _q_tilde, _g_tilde, _h

tol = 1e-10
C_0 = 1
C_sampling = 1e9
beta = 8.8
t = 20

def test_q_tilde():

	root = Subclone(node_id=0, mutation_ids=[0, 1], cell_number=100)
	v1 = Subclone(node_id=1, mutation_ids=[2], cell_number=50, parent=root)
	v2 = Subclone(node_id=2, mutation_ids=[3], cell_number=50, parent=v1)

	root.get_growth_params(None, None, beta)

	v1.growth_params = {
		"nu": 3.1 / C_0,
		"alpha": 9.1,
		"beta": beta,
		"lambda": 9.1 - beta,
		"delta": 9.1 - beta,
		"r": 1,
		"rho": 3.1 / C_0 / 9.1,
		"phi": 9.1 / (9.1 - beta),
		"gamma": 0
	}

	delta_1 = v1.growth_params["delta"]
	r_1 = v1.growth_params["r"]

	v2.growth_params = {
		"nu": 1e-5,
		"alpha": 9.1,
		"beta": beta,
		"lambda": 9.1 - beta,
		"delta": 9.1 - beta,
		"r": 2,
		"rho": 1e-5 / 9.1,
		"phi": 9.1 / (9.1 - beta),
		"gamma": 1
	}

	delta_2 = v2.growth_params["delta"]
	r_2 = v2.growth_params["r"]

	assert almosteq(
		_q_tilde(v1, t, C_sampling), 
		mpf((exp(delta_1 * t) - 1) / (delta_1 * C_sampling)),
		tol
	)

	assert almosteq(
		_q_tilde(v2, 20, C_sampling),
		mpf(((t * exp(delta_2 * t) - (exp(delta_2 * t) - 1) / delta_2) / delta_2) / C_sampling),
		tol
	)

	v2.growth_params = {
        "nu": 1e-5,
        "alpha": 9.2,
        "beta": beta,
        "lambda": 9.2 - beta,
        "delta": 9.2 - beta,
        "r": 1,
        "rho": 1e-5 / 9.2,
        "phi": 9.2 / (9.2 - beta),
        "gamma": 0,
    }

	delta_2 = v2.growth_params["delta"]
	r_2 = v2.growth_params["r"]

	assert almosteq(
		_q_tilde(v2, t, C_sampling),
		mpf((exp(delta_2 * t) - 1) / (delta_2 * C_sampling)),
		tol
	)


def test_g_tilde():

	root = Subclone(node_id=0, mutation_ids=[0, 1], cell_number=100)
	v1 = Subclone(node_id=1, mutation_ids=[2], cell_number=50, parent=root)
	v2 = Subclone(node_id=2, mutation_ids=[3], cell_number=50, parent=v1)

	root.get_growth_params(None, None, beta)

	v1.growth_params = {
		"nu": 3.1 / C_0,
		"alpha": 9.1,
		"beta": beta,
		"lambda": 9.1 - beta,
		"delta": 9.1 - beta,
		"r": 1,
		"rho": 3.1 / C_0 / 9.1,
		"phi": 9.1 / (9.1 - beta),
		"gamma": 0
	}


	v2.growth_params = {
        "nu": 1e-5,
        "alpha": 9.2,
        "beta": beta,
        "lambda": 9.2 - beta,
        "delta": 9.2 - beta,
        "r": 1,
        "rho": 1e-5 / 9.2,
        "phi": 9.2 / (9.2 - beta),
        "gamma": 0,
    }

	assert almosteq(
		_g_tilde(v2, t, C_sampling),
		_q_tilde(v2, t, C_sampling),
		tol
	)

	assert almosteq(
		_g_tilde(v1, t, C_sampling),
		_q_tilde(v1, t, C_sampling) + \
			_h(_q_tilde(v2, t, C_sampling), v2.growth_params, v1.growth_params),
		tol
	)


def test_mcdf_sampling_1():
	root = Subclone(node_id=0, mutation_ids=[0, 1], cell_number=100)
	v1 = Subclone(node_id=1, mutation_ids=[2], cell_number=50, parent=root)
	v2 = Subclone(node_id=2, mutation_ids=[3], cell_number=50, parent=v1)
	v3 = Subclone(node_id=3, mutation_ids=[4], cell_number=50, parent=v2)

	root.get_growth_params(None, None, beta)

	nu_1 = 0.7
	alpha_1 = 1
	beta_1 = 0.3
	nu_2 = 0.01
	alpha_2 = 1
	beta_2 = 1.5
	nu_3 = 0.01
	alpha_3 = 1.4
	beta_3 = 0.3
	mean_size = 0.01

	lambda_1 = alpha_1 - beta_1
	lambda_2 = alpha_2 - beta_2
	lambda_3 = alpha_3 - beta_3

	delta_1 = lambda_1
	delta_2 = max([lambda_1, lambda_2])
	delta_3 = max([lambda_1, lambda_2, lambda_3])

	r_1 = 1
	r_2 = sum([1 if x == delta_2 else 0 for x in [lambda_1, lambda_2]])
	r_3 = sum([1 if x == delta_3 else 0 for x in [lambda_1, lambda_2, lambda_3]])

	rho_1 = nu_1 / alpha_1
	rho_2 = nu_2 / alpha_2
	rho_3 = nu_3 / alpha_3

	phi_1 = alpha_1 / lambda_1
	phi_2 = -beta_2 / lambda_2
	phi_3 = alpha_3 / lambda_3

	gamma_1 = 0
	gamma_2 = delta_1 / delta_2
	gamma_3 = delta_2 / delta_3

	v1.growth_params = {
		"nu": nu_1,
		"alpha": alpha_1,
		"beta": beta_1,
		"lambda": lambda_1,
		"delta": delta_1,
		"r": r_1,
		"rho": rho_1,
		"phi": phi_1,
		"gamma": gamma_1
	}

	v2.growth_params = {
		"nu": nu_2,
		"alpha": alpha_2,
		"beta": beta_2,
		"lambda": lambda_2,
		"delta": delta_2,
		"r": r_2,
		"rho": rho_2,
		"phi": phi_2,
		"gamma": gamma_2
	}

	v3.growth_params = {
		"nu": nu_3,
		"alpha": alpha_3,
		"beta": beta_3,
		"lambda": lambda_3,
		"delta": delta_3,
		"r": r_3,
		"rho": rho_3,
		"phi": phi_3,
		"gamma": gamma_3
	}

	def F_sampling(t):
		q_tilde_3 = (exp(delta_3 * t) - 1) / (delta_3 * C_sampling)
		g_tilde_3 = q_tilde_3
		h_3 = rho_3 * power(phi_3, gamma_3) * pi / \
			sin(pi * gamma_3) * power(g_tilde_3, gamma_3)
		
		q_tilde_2 = (exp(delta_2 * t) - 1) / (delta_2 * C_sampling)
		g_tilde_2 = q_tilde_2 + h_3
		h_2 = nu_2 * g_tilde_2 / (delta_1 - lambda_2)

		q_tilde_1 = (exp(delta_1 * t) - 1) / (delta_1 * C_sampling)
		g_tilde_1 = q_tilde_1 + h_2

		return 1 - power(1 + phi_1 * g_tilde_1, -rho_1)

	assert almosteq(
		_mcdf_sampling(root, t, C_sampling, C_0), 
		F_sampling(t),
		tol
	)

	assert almosteq(
		_mcdf_sampling(root, t, C_sampling, C_0), 
		1 - power(1 + phi_1 * _g_tilde(v1, t, C_sampling), -rho_1 * C_0),
		tol
	)

	assert almosteq(
		_mcdf_sampling(root, t, C_sampling, C_0), 
		1 - power(
			1 + phi_1 * (
				_q_tilde(v1, t, C_sampling) + \
				_h(_g_tilde(v2, t, C_sampling), v2.growth_params, v1.growth_params)
			),
			-rho_1 * C_0
		),
		tol
	)

def test_mcdf_sampling_2():
	root = Subclone(node_id=0, mutation_ids=[0, 1], cell_number=100)
	v1 = Subclone(node_id=1, mutation_ids=[2], cell_number=50, parent=root)
	v2 = Subclone(node_id=2, mutation_ids=[3], cell_number=50, parent=v1)
	v3 = Subclone(node_id=3, mutation_ids=[4], cell_number=50, parent=v2)

	root.get_growth_params(None, None, beta)

	nu_1 = 0.7
	alpha_1 = 0.3
	beta_1 = 0.3
	nu_2 = 0.01
	alpha_2 = 1.4
	beta_2 = 0.3
	nu_3 = 0.01
	alpha_3 = 1.5
	beta_3 = 0.3

	lambda_1 = alpha_1 - beta_1
	lambda_2 = alpha_2 - beta_2
	lambda_3 = alpha_3 - beta_3

	delta_1 = lambda_1
	delta_2 = max([lambda_1, lambda_2])
	delta_3 = max([lambda_1, lambda_2, lambda_3])

	r_1 = 2
	r_2 = sum([1 if x == delta_2 else 0 for x in [lambda_1, lambda_2]])
	r_3 = sum([1 if x == delta_3 else 0 for x in [lambda_1, lambda_2, lambda_3]])

	rho_1 = nu_1 / alpha_1
	rho_2 = nu_2 / alpha_2
	rho_3 = nu_3 / alpha_3

	phi_1 = alpha_1
	phi_2 = alpha_2 / lambda_2
	phi_3 = alpha_3 / lambda_3

	gamma_1 = 0
	gamma_2 = delta_1 / delta_2
	gamma_3 = delta_2 / delta_3

	v1.growth_params = {
		"nu": nu_1,
		"alpha": alpha_1,
		"beta": beta_1,
		"lambda": lambda_1,
		"delta": delta_1,
		"r": r_1,
		"rho": rho_1,
		"phi": phi_1,
		"gamma": gamma_1
	}

	v2.growth_params = {
		"nu": nu_2,
		"alpha": alpha_2,
		"beta": beta_2,
		"lambda": lambda_2,
		"delta": delta_2,
		"r": r_2,
		"rho": rho_2,
		"phi": phi_2,
		"gamma": gamma_2
	}

	v3.growth_params = {
		"nu": nu_3,
		"alpha": alpha_3,
		"beta": beta_3,
		"lambda": lambda_3,
		"delta": delta_3,
		"r": r_3,
		"rho": rho_3,
		"phi": phi_3,
		"gamma": gamma_3
	}

	def F_sampling(t):	

		q_tilde_3 = (exp(delta_3 * t) - 1) / delta_3 / C_sampling
		g_tilde_3 = q_tilde_3

		h_3 = rho_3 * power(phi_3, gamma_3) * pi / \
			sin(pi * gamma_3) * power(g_tilde_3, gamma_3)
		
		q_tilde_2 = (exp(delta_2 * t) - 1) / (delta_2 * C_sampling)
		g_tilde_2 = q_tilde_2 + h_3
		h_2 = -rho_2 / lambda_2 * polylog(r_1, -phi_2 * g_tilde_2)

		q_tilde_1 = power(t, 2) / 2 / C_sampling
		g_tilde_1 = q_tilde_1 + h_2

		return 1 - power(1 + phi_1 * g_tilde_1, -rho_1)


	assert almosteq(
		_mcdf_sampling(root, t, C_sampling, C_0), 
		F_sampling(t),
		tol
	)

def test_mcdf_sampling_3():
	root = Subclone(node_id=0, mutation_ids=[0, 1], cell_number=100)
	v1 = Subclone(node_id=1, mutation_ids=[2], cell_number=50, parent=root)
	v2 = Subclone(node_id=2, mutation_ids=[3], cell_number=50, parent=v1)
	v3 = Subclone(node_id=3, mutation_ids=[4], cell_number=50, parent=v2)

	root.get_growth_params(None, None, beta)

	nu_1 = 0.7
	alpha_1 = 0.7
	beta_1 = 1.2
	nu_2 = 0.01
	alpha_2 = 1.3
	beta_2 = 0.3
	nu_3 = 0.01
	alpha_3 = 1.5
	beta_3 = 0.3

	lambda_1 = alpha_1 - beta_1
	lambda_2 = alpha_2 - beta_2
	lambda_3 = alpha_3 - beta_3

	delta_1 = 0
	delta_2 = max([delta_1, lambda_2])
	delta_3 = max([delta_2, lambda_3])

	r_1 = 1
	r_2 = sum([1 if x == delta_2 else 0 for x in [lambda_1, lambda_2]])
	r_3 = sum([1 if x == delta_3 else 0 for x in [lambda_1, lambda_2, lambda_3]])

	rho_1 = nu_1 / alpha_1
	rho_2 = nu_2 / alpha_2
	rho_3 = nu_3 / alpha_3

	phi_1 = -beta_1 / lambda_1
	phi_2 = alpha_2 / lambda_2
	phi_3 = alpha_3 / lambda_3

	gamma_1 = 0
	gamma_2 = delta_1 / delta_2
	gamma_3 = delta_2 / delta_3

	v1.growth_params = {
		"nu": nu_1,
		"alpha": alpha_1,
		"beta": beta_1,
		"lambda": lambda_1,
		"delta": delta_1,
		"r": r_1,
		"rho": rho_1,
		"phi": phi_1,
		"gamma": gamma_1
	}

	v2.growth_params = {
		"nu": nu_2,
		"alpha": alpha_2,
		"beta": beta_2,
		"lambda": lambda_2,
		"delta": delta_2,
		"r": r_2,
		"rho": rho_2,
		"phi": phi_2,
		"gamma": gamma_2
	}

	v3.growth_params = {
		"nu": nu_3,
		"alpha": alpha_3,
		"beta": beta_3,
		"lambda": lambda_3,
		"delta": delta_3,
		"r": r_3,
		"rho": rho_3,
		"phi": phi_3,
		"gamma": gamma_3
	}

	def F_sampling(t):	

		epsilon = 0.01

		t += 1 / lambda_1

		q_tilde_3 = (exp(delta_3 * t) - 1) / delta_3 / C_sampling
		g_tilde_3 = q_tilde_3

		h_3 = rho_3 * power(phi_3, gamma_3) * pi / \
			sin(pi * gamma_3) * power(g_tilde_3, gamma_3)
		
		q_tilde_2 = (exp(delta_2 * t) - 1) / (delta_2 * C_sampling)
		g_tilde_2 = q_tilde_2 + h_3
		h_2 = rho_2 * log(1 + phi_2 * g_tilde_2)

		q_tilde_1 = t / C_sampling
		g_tilde_1 = (q_tilde_1 + h_2) / t

		return 1 - power(
			phi_1 + (1 - phi_1) * exp(-g_tilde_1 * epsilon), 
			-rho_1 * t / epsilon
		)

	assert almosteq(
		_mcdf_sampling(root, t, C_sampling, C_0), 
		F_sampling(t),
		tol
	)