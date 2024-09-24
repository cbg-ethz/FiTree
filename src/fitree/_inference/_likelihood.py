import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import jax.scipy.special as jss

from ._utils import ETA_VEC, BETA_VEC, polylog, integrate
from ._wrapper import VectorizedTrees


@jax.jit
def _pt(alpha: jnp.ndarray, beta: jnp.ndarray, lam: jnp.ndarray, t: jnp.ndarray):
    """This function computes the success probability
    of the negative Binomial distribution for the one-type
    branching process. (See Lemma 1 in the supplement)
    """

    return jnp.where(
        lam == 0.0, 1.0 / (1.0 + alpha * t), lam / (alpha * jnp.exp(lam * t) - beta)
    )


@jax.jit
def _case1_var1(t, r1, delta1, alpha2, beta2, lam2):
    # Variance function for delta1 > lam2 == 0.0

    return (
        jnp.power(t, -2.0 * r1 + 2.0)
        * jnp.exp(-2.0 * delta1 * t)
        * (
            (1.0 + 2.0 * alpha2 * t) * integrate(t, r1 - 1.0, delta1)
            - 2.0 * alpha2 * integrate(t, r1, delta1)
        )
    )


@jax.jit
def _case1_var2(t, r1, delta1, alpha2, beta2, lam2):
    # Variance function for delta1 > lam2 != 0.0

    return jnp.power(t, -2.0 * r1 + 2.0) * (
        2.0
        * alpha2
        / lam2
        * jnp.exp(-2.0 * (delta1 - lam2) * t)
        * integrate(t, r1 - 1.0, delta1 - 2.0 * lam2)
        - (alpha2 + beta2)
        / lam2
        * jnp.exp(-(2.0 * delta1 - lam2) * t)
        * integrate(t, r1 - 1.0, delta1 - lam2)
    )


@jax.jit
def _lp2_case1(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    t: jnp.ndarray,
    par1: dict,
    par2: dict,
    eps: float = 1e-16,
    pdf: bool = True,
):
    """This function computes the log-likelihood of a subclone
    given its parent for the case when it is less fit than its parent.
    The means are given by Nicholson et al. (2023), and the variance is
    estimated through simulations. (See Theorem 2 in the supplement)

    Args:
        x1 : jnp.ndarray
            The number of cells in the parent subclone.
        x2 : jnp.ndarray
            The number of cells in the subclone.
        t : jnp.ndarray
            The sampling time.
        par1 : dict
            The growth parameters of the parent subclone.
        par2 : dict
            The growth parameters of the subclone.
        eps : float, optional
            The machine epsilon. Defaults to 1e-16.
    """

    delta1 = par1["delta"]
    r1 = par1["r"]
    delta2 = par2["delta"]
    lam2 = par2["lam"]
    r2 = par2["r"]
    nu2 = par2["nu"]
    alpha2 = par2["alpha"]
    beta2 = par2["beta"]

    x1_tilde = (x1 + 1.0) * jnp.exp(-delta1 * t) * jnp.power(t, 1.0 - r1)
    x2_tilde = (x2 + 1.0) * jnp.exp(-delta2 * t) * jnp.power(t, 1.0 - r2)

    gamma_mean = 1.0 / (delta1 - lam2)
    gamma_var = jax.lax.cond(
        lam2 == 0.0, _case1_var1, _case1_var2, t, r1, delta1, alpha2, beta2, lam2
    )
    gamma_scale = gamma_var / gamma_mean
    gamma_shape = gamma_mean / gamma_scale * nu2 * x1_tilde

    p2_temp = jstats.gamma.cdf(x2_tilde, a=gamma_shape, scale=gamma_scale)

    p2 = jnp.where(
        pdf & (x2 > 0.0),
        p2_temp
        - jstats.gamma.cdf(
            x2 * jnp.exp(-delta1 * t) * jnp.power(t, 1.0 - r2),
            a=gamma_shape,
            scale=gamma_scale,
        ),
        p2_temp,
    )

    p2 = jnp.max(jnp.array([p2, 0.0]))
    lp2 = jnp.log(p2 + eps)

    return lp2


@jax.jit
def _lp2_case2(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    t: jnp.ndarray,
    par1: dict,
    par2: dict,
    eps: float = 1e-16,
    pdf: bool = True,
):
    """This function computes the log-likelihood of a subclone
    given its parent when they are equally fit.
    The means are given by Nicholson et al. (2023), and the variance is
    estimated through simulations. (See Theorem 2 in the supplement)


    Args:
        x1 : jnp.ndarray
            The number of cells in the parent subclone.
        x2 : jnp.ndarray
            The number of cells in the subclone.
        t : jnp.ndarray
            The sampling time.
        par1 : dict
            The growth parameters of the parent subclone.
        par2 : dict
            The growth parameters of the subclone.
        eps : float, optional
            The machine epsilon. Defaults to 1e-16.
    """

    delta1 = par1["delta"]
    r1 = par1["r"]
    r2 = par2["r"]
    nu2 = par2["nu"]

    x1_tilde = (x1 + 1.0) * jnp.exp(-delta1 * t) * jnp.power(t, 1.0 - r1)
    x2_tilde = (x2 + 1.0) * jnp.exp(-delta1 * t) * jnp.power(t, 1.0 - r2)

    log_rate = -0.691 * jnp.log(x1_tilde) + 2.973 * jnp.log(delta1 * t + eps)
    rate = jnp.exp(log_rate)

    p2_temp = jstats.gamma.cdf(x2_tilde, a=nu2 / r1 * x1_tilde * rate, scale=1 / rate)

    p2 = jnp.where(
        pdf & (x2 > 0.0),
        p2_temp
        - jstats.gamma.cdf(
            x2 * jnp.exp(-delta1 * t) * jnp.power(t, 1.0 - r2),
            a=nu2 / r1 * x1_tilde * rate,
            scale=1 / rate,
        ),
        p2_temp,
    )

    p2 = jnp.max(jnp.array([p2, 0.0]))
    lp2 = jnp.log(p2 + eps)

    return lp2


@jax.jit
def _h(theta: jnp.ndarray, par1: dict, par2: dict):
    """This function computes the h function for the conditional
    laplace transform given by Theorem 2 and Proposition 1 in the supplement.
    """

    delta1 = par1["delta"]
    rho2 = par2["rho"]
    lam2 = par2["lam"]
    r1 = par1["r"]
    r2 = par2["r"]
    nu2 = par2["nu"]
    phi2 = par2["phi"]
    gamma2 = par2["gamma"]

    def h1(theta):
        return nu2 * theta / (delta1 - lam2)

    def h2(theta):
        return nu2 * theta / r1

    def h31(theta):
        _h = (
            -rho2 * jss.gamma(r1) / jnp.power(lam2, r1 - 1) * polylog(r1, -phi2 * theta)
        )
        return _h

    def h32(theta):
        _h = (
            rho2
            * jnp.power(phi2 * theta, gamma2)
            * jnp.pi
            / jnp.sin(jnp.pi * gamma2)
            * jss.gamma(r1)
            / jnp.power(lam2, r1 - 1)
            * jnp.power(jnp.log(theta * phi2), r2 - 1)
            / jss.gamma(r2)
        )

        return _h

    def h3(theta):
        return jax.lax.cond(gamma2 == 0.0, h31, h32, theta)

    lam_diff = lam2 - delta1

    return jax.lax.switch(
        (jnp.sign(lam_diff) + 1).astype(jnp.int32), [h1, h2, h3], theta
    )


@jax.jit
def _lp2_case3(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    t: jnp.ndarray,
    par1: dict,
    par2: dict,
    eps: float = 1e-16,
    pdf: bool = True,
):
    """This function computes the log-likelihood of a subclone
    given its parent when it is more fit. For this one, we need
    the inverse laplace transform (See Theorem 2 in the supplement)


    Args:
        x1 : jnp.ndarray
            The number of cells in the parent subclone.
        x2 : jnp.ndarray
            The number of cells in the subclone.
        t : jnp.ndarray
            The sampling time.
        par1 : dict
            The growth parameters of the parent subclone.
        par2 : dict
            The growth parameters of the subclone.
        eps : float, optional
            The machine epsilon. Defaults to 1e-16.
    """

    delta1 = par1["delta"]
    r1 = par1["r"]
    delta2 = par2["delta"]
    r2 = par2["r"]
    x1_tilde = (x1 + 1.0) * jnp.exp(-delta1 * t) * jnp.power(t, 1.0 - r1)

    def lp_func(theta):
        return jnp.exp(-_h(theta, par1, par2) * x1_tilde) / theta

    def ilp(theta):
        fp = jax.vmap(lp_func)(BETA_VEC / theta)
        return jnp.dot(ETA_VEC, fp).real / theta

    p2_temp = ilp((x2 + 1.0) * jnp.exp(-delta2 * t) * jnp.power(t, 1.0 - r2))

    p2 = jnp.where(
        pdf & (x2 > 0.0),
        p2_temp - ilp(x2 * jnp.exp(-delta2 * t) * jnp.power(t, 1.0 - r2)),
        p2_temp,
    )

    p2 = jnp.max(jnp.array([p2, 0.0]))
    lp2 = jnp.log(p2 + eps)

    return lp2


@jax.jit
def _q_tilde(t: jnp.ndarray, C_s: jnp.ndarray, r: jnp.ndarray, delta: jnp.ndarray):
    """This function computes the q_tilde function defined
    in Theorem 3 in the supplement.
    """

    return integrate(t, r - 1.0, delta) / C_s


@jax.jit
def _g(theta: jnp.ndarray, tree: VectorizedTrees, i: int, tau: float = 0.01):
    def cond_fun(val):
        i, pa_i, g = val

        return pa_i > -1

    def body_fun(val):
        i, pa_i, g = val

        par_pa, par_i = get_pars(tree, i)

        g = _h(g, par_pa, par_i)

        return pa_i, tree.parent_id[pa_i], g

    mrca, _, g = jax.lax.while_loop(cond_fun, body_fun, (i, tree.parent_id[i], theta))

    lam = tree.lam[mrca]
    rho = tree.rho[mrca]
    phi = tree.phi[mrca]
    C_0 = tree.C_0
    t = tree.sampling_time

    return jax.lax.cond(
        lam < 0.0,
        lambda: jnp.power(
            phi + (1 - phi) * jnp.exp(-g * tau / t), -rho * C_0 * t / tau
        ),
        lambda: jnp.power(1 + phi * g, -rho * C_0),
    )


@jax.jit
def _mlogp(
    tree: VectorizedTrees,
    i: int,
    x: float = jnp.inf,
    eps: float = 1e-16,
    pdf: bool = True,
    tau: float = 0.01,
):
    x = jax.lax.cond(x > tree.cell_number[i], lambda: tree.cell_number[i], lambda: x)
    t = tree.sampling_time

    def mlogp_1():
        # nodes directly following the root use exact solution
        mlogp = jax.lax.cond(
            pdf,
            lambda: jstats.nbinom.pmf(
                k=x,
                n=tree.C_0 * tree.rho[i],
                p=_pt(tree.alpha[i], tree.beta, tree.lam[i], t),
            ),
            lambda: jss.betainc(
                a=tree.C_0 * tree.rho[i],
                b=x + 1.0,
                x=_pt(tree.alpha[i], tree.beta, tree.lam[i], t),
            ),
        )

        mlogp = jnp.log(mlogp + eps)

        return mlogp

    def mlogp_2():
        # nodes with parents use the laplace transform
        x_tilde = (
            (x + 1.0) * jnp.exp(-tree.delta[i] * t) * jnp.power(t, 1.0 - tree.r[i])
        )

        def lp_func(theta):
            g = _g(theta, tree, i, tau)
            return g / theta

        def ilp(theta):
            fp = jax.vmap(lp_func)(BETA_VEC / theta)
            return jnp.dot(ETA_VEC, fp).real / theta

        mlogp = ilp(x_tilde)

        mlogp = jnp.where(
            pdf & (x > 0.0),
            mlogp
            - ilp(x * jnp.exp(-tree.delta[i] * t) * jnp.power(t, 1.0 - tree.r[i])),
            mlogp,
        )

        mlogp = jnp.log(mlogp + eps)

        return mlogp

    return jax.lax.cond(
        tree.parent_id[i] == -1,
        mlogp_1,
        mlogp_2,
    )


def get_pars(tree: VectorizedTrees, i: int):
    """Helper function to collect the parent and child parameters
    for a given node in the tree.

    Args:
        tree : VectorizedTrees
            The tree object.
        i : int
            The index of the node in the tree.

    Returns:
        (par_pa, par_i) : tuple(dict, dict)
            The parent and child parameters.
    """

    par_i = {
        "alpha": tree.alpha[i],
        "nu": tree.nu[i],
        "lam": tree.lam[i],
        "rho": tree.rho[i],
        "phi": tree.phi[i],
        "delta": tree.delta[i],
        "r": tree.r[i],
        "gamma": tree.gamma[i],
        "beta": tree.beta,
        "C_s": tree.C_s,
        "C_0": tree.C_0,
        "observed": tree.observed[i],
    }

    pa_i = tree.parent_id[i]
    par_pa = {
        "alpha": tree.alpha[pa_i],
        "nu": tree.nu[pa_i],
        "lam": tree.lam[pa_i],
        "rho": tree.rho[pa_i],
        "phi": tree.phi[pa_i],
        "delta": tree.delta[pa_i],
        "r": tree.r[pa_i],
        "gamma": tree.gamma[pa_i],
        "beta": tree.beta,
        "C_s": tree.C_s,
        "C_0": tree.C_0,
        "observed": tree.observed[pa_i],
    }

    return par_pa, par_i


@jax.jit
def jlogp_no_parent(tree: VectorizedTrees, i: int, eps: float = 1e-16):
    """This function computes the log-likelihood of a subclone
    if its parent is the root, i.e. no parent.
    (See Lemma 1 and Theorem 1 in the supplement)

    It also computes part of the log-likelihood of the sampling time
    given the subclone. (See Theorem 3 in the supplement)

    Args:
        tree : VectorizedTrees
            The tree object.
        i : int
            The index of the node in the tree.
        eps : float, optional
            The machine epsilon. Defaults to 1e-16.
    """

    x = tree.cell_number[i]
    t = tree.sampling_time

    lp = jax.lax.cond(
        tree.observed[i],
        lambda: jstats.nbinom.pmf(
            k=x,
            n=tree.C_0 * tree.rho[i],
            p=_pt(tree.alpha[i], tree.beta, tree.lam[i], t),
        ),
        lambda: jss.betainc(
            a=tree.C_0 * tree.rho[i],
            b=x + 1.0,
            x=_pt(tree.alpha[i], tree.beta, tree.lam[i], t),
        ),
    )

    lp = jnp.log(lp + eps)

    x_tilde = x * jnp.exp(-tree.delta[i] * t) * jnp.power(t, 1.0 - tree.r[i])

    lt = -_q_tilde(t, tree.C_s, tree.r[i], tree.delta[i]) * x_tilde

    return lp + lt


@jax.jit
def jlogp_w_parent(tree: VectorizedTrees, i: int, eps: float = 1e-16):
    """This function computes the log-likelihood of a subclone
    given its parent (See Theorem 2 in the supplement)

    Args:
        tree : VectorizedTrees
            The tree object.
        i : int
            The index of the node in the tree.
        eps : float, optional
            The machine epsilon. Defaults to 1e-16.
    """

    x1 = tree.cell_number[tree.parent_id[i]]
    x2 = tree.cell_number[i]
    t = tree.sampling_time

    par1, par2 = get_pars(tree, i)

    lam_diff = par2["lam"] - par1["delta"]

    lp = jax.lax.switch(
        (jnp.sign(lam_diff) + 1).astype(jnp.int32),
        [_lp2_case1, _lp2_case2, _lp2_case3],
        x1,
        x2,
        t,
        par1,
        par2,
        eps,
        True,
    )

    x2_tilde = x2 * jnp.exp(-par2["delta"] * t) * jnp.power(t, 1.0 - par2["r"])
    lt = -_q_tilde(t, par2["C_s"], par2["r"], par2["delta"]) * x2_tilde

    return lp + lt


@jax.jit
def jlogp_one_node(tree: VectorizedTrees, i: int, eps: float = 1e-16):
    return jax.lax.cond(
        tree.parent_id[i] == -1,
        lambda: jlogp_no_parent(tree, i, eps),
        lambda: jlogp_w_parent(tree, i, eps),
    )


@jax.jit
def jlogp_one_tree(tree: VectorizedTrees, eps: float = 1e-16):
    """This function computes the log-likelihood of a tree"""

    def scan_fun(jlogp, i):
        new_jlogp = jlogp_one_node(tree, i, eps) + jlogp
        return new_jlogp, new_jlogp

    jlogp, _ = jax.lax.scan(scan_fun, 0.0, tree.node_id)

    jlogp += jnp.log(jnp.sum(tree.cell_number) + eps) - jnp.log(tree.C_s)

    return jlogp


@jax.jit
def unnormalized_joint_logp(trees: VectorizedTrees, eps: float = 1e-16) -> jnp.ndarray:
    """This function computes the unnormalized joint log-likelihood
    of a set of trees. We still need to normalize it by the marginal
    probability of the sampling event occurring before some predefined
    maximum time, which is given in Theorem 3 in the supplement.
    """

    jlogp = jax.vmap(
        jlogp_one_tree,
        in_axes=(
            VectorizedTrees(
                0,  # cell_number
                0,  # observed
                0,  # sampling_time
                0,  # weight
                None,  # node_id
                None,  # parent_id
                None,  # alpha
                None,  # nu
                None,  # lam
                None,  # rho
                None,  # phi
                None,  # delta
                None,  # r
                None,  # gamma
                None,  # N_trees
                None,  # n_nodes
                None,  # beta
                None,  # C_s
                None,  # C_0
                None,  # C_min
            ),
            None,
        ),
    )(trees, eps)

    return jnp.dot(trees.weight, jlogp)
