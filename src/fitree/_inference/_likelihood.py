import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import jax.scipy.special as jss
from jax import lax
from jax._src.lax.lax import _const as _lax_const
from jax._src.scipy.special import gammaln


from fitree._inference._utils import (
    ETA_VEC,
    BETA_VEC,
    polylog,
    polylog_log_minus_z,
    integrate,
)
from fitree._trees._wrapper import VectorizedTrees

jax.config.update("jax_enable_x64", True)


@jax.jit
def get_x_tilde(
    x: jnp.ndarray,
    delta: jnp.ndarray,
    r: jnp.ndarray,
    t: jnp.ndarray,
):
    x_tilde = x * jnp.exp(-delta * t) * jnp.power(t, 1.0 - r)

    return x_tilde


@jax.jit
def nbinom_logpmf(
    x: jnp.ndarray,
    C_0: jnp.ndarray,
    rho: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    lam: jnp.ndarray,
    t: jnp.ndarray,
):
    """This function computes the negative binomial log-likelihood
    (See Lemma 1 in the supplement)
    """

    k = x
    n = C_0 * rho
    one = _lax_const(k, 1)
    comb_term = lax.sub(
        lax.sub(gammaln(lax.add(k, n)), gammaln(n)), gammaln(lax.add(k, one))
    )
    log_p = jax.lax.switch(
        (jnp.sign(lam) + 1).astype(jnp.int32),
        [
            lambda: lax.log(-lam) - lax.log(beta - alpha * lax.exp(lam * t)),
            lambda: -lax.log(1.0 + alpha * t),
            lambda: lax.log(lam) - lax.log(alpha * lax.exp(lam * t) - beta),
        ],
    )
    log_p = jnp.where(
        log_p == -jnp.inf,
        lax.log(lam) - lam * t * lax.log(alpha),
        log_p,
    )
    log_1_p = lax.log1p(-jnp.exp(log_p))  # pyright: ignore

    log_linear_term = lax.add(lax.mul(n, log_p), lax.mul(k, log_1_p))  # pyright: ignore
    log_probs = lax.add(comb_term, log_linear_term)
    return log_probs


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

    return jnp.power(t, -r1 + 1.0) * (
        (1.0 + 2.0 * alpha2 * t) * integrate(t, r1 - 1.0, delta1, -delta1 * t)
        - 2.0 * alpha2 * integrate(t, r1, delta1, -delta1 * t)
    )


@jax.jit
def _case1_var2(t, r1, delta1, alpha2, beta2, lam2):
    # Variance function for delta1 > lam2 != 0.0

    return jnp.power(t, -r1 + 1.0) * (
        2.0
        * alpha2
        / lam2
        * integrate(t, r1 - 1.0, delta1 - 2.0 * lam2, -(delta1 - 2.0 * lam2) * t)
        - (alpha2 + beta2)
        / lam2
        * integrate(t, r1 - 1.0, delta1 - lam2, -(delta1 - lam2) * t)
    )


@jax.jit
def _lp2_case1(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    t: jnp.ndarray,
    par1: dict,
    par2: dict,
    eps: float = 1e-64,
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
            The machine epsilon. Defaults to 1e-64.
    """

    delta1 = par1["delta"]
    r1 = par1["r"]
    delta2 = par2["delta"]
    lam2 = par2["lam"]
    r2 = par2["r"]
    nu2 = par2["nu"]
    alpha2 = par2["alpha"]
    beta2 = par2["beta"]

    x2_var = jax.lax.cond(
        jnp.abs(lam2) < 1e-3,
        _case1_var1,
        _case1_var2,
        t,
        r1,
        delta1,
        alpha2,
        beta2,
        lam2,
    )
    log_x1_tilde = -(r1 - 1) * jnp.log(t) - delta1 * t + jnp.log1p(x1)
    log_x2_tilde = -(r2 - 1) * jnp.log(t) - delta2 * t + jnp.log1p(x2)
    lnorm_var = jnp.log1p(x2_var / nu2 / (x1 + 1.0) * ((delta1 - lam2) ** 2))
    lnorm_mean = (
        jnp.log(nu2) + log_x1_tilde - jnp.log(delta1 - lam2)
    ) - 0.5 * lnorm_var

    p2_temp = jstats.norm.cdf(log_x2_tilde, loc=lnorm_mean, scale=jnp.sqrt(lnorm_var))

    p2 = jnp.where(
        pdf & (x2 > 0.0),
        p2_temp
        - jstats.norm.cdf(
            -(r2 - 1) * jnp.log(t) - delta2 * t + jnp.log(x2 + eps),
            loc=lnorm_mean,
            scale=jnp.sqrt(lnorm_var),
        ),
        p2_temp,
    )

    p2 = jnp.where(
        jnp.isnan(p2),
        0.0,
        p2,
    )

    p2 = jnp.max(jnp.array([p2, 0.0]))
    lp2 = jnp.log(p2 + eps)

    return lp2


@jax.jit
def _case2_var1(t, r1, delta1, alpha2, beta2):
    # Variance function for delta1 = lam2 = 0.0

    return jnp.power(t, -1.0) * (
        (1.0 + 2.0 * alpha2 * t) / r1 - 2 * alpha2 * t / (r1 + 1)
    )


@jax.jit
def _case2_var2(t, r1, delta1, alpha2, beta2):
    # Variance function for delta1 = lam2 != 0.0

    return jnp.power(t, -r1 - 1.0) * (
        2.0 * alpha2 / delta1 * integrate(t, r1 - 1.0, -delta1, delta1 * t)
        - (alpha2 + beta2) / delta1 * jnp.power(t, r1) / r1
    )


@jax.jit
def _lp2_case2(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    t: jnp.ndarray,
    par1: dict,
    par2: dict,
    eps: float = 1e-64,
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
            The machine epsilon. Defaults to 1e-64.
    """

    delta1 = par1["delta"]
    r1 = par1["r"]
    r2 = par2["r"]
    nu2 = par2["nu"]
    alpha2 = par2["alpha"]
    beta2 = par2["beta"]

    x2_var = jax.lax.cond(
        jnp.abs(delta1) < 1e-3, _case2_var1, _case2_var2, t, r1, delta1, alpha2, beta2
    )
    log_x1_tilde = -(r1 - 1) * jnp.log(t) - delta1 * t + jnp.log1p(x1)
    log_x2_tilde = -(r2 - 1) * jnp.log(t) - delta1 * t + jnp.log1p(x2)
    lnorm_var = jnp.log1p(x2_var / nu2 / (x1 + 1.0) * (r1**2))
    lnorm_mean = (jnp.log(nu2) + log_x1_tilde - jnp.log(r1)) - 0.5 * lnorm_var

    p2_temp = jstats.norm.cdf(log_x2_tilde, loc=lnorm_mean, scale=jnp.sqrt(lnorm_var))

    p2 = jnp.where(
        x2 > 0,
        p2_temp
        - jstats.norm.cdf(
            -(r2 - 1) * jnp.log(t) - delta1 * t + jnp.log(x2 + eps),
            loc=lnorm_mean,
            scale=jnp.sqrt(lnorm_var),
        ),
        p2_temp,
    )

    p2 = jnp.where(
        jnp.isnan(p2),
        0.0,
        p2,
    )

    p2 = jnp.max(jnp.array([p2, 0.0]))
    lp2 = jnp.log(p2 + eps)

    return lp2


@jax.jit
def _lp2_case3(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    t: jnp.ndarray,
    par1: dict,
    par2: dict,
    eps: float = 1e-64,
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
            The machine epsilon. Defaults to 1e-64.
    """

    delta1 = par1["delta"]
    r1 = par1["r"]
    delta2 = par2["delta"]
    r2 = par2["r"]
    rho2 = par2["rho"]
    lam2 = par2["lam"]
    phi2 = par2["phi"]
    gamma2 = par2["gamma"]
    x1_tilde = get_x_tilde(x1 + 1.0, delta1, r1, t)

    def _log_theta(beta, z):
        return jnp.log(beta) - jnp.log(z + eps) + delta2 * t

    def h1(beta, z):
        _h = (
            -rho2
            * jss.gamma(r1)
            / jnp.power(lam2, r1 - 1)
            * polylog_log_minus_z(r1, _log_theta(beta, z) + jnp.log(phi2))
        )
        return _h * x1_tilde

    def h2(beta, z):
        _h = (
            rho2
            * jnp.power(phi2, gamma2)
            * jnp.pi
            / jnp.sin(jnp.pi * gamma2)
            / jnp.power(lam2, r1 - 1)
            * jnp.power(_log_theta(beta, z) + jnp.log(phi2), r2 - 1)
            * jnp.power(beta, gamma2)
            / jnp.power(t, r1 - 1)
            * x1
            / jnp.power(z + eps, gamma2)
        )
        return _h

    def h(beta, z):
        return jax.lax.cond(gamma2 == 0.0, h1, h2, beta, z)

    def lp_func(eta, beta, z):
        return jnp.exp(-h(beta, z)) * eta / beta

    def ilp(z):
        fp = jax.vmap(lp_func, in_axes=(0, 0, None))(ETA_VEC, BETA_VEC, z)
        return jnp.sum(fp).real

    p2_temp = ilp(x2 + 1.0)

    p2 = jnp.where(
        pdf & (x2 > 0.0),
        p2_temp - ilp(x2),
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
            / jnp.power(lam2, r1 - 1)
            * jnp.power(jnp.log(theta * phi2), r2 - 1)
        )

        return _h

    def h3(theta):
        return jax.lax.cond(gamma2 == 0.0, h31, h32, theta)

    lam_diff = lam2 - delta1

    return jax.lax.switch(
        (jnp.sign(lam_diff) + 1).astype(jnp.int32), [h1, h2, h3], theta
    )


@jax.jit
def _q_tilde(t: jnp.ndarray, C_s: jnp.ndarray, r: jnp.ndarray, delta: jnp.ndarray):
    """This function computes the q_tilde function defined
    in Theorem 3 in the supplement.
    """

    return integrate(t, r - 1.0, delta, -jnp.log(C_s))


@jax.jit
def _g(theta: jnp.ndarray, tree: VectorizedTrees, i: int, tau: float = 0.01):
    """This function computes the g function defined in Corollary 1"""

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
    eps: float = 1e-64,
    pdf: bool = True,
    tau: float = 0.01,
):
    x = jax.lax.cond(x > tree.cell_number[i], lambda: tree.cell_number[i], lambda: x)
    t = tree.sampling_time

    def mlogp_1():
        # nodes directly following the root use exact solution
        mlogp = jax.lax.cond(
            pdf,
            lambda: jstats.nbinom.logpmf(  # pyright: ignore
                k=x,
                n=tree.C_0 * tree.rho[i],
                p=_pt(tree.alpha[i], tree.beta, tree.lam[i], t),
            ),
            lambda: jnp.log(
                jss.betainc(
                    a=tree.C_0 * tree.rho[i],
                    b=x + 1.0,
                    x=_pt(tree.alpha[i], tree.beta, tree.lam[i], t),
                )
                + eps
            ),
        )

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

        mlogp = jnp.max(jnp.array([mlogp, 0.0]))

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
    }

    return par_pa, par_i


@jax.jit
def jlogp_no_parent(
    x: jnp.ndarray, observed: bool, t: jnp.ndarray, par: dict, eps: float = 1e-64
):
    """This function computes the log-likelihood of a subclone
    if its parent is the root, i.e. no parent.
    (See Lemma 1 and Theorem 1 in the supplement)

    It also computes part of the log-likelihood of the sampling time
    given the subclone. (See Theorem 3 in the supplement)
    """

    lp = jax.lax.cond(
        observed,
        lambda: nbinom_logpmf(  # pyright: ignore
            x,
            par["C_0"],
            par["rho"],
            par["alpha"],
            par["beta"],
            par["lam"],
            t,
        ),
        lambda: jnp.log(
            jss.betainc(
                a=par["C_0"] * par["rho"],
                b=x + 1.0,
                x=_pt(par["alpha"], par["beta"], par["lam"], t),
            )
            + eps
        ),
    )

    return lp


@jax.jit
def jlogp_w_parent(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    observed: jnp.ndarray,
    t: jnp.ndarray,
    par1: dict,
    par2: dict,
    eps: float = 1e-64,
):
    """This function computes the log-likelihood of a subclone
    given its parent (See Theorem 2 in the supplement)
    """
    # x1 = jnp.where(observed[0], x1, 0.0)

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
        observed[1],
    )

    return lp


@jax.jit
def jlogp_one_tree(
    x_vec: jnp.ndarray,
    observed: jnp.ndarray,
    t: jnp.ndarray,
    par1: dict,
    par2: dict,
    no_parent: bool,
    eps: float = 1e-64,
):
    x1 = x_vec[0]
    x2 = x_vec[1]
    delta2 = par2["delta"]
    r2 = par2["r"]
    C_s = par2["C_s"]

    lp = jax.lax.cond(
        no_parent,
        lambda: jlogp_no_parent(x2, observed[1], t, par2, eps),
        lambda: jlogp_w_parent(x1, x2, observed, t, par1, par2, eps),
    )

    lt = -integrate(
        t,
        r2 - 1.0,
        delta2,
        jnp.log(x2) - jnp.log(C_s) - delta2 * t - (r2 - 1.0) * jnp.log(t),
    )

    return lp + lt


@jax.jit
def jlogp_one_node(
    trees: VectorizedTrees,
    i: int,
    eps: float = 1e-64,
):
    par1, par2 = get_pars(trees, i)
    pa_i = trees.parent_id[i]
    no_parent = jnp.where(pa_i == -1, True, False)

    lp = jax.vmap(
        jlogp_one_tree,
        in_axes=(
            0,
            0,
            0,
            None,
            None,
            None,
            None,
        ),
    )(
        trees.cell_number[:, [pa_i, i]],
        trees.observed[:, [pa_i, i]],
        trees.sampling_time,
        par1,
        par2,
        no_parent,
        eps,
    )

    return jnp.dot(trees.weight, lp)


@jax.jit
def unnormalized_joint_logp(trees: VectorizedTrees, eps: float = 1e-64) -> jnp.ndarray:
    """This function computes the unnormalized joint log-likelihood
    of a set of trees. We still need to normalize it by the marginal
    probability of the sampling event occurring before some predefined
    maximum time, which is given in Theorem 3 in the supplement.
    """

    def scan_fun(jlogp, i):
        new_jlogp = jlogp_one_node(trees, i, eps) + jlogp
        return new_jlogp, new_jlogp

    jlogp, _ = jax.lax.scan(scan_fun, 0.0, trees.node_id)

    jlogp += jnp.dot(
        jnp.log(jnp.sum(trees.cell_number, axis=1) + eps) - jnp.log(trees.C_s),
        trees.weight,
    )

    return jlogp


# @jax.jit
# def sum_fitness_effects(
#     genotype: jnp.ndarray,
#     F_mat: jnp.ndarray,
# ) -> jnp.ndarray:
#     """This function computes the sum of fitness effects of a genotype
#     based on the fitness matrix F_mat.

#     Suppose indices {2,3,5} in genotype vector are non-zero. Then the
#     sum of fitness effects of this genotype is given by:
#     F_mat[2,2] + F_mat[3,3] + F_mat[5,5] + F_mat[2,3] + F_mat[2,5] + F_mat[3,5]
#     """
#     # Ensure genotype is boolean
#     genotype_bool = genotype.astype(bool)

#     # Create a mask for the outer product of the genotype with itself
#     mask = genotype_bool[:, None] & genotype_bool[None, :]  # Shape: (N, N)

#     # Create an upper-triangular mask to avoid double-counting symmetric pairs
#     upper_tri_mask = jnp.triu(jnp.ones_like(F_mat, dtype=bool))

#     # Combine the genotype mask with the upper-triangular mask
#     combined_mask = mask & upper_tri_mask

#     # Apply the combined mask to F_mat and sum the selected elements
#     total_sum = jnp.sum(jnp.where(combined_mask, F_mat, 0.0))

#     return total_sum


@jax.jit
def compute_fitness_effects(
    genotype: jnp.ndarray,
    F_mat: jnp.ndarray,
) -> jnp.ndarray:
    """This function computes the sum of fitness effects of a genotype
    based on the fitness matrix F_mat.

    Suppose indices {2,3,5} in genotype vector are non-zero. Then the
    sum of fitness effects of this genotype is given by:
    max(F_mat[2,2] + F_mat[3,3] + F_mat[5,5]) + F_mat[2,3] + F_mat[2,5] + F_mat[3,5]
    """

    # Ensure genotype is boolean
    genotype_bool = genotype.astype(bool)

    # Create a mask for the outer product of the genotype with itself
    mask = genotype_bool[:, None] & genotype_bool[None, :]

    # Create an upper-triangular mask to avoid double-counting symmetric pairs
    upper_tri_mask = jnp.triu(jnp.ones_like(F_mat, dtype=bool), k=1)

    # Combine the genotype mask with the upper-triangular mask
    combined_mask = mask & upper_tri_mask

    # Apply the combined mask to F_mat and sum the selected elements
    total_sum = jnp.sum(jnp.where(combined_mask, F_mat, 0.0))

    # Add the maximum diagonal element to the total sum
    total_sum += jnp.max(jnp.where(genotype_bool, jnp.diag(F_mat), 0.0))

    return total_sum


@jax.jit
def update_params(
    trees: VectorizedTrees,
    F_mat: jnp.ndarray,
    zero_window: float = 1e-2,
):
    """This function updates the growth parameters of the tree
    based on the fitness matrix.
    """

    def scan_fun(trees, i):
        pa_i = trees.parent_id[i]
        delta_pa = jnp.where(pa_i == -1, 0.0, trees.delta[pa_i])
        r_pa = jnp.where(pa_i == -1, 1.0, trees.r[pa_i])

        log_alpha = jnp.log(trees.beta) + compute_fitness_effects(
            trees.genotypes[i, :], F_mat
        )
        new_alpha_i = jnp.exp(log_alpha)

        new_lam_i = new_alpha_i - trees.beta
        new_lam_i = jnp.where(
            jnp.abs(new_lam_i - delta_pa) < zero_window,
            delta_pa,  # pyright: ignore
            new_lam_i,  # pyright: ignore
        )

        new_alpha_i = new_lam_i + trees.beta

        trees_alpha = trees.alpha.at[i].set(new_alpha_i)
        trees_lam = trees.lam.at[i].set(new_lam_i)

        new_rho_i = trees.nu[i] / new_alpha_i
        trees_rho = trees.rho.at[i].set(new_rho_i)

        new_phi_i = jnp.select(
            [
                new_lam_i > 0.0,
                new_lam_i == 0.0,
                new_lam_i < 0.0,
            ],
            [
                new_alpha_i / new_lam_i,
                new_alpha_i,
                -trees.beta / new_lam_i,
            ],
        )
        trees_phi = trees.phi.at[i].set(new_phi_i)

        new_delta_i = jnp.max(jnp.array([delta_pa, new_lam_i]))
        trees_delta = trees.delta.at[i].set(new_delta_i)

        new_r_i = jnp.select(
            [
                new_lam_i > delta_pa,
                new_lam_i == delta_pa,
                new_lam_i < delta_pa,
            ],
            [1.0, r_pa + 1.0, r_pa],  # pyright: ignore
        )
        trees_r = trees.r.at[i].set(new_r_i)

        new_gamma_i = jnp.where(
            new_delta_i == 0.0,
            0.0,
            trees.delta[pa_i] / new_delta_i,
        )
        trees_gamma = trees.gamma.at[i].set(new_gamma_i)

        # Update the trees object with the new parameters
        trees = trees._replace(
            alpha=trees_alpha,
            lam=trees_lam,
            rho=trees_rho,
            phi=trees_phi,
            delta=trees_delta,
            r=trees_r,
            gamma=trees_gamma,
        )

        return trees, None

    updated_trees, _ = jax.lax.scan(scan_fun, trees, trees.node_id)

    return updated_trees


@jax.jit
def compute_g_tilde_vec(trees: VectorizedTrees, t: float) -> jnp.ndarray:
    """This function computes the g_tilde vector for the tree
    at maximum time t_max. (See Theorem 3 in the supplement)
    """

    g_tilde_vec = jnp.zeros_like(trees.node_id, dtype=jnp.float64)

    # Nodes to scan: reverse order of the nodes
    # This is to compute g_tilde in reverse topological order of
    # the union_tree (i.e. starting from the leaves)
    nodes_to_scan = jnp.flip(trees.node_id)

    # Compute the recursive g_tilde vector
    def scan_fun(g_tilde_vec, i):
        # Compute q_tilde for the node
        q_tilde_i = _q_tilde(t, trees.C_s, trees.r[i], trees.delta[i])
        g_tilde_i = g_tilde_vec[i] + q_tilde_i
        g_tilde_vec = g_tilde_vec.at[i].set(g_tilde_i)

        # Update the g_tilde vector for the parent node
        pa_i = trees.parent_id[i]
        par1, par2 = get_pars(trees, i)
        g_tilde_pa_i = g_tilde_vec[pa_i] + jax.lax.cond(
            pa_i == -1,
            lambda: 0.0,
            lambda: _h(g_tilde_i, par1, par2),
        )
        g_tilde_vec = g_tilde_vec.at[pa_i].set(g_tilde_pa_i)

        return g_tilde_vec, None

    g_tilde_vec, _ = jax.lax.scan(scan_fun, g_tilde_vec, nodes_to_scan)

    return g_tilde_vec


@jax.jit
def _log_pt(
    trees: VectorizedTrees, t: float, eps: float = 1e-64, tau: float = 1e-2
) -> float:
    """This function computes the log probability P(T_s > t_max)
    defined in Theorem 3.
    """

    g_tilde_vec = compute_g_tilde_vec(trees, t)
    C_0 = trees.C_0

    def scan_fun(log_pt, i):
        lam = trees.lam[i]
        rho = trees.rho[i]
        phi = trees.phi[i]
        g = g_tilde_vec[i]

        log_pt += jax.lax.cond(
            trees.parent_id[i] == -1,
            lambda: jax.lax.cond(
                lam < 0.0,
                lambda: -rho
                * C_0
                * t
                / tau
                * jnp.log(phi + (1 - phi) * jnp.exp(-g * tau / t) + eps),
                lambda: -rho * C_0 * jnp.log1p(phi * g),
            ),
            lambda: 0.0,
        )

        return log_pt, None

    log_pt, _ = jax.lax.scan(scan_fun, 0.0, trees.node_id)

    return log_pt


@jax.jit
def compute_normalizing_constant(
    trees: VectorizedTrees, eps: float = 1e-64, tau: float = 1e-2
) -> jnp.ndarray:
    """This function computes the normalizing constant for the
    joint likelihood of the trees.
    P(T_s < t_max) = 1 - P(T_s > t_max)
    """

    t = trees.t_max
    log_pt = _log_pt(trees, t, eps, tau)

    return -jnp.expm1(log_pt)


@jax.jit
def estimate_cell_numbers(trees: VectorizedTrees):
    # Estimate observed cell numbers using sequenced cell numbers
    frequencies = trees.seq_cell_number / trees.seq_cell_number.sum(axis=1).reshape(
        -1, 1
    )
    cell_number = jnp.round(jnp.array(frequencies * trees.tumor_size.reshape(-1, 1)))

    t = trees.sampling_time
    C_0 = trees.C_0
    beta = trees.beta

    # Compute the unobserved cell numbers using expected values
    def scan_fun(cell_number, i):
        pa_i = trees.parent_id[i]
        lam_i = trees.lam[i]
        nu_i = trees.nu[i]
        alpha_i = trees.alpha[i]
        rho_i = trees.rho[i]
        delta_pa_i = trees.delta[pa_i]
        r_pa_i = trees.r[pa_i]
        observed = trees.observed[:, i].astype(jnp.float64)
        cell_number_i = cell_number[:, i]
        cell_number_i *= observed

        def expected_no_parent():
            p = _pt(alpha_i, beta, lam_i, t)
            exp_num = C_0 * rho_i * (1 - p) / p
            exp_num = jnp.where(
                lam_i > 0.0,
                0.0,
                exp_num,
            )
            return exp_num

        def expected_w_parent():
            lam_diff = lam_i - delta_pa_i
            exp_num = (
                nu_i
                * cell_number[:, pa_i]
                * jax.lax.switch(
                    (jnp.sign(lam_diff) + 1).astype(jnp.int32),
                    [
                        lambda: 1.0 / (delta_pa_i - lam_i) * jnp.ones_like(t),
                        lambda: 1.0 / r_pa_i * t,
                        lambda: (
                            jss.gamma(r_pa_i)
                            / jnp.power(lam_i - delta_pa_i, r_pa_i)
                            * jnp.ones_like(t)
                        ),
                    ],
                )
            )
            exp_num = jnp.where(
                lam_i > jnp.max(jnp.array([delta_pa_i, 0.0])),
                0.0,
                exp_num,
            )
            return exp_num

        unobserved_number_i = jax.lax.cond(
            pa_i == -1,
            expected_no_parent,
            expected_w_parent,
        )
        cell_number_i += (1.0 - observed) * unobserved_number_i
        cell_number = cell_number.at[:, i].set(jnp.round(cell_number_i))

        return cell_number, None

    cell_number, _ = jax.lax.scan(scan_fun, cell_number, trees.node_id)

    tumor_size = jnp.sum(cell_number, axis=1)  # pyright: ignore

    trees = trees._replace(cell_number=cell_number, tumor_size=tumor_size)

    return trees


@jax.jit
def log_n_choose_k(
    n: jnp.ndarray,
    k: jnp.ndarray,
):
    """This function computes the log of the binomial coefficient
    n choose k.
    """

    def case_0():
        # k is 0 or n
        return 0.0

    def case_1():
        # k is 1 or n - 1
        return jnp.log(n)

    def case_2():
        # both n and k are small
        return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

    def case_3():
        # n is large and k is small
        return (
            k * (jnp.log(n) + 1 - jnp.log(k))
            - 1 / 2 * jnp.log(2 * jnp.pi * k)
            - k**2 / (2 * n)
        )

    def case_4():
        # n is large and k is large
        return (
            1 / 2 * (jnp.log(n) - jnp.log(2 * jnp.pi * k) - jnp.log(n - k))
            + n * jnp.log(n)
            - k * jnp.log(k)
            - (n - k) * jnp.log(n - k)
        )

    return jnp.select(
        [
            (k == 0.0) | (k == n),
            (k == 1.0) | (k == n - 1),
            n <= 1e8,
            (n > 1e8) & (k <= 0.1 * n),
            (n > 1e8) & (k > 0.1 * n),
        ],
        [
            case_0(),
            case_1(),
            case_2(),
            case_3(),
            case_4(),
        ],
    )


@jax.jit
def log_multi_hypergeo(
    K: jnp.ndarray,
    k: jnp.ndarray,
    N: jnp.ndarray,
    n: jnp.ndarray,
):
    """This function computes the log density of the multivariate hypergeometric
    distribution.
    """

    log_p = jnp.sum(log_n_choose_k(K, k), axis=1)
    log_p -= log_n_choose_k(N, n)

    return jnp.sum(log_p)


@jax.jit
def jlogp(
    trees: VectorizedTrees,
    F_mat: jnp.ndarray,
    C_s: jnp.ndarray,
    eps: float = 1e-64,
    tau: float = 1e-2,
):
    """This function computes the joint log-likelihood of a set of trees
    given the fitness matrix F_mat after normalizing it by the marginal
    likelihood of the sampling event occurring before some predefined
    maximum time. (See Theorem 3 in the supplement)
    """

    # Update C_s in the trees object
    trees = trees._replace(C_s=C_s)

    # Update the growth parameters based on the fitness matrix
    trees = update_params(trees, F_mat)

    # Estimate the cell numbers
    trees = estimate_cell_numbers(trees)

    # Compute the unnormalized joint log-likelihood
    jlogp_unnormalized = unnormalized_joint_logp(trees, eps)

    # # Compute the probability of sampling C_seq cells
    log_p_seq = log_multi_hypergeo(
        trees.cell_number,
        trees.seq_cell_number,
        trees.tumor_size,
        trees.seq_cell_number.sum(axis=1),
    )

    # Normalizing constant on the full F_mat with positive diagonals
    # plus positive off-diagonals is numerically unstable
    # [TODO] Investigate why this is the case
    # For now, we will use the diagonal of F_mat to compute the normalizing constant
    # as the base fitness values influence the normalizing constant the most
    trees = update_params(trees, jnp.diag(jnp.diag(F_mat)))
    normalizing_constant = compute_normalizing_constant(trees, eps=eps, tau=tau)

    jlogp = jlogp_unnormalized + log_p_seq

    return jlogp, normalizing_constant
