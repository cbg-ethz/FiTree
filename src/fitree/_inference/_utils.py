import jax
import jax.numpy as jnp
import jax.scipy.special as jss
from jax import config

config.update("jax_enable_x64", True)

# Optimized parameters (order = 20) of the CME method for
# numerical inverse laplace transform
# Reference: Horvath et al. (2020) Numerical inverse Laplace
# transformation using concentrated matrix exponential distributions
# Website: http://inverselaplace.org
# Values obtained from https://github.com/ghorvath78/iltcme/blob/master/iltcme_ext.json

d = {
    "eta_re": [
        4892.76882097293,
        3639.5638950007028,
        -6636.7185148917088,
        -7924.5037083388106,
        279.75212465251906,
        6642.49195098979,
        4073.2762452929469,
        -2332.684484151443,
        -4287.6906599151671,
        -1065.2964872403713,
        1968.3636074964643,
        1687.7593382899179,
        -86.121823891971943,
        -778.77770298194059,
        -307.99553346440314,
        128.81526084347016,
        119.22350550437042,
        4.3646093646791639,
        -15.199431391585998,
        -1.7882507296626056,
        0.39949123248368934,
    ],
    "eta_im": [
        0,
        -8938.026629493128,
        -6460.8180356809908,
        3500.2182722823027,
        7904.3803866231929,
        2379.587634928153,
        -4604.8090778887781,
        -4669.403269710524,
        324.36244532844285,
        3247.0197930214563,
        1702.180809603379,
        -838.19757742216962,
        -1283.834059562857,
        -257.60540657904835,
        371.08201509137956,
        222.6472363948352,
        -24.026085178886035,
        -49.001817877703836,
        -5.1334873791638493,
        3.3112869444689634,
        0.25951100825067525,
    ],
    "beta_re": [
        8.6736267672444,
        8.6736267672444,
        8.6736267672444,
        8.6736267672444,
        8.6736267672444,
        8.6736267672444,
        8.6736267672444,
        8.6736267672444,
        8.6736267672444,
        8.6736267672444,
        8.6736267672444,
        8.6736267672444,
        8.6736267672444,
        8.6736267672444,
        8.6736267672444,
        8.6736267672444,
        8.6736267672444,
        8.6736267672444,
        8.6736267672444,
        8.6736267672444,
        8.6736267672444,
    ],
    "beta_im": [
        0,
        5.0526738216031815,
        10.105347643206363,
        15.158021464809545,
        20.210695286412726,
        25.263369108015908,
        30.316042929619091,
        35.368716751222273,
        40.421390572825452,
        45.474064394428638,
        50.526738216031816,
        55.579412037635,
        60.632085859238181,
        65.684759680841367,
        70.737433502444546,
        75.790107324047725,
        80.8427811456509,
        85.895454967254082,
        90.948128788857275,
        96.000802610460454,
        101.05347643206363,
    ],
}

# convert to eta and beta vectors
ETA_VEC = jnp.array(d["eta_re"]) + jnp.array(d["eta_im"]) * 1j
BETA_VEC = jnp.array(d["beta_re"]) + jnp.array(d["beta_im"]) * 1j


@jax.jit
def altzeta(n):
    """The dirichlet eta function is defined as the alternating zeta function."""
    return jnp.where(n == 0.0, 0.5, (1 - 2.0 ** (1 - n)) * jss.zeta(n, q=1))


@jax.jit
def polylog(n, z):
    """This function computes the approximation of the polylogarithm of order n at z
    for positive integers n and large negative z. The approximation is based on
    equation (11.1) in the technical report "The Computation of Polylogarithms" by
    Wood, David C. (1992)
    """

    max_k = jnp.floor(n / 2.0).astype(jnp.int32)

    def body_fun(k, carry):
        carry += (
            altzeta(2.0 * k)
            * jnp.power(jnp.log(1 - z), n - 2.0 * k)
            / jss.gamma(n - 2.0 * k + 1.0)
        )
        return carry

    return jax.lax.fori_loop(0, max_k + 1, body_fun, 0.0) * (-2.0)


@jax.jit
def polylog_log_minus_z(n, log_minus_z):
    """This function computes the approximation of the polylogarithm of order n at z
    for positive integers n and large negative z. The approximation is based on
    equation (11.1) in the technical report "The Computation of Polylogarithms" by
    Wood, David C. (1992)

    Here, the term log(1 - z) or log(-z) is passed as an argument.
    """

    max_k = jnp.floor(n / 2.0).astype(jnp.int32)

    def body_fun(k, carry):
        carry += (
            altzeta(2.0 * k)
            * jnp.power(log_minus_z, n - 2.0 * k)
            / jss.gamma(n - 2.0 * k + 1.0)
        )
        return carry

    return jax.lax.fori_loop(0, max_k + 1, body_fun, 0.0) * (-2.0)


@jax.jit
def integrate_by_parts(
    t: jnp.ndarray,
    rr: jnp.ndarray,
    delta: jnp.ndarray,
    a: jnp.ndarray,
    integral: jnp.ndarray,
):
    """Helper function for integrate function."""

    def I1():
        return (rr + 1.0, (jnp.exp(delta * t + a) - jnp.exp(a)) / delta)

    def I2():
        return (
            rr + 1.0,
            (jnp.power(t, rr) * jnp.exp(delta * t + a) - rr * integral) / delta,
        )

    return jax.lax.cond(rr == 0.0, I1, I2)


@jax.jit
def integrate2(t: jnp.ndarray, r: jnp.ndarray, delta: jnp.ndarray, a: jnp.ndarray):
    """Helper function for integrate function."""

    def cond_fun(val):
        rr, integral = val
        return rr <= r

    def body_fun(val):
        rr, integral = val
        return integrate_by_parts(t, rr, delta, a, integral)

    _, integral = jax.lax.while_loop(cond_fun, body_fun, (0.0, 0.0))

    return integral


@jax.jit
def integrate1(t: jnp.ndarray, r: jnp.ndarray, delta: jnp.ndarray, a: jnp.ndarray):
    """Helper function for integrate function."""

    return jnp.power(t, r + 1.0) / (r + 1.0) * jnp.exp(a)


@jax.jit
def integrate(t: jnp.ndarray, r: jnp.ndarray, delta: jnp.ndarray, a: jnp.ndarray):
    """This function implements the integral of the form
    $$ \int_0^t s^r \exp(\delta * s + a) ds $$
    using the method of integration by parts.
    r is a non-negative integer and delta is a real number.
    """

    return jax.lax.cond(delta == 0.0, integrate1, integrate2, t, r, delta, a)
