import jax
import jax.numpy as jnp
from jax import jit
from functools import partial


@partial(jit, static_argnums=(0,))
def ilp_jax(f, t, dps=20):
    """
    Compute the inverse Laplace transform of a function f at time t using the Cohen algorithm.
    """

    # set parameters
    dps *= 1.74
    degree = int(dps * 1.31)
    M = degree + 1
    alpha = 2 / 3 * (dps + jnp.log(10) + jnp.log(2 * t))
    a_t = alpha / (2 * t)
    p_t = jnp.pi * 1j / t
    tmp = jnp.arange(M)
    p = a_t + tmp * p_t

    # apply f to all p values using jax.vmap
    fp = jax.vmap(f)(p)

    # calculate the time-domain solution
    n = degree
    A = fp.real
    d = jnp.power(3 + jnp.sqrt(8), n)
    d = (d + 1 / d) / 2
    b = -1
    c = -d
    s = 0

    for k in range(n):
        c = b - c
        s += c * A[k + 1]
        b = 2 * (k + n) * (k - n) * b / ((2 * k + 1) * (k + 1))

    result = jnp.exp(alpha / 2) / t * (A[0] / 2 - s / d)

    return result
