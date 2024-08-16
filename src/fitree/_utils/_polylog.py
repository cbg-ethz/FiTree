import jax
import jax.numpy as jnp
import jax.scipy.special as jss
from jax import config

config.update("jax_enable_x64", True)


@jax.jit
def altzeta(n):
    return jnp.where(n == 0.0, 0.5, (1 - 2.0 ** (1 - n)) * jss.zeta(n, q=1))


@jax.jit
def polylog_jax(n, z):
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
