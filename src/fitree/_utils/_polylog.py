import jax
import jax.numpy as jnp
import jax.scipy.special as jss


@jax.jit
def complex_log(z):
    return jnp.log(jnp.abs(z)) + 1j * jnp.angle(z)


@jax.jit
def polylog_series(s, z, eps=jnp.finfo(float).eps):
    tol = eps

    def body_fun(carry):
        l, zk, k, term = carry
        l += term
        zk *= z
        k += 1
        term = zk / jnp.power(k, s)
        return l, zk, k, term

    def cond_fun(carry):
        _, _, _, term = carry
        return jnp.abs(term) >= tol

    initial_term = z
    initial_carry = (0.0, z, 1, initial_term)
    l, _, _, _ = jax.lax.while_loop(cond_fun, body_fun, initial_carry)
    return l


@jax.jit
def bernoulli(n):
    def small_n_case(n):
        return jnp.select([n == 0, n == 1, n == 2, n == 3], [1.0, -0.5, 1 / 6, 0.0])

    def bernoulli_term_even(n):
        k = jnp.arange(2, 50)
        q1 = 1.0
        q2 = 0.0
        m = 4.0
        initial_carry = (q1, q2, m, n)

        def body_fun(carry):
            q1, q2, m, n = carry
            q1 *= -(m - 1) * m / 4.0 / jnp.pi**2.0
            q2 = jnp.sum(k**-m)
            m += 2.0
            return (q1, q2, m, n)

        def cond_fun(carry):
            _, _, m, n = carry
            return m < n + 1

        q1, q2, _, _ = jax.lax.while_loop(cond_fun, body_fun, initial_carry)
        q1 *= 1.0 / jnp.pi**2

        return q1 * (1 + q2)

    def bernoulli_term(n):
        return jax.lax.cond(n % 2 == 1, lambda n: 0.0, bernoulli_term_even, n)

    b = jax.lax.cond(n < 4, small_n_case, bernoulli_term, n)

    return b


@jax.jit
def bernpoly(n, z):
    def small_n_case(n, z):
        res = (
            jnp.select(
                [n == 0, n == 1, n == 2, n == 3],
                [z**0, z - 0.5, (6 * z * (z - 1) + 1) / 6, z * (z * (z - 1.5) + 0.5)],
            )
            + 0.0j
        )
        return res

    def special_z_case(n, z):
        bn = bernoulli(n) + 0.0j
        res = (
            jnp.select(
                [z == 0, z == 1, z == 0.5, jnp.isinf(z), jnp.isnan(z)],
                [bn, bn, (jnp.power(2, 1 - n) - 1) * bn, z**n, z],
            )
            + 0.0j
        )
        return res

    def check_special_cond(z):
        return (z == 0) | (z == 1) | (z == 0.5) | jnp.isinf(z) | jnp.isnan(z)

    def general_case(n, z):
        def term_1(n, z):
            t = 1.0 + 0.0j
            s = t
            r = 1 / z
            t = t * n * r
            s -= t / 2
            k = 2.0

            def body_fun(carry):
                t, k, s = carry
                t *= (n + 1 - k) / k * (1 / z)
                s = jnp.where(k % 2 == 0, s + t * bernoulli(k), s)
                return t, k + 1, s

            def cond_fun(carry):
                _, k, _ = carry
                return k <= n

            _, _, s = jax.lax.while_loop(cond_fun, body_fun, (t, k, s))
            return s * z**n

        def term_2(n, z):
            s = bernoulli(n) + 0.0j
            t = 1.0 + 0.0j
            k = 1.0

            def body_fun(carry):
                t, k, s = carry
                t *= (n + 1 - k) / k * z
                m = n - k
                s = jnp.where(m % 2 == 0, s + t * bernoulli(m), s)
                return t, k + 1, s

            def cond_fun(carry):
                _, k, _ = carry
                return k < n - 1

            t, _, s = jax.lax.while_loop(cond_fun, body_fun, (t, k, s))
            t = t * 2 / (n - 1) * z
            s -= t / 2
            t = t / n * z
            s += t

            return s

        return jax.lax.cond(jnp.abs(z) > 2, term_1, term_2, n, z)

    def large_n_case(n, z):
        return jax.lax.cond(check_special_cond(z), special_z_case, general_case, n, z)

    return jax.lax.cond(n < 4, small_n_case, large_n_case, n, z)


@jax.jit
def polylog_continuation(n, z):
    twopij = 2j * jnp.pi
    a = -jnp.power(twopij, n) / jss.gamma(n + 1) * bernpoly(n, complex_log(z) / twopij)

    def check_cond_1(z):
        return jnp.isreal(z) & (z < 0)

    def check_cond_2(z):
        return jnp.imag(z) < 0 | ((jnp.imag(z) == 0) & (jnp.real(z) >= 1))

    a = jax.lax.cond(check_cond_1(z), lambda a: jnp.real(a) + 0.0j, lambda a: a, a)

    a = jax.lax.cond(
        check_cond_2(z),
        lambda a: a - twopij * jnp.power(complex_log(z), n - 1) / jss.gamma(n),
        lambda a: a,
        a,
    )

    return jax.lax.cond(n < 0, lambda z: z * 0.0 + 0.0j, lambda z: a, z)


@jax.jit
def harmonic(n):
    def body_fun(i, val):
        return val + 1.0 / (i + 1)

    return jax.lax.fori_loop(0, n, body_fun, 0.0)


# @jax.jit
# def riemann_zeta(s):
#     return jss.zeta(s, q=1)


@jax.jit
def riemann_zeta(n):
    def zeta_pos(n):
        return jss.zeta(n, q=1)

    def zeta_neg(n):
        return (-1) ** (-n) * bernoulli(-n + 1) / (-n + 1)

    return jax.lax.cond(
        n > 0, zeta_pos, lambda n: jax.lax.cond(n == 0, lambda n: -0.5, zeta_neg, n), n
    )


@jax.jit
def altzeta(n):
    return (1 - 2 ** (1 - n)) * riemann_zeta(n)


@jax.jit
def polylog_unitcircle(n, z, eps=jnp.finfo(float).eps):
    tol = eps

    def body_fun(carry):
        l, logz, logmz, m, term = carry
        l += term
        logmz *= logz
        m += 1
        term = jnp.where(
            n - m != 1, riemann_zeta(n - m) * logmz / jss.gamma(m + 1), 0.0
        )
        return (l, logz, logmz, m, term)

    def cond_fun(carry):
        _, _, _, _, term = carry
        return (term == 0.0) | (jnp.abs(term) >= tol)

    l = 0.0 + 0.0j
    logz = complex_log(z)
    logmz = 1.0 + 0.0j
    term = jnp.where(n != 1, riemann_zeta(n), 0.0) + 0.0j

    initial_carry = (l, logz, logmz, 0.0, term)
    l, _, _, _, _ = jax.lax.while_loop(cond_fun, body_fun, initial_carry)
    l += (
        jnp.power(complex_log(z), n - 1)
        / jss.gamma(n)
        * (harmonic(n - 1) - complex_log(-complex_log(z)))
    )

    return l


@jax.jit
def polylog_jax(n, z):
    def check_special_cases(n, z):
        return (z == 1) | (z == -1) | (n == 0) | (n == 1) | (n == -1)

    def special_cases(n, z):
        res = jnp.select(
            [z == 1, z == -1, n == 0, n == 1, n == -1],
            [
                riemann_zeta(n),
                altzeta(n),
                z / (1 - z),
                -complex_log(1 - z),
                z / (1 - z) ** 2,
            ],
        )
        return jnp.complex64(res)

    def case_1(n, z):
        res = polylog_series(n, z)
        return jnp.complex64(res)

    def case_2(n, z):
        res = (-1) ** (n + 1) * polylog_series(n, 1 / z) + polylog_continuation(n, z)
        return jnp.complex64(res)

    def case_3(n, z):
        res = polylog_unitcircle(n, z)
        return jnp.complex64(res)

    def general_cases(n, z):
        return jax.lax.cond(
            jnp.abs(z) <= 0.75,
            case_1,
            lambda n, z: jax.lax.cond(jnp.abs(z) >= 1.4, case_2, case_3, n, z),
            n,
            z,
        )

    return jax.lax.cond(check_special_cases(n, z), special_cases, general_cases, n, z)
