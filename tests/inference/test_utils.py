import mpmath as mp
import jax.numpy as jnp

from fitree._inference._utils import polylog, integrate


def test_polylog():

	z = -100000

	assert jnp.allclose(float(polylog(1, z)), float(mp.polylog(1, z)), atol=1e-2)
	assert jnp.allclose(float(polylog(2, z)), float(mp.polylog(2, z)), atol=1e-2)
	assert jnp.allclose(float(polylog(3, z)), float(mp.polylog(3, z)), atol=1e-2)
	assert jnp.allclose(float(polylog(4, z)), float(mp.polylog(4, z)), atol=1e-2)


def test_integrate():

    test_cases = [
        {"t": 1.0, "r": 0.0, "delta": 0.0, "expected": 1.0},
        {"t": 1.0, "r": 1.0, "delta": 0.0, "expected": 0.5},
        {"t": 2.0, "r": 1.0, "delta": 0.0, "expected": 2.0},
        {"t": 1.0, "r": 2.0, "delta": 0.0, "expected": 1/3},
        {"t": 2.0, "r": 2.0, "delta": 0.0, "expected": 8/3},
        
        # Exponential cases (delta != 0)
        {"t": 1.0, "r": 0.0, "delta": 1.0, "expected": jnp.exp(1.0) - 1.0},
        {"t": 2.0, "r": 0.0, "delta": 1.0, "expected": jnp.exp(2.0) - 1.0},
        {"t": 1.0, "r": 1.0, "delta": 1.0, "expected": 1.0},
        {"t": 2.0, "r": 1.0, "delta": 1.0, "expected": jnp.exp(2.0) + 1.0},
        {"t": 1.0, "r": 2.0, "delta": 0.1, "expected": 0.359362},
        {"t": 2.0, "r": 3.0, "delta": -0.5, "expected": 1.82286},
    ]

    for i, test in enumerate(test_cases):
        t = jnp.array(test["t"])
        r = jnp.array(test["r"])
        delta = jnp.array(test["delta"])
        expected = test["expected"]

        result = integrate(t, r, delta)

        assert jnp.allclose(result, expected, atol=1e-5), f"Test case {i + 1} failed: expected {expected}, got {result}"
