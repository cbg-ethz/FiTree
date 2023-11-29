import numpy as np
import mpmath as mp

from typing import Any

from fitree._trees import Subclone
from ._conditional import _h


def _g(
    theta: float | np.ndarray | mp.mpf,
    v: Subclone | Any,
    C_0: int | np.ndarray,
    t: float | np.ndarray,
    epsilon: float | np.ndarray | mp.mpf = 0.01,
) -> float | np.ndarray | mp.mpf:
    gpar_v = v.growth_params

    if v.parent.is_root:
        phi = gpar_v["phi"]
        rho = gpar_v["rho"]

        if gpar_v["lambda"] < 0:
            return mp.power(
                phi + (1 - phi) * mp.exp(-theta * epsilon / t), -rho * C_0 * t / epsilon
            )
        else:
            return mp.power(1 + phi * theta, -rho * C_0)

    else:
        gpar_pa_v = v.parent.growth_params
        h = _h(theta, gpar_v, gpar_pa_v)
        return _g(h, v.parent, C_0, t, epsilon)


def _mcdf(
    v: Subclone,
    t: float | np.ndarray,
    C_0: int | np.ndarray,
    C_tilde_v: int | float | np.ndarray | None = None,
) -> float | np.ndarray | mp.mpf:
    if C_tilde_v is None:
        C_tilde_v = v.get_C_tilde(t)

    def lp_func(theta):
        return _g(theta, v, C_0, t) / theta

    return mp.invertlaplace(lp_func, C_tilde_v)


def _mpdf(
    v: Subclone,
    t: float | np.ndarray,
    C_0: int | np.ndarray,
    C_tilde_v: int | float | np.ndarray | None = None,
) -> float | np.ndarray | mp.mpf:
    if C_tilde_v is None:
        C_tilde_v = v.get_C_tilde(t)

    def lp_func(theta):
        return _g(theta, v, C_0, t)

    return mp.invertlaplace(lp_func, C_tilde_v)
