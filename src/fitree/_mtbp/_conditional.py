import numpy as np
import mpmath as mp

from fitree._trees import Subclone


def _h(
    theta: float | np.ndarray | mp.mpf, gpar_v: dict, gpar_pa_v: dict
) -> float | np.ndarray | mp.mpf:
    if gpar_pa_v["delta"] > gpar_v["lambda"]:
        return gpar_v["nu"] * theta / (gpar_pa_v["delta"] - gpar_v["lambda"])

    elif gpar_pa_v["delta"] == gpar_v["lambda"]:
        return gpar_v["nu"] * theta / gpar_pa_v["r"]

    else:
        if gpar_v["gamma"] == 0:
            h = -gpar_v["rho"] * mp.fac(gpar_pa_v["r"] - 1)
            h /= mp.power(gpar_v["lambda"], gpar_pa_v["r"] - 1)
            h *= mp.polylog(gpar_pa_v["r"], -gpar_v["phi"] * theta)

            return h

        else:
            h = gpar_v["rho"] * mp.power(gpar_v["phi"], gpar_v["gamma"])
            h *= mp.pi / mp.sin(mp.pi * gpar_v["gamma"])
            h *= mp.fac(gpar_pa_v["r"] - 1)
            h /= mp.power(gpar_v["lambda"], gpar_pa_v["r"] - 1)
            h *= mp.power(mp.log(theta * gpar_v["phi"]), gpar_v["r"] - 1)
            h /= mp.fac(gpar_v["r"] - 1)
            h *= mp.power(theta, gpar_v["gamma"])

            return h


def _ccdf(
    v: Subclone,
    t: float | np.ndarray,
    C_tilde_v: int | float | np.ndarray | None = None,
) -> float | np.ndarray | mp.mpf:
    if C_tilde_v is None:
        C_tilde_v = v.get_C_tilde(t)

    C_tilde_pa_v = v.parent.get_C_tilde(t)

    gpar_v = v.growth_params
    gpar_pa_v = v.parent.growth_params

    def lp_func(theta):
        return mp.exp(-_h(theta, gpar_v, gpar_pa_v) * C_tilde_pa_v) / theta

    return mp.invertlaplace(lp_func, C_tilde_v)


def _cpdf(
    v: Subclone,
    t: float | np.ndarray,
    C_tilde_v: int | float | np.ndarray | None = None,
) -> float | np.ndarray | mp.mpf:
    if C_tilde_v is None:
        C_tilde_v = v.get_C_tilde(t)

    C_tilde_pa_v = v.parent.get_C_tilde(t)

    gpar_v = v.growth_params
    gpar_pa_v = v.parent.growth_params

    def lp_func(theta):
        return mp.exp(-_h(theta, gpar_v, gpar_pa_v) * C_tilde_pa_v)

    return mp.invertlaplace(lp_func, C_tilde_v)
