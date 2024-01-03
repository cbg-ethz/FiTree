import numpy as np
import mpmath as mp

from anytree import PreOrderIter

from fitree._trees import Subclone
from ._conditional import _h


def _integrate(
    t: float | np.ndarray, r: float | np.ndarray, delta: float | np.ndarray
) -> float | np.ndarray:
    if delta == 0:
        return mp.power(t, r) / r
    else:
        return _integrate_by_parts(t, r, delta)


def _integrate_by_parts(
    t: float | np.ndarray, r: float | np.ndarray, delta: float | np.ndarray
) -> float | np.ndarray:
    if r == 1:
        return (mp.exp(delta * t) - 1) / delta
    else:
        return (
            mp.power(t, r - 1) * mp.exp(delta * t)
            - (r - 1) * _integrate_by_parts(t, r - 1, delta)
        ) / delta


def _q_tilde(
    v: Subclone, t: float | np.ndarray, C_sampling: int | np.ndarray
) -> float | np.ndarray:
    gpar_v = v.growth_params
    r = gpar_v["r"]
    delta = gpar_v["delta"]

    return _integrate(t, r, delta) / C_sampling


def _g_tilde(
    v: Subclone, t: float | np.ndarray, C_sampling: int | np.ndarray
) -> float | np.ndarray:
    gpar_v = v.growth_params
    gpar_v["r"]
    gpar_v["delta"]

    g_tilde = _q_tilde(v, t, C_sampling)
    for ch in v.children:
        gpar_ch = ch.growth_params
        g_tilde += _h(_g_tilde(ch, t, C_sampling), gpar_ch, gpar_v)

    return g_tilde


def _ccdf_sampling(
    tree: Subclone, t: float | np.ndarray, C_sampling: int | np.ndarray
) -> float | np.ndarray:
    if not tree.is_root:
        raise ValueError("The tree given is not a root node!")

    log_ccdf = 0
    # loop through all nodes except the root in the tree
    # add up -q_tilde * C_tilde for each node
    tree_iter = PreOrderIter(tree)
    next(tree_iter)
    for node in tree_iter:
        log_ccdf -= _q_tilde(node, t, C_sampling) * node.get_C_tilde(t)

    return 1 - mp.exp(log_ccdf)


def _mcdf_sampling(
    tree: Subclone,
    t: float | np.ndarray,
    C_sampling: int | np.ndarray,
    C_0: int | np.ndarray,
    epsilon: float | np.ndarray | mp.mpf = 0.01,
) -> float | np.ndarray:
    if not tree.is_root:
        raise ValueError("The tree given is not a root node!")

    log_mcdf = 0
    for ch in tree.children:
        gpar_ch = ch.growth_params
        if gpar_ch["lambda"] >= 0:
            log_mcdf -= (
                C_0
                * gpar_ch["rho"]
                * mp.log(1 + gpar_ch["phi"] * _g_tilde(ch, t, C_sampling))
            )
        else:
            t_ = t + 1 / gpar_ch["lambda"]
            log_mcdf -= (
                C_0
                * gpar_ch["rho"]
                * t_
                / epsilon
                * mp.log(
                    gpar_ch["phi"]
                    + (1 - gpar_ch["phi"])
                    * mp.exp(-_g_tilde(ch, t_, C_sampling) / t_ * epsilon)
                )
            )

    return 1 - mp.exp(log_mcdf)
