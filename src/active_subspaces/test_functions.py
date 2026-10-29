"""Built-in test problems: five engineering functions and two known-answer factories.

The five engineering functions (borehole, OTL circuit, piston, robot arm and
wing weight) are defined in **physical** coordinates on ``Uniform`` input
spaces with the ranges of Surjanovic and Bingham [1]_. Each comes as a value
function, an analytic gradient function and a ready-made :class:`~active_subspaces.Model`
constant (``BOREHOLE``, ...) with input names, units, descriptions and stored
Monte Carlo reference results (eigenvalues and eigenvectors of the matrix
``C`` in normalized coordinates, from :math:`10^6` samples).

The factories :func:`quadratic` and :func:`ridge` build models with an
**exact** reference result, defined in the normalized coordinates of any
``Uniform`` or ``Gaussian`` space. :func:`builtin_models` returns the nine
built-in models as a :class:`~active_subspaces.ModelRegistry`.

This module is loaded on first use as ``active_subspaces.test_functions``.

References
----------
.. [1] S. Surjanovic, D. Bingham. Virtual Library of Simulation Experiments:
   Test Functions and Datasets. https://www.sfu.ca/~ssurjano/.

Examples
--------
>>> import active_subspaces as ac
>>> registry = ac.test_functions.builtin_models()
>>> len(registry.names())
9
>>> print(", ".join(registry.names()[:3]))
borehole, otl_circuit, piston
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from active_subspaces import _reference_data
from active_subspaces import _validation as _v
from active_subspaces.errors import InputError
from active_subspaces.models import Model, ReferenceResults
from active_subspaces.registry import ModelRegistry
from active_subspaces.results import _make_provenance
from active_subspaces.spaces import Gaussian, InputSpace, Uniform

__all__ = (
    "BOREHOLE",
    "OTL_CIRCUIT",
    "PISTON",
    "ROBOT_ARM",
    "WING_WEIGHT",
    "borehole",
    "borehole_grad",
    "builtin_models",
    "dct_basis",
    "otl_circuit",
    "otl_circuit_grad",
    "piston",
    "piston_grad",
    "quadratic",
    "ridge",
    "robot_arm",
    "robot_arm_grad",
    "wing_weight",
    "wing_weight_grad",
)

_TWO_PI = 2.0 * np.pi


def _inputs(X: Any, m: int) -> NDArray[np.float64]:
    """Return a 2-D float64 view of ``X`` with ``m`` columns (``input.shape`` otherwise).

    No finiteness or support check.
    """
    arr: NDArray[np.float64] = _v.coerce(X, "X")
    if arr.ndim != 2 or arr.shape[1] != m:
        raise InputError(
            f"X: expected shape (k, {m}), got {arr.shape}",
            code="input.shape",
            details={"param": "X", "expected": ["k", m], "got": list(arr.shape)},
        )
    return arr


# ---------------------------------------------------------------------------
# Borehole, uniform variant [D-12, D-21]
# ---------------------------------------------------------------------------


def _borehole_parts(X: Any) -> tuple[NDArray[np.float64], ...]:
    X = _inputs(X, 8)
    rw, r, Tu, Hu, Tl, Hl, L, Kw = (X[:, j] for j in range(8))
    ell = np.log(r / rw)
    D = ell * (1.0 + Tu / Tl) + 2.0 * L * Tu / (rw**2 * Kw)
    return rw, r, Tu, Hu, Tl, Hl, L, Kw, ell, D


def borehole(X: ArrayLike) -> NDArray[np.float64]:
    r"""Evaluate the borehole function: water flow rate through a borehole (m^3/yr).

    Parameters
    ----------
    X : array_like, shape (M, 8)
        Physical inputs, one row per point, in the order ``rw, r, Tu, Hu, Tl,
        Hl, L, Kw`` (units and ranges in ``BOREHOLE.space``). No support
        check is made, so points outside the box are evaluated as given.

    Returns
    -------
    ndarray, shape (M,)
        Flow rates, float64.

    Raises
    ------
    InputError
        ``input.shape`` if `X` is not two-dimensional with 8 columns;
        ``input.dtype`` if `X` is not a real numeric array.

    See Also
    --------
    borehole_grad : The physical gradient.
    BOREHOLE : The function as a ``Model`` on its uniform input space.

    Notes
    -----
    With :math:`\ell = \ln(r / r_w)`,

    .. math::

        D = \ell \Bigl(1 + \frac{T_u}{T_l}\Bigr) + \frac{2 L T_u}{r_w^2 K_w},
        \qquad f = \frac{2 \pi T_u (H_u - H_l)}{D},

    which equals the published form [1]_. This is the uniform variant: every
    input, including :math:`r_w \in [0.05, 0.15]` and
    :math:`r \in [100, 50000]`, is uniform on its range. The 2016 tutorial
    version used a normal :math:`r_w` and a log-normal :math:`r`. Since
    :math:`f` depends on :math:`H_u` and :math:`H_l` only through their
    difference and both ranges have width 120, the normalized gradient is
    orthogonal to :math:`e_4 + e_6`.

    References
    ----------
    .. [1] S. Surjanovic, D. Bingham. Virtual Library of Simulation Experiments:
       Test Functions and Datasets. https://www.sfu.ca/~ssurjano/.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.test_functions.BOREHOLE.space
    >>> x_mid = (space.lower + space.upper) / 2
    >>> print(np.round(ac.test_functions.borehole(x_mid[None, :]), 3))
    [70.873]
    """
    _, _, Tu, Hu, _, Hl, _, _, _, D = _borehole_parts(X)
    return _TWO_PI * Tu * (Hu - Hl) / D


def borehole_grad(X: ArrayLike) -> NDArray[np.float64]:
    r"""Evaluate the gradient of :func:`borehole` with respect to its physical inputs.

    Parameters
    ----------
    X : array_like, shape (M, 8)
        Physical inputs ``rw, r, Tu, Hu, Tl, Hl, L, Kw``; no support check.

    Returns
    -------
    ndarray, shape (M, 8)
        Physical partial derivatives :math:`\partial f / \partial \xi_j`, float64.

    Raises
    ------
    InputError
        ``input.shape`` if `X` is not two-dimensional with 8 columns;
        ``input.dtype`` if `X` is not a real numeric array.

    See Also
    --------
    borehole : The function values.

    Notes
    -----
    With :math:`D` as in :func:`borehole`,
    :math:`\partial_\theta f = -f\, \partial_\theta D / D` for
    :math:`\theta \in \lbrace r_w, r, T_l, L, K_w \rbrace`,
    :math:`\partial_{T_u} f = f\,(1 / T_u - \partial_{T_u} D / D)` and
    :math:`\partial_{H_u} f = -\partial_{H_l} f = 2 \pi T_u / D`.

    References
    ----------
    .. [1] S. Surjanovic, D. Bingham. Virtual Library of Simulation Experiments:
       Test Functions and Datasets. https://www.sfu.ca/~ssurjano/.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.test_functions.BOREHOLE.space
    >>> G = ac.test_functions.borehole_grad(space.sample(4, rng=0))
    >>> G.shape
    (4, 8)
    >>> print(bool(np.allclose(G[:, 3], -G[:, 5])))
    True
    """
    rw, r, Tu, Hu, Tl, Hl, L, Kw, ell, D = _borehole_parts(X)
    f = _TWO_PI * Tu * (Hu - Hl) / D
    a = 1.0 + Tu / Tl
    c = rw**2 * Kw
    dD_rw = -a / rw - 4.0 * L * Tu / (rw**3 * Kw)
    dD_r = a / r
    dD_Tu = ell / Tl + 2.0 * L / c
    dD_Tl = -ell * Tu / Tl**2
    dD_L = 2.0 * Tu / c
    dD_Kw = -2.0 * L * Tu / (c * Kw)
    q = f / D
    dHu = _TWO_PI * Tu / D
    return np.column_stack(
        (
            -q * dD_rw,
            -q * dD_r,
            f * (1.0 / Tu) - q * dD_Tu,
            dHu,
            -q * dD_Tl,
            -dHu,
            -q * dD_L,
            -q * dD_Kw,
        )
    )


# ---------------------------------------------------------------------------
# OTL circuit
# ---------------------------------------------------------------------------


def _otl_parts(X: Any) -> tuple[NDArray[np.float64], ...]:
    X = _inputs(X, 6)
    Rb1, Rb2, Rf, Rc1, Rc2, beta = (X[:, j] for j in range(6))
    B = beta * (Rc2 + 9.0)
    Delta = B + Rf
    Vb1 = 12.0 * Rb2 / (Rb1 + Rb2)
    N = (Vb1 + 0.74) * B + 11.35 * Rf + 0.74 * Rf * B / Rc1
    return Rb1, Rb2, Rf, Rc1, Rc2, beta, B, Delta, Vb1, N


def otl_circuit(X: ArrayLike) -> NDArray[np.float64]:
    r"""Evaluate the OTL circuit function: midpoint voltage of a push-pull circuit (V).

    Parameters
    ----------
    X : array_like, shape (M, 6)
        Physical inputs ``Rb1, Rb2, Rf, Rc1, Rc2, beta`` (units and ranges in
        ``OTL_CIRCUIT.space``); no support check.

    Returns
    -------
    ndarray, shape (M,)
        Midpoint voltages, float64.

    Raises
    ------
    InputError
        ``input.shape`` if `X` is not two-dimensional with 6 columns;
        ``input.dtype`` if `X` is not a real numeric array.

    See Also
    --------
    otl_circuit_grad : The physical gradient.
    OTL_CIRCUIT : The function as a ``Model`` on its uniform input space.

    Notes
    -----
    With :math:`B = \beta (R_{c2} + 9)`, :math:`\Delta = B + R_f` and
    :math:`V_{b1} = 12 R_{b2} / (R_{b1} + R_{b2})`,

    .. math::

        V_m = \frac{(V_{b1} + 0.74) B + 11.35 R_f + 0.74 R_f B / R_{c1}}{\Delta}.

    References
    ----------
    .. [1] S. Surjanovic, D. Bingham. Virtual Library of Simulation Experiments:
       Test Functions and Datasets. https://www.sfu.ca/~ssurjano/.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.test_functions.OTL_CIRCUIT.space
    >>> x_mid = (space.lower + space.upper) / 2
    >>> print(np.round(ac.test_functions.otl_circuit(x_mid[None, :]), 3))
    [5.311]
    """
    *_, Delta, _Vb1, N = _otl_parts(X)
    return N / Delta


def otl_circuit_grad(X: ArrayLike) -> NDArray[np.float64]:
    r"""Evaluate the gradient of :func:`otl_circuit` with respect to its physical inputs.

    Parameters
    ----------
    X : array_like, shape (M, 6)
        Physical inputs ``Rb1, Rb2, Rf, Rc1, Rc2, beta``; no support check.

    Returns
    -------
    ndarray, shape (M, 6)
        Physical partial derivatives, float64.

    Raises
    ------
    InputError
        ``input.shape`` if `X` is not two-dimensional with 6 columns;
        ``input.dtype`` if `X` is not a real numeric array.

    See Also
    --------
    otl_circuit : The function values.

    Notes
    -----
    With :math:`N` the numerator and :math:`\Delta` the denominator of
    :func:`otl_circuit`,
    :math:`\partial_\theta V_m = (\partial_\theta N - V_m\, \partial_\theta \Delta) / \Delta`.

    References
    ----------
    .. [1] S. Surjanovic, D. Bingham. Virtual Library of Simulation Experiments:
       Test Functions and Datasets. https://www.sfu.ca/~ssurjano/.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.test_functions.OTL_CIRCUIT.space
    >>> ac.test_functions.otl_circuit_grad(space.sample(4, rng=0)).shape
    (4, 6)
    """
    Rb1, Rb2, Rf, Rc1, Rc2, beta, B, Delta, Vb1, N = _otl_parts(X)
    Vm = N / Delta
    s = (Rb1 + Rb2) ** 2
    e = Vb1 + 0.74 + 0.74 * Rf / Rc1
    dN = (
        -12.0 * Rb2 * B / s,
        12.0 * Rb1 * B / s,
        11.35 + 0.74 * B / Rc1,
        -0.74 * Rf * B / Rc1**2,
        beta * e,
        (Rc2 + 9.0) * e,
    )
    dDelta: tuple[float | NDArray[np.float64], ...] = (0.0, 0.0, 1.0, 0.0, beta, Rc2 + 9.0)
    return np.column_stack([(dn - Vm * dd) / Delta for dn, dd in zip(dN, dDelta, strict=False)])


# ---------------------------------------------------------------------------
# Piston
# ---------------------------------------------------------------------------


def _piston_parts(X: Any) -> tuple[NDArray[np.float64], ...]:
    X = _inputs(X, 7)
    M, S, V0, k, P0, Ta, T0 = (X[:, j] for j in range(7))
    K = P0 * V0 * Ta / T0
    Ap = P0 * S + 19.62 * M - k * V0 / S
    R = Ap**2 + 4.0 * k * K
    sqR = np.sqrt(R)
    V = S / (2.0 * k) * (sqR - Ap)
    Q = k + S**2 * K / V**2
    C = _TWO_PI * np.sqrt(M / Q)
    return M, S, V0, k, P0, Ta, T0, K, Ap, sqR, V, Q, C


def piston(X: ArrayLike) -> NDArray[np.float64]:
    r"""Evaluate the piston function: cycle time of a piston in a cylinder (s).

    Parameters
    ----------
    X : array_like, shape (M, 7)
        Physical inputs ``M, S, V0, k, P0, Ta, T0`` (units and ranges in
        ``PISTON.space``); no support check.

    Returns
    -------
    ndarray, shape (M,)
        Cycle times, float64.

    Raises
    ------
    InputError
        ``input.shape`` if `X` is not two-dimensional with 7 columns;
        ``input.dtype`` if `X` is not a real numeric array.

    See Also
    --------
    piston_grad : The physical gradient.
    PISTON : The function as a ``Model`` on its uniform input space.

    Notes
    -----
    With :math:`K = P_0 V_0 T_a / T_0`,

    .. math::

        A_p = P_0 S + 19.62 M - \frac{k V_0}{S}, \quad
        V = \frac{S}{2k} \Bigl(\sqrt{A_p^2 + 4 k K} - A_p\Bigr), \quad
        C = 2 \pi \sqrt{\frac{M}{k + S^2 K / V^2}}.

    References
    ----------
    .. [1] S. Surjanovic, D. Bingham. Virtual Library of Simulation Experiments:
       Test Functions and Datasets. https://www.sfu.ca/~ssurjano/.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.test_functions.PISTON.space
    >>> x_mid = (space.lower + space.upper) / 2
    >>> print(np.round(ac.test_functions.piston(x_mid[None, :]), 3))
    [0.464]
    """
    return _piston_parts(X)[-1]


def piston_grad(X: ArrayLike) -> NDArray[np.float64]:
    r"""Evaluate the gradient of :func:`piston` with respect to its physical inputs.

    Parameters
    ----------
    X : array_like, shape (M, 7)
        Physical inputs ``M, S, V0, k, P0, Ta, T0``; no support check.

    Returns
    -------
    ndarray, shape (M, 7)
        Physical partial derivatives, float64.

    Raises
    ------
    InputError
        ``input.shape`` if `X` is not two-dimensional with 7 columns;
        ``input.dtype`` if `X` is not a real numeric array.

    See Also
    --------
    piston : The function values.

    Notes
    -----
    The forward chain rule through the intermediates :math:`K`, :math:`A_p`,
    :math:`R = A_p^2 + 4kK`, :math:`V`, :math:`Q = k + S^2 K / V^2` and
    :math:`C = 2\pi\sqrt{M/Q}` of :func:`piston`, ending with
    :math:`\partial C = C\,(\partial M / (2M) - \partial Q / (2Q))`.

    References
    ----------
    .. [1] S. Surjanovic, D. Bingham. Virtual Library of Simulation Experiments:
       Test Functions and Datasets. https://www.sfu.ca/~ssurjano/.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.test_functions.PISTON.space
    >>> ac.test_functions.piston_grad(space.sample(4, rng=0)).shape
    (4, 7)
    """
    M, S, V0, k, P0, Ta, T0, K, Ap, sqR, V, Q, C = _piston_parts(X)
    n = M.shape[0]
    zero = np.zeros(n)
    one = np.ones(n)
    # columns: M, S, V0, k, P0, Ta, T0
    dM = (one, zero, zero, zero, zero, zero, zero)
    dS = (zero, one, zero, zero, zero, zero, zero)
    dk = (zero, zero, zero, one, zero, zero, zero)
    dAp = (19.62 * one, P0 + k * V0 / S**2, -k / S, -V0 / S, S, zero, zero)
    dK = (zero, zero, K / V0, zero, K / P0, K / Ta, -K / T0)
    cols = []
    for j in range(7):
        dR = 2.0 * Ap * dAp[j] + 4.0 * (K * dk[j] + k * dK[j])
        dV = V * (dS[j] / S - dk[j] / k) + S / (2.0 * k) * (dR / (2.0 * sqR) - dAp[j])
        dQ = dk[j] + S**2 / V**2 * dK[j] + 2.0 * S * K / V**2 * dS[j] - 2.0 * S**2 * K / V**3 * dV
        cols.append(C * (dM[j] / (2.0 * M) - dQ / (2.0 * Q)))
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Robot arm
# ---------------------------------------------------------------------------


def _robot_parts(X: Any) -> tuple[NDArray[np.float64], ...]:
    X = _inputs(X, 8)
    theta = X[:, :4]
    Ls = X[:, 4:]
    phi = np.cumsum(theta, axis=1)
    cphi, sphi = np.cos(phi), np.sin(phi)
    u = np.sum(Ls * cphi, axis=1)
    v = np.sum(Ls * sphi, axis=1)
    return Ls, cphi, sphi, u, v


def robot_arm(X: ArrayLike) -> NDArray[np.float64]:
    r"""Evaluate the robot arm function: distance of the arm's end from the origin.

    Parameters
    ----------
    X : array_like, shape (M, 8)
        Physical inputs ``theta1, ..., theta4`` (rad) and ``L1, ..., L4``
        (ranges in ``ROBOT_ARM.space``); no support check.

    Returns
    -------
    ndarray, shape (M,)
        Distances, in the units of the lengths ``L``, float64.

    Raises
    ------
    InputError
        ``input.shape`` if `X` is not two-dimensional with 8 columns;
        ``input.dtype`` if `X` is not a real numeric array.

    See Also
    --------
    robot_arm_grad : The physical gradient.
    ROBOT_ARM : The function as a ``Model`` on its uniform input space.

    Notes
    -----
    With :math:`\varphi_i = \sum_{j \le i} \theta_j`,
    :math:`u = \sum_{i=1}^4 L_i \cos\varphi_i` and
    :math:`v = \sum_{i=1}^4 L_i \sin\varphi_i`, :math:`f = \sqrt{u^2 + v^2}`.

    References
    ----------
    .. [1] S. Surjanovic, D. Bingham. Virtual Library of Simulation Experiments:
       Test Functions and Datasets. https://www.sfu.ca/~ssurjano/.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> x = np.array([[0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25]])
    >>> print(np.round(ac.test_functions.robot_arm(x), 3))
    [1.]
    """
    _, _, _, u, v = _robot_parts(X)
    return np.sqrt(u**2 + v**2)


def robot_arm_grad(X: ArrayLike) -> NDArray[np.float64]:
    r"""Evaluate the gradient of :func:`robot_arm` with respect to its physical inputs.

    Parameters
    ----------
    X : array_like, shape (M, 8)
        Physical inputs ``theta1, ..., theta4, L1, ..., L4``; no support check.

    Returns
    -------
    ndarray, shape (M, 8)
        Physical partial derivatives, float64. The derivative with respect to
        ``theta1`` is exactly 0. Rows where :math:`u^2 + v^2 = 0` are zero.

    Raises
    ------
    InputError
        ``input.shape`` if `X` is not two-dimensional with 8 columns;
        ``input.dtype`` if `X` is not a real numeric array.

    See Also
    --------
    robot_arm : The function values.

    Notes
    -----
    With :math:`u`, :math:`v` and :math:`\varphi_i` as in :func:`robot_arm`,

    .. math::

        \partial_{L_i} f = \frac{u \cos\varphi_i + v \sin\varphi_i}{f}, \qquad
        \partial_{\theta_j} f = \frac{1}{f} \sum_{i \ge j}
        L_i \bigl(v \cos\varphi_i - u \sin\varphi_i\bigr).

    For :math:`j = 1` the sum is :math:`vu - uv = 0` (the function is
    invariant under rotating the whole arm), and 0 is returned exactly. Where
    :math:`u^2 + v^2 = 0` the function is not differentiable; the zero vector
    is returned, an element of the Clarke generalized gradient.

    References
    ----------
    .. [1] S. Surjanovic, D. Bingham. Virtual Library of Simulation Experiments:
       Test Functions and Datasets. https://www.sfu.ca/~ssurjano/.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.test_functions.ROBOT_ARM.space
    >>> G = ac.test_functions.robot_arm_grad(space.sample(4, rng=0))
    >>> G.shape
    (4, 8)
    >>> print(G[:, 0])
    [0. 0. 0. 0.]
    """
    Ls, cphi, sphi, u, v = _robot_parts(X)
    n = u.shape[0]
    r2 = u**2 + v**2
    ok = r2 > 0
    f = np.sqrt(r2)
    inv = np.zeros(n)
    np.divide(1.0, f, out=inv, where=ok)
    G = np.zeros((n, 8))
    terms = Ls * (v[:, None] * cphi - u[:, None] * sphi)  # L_i (v cos phi_i - u sin phi_i)
    tail = np.cumsum(terms[:, ::-1], axis=1)[:, ::-1]  # sum over i >= j
    G[:, 1:4] = tail[:, 1:4] * inv[:, None]
    G[:, 4:8] = (u[:, None] * cphi + v[:, None] * sphi) * inv[:, None]
    G[~ok] = 0.0
    return G


# ---------------------------------------------------------------------------
# Wing weight
# ---------------------------------------------------------------------------


_WING_EXP = np.array([0.758, 0.0035, 0.6, 0.0, 0.006, 0.04, -0.3, 0.49, 0.49, 0.0])


def _wing_parts(X: Any) -> tuple[NDArray[np.float64], ...]:
    X = _inputs(X, 10)
    Sw, Wfw, A, Lam, q, lam, tc, Nz, Wdg, Wp = (X[:, j] for j in range(10))
    Lr = np.pi * Lam / 180.0
    P = (
        0.036
        * Sw**0.758
        * Wfw**0.0035
        * A**0.6
        * np.cos(Lr) ** -0.9
        * q**0.006
        * lam**0.04
        * (100.0 * tc) ** -0.3
        * (Nz * Wdg) ** 0.49
    )
    return X, Sw, Wp, Lr, P


def wing_weight(X: ArrayLike) -> NDArray[np.float64]:
    r"""Evaluate the wing weight function: weight of a light aircraft wing (lb).

    Parameters
    ----------
    X : array_like, shape (M, 10)
        Physical inputs ``Sw, Wfw, A, Lambda, q, lambda, tc, Nz, Wdg, Wp``
        (units and ranges in ``WING_WEIGHT.space``; the sweep ``Lambda`` is
        in degrees); no support check.

    Returns
    -------
    ndarray, shape (M,)
        Wing weights, float64.

    Raises
    ------
    InputError
        ``input.shape`` if `X` is not two-dimensional with 10 columns;
        ``input.dtype`` if `X` is not a real numeric array.

    See Also
    --------
    wing_weight_grad : The physical gradient.
    WING_WEIGHT : The function as a ``Model`` on its uniform input space.

    Notes
    -----
    With :math:`\Lambda_r = \pi \Lambda / 180`,

    .. math::

        f = 0.036\, S_w^{0.758} W_{fw}^{0.0035}
        \Bigl(\frac{A}{\cos^2 \Lambda_r}\Bigr)^{0.6} q^{0.006} \lambda^{0.04}
        \Bigl(\frac{100\, t_c}{\cos \Lambda_r}\Bigr)^{-0.3} (N_z W_{dg})^{0.49}
        + S_w W_p .

    References
    ----------
    .. [1] S. Surjanovic, D. Bingham. Virtual Library of Simulation Experiments:
       Test Functions and Datasets. https://www.sfu.ca/~ssurjano/.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.test_functions.WING_WEIGHT.space
    >>> x_mid = (space.lower + space.upper) / 2
    >>> print(np.round(ac.test_functions.wing_weight(x_mid[None, :]), 3))
    [267.625]
    """
    _, Sw, Wp, _, P = _wing_parts(X)
    return P + Sw * Wp


def wing_weight_grad(X: ArrayLike) -> NDArray[np.float64]:
    r"""Evaluate the gradient of :func:`wing_weight` with respect to its physical inputs.

    Parameters
    ----------
    X : array_like, shape (M, 10)
        Physical inputs ``Sw, Wfw, A, Lambda, q, lambda, tc, Nz, Wdg, Wp``;
        no support check.

    Returns
    -------
    ndarray, shape (M, 10)
        Physical partial derivatives, float64; the derivative with respect to
        the sweep ``Lambda`` is per degree.

    Raises
    ------
    InputError
        ``input.shape`` if `X` is not two-dimensional with 10 columns;
        ``input.dtype`` if `X` is not a real numeric array.

    See Also
    --------
    wing_weight : The function values.

    Notes
    -----
    With :math:`P = f - S_w W_p` the power-law term of :func:`wing_weight`:
    :math:`\partial_{S_w} f = 0.758 P / S_w + W_p`,
    :math:`\partial_{W_p} f = S_w`,
    :math:`\partial_\Lambda f = 0.9 P \tan\Lambda_r \cdot \pi / 180`, and
    :math:`\partial_\theta f = e_\theta P / \theta` for the other inputs, with
    :math:`e_\theta` the exponent of :math:`\theta` in :math:`P`.

    References
    ----------
    .. [1] S. Surjanovic, D. Bingham. Virtual Library of Simulation Experiments:
       Test Functions and Datasets. https://www.sfu.ca/~ssurjano/.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.test_functions.WING_WEIGHT.space
    >>> ac.test_functions.wing_weight_grad(space.sample(4, rng=0)).shape
    (4, 10)
    """
    X, Sw, Wp, Lr, P = _wing_parts(X)
    G = np.empty(X.shape)
    for j, e in enumerate(_WING_EXP):
        if e != 0.0:
            G[:, j] = e * P / X[:, j]
    G[:, 0] += Wp
    G[:, 3] = 0.9 * P * np.tan(Lr) * (np.pi / 180.0)
    G[:, 9] = Sw
    return G


# ---------------------------------------------------------------------------
# Engineering Model constants (§4.6)
# ---------------------------------------------------------------------------


def _stored_reference(name: str) -> ReferenceResults | None:
    d = _reference_data.REFERENCE_RESULTS.get(name)
    if d is None:
        return None
    rr = ReferenceResults.from_dict(d)
    assert isinstance(rr, ReferenceResults)
    return rr


def _engineering(
    name: str,
    value: Any,
    grad: Any,
    table: tuple[tuple[str, str | None, float, float, str], ...],
    description: str,
    qoi_units: str | None,
    qoi_description: str,
) -> Model:
    names, units, lower, upper, desc = zip(*table, strict=False)
    return Model(
        space=Uniform(
            np.array(lower, dtype=float), np.array(upper, dtype=float), names=names, units=units
        ),
        value=value,
        grad=grad,
        name=name,
        description=description,
        input_descriptions=desc,
        qoi_units=qoi_units,
        qoi_description=qoi_description,
        cost=1e-6,
        batch_size=None,
        references=("SurjanovicBingham",),
        reference_results=_stored_reference(name),
    )


_BOREHOLE_INPUTS = (
    ("rw", "m", 0.05, 0.15, "radius of the borehole"),
    ("r", "m", 100.0, 50000.0, "radius of influence"),
    ("Tu", "m^2/yr", 63070.0, 115600.0, "transmissivity of the upper aquifer"),
    ("Hu", "m", 990.0, 1110.0, "potentiometric head of the upper aquifer"),
    ("Tl", "m^2/yr", 63.1, 116.0, "transmissivity of the lower aquifer"),
    ("Hl", "m", 700.0, 820.0, "potentiometric head of the lower aquifer"),
    ("L", "m", 1120.0, 1680.0, "length of the borehole"),
    ("Kw", "m/yr", 9855.0, 12045.0, "hydraulic conductivity of the borehole"),
)
_OTL_INPUTS = (
    ("Rb1", "kOhm", 50.0, 150.0, "resistance b1"),
    ("Rb2", "kOhm", 25.0, 70.0, "resistance b2"),
    ("Rf", "kOhm", 0.5, 3.0, "resistance f"),
    ("Rc1", "kOhm", 1.2, 2.5, "resistance c1"),
    ("Rc2", "kOhm", 0.25, 1.2, "resistance c2"),
    ("beta", None, 50.0, 300.0, "current gain (a ratio; the source lists amperes)"),
)
_PISTON_INPUTS = (
    ("M", "kg", 30.0, 60.0, "piston weight"),
    ("S", "m^2", 0.005, 0.020, "piston surface area"),
    ("V0", "m^3", 0.002, 0.010, "initial gas volume"),
    ("k", "N/m", 1000.0, 5000.0, "spring coefficient"),
    ("P0", "N/m^2", 90000.0, 110000.0, "atmospheric pressure"),
    ("Ta", "K", 290.0, 296.0, "ambient temperature"),
    ("T0", "K", 340.0, 360.0, "filling gas temperature"),
)
_ROBOT_INPUTS = tuple(
    (f"theta{i}", "rad", 0.0, _TWO_PI, f"angle of arm segment {i} relative to the previous one")
    for i in range(1, 5)
) + tuple((f"L{i}", None, 0.0, 1.0, f"length of arm segment {i}") for i in range(1, 5))
_WING_INPUTS = (
    ("Sw", "ft^2", 150.0, 200.0, "wing area"),
    ("Wfw", "lb", 220.0, 300.0, "weight of fuel in the wing"),
    ("A", None, 6.0, 10.0, "aspect ratio"),
    ("Lambda", "deg", -10.0, 10.0, "quarter-chord sweep"),
    ("q", "lb/ft^2", 16.0, 45.0, "dynamic pressure at cruise"),
    ("lambda", None, 0.5, 1.0, "taper ratio"),
    ("tc", None, 0.08, 0.18, "aerofoil thickness-to-chord ratio"),
    ("Nz", None, 2.5, 6.0, "ultimate load factor"),
    ("Wdg", "lb", 1700.0, 2500.0, "flight design gross weight"),
    ("Wp", "lb/ft^2", 0.025, 0.08, "paint weight"),
)

BOREHOLE = _engineering(
    "borehole",
    borehole,
    borehole_grad,
    _BOREHOLE_INPUTS,
    "Borehole water flow, uniform variant (8 inputs)",
    "m^3/yr",
    "water flow rate",
)
"""The borehole function as a ``Model``: 8 uniform physical inputs, value in m^3/yr.

Its ``space`` is ``Uniform(lower, upper, names=..., units=...)`` with the
ranges of Surjanovic and Bingham; ``value`` is :func:`borehole` and ``grad`` is
:func:`borehole_grad`, both in physical coordinates. ``reference_results`` holds
Monte Carlo estimates of the eigenvalues (with standard errors) and
eigenvectors of ``C`` in normalized coordinates, and the mean and variance of
the value, from :math:`10^6` points drawn with seed 2016.

Examples
--------
>>> import active_subspaces as ac
>>> model = ac.test_functions.BOREHOLE
>>> model.space.dim
8
>>> print(model.space.names[:3])
('rw', 'r', 'Tu')
"""
OTL_CIRCUIT = _engineering(
    "otl_circuit",
    otl_circuit,
    otl_circuit_grad,
    _OTL_INPUTS,
    "Output transformerless push-pull circuit (6 inputs)",
    "V",
    "midpoint voltage",
)
"""The OTL circuit function as a ``Model``: 6 uniform physical inputs, value in V.

Its ``space`` is ``Uniform(lower, upper, names=..., units=...)`` with the
ranges of Surjanovic and Bingham; ``value`` is :func:`otl_circuit` and ``grad`` is
:func:`otl_circuit_grad`, both in physical coordinates. ``reference_results`` holds
Monte Carlo estimates of the eigenvalues (with standard errors) and
eigenvectors of ``C`` in normalized coordinates, and the mean and variance of
the value, from :math:`10^6` points drawn with seed 2016.

Examples
--------
>>> import active_subspaces as ac
>>> model = ac.test_functions.OTL_CIRCUIT
>>> model.space.dim
6
>>> print(model.space.names[:3])
('Rb1', 'Rb2', 'Rf')
"""
PISTON = _engineering(
    "piston",
    piston,
    piston_grad,
    _PISTON_INPUTS,
    "Piston simulation (7 inputs)",
    "s",
    "cycle time",
)
"""The piston function as a ``Model``: 7 uniform physical inputs, value in s.

Its ``space`` is ``Uniform(lower, upper, names=..., units=...)`` with the
ranges of Surjanovic and Bingham; ``value`` is :func:`piston` and ``grad`` is
:func:`piston_grad`, both in physical coordinates. ``reference_results`` holds
Monte Carlo estimates of the eigenvalues (with standard errors) and
eigenvectors of ``C`` in normalized coordinates, and the mean and variance of
the value, from :math:`10^6` points drawn with seed 2016.

Examples
--------
>>> import active_subspaces as ac
>>> model = ac.test_functions.PISTON
>>> model.space.dim
7
>>> print(model.space.names[:3])
('M', 'S', 'V0')
"""
ROBOT_ARM = _engineering(
    "robot_arm",
    robot_arm,
    robot_arm_grad,
    _ROBOT_INPUTS,
    "Robot arm with four segments (8 inputs)",
    None,
    "distance of the arm's end from the origin (units of L)",
)
"""The robot arm function as a ``Model``: 8 uniform physical inputs.

Its ``space`` is ``Uniform(lower, upper, names=..., units=...)`` with the
ranges of Surjanovic and Bingham; ``value`` is :func:`robot_arm` and ``grad`` is
:func:`robot_arm_grad`, both in physical coordinates. ``reference_results`` holds
Monte Carlo estimates of the eigenvalues (with standard errors) and
eigenvectors of ``C`` in normalized coordinates, and the mean and variance of
the value, from :math:`10^6` points drawn with seed 2016.

Examples
--------
>>> import active_subspaces as ac
>>> model = ac.test_functions.ROBOT_ARM
>>> model.space.dim
8
>>> print(model.space.names[:3])
('theta1', 'theta2', 'theta3')
"""
WING_WEIGHT = _engineering(
    "wing_weight",
    wing_weight,
    wing_weight_grad,
    _WING_INPUTS,
    "Light aircraft wing weight (10 inputs)",
    "lb",
    "wing weight",
)
"""The wing weight function as a ``Model``: 10 uniform physical inputs, value in lb.

Its ``space`` is ``Uniform(lower, upper, names=..., units=...)`` with the
ranges of Surjanovic and Bingham; ``value`` is :func:`wing_weight` and ``grad`` is
:func:`wing_weight_grad`, both in physical coordinates. ``reference_results`` holds
Monte Carlo estimates of the eigenvalues (with standard errors) and
eigenvectors of ``C`` in normalized coordinates, and the mean and variance of
the value, from :math:`10^6` points drawn with seed 2016.

Examples
--------
>>> import active_subspaces as ac
>>> model = ac.test_functions.WING_WEIGHT
>>> model.space.dim
10
>>> print(model.space.names[:3])
('Sw', 'Wfw', 'A')
"""


# ---------------------------------------------------------------------------
# Known-answer problems (§3.3, §4.7) [D-122]
# ---------------------------------------------------------------------------


def _bad(message: str, param: str, **details: Any) -> InputError:
    return InputError(message, code="input.parameter", details={"param": param, **details})


def dct_basis(m: int) -> NDArray[np.float64]:
    r"""Return the orthonormal DCT-II basis of dimension `m`.

    It is the default set of eigenvectors of :func:`quadratic`: a dense
    orthogonal matrix, so no eigenvector is aligned with a coordinate axis.

    Parameters
    ----------
    m : int
        Dimension, at least 1.

    Returns
    -------
    ndarray, shape (m, m)
        Orthogonal matrix ``W``, float64; its first column is constant.

    Raises
    ------
    InputError
        ``input.parameter`` if `m` is not an integer at least 1.

    See Also
    --------
    quadratic : Builds a quadratic with these eigenvectors by default.

    Notes
    -----
    .. math::

        W_{jk} = c_k \cos\Bigl(\frac{\pi (2j - 1)(k - 1)}{2m}\Bigr),
        \qquad c_1 = m^{-1/2}, \quad c_k = (2/m)^{1/2} \ (k \ge 2),

    for :math:`j, k = 1, \dots, m`.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> W = ac.test_functions.dct_basis(4)
    >>> print(np.round(W[:, 0], 3))
    [0.5 0.5 0.5 0.5]
    >>> print(bool(np.allclose(W.T @ W, np.eye(4))))
    True
    """
    m = _v.integer(m, "m", lo=1)
    j = np.arange(m)[:, None]
    k = np.arange(m)[None, :]
    W: NDArray[np.float64] = np.cos(np.pi * (2 * j + 1) * k / (2 * m))
    c: NDArray[np.float64] = np.full(m, np.sqrt(2.0 / m))
    c[0] = np.sqrt(1.0 / m)
    return W * c


def _known_space(space: Any, m: int) -> Uniform | Gaussian:
    if space is None:
        return Uniform.reference(m)
    if not isinstance(space, (Uniform, Gaussian)):
        raise _bad(
            f"space: expected a Uniform or Gaussian input space, got {type(space).__name__}",
            "space",
        )
    if space.dim != m:
        raise _bad(
            f"space: expected dimension {m}, got {space.dim}", "space", expected=m, got=space.dim
        )
    return space


def _moments(space: Uniform | Gaussian) -> tuple[float, float]:
    """(sigma^2, mu_4) of one normalized coordinate."""
    return (1.0 / 3.0, 1.0 / 5.0) if isinstance(space, Uniform) else (1.0, 3.0)


def _real_vector(a: Any, name: str) -> NDArray[np.float64]:
    try:
        arr = _v.coerce(a, name)
    except InputError as exc:
        raise _bad(exc.message, name) from exc
    if arr.ndim != 1 or arr.shape[0] < 1:
        raise _bad(f"{name}: expected a non-empty 1-D array, got shape {arr.shape}", name)
    if not np.all(np.isfinite(arr)):
        raise _bad(f"{name}: must be finite", name)
    return np.array(arr, dtype=np.float64, copy=True)


def _check_name(name: Any) -> None:
    if name is not None and not isinstance(name, str):
        raise _bad("name: expected a string or None", "name")


class _Quadratic:
    """Batch callables of ``f(x) = x^T H x / 2`` in normalized coordinates; pickles."""

    __slots__ = ("H", "output", "space")

    def __init__(self, space: Uniform | Gaussian, H: NDArray[np.float64], output: str) -> None:
        self.space, self.H, self.output = space, H, output

    def __getstate__(self) -> tuple[Uniform | Gaussian, NDArray[np.float64], str]:
        return (self.space, self.H, self.output)

    def __setstate__(self, state: tuple[Uniform | Gaussian, NDArray[np.float64], str]) -> None:
        self.space, self.H, self.output = state

    def __call__(self, X: Any) -> Any:
        x = self.space.to_normalized(X)
        Hx = x @ self.H
        if self.output == "value":
            return 0.5 * np.einsum("ij,ij->i", x, Hx)
        g = self.space.grads_to_physical(Hx)
        if self.output == "grad":
            return g
        return 0.5 * np.einsum("ij,ij->i", x, Hx), g


class _Ridge:
    """Batch callables of ``f(x) = g(a^T x)`` in normalized coordinates; pickles."""

    __slots__ = ("a", "output", "profile", "space")

    def __init__(
        self, space: Uniform | Gaussian, a: NDArray[np.float64], profile: str, output: str
    ) -> None:
        self.space, self.a, self.profile, self.output = space, a, profile, output

    def __getstate__(self) -> tuple[Uniform | Gaussian, NDArray[np.float64], str, str]:
        return (self.space, self.a, self.profile, self.output)

    def __setstate__(
        self, state: tuple[Uniform | Gaussian, NDArray[np.float64], str, str]
    ) -> None:
        self.space, self.a, self.profile, self.output = state

    def _g(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.profile == "linear":
            return t.copy()
        if self.profile == "sin":
            return np.sin(t)
        return 0.5 * t**2

    def _dg(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.profile == "linear":
            return np.ones_like(t)
        if self.profile == "sin":
            return np.cos(t)
        return t.copy()

    def __call__(self, X: Any) -> Any:
        x = self.space.to_normalized(X)
        t = x @ self.a
        if self.output == "value":
            return self._g(t)
        g = self.space.grads_to_physical(self._dg(t)[:, None] * self.a[None, :])
        if self.output == "grad":
            return g
        return self._g(t), g


def quadratic(
    eigenvalues: ArrayLike,
    *,
    directions: ArrayLike | None = None,
    space: InputSpace | None = None,
    name: str | None = None,
) -> Model:
    r"""Build a quadratic model whose matrix ``C`` has a prescribed spectrum.

    The quadratic :math:`f(x) = \tfrac12 x^\top H x` is defined in the
    normalized coordinates :math:`x` of `space`, and its active-subspace
    matrix :math:`C` in those coordinates has exactly the given eigenvalues
    and eigenvectors. The model's callables take physical points and return
    physical gradients.

    Parameters
    ----------
    eigenvalues : array_like, shape (m,)
        Eigenvalues :math:`\lambda_1 \ge \dots \ge \lambda_m \ge 0` of
        :math:`C`, finite, with :math:`\lambda_1 > 0`.
    directions : array_like, shape (m, m), optional
        Orthogonal matrix :math:`W` of eigenvectors in normalized coordinates
        (:math:`\max\lvert W^\top W - I\rvert \le 10^{-12}`). Default
        :func:`dct_basis` ``(m)``.
    space : Uniform or Gaussian, optional
        Input space of dimension `m`. Default ``Uniform.reference(m)``, the
        box :math:`[-1, 1]^m`, where physical and normalized coordinates
        coincide.
    name : str, optional
        Name of the model, also recorded in its reference results.

    Returns
    -------
    Model
        With ``value``, ``grad`` and ``value_and_grad`` (batch callables on
        physical points), ``cost=1e-6`` and exact ``reference_results``
        (eigenvalues, eigenvectors `directions`, mean and variance).

    Raises
    ------
    InputError
        ``input.parameter`` if `eigenvalues` is not a finite, non-empty 1-D
        array of non-increasing, non-negative values with a positive first
        entry; if `directions` is not an orthogonal (m, m) matrix; if `space`
        is not a ``Uniform`` or ``Gaussian`` space of dimension `m`; or if
        `name` is not a string.

    See Also
    --------
    ridge : A ridge function with an exact reference result.
    dct_basis : The default eigenvectors.

    Notes
    -----
    Let :math:`\Sigma = \sigma^2 I` be the covariance of the normalized
    inputs, with :math:`\sigma^2 = 1/3` and fourth moment
    :math:`\mu_4 = 1/5` for a uniform space, or :math:`\sigma^2 = 1` and
    :math:`\mu_4 = 3` for a Gaussian one. With
    :math:`h_i = \sqrt{\lambda_i / \sigma^2}`,

    .. math::

        H = W \operatorname{diag}(h) W^\top, \qquad \nabla_x f = H x, \qquad
        C = H \Sigma H = W \operatorname{diag}(\lambda) W^\top.

    The mean of :math:`f` is :math:`\tfrac12 \sigma^2 \operatorname{tr} H`
    and its variance is

    .. math::

        \operatorname{Var} f = \tfrac14 \Bigl[(\mu_4 - 3\sigma^4)
        \sum_{j=1}^m H_{jj}^2 + 2 \sigma^4 \sum_{i=1}^m h_i^2\Bigr].

    The reference eigenvectors are `directions` as given; the sign rule of
    the subspace estimators is not applied to them, so compare them with an
    estimate by span or up to sign.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.0])
    >>> print(model.reference_results.eigenvalues)
    [1.  0.1 0. ]
    >>> X = model.space.sample(5, rng=0)
    >>> print(np.round(ac.evaluate(model, X=X), 3))
    [0.547 0.508 0.169 0.55  0.026]
    """
    lam = _real_vector(eigenvalues, "eigenvalues")
    m = lam.shape[0]
    if np.any(lam < 0) or np.any(np.diff(lam) > 0) or not lam[0] > 0:
        raise _bad(
            "eigenvalues: expected non-negative, non-increasing values with eigenvalues[0] > 0",
            "eigenvalues",
        )
    if directions is None:
        W = dct_basis(m)
    else:
        try:
            W = _v.matrix(directions, "directions", rows=m, cols=m)
        except InputError as exc:
            raise _bad(exc.message, "directions") from exc
        if np.max(np.abs(W.T @ W - np.eye(m))) > 1e-12:
            raise _bad(
                "directions: expected an orthogonal (m, m) matrix (|W^T W - I|_max <= 1e-12)",
                "directions",
            )
    sp = _known_space(space, m)
    _check_name(name)
    sigma2, mu4 = _moments(sp)
    h = np.sqrt(lam / sigma2)
    H = (W * h) @ W.T
    H = 0.5 * (H + H.T)
    H.flags.writeable = False
    mean = 0.5 * sigma2 * float(np.sum(h))
    variance = 0.25 * (
        (mu4 - 3.0 * sigma2**2) * float(np.sum(np.diag(H) ** 2))
        + 2.0 * sigma2**2 * float(np.sum(h**2))
    )
    prov = _make_provenance(
        "test_functions.quadratic",
        {"eigenvalues": lam, "directions": None if directions is None else W, "name": name},
        space=sp,
    )
    ref = ReferenceResults(
        model=name or "<unnamed>",
        exact=True,
        eigenvalues=lam,
        eigenvectors=W,
        eigenvalue_se=None,
        mean=mean,
        mean_se=None,
        variance=max(variance, 0.0),
        n_samples=None,
        seed=None,
        provenance=prov,
    )
    return Model(
        space=sp,
        value=_Quadratic(sp, H, "value"),
        grad=_Quadratic(sp, H, "grad"),
        value_and_grad=_Quadratic(sp, H, "value_and_grad"),
        name=name,
        description=f"quadratic with a prescribed spectrum (m={m})",
        cost=1e-6,
        reference_results=ref,
    )


def ridge(
    direction: ArrayLike,
    *,
    profile: Literal["linear", "sin", "square"] = "sin",
    space: InputSpace | None = None,
    name: str | None = None,
) -> Model:
    r"""Build a ridge function model :math:`f(x) = g(a^\top x)` with an exact reference result.

    The function is defined in the normalized coordinates :math:`x` of
    `space`, with :math:`a = d / \lVert d \rVert` for the given direction
    :math:`d`. Its active-subspace matrix is :math:`C = \lambda_1 a a^\top`,
    so it has a one-dimensional active subspace spanned by :math:`a`. The
    model's callables take physical points and return physical gradients.

    Parameters
    ----------
    direction : array_like, shape (m,)
        Ridge direction :math:`d` in normalized coordinates; finite and
        nonzero. It is normalized to unit length.
    profile : {"sin", "linear", "square"}, default "sin"
        The profile :math:`g(t)`: :math:`\sin t`, :math:`t` or :math:`t^2/2`.
    space : Uniform or Gaussian, optional
        Input space of dimension `m`. Default ``Uniform.reference(m)``, the
        box :math:`[-1, 1]^m`, where physical and normalized coordinates
        coincide.
    name : str, optional
        Name of the model, also recorded in its reference results.

    Returns
    -------
    Model
        With ``value``, ``grad`` and ``value_and_grad`` (batch callables on
        physical points), ``cost=1e-6`` and exact ``reference_results``
        (eigenvalues :math:`(\lambda_1, 0, \dots, 0)`, eigenvectors whose
        first column is :math:`a`, mean and variance).

    Raises
    ------
    InputError
        ``input.parameter`` if `direction` is not a finite, nonzero, non-empty
        1-D array; if `profile` is not one of the three names; if `space` is
        not a ``Uniform`` or ``Gaussian`` space of dimension `m`; or if `name`
        is not a string.

    See Also
    --------
    quadratic : A quadratic with a prescribed spectrum.

    Notes
    -----
    :math:`\nabla_x f = g'(a^\top x)\, a` and
    :math:`\lambda_1 = \mathbb{E}[g'(a^\top x)^2]`. With
    :math:`\sigma^2 = 1/3`, :math:`\mu_4 = 1/5` (uniform) or
    :math:`\sigma^2 = 1`, :math:`\mu_4 = 3` (Gaussian), and the
    characteristic function of :math:`t = a^\top x`,
    :math:`\phi(s) = \prod_{j=1}^m \sin(s a_j) / (s a_j)` (uniform; a
    factor is 1 when :math:`a_j = 0`) or :math:`\phi(s) = e^{-s^2/2}`
    (Gaussian):

    - ``"linear"``: :math:`\lambda_1 = 1`, mean 0, variance :math:`\sigma^2`;
    - ``"sin"``: :math:`\lambda_1 = (1 + \phi(2))/2`, mean 0, variance
      :math:`(1 - \phi(2))/2`;
    - ``"square"``: :math:`\lambda_1 = \sigma^2`, mean :math:`\sigma^2/2`,
      variance :math:`\tfrac14 [(\mu_4 - 3\sigma^4) \sum_j a_j^4 + 2\sigma^4]`.

    The reference eigenvectors are the columns of the Householder reflector
    :math:`P = I - 2 v v^\top / (v^\top v)` with :math:`v = a - e_1`
    (:math:`P = I` if :math:`a = e_1`), so the first column is :math:`a`;
    only its span is determined by :math:`f`.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> model = ac.test_functions.ridge([1.0, 2.0], profile="sin")
    >>> ref = model.reference_results
    >>> print(np.round(ref.eigenvalues, 4))
    [0.7379 0.    ]
    >>> print(np.round(ref.eigenvectors[:, 0], 4))
    [0.4472 0.8944]
    """
    d = _real_vector(direction, "direction")
    m = d.shape[0]
    norm = float(np.linalg.norm(d))
    if norm == 0.0:
        raise _bad("direction: must be nonzero", "direction")
    a = d / norm
    a.flags.writeable = False
    _v.choice(profile, "profile", ("linear", "sin", "square"))
    sp = _known_space(space, m)
    _check_name(name)
    sigma2, mu4 = _moments(sp)
    if isinstance(sp, Uniform):
        phi2 = float(np.prod(np.sinc(2.0 * a / np.pi)))
    else:
        phi2 = float(np.exp(-2.0))
    if profile == "linear":
        lam1, mean, variance = 1.0, 0.0, sigma2
    elif profile == "sin":
        lam1, mean, variance = 0.5 * (1.0 + phi2), 0.0, 0.5 * (1.0 - phi2)
    else:
        lam1, mean = sigma2, sigma2 / 2.0
        variance = 0.25 * ((mu4 - 3.0 * sigma2**2) * float(np.sum(a**4)) + 2.0 * sigma2**2)
    e1 = np.zeros(m)
    e1[0] = 1.0
    v = a - e1
    vv = float(v @ v)
    P = np.eye(m) if vv == 0.0 else np.eye(m) - 2.0 * np.outer(v, v) / vv
    eig = np.zeros(m)
    eig[0] = lam1
    prov = _make_provenance(
        "test_functions.ridge", {"direction": d, "profile": profile, "name": name}, space=sp
    )
    ref = ReferenceResults(
        model=name or "<unnamed>",
        exact=True,
        eigenvalues=eig,
        eigenvectors=P,
        eigenvalue_se=None,
        mean=mean,
        mean_se=None,
        variance=max(variance, 0.0),
        n_samples=None,
        seed=None,
        provenance=prov,
    )
    return Model(
        space=sp,
        value=_Ridge(sp, a, profile, "value"),
        grad=_Ridge(sp, a, profile, "grad"),
        value_and_grad=_Ridge(sp, a, profile, "value_and_grad"),
        name=name,
        description=f"ridge function g(a^T x) with g = {profile} (m={m})",
        cost=1e-6,
        reference_results=ref,
    )


# ---------------------------------------------------------------------------
# The built-in registry (§4.5-4.7)
# ---------------------------------------------------------------------------

_LAM10 = np.array([1.0, 1e-1] + [1e-4 * 2.0 ** -(j - 3) for j in range(3, 11)])

QUADRATIC_UNIFORM_10 = quadratic(_LAM10, space=Uniform.reference(10), name="quadratic_uniform_10")
QUADRATIC_GAUSSIAN_10 = quadratic(
    _LAM10, space=Gaussian.reference(10), name="quadratic_gaussian_10"
)
RIDGE_SIN_UNIFORM_10 = ridge(
    np.arange(1, 11), profile="sin", space=Uniform.reference(10), name="ridge_sin_uniform_10"
)
RIDGE_SIN_GAUSSIAN_10 = ridge(
    np.arange(1, 11), profile="sin", space=Gaussian.reference(10), name="ridge_sin_gaussian_10"
)

_BUILTINS = (
    BOREHOLE,
    OTL_CIRCUIT,
    PISTON,
    ROBOT_ARM,
    WING_WEIGHT,
    QUADRATIC_UNIFORM_10,
    QUADRATIC_GAUSSIAN_10,
    RIDGE_SIN_UNIFORM_10,
    RIDGE_SIN_GAUSSIAN_10,
)


def builtin_models() -> ModelRegistry:
    r"""Return a new registry of the nine built-in models.

    The registry holds the five engineering models (``borehole``,
    ``otl_circuit``, ``piston``, ``robot_arm``, ``wing_weight``) and four
    known-answer models with :math:`m = 10`: ``quadratic_uniform_10`` and
    ``quadratic_gaussian_10`` (:func:`quadratic` on the reference uniform and
    Gaussian spaces) and ``ridge_sin_uniform_10`` and
    ``ridge_sin_gaussian_10`` (:func:`ridge` with direction
    :math:`(1, 2, \dots, 10)` and the ``"sin"`` profile).

    Returns
    -------
    ModelRegistry
        A new registry on each call, holding the same ``Model`` objects each
        time. Use ``registry.with_models(...)`` to extend it.

    See Also
    --------
    active_subspaces.ModelRegistry : The registry type.

    Notes
    -----
    The quadratics have eigenvalues :math:`\lambda_1 = 1`,
    :math:`\lambda_2 = 10^{-1}` and
    :math:`\lambda_j = 10^{-4} \cdot 2^{-(j-3)}` for :math:`j = 3, \dots, 10`,
    with the default :func:`dct_basis` eigenvectors. The largest gap between
    eigenvalues is after :math:`\lambda_1` and the largest gap between their
    logarithms is after :math:`\lambda_2`.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> registry = ac.test_functions.builtin_models()
    >>> print(registry.names()[-2:])
    ('robot_arm', 'wing_weight')
    """
    return ModelRegistry(_BUILTINS)
