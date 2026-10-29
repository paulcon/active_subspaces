r"""Integration with active and inactive variables.

Write a normalized input as :math:`x = W_1 y + W_2 z`, with active variables
:math:`y = W_1^\top x` and inactive variables :math:`z = W_2^\top x`. The integral
:math:`\mu = \int f(x) \rho(x)\, dx` is estimated with a quadrature rule
:math:`\lbrace (y_k, \omega_k) \rbrace` on the active variables and
:math:`N_{MC}` draws of the inactive variables per node [1]_:

.. math::

    \hat G_k = \frac{1}{N_{MC}} \sum_{j=1}^{N_{MC}} f(W_1 y_k + W_2 z_{kj}), \qquad
    \hat\mu = \sum_{k=1}^{N_q} \omega_k \hat G_k .

The rule is a tensor Gauss-Hermite rule for a Gaussian space, the Gauss rule of
the box-spline density of :math:`w^\top x` for a uniform space with
:math:`n = 1`, and a Delaunay-centroid rule with Monte Carlo cell masses for a
uniform space with :math:`n \ge 2`.

Errors are reported as standard errors, never as intervals.
``inactive_std_error`` covers the inactive Monte Carlo only (with Geyer's
autocorrelation time for hit-and-run draws [2]_); ``total_std_error`` comes from
:math:`R \ge 2` independent replicates and covers every random component, but
not the deterministic bias of the rule in :math:`y`. Everything is computed in
normalized coordinates; models are evaluated at the physical points.

References
----------
.. [1] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
   Reduction in Parameter Studies.* SIAM, 2015.
.. [2] C. J. Geyer. Practical Markov chain Monte Carlo. *Statist. Sci.*
   7(4):473-483, 1992.
"""

from __future__ import annotations

import concurrent.futures
import math
from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import numpy as np
import scipy.spatial
from numpy.typing import ArrayLike, NDArray

from active_subspaces import _random
from active_subspaces import _validation as _v
from active_subspaces._quadrature import box_spline_gauss, gauss_hermite, tensor_rule
from active_subspaces.designs import active_design
from active_subspaces.errors import (
    AccuracyWarning,
    InputError,
    NumericalError,
    UnsupportedError,
    _warn,
)
from active_subspaces.geometry import ActiveDomain, _to_physical, inverse_map
from active_subspaces.models import Model, as_model, evaluate
from active_subspaces.results import (
    QoIResult,
    Result,
    _invariant,
    _make_provenance,
    _register_result,
)
from active_subspaces.solvers import SolverBackend, get_backend
from active_subspaces.surrogates import PolynomialSurrogate, RBFSurrogate, Surrogate

__all__ = (
    "ActiveQuadratureRule",
    "IntegrationDesign",
    "IntegrationResult",
    "active_quadrature_rule",
    "estimate_integral",
    "integrate",
    "integrate_surrogate",
    "integration_design",
)

_RULE_KINDS: dict[str, Literal["gauss_hermite", "box_spline_gauss", "delaunay_centroid"]] = {
    "unbounded": "gauss_hermite",
    "interval": "box_spline_gauss",
    "zonotope": "delaunay_centroid",
}
_SAMPLERS = ("none", "gaussian", "exact_interval", "hit_and_run")
_SUM_TOL = 1e-12
#: Rows of the N_X x m weight-sample draw generated per chunk (the stream equals one draw).
_WEIGHT_CHUNK = 1 << 16
#: Relative tolerance (times the vertex diameter) of the brute-force simplex retry [D-105].
_RETRY_TOL = 1e-10


# ---------------------------------------------------------------------------
# Result types (§4.2)
# ---------------------------------------------------------------------------


@_register_result
class ActiveQuadratureRule(Result):
    r"""A quadrature rule on the active variables.

    The rule :math:`\lbrace (y_k, \omega_k) \rbrace_{k=1}^{N_q}` approximates
    integrals against the marginal density :math:`\pi(y)` of the active
    variables :math:`y = W_1^\top x`. Rules are built by
    :func:`active_quadrature_rule`; the constructor is used when a saved result
    is loaded.

    Parameters
    ----------
    domain : ActiveDomain
        The active domain the rule lives on.
    kind : {"gauss_hermite", "box_spline_gauss", "delaunay_centroid"}
        The rule: tensor Gauss-Hermite (Gaussian space), the Gauss rule of the
        box-spline density (uniform space, ``n = 1``) or the Delaunay-centroid
        rule (uniform space, ``n >= 2``).
    nodes : ndarray, shape (N_q, n)
        Nodes :math:`y_k` in normalized active coordinates, float64.
    weights : ndarray, shape (N_q,)
        Weights :math:`\omega_k`, nonnegative and summing to 1.
    n_requested : tuple of int
        ``n_nodes`` as given: ``(N,)`` or one count per dimension.
    degree : int or None
        Polynomial degree integrated exactly against :math:`\pi`;
        ``None`` for the Delaunay-centroid rule.
    design : ndarray, shape (N_d, n), or None
        The interior maximin points in normalized active coordinates
        (Delaunay-centroid rule only).
    n_weight_samples : int or None
        :math:`N_X`, the Monte Carlo samples of the cell masses (Delaunay-centroid
        rule only).
    weight_deficit : float
        Fraction :math:`d` of weight samples that fell outside the triangulation;
        0.0 unless the vertex set of the domain was incomplete.
    provenance : Provenance
        How the rule was computed.

    See Also
    --------
    active_quadrature_rule : Build a rule.
    integrate_surrogate : Integrate a function of the active variables with a rule.

    Notes
    -----
    For a Gaussian space the nodes are a tensor product of :math:`k_i`-point
    probabilists' Gauss-Hermite rules, exact for polynomials of degree
    :math:`\le 2 k_i - 1` in :math:`y_i`. For a uniform space with :math:`n = 1`
    the rule is the :math:`N_q`-point Gauss rule of the density of
    :math:`y = w^\top x`, :math:`x \sim U[-1, 1]^m` (a box spline), exact to
    degree :math:`2 N_q - 1`. For :math:`n \ge 2` the nodes are the centroids
    :math:`c_k` of the Delaunay simplices :math:`T_k` of the zonotope vertices and
    a maximin design, with Monte Carlo cell masses

    .. math::

        \omega_k = \frac{\#\lbrace i : W_1^\top x_i \in T_k \rbrace}{N_X (1 - d)},
        \qquad x_i \sim U[-1, 1]^m .

    The Delaunay-centroid rule integrates constants exactly; its error is
    :math:`O(h)` in the largest cell diameter for Lipschitz integrands.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
    >>> sub = ac.subspace_from_basis([[1.0], [0.5]], space=space)
    >>> rule = ac.active_quadrature_rule(ac.active_domain(sub), n_nodes=4)
    >>> print(rule.kind, rule.n_nodes, rule.degree)
    box_spline_gauss 4 7
    >>> print(np.round(rule.weights, 3))
    [0.099 0.401 0.401 0.099]
    """

    TYPE = "ActiveQuadratureRule"
    SCHEMA_VERSION = 1

    domain: ActiveDomain
    kind: Literal["gauss_hermite", "box_spline_gauss", "delaunay_centroid"]
    nodes: NDArray[np.float64]
    weights: NDArray[np.float64]
    n_requested: tuple[int, ...]
    degree: int | None
    design: NDArray[np.float64] | None
    n_weight_samples: int | None
    weight_deficit: float

    @property
    def n_nodes(self) -> int:
        """Number of nodes :math:`N_q`.

        Returns
        -------
        int
            ``nodes.shape[0]``.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Gaussian(np.zeros(3), np.eye(3))
        >>> sub = ac.subspace_from_basis([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], space=space)
        >>> rule = ac.active_quadrature_rule(ac.active_domain(sub), n_nodes=10)
        >>> rule.n_nodes
        9
        """
        return int(self.nodes.shape[0])

    def _check(self) -> None:
        dom = self.domain
        _invariant(isinstance(dom, ActiveDomain), "domain must be an ActiveDomain")
        n = dom.n
        _invariant(
            self.kind == _RULE_KINDS[dom.kind],
            f"kind {self.kind!r} does not match a {dom.kind} domain",
        )
        Nq = self.nodes.shape[0] if self.nodes.ndim == 2 else -1
        _invariant(self.nodes.shape == (Nq, n) and Nq >= 1, "nodes must be (N_q, n) with N_q >= 1")
        _invariant(bool(np.isfinite(self.nodes).all()), "nodes must be finite")
        _invariant(self.weights.shape == (Nq,), "weights must be (N_q,)")
        _check_weights(self.weights, "weights")
        _invariant(len(self.n_requested) >= 1, "n_requested must be non-empty")
        zon = self.kind == "delaunay_centroid"
        _invariant((self.degree is None) == zon, "degree is None exactly for the Delaunay rule")
        _invariant((self.design is None) != zon, "design is given exactly for the Delaunay rule")
        _invariant(
            (self.n_weight_samples is None) != zon,
            "n_weight_samples is given exactly for the Delaunay rule",
        )
        if zon:
            # both are not None here: the invariants above hold
            assert self.design is not None
            assert self.n_weight_samples is not None
            _invariant(
                self.design.ndim == 2 and self.design.shape[1] == n, "design must be (N_d, n)"
            )
            _invariant(self.n_weight_samples >= 1, "n_weight_samples must be >= 1")
        _invariant(0.0 <= self.weight_deficit < 1.0, "weight_deficit must be in [0, 1)")
        _invariant(
            zon or self.weight_deficit == 0.0, "weight_deficit is 0 except for the Delaunay rule"
        )


@_register_result
class IntegrationDesign(Result):
    r"""Rule nodes and inactive draws for one or more replicates.

    A design holds the points at which to evaluate the model for an
    integration: for each node :math:`y_k` of the rule, :math:`N_{MC}` draws
    :math:`x_{kj} = W_1 y_k + W_2 z_{kj}` with :math:`z_{kj}` from the
    conditional density of the inactive variables given :math:`y_k`. Designs are
    built by :func:`integration_design`; the constructor is used when a saved
    result is loaded.

    Parameters
    ----------
    domain : ActiveDomain
        The active domain of the rule.
    rule_kind : {"gauss_hermite", "box_spline_gauss", "delaunay_centroid"}
        As :attr:`ActiveQuadratureRule.kind`.
    n_replicates : int
        Number :math:`R \ge 1` of independent replicates.
    nodes : ndarray, shape (Q, n)
        Nodes of all replicates in normalized active coordinates, stacked
        replicate by replicate.
    node_weights : ndarray, shape (Q,)
        Rule weights; each replicate's block sums to 1.
    node_replicate : ndarray of int64, shape (Q,)
        Replicate of each node, non-decreasing from 0 to ``R - 1``.
    weight_deficits : ndarray, shape (R,)
        Weight deficit :math:`d` of each replicate's rule.
    n_weight_samples : int or None
        :math:`N_X` for the Delaunay-centroid rule, else ``None``.
    n_mc : int
        Inactive draws per node, :math:`N_{MC} \ge 0`.
    X : ndarray, shape (Q * N_MC, m)
        Points in normalized coordinates; row ``q * N_MC + j`` is draw ``j``
        of node ``q``.
    inactive_sampler : {"none", "gaussian", "exact_interval", "hit_and_run"}
        How the inactive variables were drawn (``"none"`` when ``n_mc == 0``).
    burn_in : int or None
        Burn-in of the hit-and-run sampler as used; ``None`` for other samplers.
    thin : int or None
        Thinning of the hit-and-run sampler as used; ``None`` for other samplers.
    provenance : Provenance
        How the design was computed.

    See Also
    --------
    integration_design : Build a design.
    estimate_integral : Combine model values at ``X_physical`` into an estimate.

    Notes
    -----
    The pairs :math:`(x_{kj}, \omega_k / (N_{MC} R))` form a quadrature rule on
    :math:`x` whose weights sum to 1 (:attr:`point_weights`).

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
    >>> sub = ac.subspace_from_basis([[1.0], [0.5]], space=space)
    >>> design = ac.integration_design(ac.active_domain(sub), n_nodes=4, n_mc=10, rng=0)
    >>> design.X.shape
    (40, 2)
    >>> design.inactive_sampler
    'exact_interval'
    """

    TYPE = "IntegrationDesign"
    SCHEMA_VERSION = 1

    domain: ActiveDomain
    rule_kind: Literal["gauss_hermite", "box_spline_gauss", "delaunay_centroid"]
    n_replicates: int
    nodes: NDArray[np.float64]
    node_weights: NDArray[np.float64]
    node_replicate: NDArray[np.int64]
    weight_deficits: NDArray[np.float64]
    n_weight_samples: int | None
    n_mc: int
    X: NDArray[np.float64]
    inactive_sampler: Literal["none", "gaussian", "exact_interval", "hit_and_run"]
    burn_in: int | None
    thin: int | None

    @property
    def X_physical(self) -> NDArray[np.float64]:
        """The points ``X`` in physical coordinates.

        For a uniform space the points are clipped to the box ``[lower, upper]``.

        Returns
        -------
        ndarray, shape (Q * N_MC, m)
            The rows of ``X`` mapped to physical coordinates, in the same order.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
        >>> sub = ac.subspace_from_basis([[1.0], [0.5]], space=space)
        >>> design = ac.integration_design(ac.active_domain(sub), n_nodes=4, n_mc=10, rng=0)
        >>> Xp = design.X_physical
        >>> print(bool(np.all(Xp >= 0.0)), bool(np.all(Xp <= [1.0, 2.0])))
        True True
        """
        return _to_physical(self.domain.subspace.space, self.X)

    @property
    def point_node(self) -> NDArray[np.int64]:
        """Node index of each row of ``X``.

        Returns
        -------
        ndarray of int64, shape (Q * N_MC,)
            ``numpy.repeat(numpy.arange(Q), N_MC)``.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
        >>> sub = ac.subspace_from_basis([[1.0], [0.5]], space=space)
        >>> design = ac.integration_design(ac.active_domain(sub), n_nodes=2, n_mc=3, rng=0)
        >>> print(design.point_node)
        [0 0 0 1 1 1]
        """
        return np.repeat(np.arange(self.nodes.shape[0], dtype=np.int64), self.n_mc)

    @property
    def point_weights(self) -> NDArray[np.float64]:
        r"""Quadrature weights of the rows of ``X`` on the full input space.

        Row ``q * N_MC + j`` has weight :math:`\omega_q / (N_{MC} R)`, so the
        weights sum to 1 and can be passed as ``quad_weights`` to
        :func:`~active_subspaces.active_subspace` together with ``X_physical``.

        Returns
        -------
        ndarray, shape (Q * N_MC,)
            The weights; empty when ``n_mc == 0``.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
        >>> sub = ac.subspace_from_basis([[1.0], [0.5]], space=space)
        >>> design = ac.integration_design(ac.active_domain(sub), n_nodes=4, n_mc=10, rng=0)
        >>> print(f"{design.point_weights.sum():.6f}")
        1.000000
        """
        if self.n_mc == 0:
            return np.zeros(0)
        return self.node_weights[self.point_node] / (self.n_mc * self.n_replicates)

    def _check(self) -> None:
        dom = self.domain
        _invariant(isinstance(dom, ActiveDomain), "domain must be an ActiveDomain")
        m, n, R = dom.m, dom.n, self.n_replicates
        _invariant(
            self.rule_kind == _RULE_KINDS[dom.kind],
            f"rule_kind {self.rule_kind!r} does not match a {dom.kind} domain",
        )
        _invariant(R >= 1, "n_replicates must be >= 1")
        Q = self.nodes.shape[0] if self.nodes.ndim == 2 else -1
        _invariant(self.nodes.shape == (Q, n) and Q >= R, "nodes must be (Q, n) with Q >= R")
        _invariant(bool(np.isfinite(self.nodes).all()), "nodes must be finite")
        _invariant(
            self.node_weights.shape == (Q,) and self.node_replicate.shape == (Q,),
            "node_weights and node_replicate must be (Q,)",
        )
        rep = self.node_replicate
        _invariant(
            bool(np.all(np.diff(rep) >= 0)) and rep[0] == 0 and rep[-1] == R - 1,
            "node_replicate must be non-decreasing from 0 to R-1",
        )
        _invariant(np.unique(rep).size == R, "every replicate must have a node")
        for r in range(R):
            _check_weights(self.node_weights[rep == r], f"node_weights of replicate {r}")
        _invariant(self.weight_deficits.shape == (R,), "weight_deficits must be (R,)")
        _invariant(
            bool(np.all((self.weight_deficits >= 0) & (self.weight_deficits < 1))),
            "weight_deficits must be in [0, 1)",
        )
        zon = self.rule_kind == "delaunay_centroid"
        _invariant(
            (self.n_weight_samples is None) != zon,
            "n_weight_samples is given exactly for the Delaunay rule",
        )
        _invariant(self.n_mc >= 0, "n_mc must be >= 0")
        _invariant(self.X.shape == (Q * self.n_mc, m), "X must be (Q * n_mc, m)")
        _invariant(bool(np.isfinite(self.X).all()), "X must be finite")
        _invariant(
            (self.inactive_sampler == "none") == (self.n_mc == 0),
            "inactive_sampler is 'none' exactly when n_mc == 0",
        )
        hr = self.inactive_sampler == "hit_and_run"
        _invariant(
            (self.burn_in is not None and self.thin is not None) == hr,
            "burn_in and thin are recorded exactly for hit-and-run",
        )


@_register_result
class IntegrationResult(QoIResult):
    r"""Estimate of :math:`\int f(x) \rho(x)\, dx` and its error components.

    Results are produced by :func:`integrate`, :func:`estimate_integral` and
    :func:`integrate_surrogate`; the constructor is used when a saved result is
    loaded.

    Parameters
    ----------
    source : {"model", "surrogate"}
        Whether model values or a function of the active variables were
        integrated.
    design : IntegrationDesign
        The design; ``design.n_mc == 0`` for ``"surrogate"``.
    surrogate : PolynomialSurrogate, RBFSurrogate or None
        The built-in surrogate integrated (for a reduced model, its surrogate);
        ``None`` for a model, a plain callable or a user-defined surrogate.
    f : ndarray, shape (Q * N_MC,)
        Model values at the rows of ``design.X_physical``; shape ``(0,)`` for
        ``"surrogate"``.
    node_means : ndarray, shape (Q,)
        :math:`\hat G_k`, or :math:`g(y_k)` for ``"surrogate"``.
    node_variances : ndarray, shape (Q,)
        :math:`\hat V_k` with ddof = 1; NaN where undefined
        (:math:`N_{MC} < 2` or ``"surrogate"``).
    estimate : float
        :math:`\hat\mu`, the mean of the replicate estimates.
    inactive_std_error : float or None
        :math:`\hat\sigma_{in}`, the standard error of the inactive Monte Carlo
        only; ``None`` for ``"surrogate"`` or :math:`N_{MC} < 2`.
    total_std_error : float or None
        :math:`\hat\sigma_{tot} = s_R / \sqrt{R}` from the replicates; ``None``
        unless :math:`R \ge 2`.
    replicate_estimates : ndarray, shape (R,)
        :math:`\hat\mu_1, \dots, \hat\mu_R`.
    autocorrelation_time : float or None
        :math:`\hat\tau` used in ``inactive_std_error``: 1.0 for independent
        draws; ``None`` when ``inactive_std_error`` is ``None``.
    provenance : Provenance
        How the result was computed.
    qoi : str
        Name of the quantity of interest, ``domain.subspace.qoi``.

    See Also
    --------
    integrate : Integrate a model.
    integrate_surrogate : Integrate a function of the active variables.

    Notes
    -----
    With :math:`\hat V_k` the ddof = 1 variance of the :math:`N_{MC}` values at
    node :math:`k`,

    .. math::

        \hat\sigma_{in} = \Bigl( \sum_k \omega_k^2 \, \hat\tau \, \hat V_k / N_{MC}
        \Bigr)^{1/2}

    (per replicate; across replicates
    :math:`\hat\sigma_{in} = R^{-1} (\sum_r \hat\sigma_{in,r}^2)^{1/2}`), and

    .. math::

        \hat\mu = \frac{1}{R} \sum_{r=1}^{R} \hat\mu_r, \qquad
        \hat\sigma_{tot} = \frac{s_R}{\sqrt{R}}, \qquad
        s_R^2 = \frac{1}{R-1} \sum_{r=1}^{R} (\hat\mu_r - \hat\mu)^2 .

    :math:`\hat\sigma_{tot}` captures every random component (inactive draws,
    the Monte Carlo cell masses and the random design of the
    Delaunay-centroid rule) but not the deterministic error of the rule in
    :math:`y`. No confidence interval is formed.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
    >>> model = ac.test_functions.ridge([1.0, 0.5], profile="sin", space=space)
    >>> sub = ac.subspace_from_basis([[1.0], [0.5]], space=space)
    >>> res = ac.integrate(model, domain=ac.active_domain(sub), n_nodes=4, n_mc=10, rng=0)
    >>> print(res.source, res.n_evaluations, res.total_std_error)
    model 40 None
    >>> print(abs(res.estimate) < 3 * res.inactive_std_error)
    True
    """

    TYPE = "IntegrationResult"
    SCHEMA_VERSION = 1

    source: Literal["model", "surrogate"]
    design: IntegrationDesign
    surrogate: PolynomialSurrogate | RBFSurrogate | None
    f: NDArray[np.float64]
    node_means: NDArray[np.float64]
    node_variances: NDArray[np.float64]
    estimate: float
    inactive_std_error: float | None
    total_std_error: float | None
    replicate_estimates: NDArray[np.float64]
    autocorrelation_time: float | None

    @property
    def n_evaluations(self) -> int:
        """Number of model evaluations.

        Returns
        -------
        int
            ``f.size``; 0 for ``source == "surrogate"``.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
        >>> sub = ac.subspace_from_basis([[1.0], [0.5]], space=space)
        >>> res = ac.integrate_surrogate(
        ...     lambda Y: Y[:, 0] ** 2, domain=ac.active_domain(sub), n_nodes=3
        ... )
        >>> res.n_evaluations
        0
        """
        return int(self.f.size)

    def _check(self) -> None:
        d = self.design
        _invariant(isinstance(d, IntegrationDesign), "design must be an IntegrationDesign")
        Q, R = d.nodes.shape[0], d.n_replicates
        _invariant(
            (self.source == "surrogate") == (d.n_mc == 0),
            "source is 'surrogate' exactly when design.n_mc == 0",
        )
        _invariant(
            self.source == "surrogate" or self.surrogate is None,
            "surrogate is None for source 'model'",
        )
        _invariant(self.f.shape == (Q * d.n_mc,), "f must be (Q * n_mc,)")
        _invariant(
            self.node_means.shape == (Q,) and self.node_variances.shape == (Q,),
            "node_means and node_variances must be (Q,)",
        )
        _invariant(self.replicate_estimates.shape == (R,), "replicate_estimates must be (R,)")
        _invariant(
            (self.total_std_error is None) == (R == 1),
            "total_std_error is None exactly when R == 1",
        )
        _invariant(
            (self.autocorrelation_time is None) == (self.inactive_std_error is None),
            "autocorrelation_time is None exactly when inactive_std_error is",
        )
        _invariant(
            self.source == "model" or self.inactive_std_error is None,
            "a surrogate result has no inactive_std_error",
        )
        _invariant(self.qoi == d.domain.subspace.qoi, "qoi must equal domain.subspace.qoi")


def _check_weights(w: np.ndarray, what: str) -> None:
    _invariant(
        bool(np.all(np.isfinite(w))) and bool(np.all(w >= 0)), f"{what} must be finite and >= 0"
    )
    _invariant(abs(float(np.sum(w)) - 1.0) <= _SUM_TOL, f"{what} must sum to 1")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _param(message: str, param: str, **details: Any) -> InputError:
    return InputError(message, code="input.parameter", details={"param": param, **details})


def _check_domain(domain: Any) -> ActiveDomain:
    if not isinstance(domain, ActiveDomain):
        raise _param(f"domain: expected an ActiveDomain, got {type(domain).__name__}", "domain")
    if domain.kind not in _RULE_KINDS:
        raise UnsupportedError(
            f"integration is not defined for an active domain of kind {domain.kind!r}",
            code="unsupported.space",
            details={"kind": domain.kind},
        )
    return domain


def _is_sequence(x: Any) -> bool:
    if isinstance(x, np.ndarray):
        return x.ndim >= 1
    return isinstance(x, Sequence) and not isinstance(x, (str, bytes))


def _n_nodes(domain: ActiveDomain, n_nodes: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``(n_requested, counts)``.

    Per-dimension ``k_i`` (Gaussian), ``(N_q,)`` or ``(N_d,)``.
    """
    n = domain.n
    if _is_sequence(n_nodes):
        if domain.kind != "unbounded":
            raise _param(
                "n_nodes: a per-dimension sequence is accepted only for a Gaussian domain",
                "n_nodes",
                reason="sequence_for_uniform",
            )
        ks = tuple(_v.integer(k, "n_nodes", lo=1) for k in n_nodes)
        if len(ks) != n:
            raise _param(
                f"n_nodes: expected {n} per-dimension counts, got {len(ks)}",
                "n_nodes",
                expected=n,
                got=len(ks),
            )
        return ks, ks
    lo = 0 if domain.kind == "zonotope" else 1
    N = _v.integer(n_nodes, "n_nodes", lo=lo)
    if domain.kind == "unbounded":
        k = _int_root(N, n)
        return (N,), (k,) * n
    return (N,), (N,)


def _int_root(N: int, n: int) -> int:
    """Largest integer ``k`` with ``k**n <= N`` (exact integer arithmetic)."""
    k: int = max(1, round(cast(float, N ** (1.0 / n))))
    while k > 1 and k**n > N:
        k -= 1
    while (k + 1) ** n <= N:
        k += 1
    return k


def _opt_int(x: Any, name: str, lo: int) -> int | None:
    return None if x is None else _v.integer(x, name, lo=lo)


def _check_model(model: Any, domain: ActiveDomain) -> Model:
    """Resolve ``model`` in the domain's space.

    ``space_mismatch`` if a ``Model`` has another space [D-76].
    """
    return as_model(model, space=domain.subspace.space)


def _check_surrogate(
    surrogate: Any, domain: ActiveDomain
) -> tuple[Callable[[np.ndarray], Any], Any]:
    """Return ``(evaluate, the protocol object or None)``; a callable is a function of ``y``."""
    if isinstance(surrogate, Surrogate):
        sub = surrogate.subspace
        if sub is None or not (type(sub) is type(domain.subspace) and sub == domain.subspace):
            raise _param(
                "surrogate: its subspace must equal domain.subspace (a surrogate on all inputs "
                "is not a function of y)",
                "surrogate",
                reason="subspace_mismatch",
            )
        return surrogate.evaluate, surrogate
    if callable(surrogate):
        return surrogate, None
    raise _param(
        f"surrogate: expected a Surrogate or a callable, got {type(surrogate).__name__}",
        "surrogate",
        got=type(surrogate).__name__,
    )


def _builtin_surrogate(obj: Any) -> PolynomialSurrogate | RBFSurrogate | None:
    if isinstance(obj, (PolynomialSurrogate, RBFSurrogate)):
        return obj
    inner = getattr(obj, "surrogate", None)  # a ReducedModel delegates to its surrogate [D-104]
    if isinstance(inner, (PolynomialSurrogate, RBFSurrogate)):
        return inner
    return None


# ---------------------------------------------------------------------------
# Rules (§5.1)
# ---------------------------------------------------------------------------


class _Rule:
    __slots__ = ("deficit", "degree", "design", "nodes", "weights")

    def __init__(
        self,
        nodes: NDArray[np.float64],
        weights: NDArray[np.float64],
        degree: int | None,
        design: NDArray[np.float64] | None = None,
        deficit: float = 0.0,
    ) -> None:
        self.nodes, self.weights, self.degree, self.design, self.deficit = (
            nodes,
            weights,
            degree,
            design,
            deficit,
        )


def _gauss_rule(ks: tuple[int, ...]) -> _Rule:
    nodes, weights = tensor_rule([gauss_hermite(k) for k in ks])
    weights = weights / np.sum(weights)
    return _Rule(nodes, weights, min(2 * k - 1 for k in ks))


def _interval_rule(domain: ActiveDomain, N: int) -> _Rule:
    t, p = box_spline_gauss(np.asarray(domain.subspace.W1)[:, 0], N)
    return _Rule(t.reshape(-1, 1), p, 2 * N - 1)


def _delaunay_rule(domain: ActiveDomain, N_d: int, N_X: int, gen: np.random.Generator) -> _Rule:
    s_design, s_weights = _random.spawn(gen, 2)
    n, m = domain.n, domain.m
    V = np.asarray(domain.vertices, dtype=np.float64)
    if N_d > 0:
        D = np.array(
            active_design(domain=domain, n_points=N_d, rng=s_design).points, dtype=np.float64
        )
    else:
        D = np.zeros((0, n))
    P = np.vstack([V, D])
    try:
        T = scipy.spatial.Delaunay(P)
    except scipy.spatial.QhullError as exc:
        raise NumericalError(
            f"Delaunay triangulation of the active domain failed: {exc}",
            code="numerical.singular",
            details={"what": "delaunay", "n_points": int(P.shape[0])},
        ) from exc
    W1 = np.array(domain.subspace.W1, dtype=np.float64)
    diam = float(scipy.spatial.distance.pdist(V).max())
    counts = np.zeros(T.nsimplex, dtype=np.int64)
    n_out = 0
    for start in range(0, N_X, _WEIGHT_CHUNK):
        size = min(_WEIGHT_CHUNK, N_X - start)
        Ys = s_weights.uniform(-1.0, 1.0, size=(size, m)) @ W1
        idx = np.asarray(T.find_simplex(Ys), dtype=np.int64)
        miss = idx == -1
        if miss.any():
            idx[miss] = T.find_simplex(Ys[miss], bruteforce=True, tol=_RETRY_TOL * diam)
        inside = idx >= 0
        n_out += int(np.count_nonzero(~inside))
        counts += np.bincount(idx[inside], minlength=T.nsimplex)
    if n_out == N_X:
        raise NumericalError(
            "every weight sample fell outside the triangulation of the active domain",
            code="numerical.singular",
            details={"what": "weight_samples", "n_weight_samples": int(N_X)},
        )
    d = n_out / N_X
    if n_out > 0:
        _warn(
            AccuracyWarning,
            f"{n_out} of {N_X} weight samples (d = {d:.3g}) fell outside the triangulated hull, "
            "whose vertex set is "
            "incomplete; the weights were renormalized, so the estimate is of E[G | y in hull]",
            code="accuracy.weight_deficit",
            details={"deficit": d, "n_outside": n_out, "n_weight_samples": int(N_X)},
        )
    keep = counts > 0
    centroids = T.points[T.simplices].mean(axis=1)
    weights = counts[keep] / float(N_X - n_out)
    return _Rule(np.ascontiguousarray(centroids[keep]), weights, None, D, d)


def _rule(
    domain: ActiveDomain, counts: tuple[int, ...], N_X: int, gen: np.random.Generator | None
) -> _Rule:
    if domain.kind == "unbounded":
        return _gauss_rule(counts)
    if domain.kind == "interval":
        return _interval_rule(domain, counts[0])
    assert gen is not None
    return _delaunay_rule(domain, counts[0], N_X, gen)


def active_quadrature_rule(
    domain: ActiveDomain,
    *,
    n_nodes: int | Sequence[int],
    n_weight_samples: int = 100_000,
    rng: _random.RngLike = None,
) -> ActiveQuadratureRule:
    r"""Build a quadrature rule on the active variables of a domain.

    The rule integrates functions of the active variables :math:`y` against
    their marginal density :math:`\pi(y)` [1]_; its nodes are in normalized
    active coordinates and its weights sum to 1.

    Parameters
    ----------
    domain : ActiveDomain
        The active domain, from :func:`~active_subspaces.active_domain`.
    n_nodes : int or sequence of int
        Gaussian domain: the target total :math:`N`, giving
        :math:`k = \lfloor N^{1/n} \rfloor` points per dimension (computed
        exactly, so :math:`k^n \le N`), or a sequence of ``n`` per-dimension
        counts. Interval: the number of nodes, ``>= 1``. Zonotope: the number
        :math:`N_d \ge 0` of interior maximin points; the node count is then
        the number of Delaunay simplices with positive weight.
    n_weight_samples : int, default 100000
        :math:`N_X`, the Monte Carlo samples for the cell masses of the
        zonotope rule.
    rng : None, int, SeedSequence or Generator, optional
        Random source of the zonotope rule (maximin design and cell masses);
        unused for the other domains.

    Returns
    -------
    ActiveQuadratureRule
        The nodes, weights, kind and degree of exactness of the rule.

    Raises
    ------
    InputError
        ``input.parameter`` if `domain` is not an ``ActiveDomain``, if `n_nodes`
        or `n_weight_samples` is not a valid count, or if a sequence `n_nodes`
        is given for a uniform domain.
    NumericalError
        ``numerical.singular`` if the Delaunay triangulation fails or every
        weight sample falls outside it, or if the Gauss rule cannot be computed.

    Warns
    -----
    AccuracyWarning
        ``accuracy.weight_deficit`` if weight samples fall outside the
        triangulation of an incomplete vertex set; the weights are then
        renormalized.

    See Also
    --------
    integration_design : The rule together with inactive draws.
    integrate_surrogate : Integrate a function of the active variables.

    Notes
    -----
    **Gaussian space.** A tensor product of probabilists' Gauss-Hermite rules,
    exact for polynomials of degree :math:`\le 2 k_i - 1` in :math:`y_i`.

    **Uniform space,** :math:`n = 1`. The :math:`N_q`-point Gauss rule of the
    density :math:`\pi` of :math:`y = w^\top x = \sum_j w_j x_j` with
    :math:`x_j \sim U[-1, 1]` independent (a box spline), exact to degree
    :math:`2 N_q - 1` against :math:`\pi`. It is computed by convolving
    Gauss-Legendre rules one input at a time and compressing each convolution
    back to :math:`N_q` atoms with a Lanczos (Stieltjes) step, which preserves the
    moments up to degree :math:`2 N_q - 1` and is numerically stable [2]_, [3]_.

    **Uniform space,** :math:`n \ge 2`. The centroids :math:`c_k` of the
    Delaunay simplices :math:`T_k` of the zonotope vertices together with
    :math:`N_d` maximin points, with Monte Carlo cell masses

    .. math::

        \omega_k = \frac{\#\lbrace i : W_1^\top x_i \in T_k \rbrace}{N_X (1 - d)},
        \qquad x_i \sim U[-1, 1]^m,

    where :math:`d` is the fraction of samples outside the triangulation (0
    when the vertex set is complete). Simplices with :math:`\omega_k = 0` are
    dropped. The cell masses carry a Monte Carlo error of order
    :math:`\sigma_G / \sqrt{N_X}`, where :math:`\sigma_G` is the standard
    deviation of the integrand; the rule integrates constants exactly and has
    error :math:`O(h)` in the largest cell diameter :math:`h`.

    *Departures from the 2016 package.* The 2016 interval rule used midpoints
    of equal cells with Monte Carlo histogram weights; v1 uses the Gauss rule
    of the box spline, which is deterministic. The 2016 zonotope rule fixed
    :math:`N_X = 10^4` and silently left :math:`\sum_k \omega_k < 1` when
    samples fell outside the hull, biasing :math:`\hat\mu` toward 0; v1 makes
    :math:`N_X` a parameter and renormalizes with a warning, which estimates
    :math:`E[G(y) \mid y \in \text{hull}]`.

    References
    ----------
    .. [1] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.
    .. [2] G. H. Golub, J. H. Welsch. Calculation of Gauss quadrature rules.
       *Math. Comp.* 23(106):221-230, 1969.
    .. [3] W. Gautschi. *Orthogonal Polynomials: Computation and
       Approximation.* Oxford University Press, 2004.

    Examples
    --------
    A uniform space with one active variable: the Gauss rule of the box
    spline integrates :math:`y^2` exactly with two nodes
    (:math:`E[y^2] = 1/3` here).

    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
    >>> sub = ac.subspace_from_basis([[1.0], [0.5]], space=space)
    >>> rule = ac.active_quadrature_rule(ac.active_domain(sub), n_nodes=2)
    >>> print(f"{np.sum(rule.weights * rule.nodes[:, 0] ** 2):.6f}")
    0.333333

    A Gaussian space with two active variables: :math:`N = 10` gives
    :math:`3 \times 3` nodes.

    >>> space = ac.Gaussian(np.zeros(3), np.eye(3))
    >>> sub = ac.subspace_from_basis([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], space=space)
    >>> rule = ac.active_quadrature_rule(ac.active_domain(sub), n_nodes=10)
    >>> print(rule.kind, rule.nodes.shape, rule.degree)
    gauss_hermite (9, 2) 5
    """
    domain = _check_domain(domain)
    requested, counts = _n_nodes(domain, n_nodes)
    N_X = _v.integer(n_weight_samples, "n_weight_samples", lo=1)
    zon = domain.kind == "zonotope"
    gen, desc = _random.generator(rng) if zon else (None, None)
    rule = _rule(domain, counts, N_X, gen)
    params = {"domain": domain, "n_nodes": list(requested), "n_weight_samples": N_X}
    prov = _make_provenance(
        "active_quadrature_rule", params, space=domain.subspace.space, rng_state=desc
    )
    return ActiveQuadratureRule(
        domain=domain,
        kind=_RULE_KINDS[domain.kind],
        nodes=rule.nodes,
        weights=rule.weights,
        n_requested=requested,
        degree=rule.degree,
        design=rule.design,
        n_weight_samples=N_X if zon else None,
        weight_deficit=float(rule.deficit),
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# Designs and estimates (§5.2)
# ---------------------------------------------------------------------------


def integration_design(
    domain: ActiveDomain,
    *,
    n_nodes: int | Sequence[int],
    n_mc: int = 50,  # [D-171]
    n_replicates: int = 1,
    n_weight_samples: int = 100_000,
    burn_in: int | None = None,
    thin: int | None = None,
    backend: SolverBackend | str | None = None,
    rng: _random.RngLike = None,
) -> IntegrationDesign:
    r"""Build the rule nodes and inactive draws for one or more replicates.

    For each of :math:`R` independent replicates, build a rule
    :math:`\lbrace (y_k, \omega_k) \rbrace` with :func:`active_quadrature_rule`
    and draw :math:`N_{MC}` inactive points per node with
    :func:`~active_subspaces.inverse_map`. Evaluate the model at
    ``design.X_physical`` and pass the values to :func:`estimate_integral`.

    Parameters
    ----------
    domain : ActiveDomain
        The active domain, from :func:`~active_subspaces.active_domain`.
    n_nodes : int or sequence of int
        As for :func:`active_quadrature_rule`.
    n_mc : int, default 50
        Inactive draws per node, :math:`N_{MC} \ge 0`; 0 gives the rule only.
    n_replicates : int, default 1
        Number of independent replicates :math:`R \ge 1`; :math:`R \ge 2`
        enables ``total_std_error``.
    n_weight_samples : int, default 100000
        :math:`N_X` for the zonotope rule.
    burn_in : int, optional
        Burn-in of the hit-and-run sampler (``>= 0``); default as in
        :func:`~active_subspaces.inverse_map`.
    thin : int, optional
        Thinning of the hit-and-run sampler (``>= 1``); default as in
        :func:`~active_subspaces.inverse_map`.
    backend : SolverBackend, "scipy" or None, optional
        LP backend for the start of the hit-and-run sampler.
    rng : None, int, SeedSequence or Generator, optional
        Random source. Replicate ``r`` uses child ``r`` of the generator, split
        into one stream for the rule and one for the inactive draws.

    Returns
    -------
    IntegrationDesign
        Nodes, weights and the points ``X`` (normalized) and ``X_physical``.

    Raises
    ------
    InputError
        ``input.parameter`` if `domain` is not an ``ActiveDomain``, if a count
        (`n_nodes`, `n_mc`, `n_replicates`, `n_weight_samples`, `burn_in`,
        `thin`) is invalid, if a sequence `n_nodes` is given for a uniform
        domain, or if `backend` is not a solver backend.
    NumericalError
        ``numerical.singular`` as for :func:`active_quadrature_rule`.
    SolverError
        ``solver.failed`` if the LP that starts the hit-and-run sampler fails.

    Warns
    -----
    AccuracyWarning
        ``accuracy.weight_deficit`` as for :func:`active_quadrature_rule`;
        ``accuracy.correlated_draws`` and ``accuracy.degenerate_fiber`` from
        the inactive sampler of :func:`~active_subspaces.inverse_map`.

    See Also
    --------
    estimate_integral : Combine model values at the design into an estimate.
    integrate : Design, model evaluations and estimate in one call.

    Notes
    -----
    The inactive draws come from the conditional density :math:`\pi(z \mid y_k)`:
    independent Gaussian draws for a Gaussian space, exact uniform draws on an
    interval when :math:`m - n = 1`, and hit-and-run draws [1]_ on the polytope
    :math:`\lbrace z : -1 \le W_1 y_k + W_2 z \le 1 \rbrace` otherwise.

    References
    ----------
    .. [1] R. L. Smith. Efficient Monte Carlo procedures for generating points
       uniformly distributed over bounded regions. *Oper. Res.*
       32(6):1296-1308, 1984.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
    >>> sub = ac.subspace_from_basis([[1.0], [0.5]], space=space)
    >>> design = ac.integration_design(
    ...     ac.active_domain(sub), n_nodes=4, n_mc=10, n_replicates=2, rng=0
    ... )
    >>> print(design.nodes.shape, design.X_physical.shape)
    (8, 1) (80, 2)
    >>> print(f"{design.point_weights.sum():.6f}")
    1.000000
    """
    domain = _check_domain(domain)
    requested, counts = _n_nodes(domain, n_nodes)
    n_mc = _v.integer(n_mc, "n_mc", lo=0)
    R = _v.integer(n_replicates, "n_replicates", lo=1)
    N_X = _v.integer(n_weight_samples, "n_weight_samples", lo=1)
    burn_in = _opt_int(burn_in, "burn_in", 0)
    thin = _opt_int(thin, "thin", 1)
    be = get_backend(backend)
    gen, desc = _random.generator(rng)
    reps = _random.spawn(gen, R)
    sub = domain.subspace
    nodes, weights, rep_idx, deficits, Xs = [], [], [], [], []
    sampler: Literal["none", "gaussian", "exact_interval", "hit_and_run"] = "none"
    used_burn: int | None = None
    used_thin: int | None = None
    for r in range(R):
        s_rule, s_inactive = _random.spawn(reps[r], 2)
        rule = _rule(domain, counts, N_X, s_rule)
        nodes.append(rule.nodes)
        weights.append(rule.weights)
        rep_idx.append(np.full(rule.nodes.shape[0], r, dtype=np.int64))
        deficits.append(rule.deficit)
        if n_mc > 0:
            inv = inverse_map(
                sub,
                Y=rule.nodes,
                n_draws=n_mc,
                burn_in=burn_in,
                thin=thin,
                backend=be,
                rng=s_inactive,
            )
            Xs.append(np.asarray(inv.X))
            sampler, used_burn, used_thin = inv.sampler, inv.burn_in, inv.thin
    X = np.vstack(Xs) if Xs else np.zeros((0, domain.m))
    params = {
        "domain": domain,
        "n_nodes": list(requested),
        "n_mc": n_mc,
        "n_replicates": R,
        "n_weight_samples": N_X,
        "burn_in": burn_in,
        "thin": thin,
        "backend": str(getattr(be, "name", type(be).__name__)),
    }
    prov = _make_provenance("integration_design", params, space=sub.space, rng_state=desc)
    return IntegrationDesign(
        domain=domain,
        rule_kind=_RULE_KINDS[domain.kind],
        n_replicates=R,
        nodes=np.vstack(nodes),
        node_weights=np.concatenate(weights),
        node_replicate=np.concatenate(rep_idx),
        weight_deficits=np.array(deficits, dtype=np.float64),
        n_weight_samples=N_X if domain.kind == "zonotope" else None,
        n_mc=n_mc,
        X=X,
        inactive_sampler=sampler,
        burn_in=used_burn,
        thin=used_thin,
        provenance=prov,
    )


#: Hit-and-run chains shorter than this warn accuracy.correlated_draws (D-114, amended 2026-10-29).
_MIN_CHAIN = 20


def _geyer_tau(F: np.ndarray) -> float:
    """Pooled integrated autocorrelation time of chains ``F (C, N)``.

    Geyer's initial positive sequence [D-114].
    """
    C, N = F.shape
    D = F - F.mean(axis=1, keepdims=True)
    gamma0 = float(np.sum(D * D)) / (C * N)
    if not gamma0 > 0:
        return 1.0
    r = np.array(
        [float(np.sum(D[:, : N - lag] * D[:, lag:])) / (C * N) / gamma0 for lag in range(N)]
    )
    total = 0.0
    t = 0
    while 2 * t + 1 <= N - 1:
        Gamma = r[2 * t] + r[2 * t + 1]
        if Gamma <= 0:
            break
        total += Gamma
        t += 1
    return float(min(N, max(1.0, -1.0 + 2.0 * total)))


def _nonfinite(what: str) -> NumericalError:
    return NumericalError(
        f"integration produced a non-finite {what}",
        code="numerical.nonfinite",
        details={"what": what},
    )


def _combine(
    design: IntegrationDesign, node_means: np.ndarray
) -> tuple[np.ndarray, float, float | None]:
    """Replicate estimates, their mean, and ``s_R / sqrt(R)`` (``None`` for ``R = 1``) [D-111]."""
    R = design.n_replicates
    rep = design.node_replicate
    w = design.node_weights
    mu_r = np.array([float(np.sum(w[rep == r] * node_means[rep == r])) for r in range(R)])
    if not np.isfinite(mu_r).all():
        raise _nonfinite("replicate estimate")
    mu = float(np.mean(mu_r))
    total = float(np.std(mu_r, ddof=1) / math.sqrt(R)) if R >= 2 else None
    if not math.isfinite(mu) or (total is not None and not math.isfinite(total)):
        raise _nonfinite("estimate")
    return mu_r, mu, total


def _estimate(design: IntegrationDesign, f: np.ndarray) -> dict[str, Any]:
    Q, N = design.nodes.shape[0], design.n_mc
    F = f.reshape(Q, N)
    G = F.mean(axis=1)
    V = F.var(axis=1, ddof=1) if N >= 2 else np.full(Q, np.nan)
    if not np.isfinite(G).all() or (N >= 2 and not np.isfinite(V).all()):
        raise _nonfinite("node mean or variance")
    mu_r, mu, total = _combine(design, G)
    tau: float | None = None
    se: float | None = None
    if N >= 2:
        if design.inactive_sampler == "hit_and_run":
            tau = _geyer_tau(F) if N >= 4 else 1.0
            if N < _MIN_CHAIN:
                what = (
                    "too short to estimate the autocorrelation time; inactive_std_error assumes "
                    "independent draws"
                    if N < 4
                    else "too short for a reliable autocorrelation time; inactive_std_error may "
                    "be too small"
                )
                _warn(
                    AccuracyWarning,
                    f"hit-and-run chains of n_mc = {N} < {_MIN_CHAIN} steps are {what}",
                    code="accuracy.correlated_draws",
                    details={"n_mc": N},
                )
        else:
            tau = 1.0
        rep, w = design.node_replicate, design.node_weights
        var_r = np.array(
            [
                float(np.sum(w[rep == r] ** 2 * tau * V[rep == r] / N))
                for r in range(design.n_replicates)
            ]
        )
        se = float(math.sqrt(float(np.sum(var_r))) / design.n_replicates)
        if not math.isfinite(se):
            raise _nonfinite("inactive standard error")
    return {
        "node_means": G,
        "node_variances": V,
        "estimate": mu,
        "inactive_std_error": se,
        "total_std_error": total,
        "replicate_estimates": mu_r,
        "autocorrelation_time": tau,
    }


def _check_design(design: Any) -> IntegrationDesign:
    if not isinstance(design, IntegrationDesign):
        raise _param(
            f"design: expected an IntegrationDesign, got {type(design).__name__}", "design"
        )
    if design.n_mc == 0:
        raise _param(
            "design: has no inactive draws (n_mc = 0); use integrate_surrogate for a function of "
            "y",
            "design",
            reason="no_draws",
        )
    return design


def estimate_integral(design: IntegrationDesign, *, f: ArrayLike) -> IntegrationResult:
    r"""Combine model values at a design into an integral estimate.

    Use this when the model is evaluated outside the library: build the design
    with :func:`integration_design`, evaluate :math:`f` at
    ``design.X_physical``, and pass the values here. Deterministic.

    Parameters
    ----------
    design : IntegrationDesign
        A design with ``n_mc >= 1``.
    f : array_like, shape (Q * N_MC,) or (Q * N_MC, 1)
        Values of the model at the rows of ``design.X_physical`` (physical
        coordinates), in row order.

    Returns
    -------
    IntegrationResult
        ``estimate``, ``inactive_std_error``, ``total_std_error`` (when the
        design has :math:`R \ge 2` replicates) and the per-node means and
        variances.

    Raises
    ------
    InputError
        ``input.parameter`` if `design` is not an ``IntegrationDesign`` or has
        no inactive draws (``n_mc == 0``); ``input.size_mismatch``,
        ``input.shape``, ``input.dtype``, ``input.nonfinite`` or
        ``input.vector_qoi`` if `f` does not hold one finite value per row of
        ``design.X``.
    NumericalError
        ``numerical.nonfinite`` if an intermediate quantity is not finite.

    Warns
    -----
    AccuracyWarning
        ``accuracy.correlated_draws`` for hit-and-run draws with
        :math:`2 \le N_{MC} < 20`, for which the autocorrelation time is
        unreliable and ``inactive_std_error`` may be too small.

    See Also
    --------
    integrate : Design, model evaluations and estimate in one call.

    Notes
    -----
    Per node, :math:`\hat G_k` and :math:`\hat V_k` are the mean and the
    ddof = 1 variance of the :math:`N_{MC}` values; per replicate,
    :math:`\hat\mu_r = \sum_k \omega_k \hat G_k` and

    .. math::

        \hat\sigma_{in,r} = \Bigl( \sum_k \omega_k^2 \, \hat\tau \, \hat V_k / N_{MC}
        \Bigr)^{1/2} .

    :math:`\hat\tau = 1` for independent draws. For hit-and-run draws,
    :math:`\hat\tau` is the integrated autocorrelation time of :math:`f`
    along the chains, pooled over nodes, by Geyer's initial positive sequence
    [1]_: with :math:`\hat r_\ell` the pooled lag-:math:`\ell` autocorrelation
    and :math:`\Gamma_t = \hat r_{2t} + \hat r_{2t+1}`,
    :math:`\hat\tau = \min(N_{MC}, \max(1, -1 + 2 \sum_{t=0}^{T} \Gamma_t))`,
    where :math:`T` is the last :math:`t` before the first
    :math:`\Gamma_t \le 0` (:math:`\hat\tau = 1` when :math:`N_{MC} < 4`). The
    estimate and the replicate standard error are as in
    :class:`IntegrationResult`.

    *Departure from the 2016 package.* The 2016 ``integrate`` used ddof = 0,
    treated hit-and-run draws as independent and returned
    :math:`\hat\mu \pm 1.96 \hat\sigma`; v1 uses ddof = 1, the autocorrelation
    time, and reports standard errors only.

    References
    ----------
    .. [1] C. J. Geyer. Practical Markov chain Monte Carlo. *Statist. Sci.*
       7(4):473-483, 1992.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
    >>> model = ac.test_functions.ridge([1.0, 0.5], profile="sin", space=space)
    >>> sub = ac.subspace_from_basis([[1.0], [0.5]], space=space)
    >>> design = ac.integration_design(
    ...     ac.active_domain(sub), n_nodes=4, n_mc=10, n_replicates=2, rng=0
    ... )
    >>> f = ac.evaluate(model, X=design.X_physical)
    >>> res = ac.estimate_integral(design, f=f)
    >>> res.replicate_estimates.shape
    (2,)
    >>> print(abs(res.estimate) < 3 * res.total_std_error)
    True
    """
    design = _check_design(design)
    fv = _v.values(f, M=design.X.shape[0])
    prov = _make_provenance(
        "estimate_integral",
        {"design": design},
        space=design.domain.subspace.space,
        digests={"f": _v.digest(fv)},
    )
    return IntegrationResult(
        source="model",
        design=design,
        surrogate=None,
        f=fv,
        **_estimate(design, fv),
        qoi=design.domain.subspace.qoi,
        provenance=prov,
    )


def integrate(
    model: Model | Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    domain: ActiveDomain,
    n_nodes: int | Sequence[int],
    n_mc: int = 50,  # [D-171]
    n_replicates: int = 1,
    n_weight_samples: int = 100_000,
    burn_in: int | None = None,
    thin: int | None = None,
    backend: SolverBackend | str | None = None,
    executor: concurrent.futures.Executor | None = None,
    batch_size: int | None = None,
    rng: _random.RngLike = None,
) -> IntegrationResult:
    r"""Integrate a model with a rule on the active variables and Monte Carlo on the rest.

    Estimates :math:`\mu = \int f(x) \rho(x)\, dx` by
    :func:`integration_design`, then the model at ``design.X_physical``, then
    :func:`estimate_integral`, with the same `rng`. The cost is
    :math:`Q \cdot N_{MC}` model evaluations, :math:`Q` being the total number
    of nodes over the replicates.

    Parameters
    ----------
    model : Model or callable
        A ``Model`` whose space equals ``domain.subspace.space``, or a batch
        callable mapping physical points ``(k, m)`` to values ``(k,)``.
    domain : ActiveDomain
        The active domain, from :func:`~active_subspaces.active_domain`.
    n_nodes : int or sequence of int
        As for :func:`active_quadrature_rule`.
    n_mc : int, default 50
        Inactive draws per node, :math:`N_{MC} \ge 1`.
    n_replicates : int, default 1
        Number of independent replicates :math:`R`; :math:`R \ge 2` gives
        ``total_std_error``.
    n_weight_samples : int, default 100000
        :math:`N_X` for the zonotope rule.
    burn_in : int, optional
        As for :func:`integration_design`.
    thin : int, optional
        As for :func:`integration_design`.
    backend : SolverBackend, "scipy" or None, optional
        As for :func:`integration_design`.
    executor : concurrent.futures.Executor, optional
        Runs the model evaluations; see :func:`~active_subspaces.evaluate`.
        The result does not depend on the number of workers.
    batch_size : int, optional
        Rows per model call; see :func:`~active_subspaces.evaluate`.
    rng : None, int, SeedSequence or Generator, optional
        Random source of the design.

    Returns
    -------
    IntegrationResult
        ``estimate``, ``inactive_std_error``, ``total_std_error`` and the
        design and model values.

    Raises
    ------
    InputError
        ``input.parameter`` for an invalid argument as in
        :func:`integration_design`, an invalid `batch_size`, or a ``Model``
        whose space differs from ``domain.subspace.space``.
    ModelError
        ``model.*`` if the model raises, or returns values of the wrong shape,
        dtype or that are not finite.
    NumericalError
        ``numerical.singular`` or ``numerical.nonfinite`` as in
        :func:`integration_design` and :func:`estimate_integral`.
    SolverError
        ``solver.failed`` as in :func:`integration_design`.

    Warns
    -----
    AccuracyWarning
        ``accuracy.weight_deficit``, ``accuracy.correlated_draws`` and
        ``accuracy.degenerate_fiber`` as in :func:`integration_design` and
        :func:`estimate_integral`.

    See Also
    --------
    integration_design, estimate_integral : The two steps, for models evaluated elsewhere.
    integrate_surrogate : Integrate a function of the active variables only.

    Notes
    -----
    Write :math:`x = W_1 y + W_2 z`. Then [1]_

    .. math::

        \mu = \int G(y) \pi(y)\, dy, \qquad
        G(y) = \int f(W_1 y + W_2 z) \pi(z \mid y)\, dz,

    and the estimate is :math:`\hat\mu = \sum_k \omega_k \hat G_k` with
    :math:`\hat G_k` the mean of :math:`f` over the :math:`N_{MC}` inactive
    draws at node :math:`y_k`. The error components are described in
    :class:`IntegrationResult`. For hit-and-run draws with the default
    :math:`N_{MC} = 50`, ``inactive_std_error`` was within about 10% of the
    empirical spread in calibration runs with up to 19 inactive variables; the
    shortfall grows with :math:`m - n`, so use a larger `n_mc` or `thin` when
    :math:`m - n` is large.

    References
    ----------
    .. [1] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.

    Examples
    --------
    The ridge :math:`f(x) = \sin(a^\top x)` has mean 0 over the box.

    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
    >>> model = ac.test_functions.ridge([1.0, 0.5], profile="sin", space=space)
    >>> sub = ac.subspace_from_basis([[1.0], [0.5]], space=space)
    >>> res = ac.integrate(model, domain=ac.active_domain(sub), n_nodes=4, n_mc=10, rng=0)
    >>> print(res.n_evaluations, res.design.inactive_sampler)
    40 exact_interval
    >>> print(abs(res.estimate) < 3 * res.inactive_std_error)
    True
    """
    domain = _check_domain(domain)
    mdl = _check_model(model, domain)
    _v.integer(n_mc, "n_mc", lo=1)
    if batch_size is not None:
        _v.integer(batch_size, "batch_size", lo=1)
    design = integration_design(
        domain,
        n_nodes=n_nodes,
        n_mc=n_mc,
        n_replicates=n_replicates,
        n_weight_samples=n_weight_samples,
        burn_in=burn_in,
        thin=thin,
        backend=backend,
        rng=rng,
    )
    space = domain.subspace.space
    fv = evaluate(mdl, X=design.X_physical, space=space, executor=executor, batch_size=batch_size)
    fv = _v.values(fv, M=design.X.shape[0])
    params = dict(design.provenance.parameters)
    params.update({"model": mdl.name, "batch_size": batch_size})
    prov = _make_provenance("integrate", params, space=space, rng_state=design.provenance.rng)
    return IntegrationResult(
        source="model",
        design=design,
        surrogate=None,
        f=fv,
        **_estimate(design, fv),
        qoi=domain.subspace.qoi,
        provenance=prov,
    )


def integrate_surrogate(
    surrogate: Surrogate | Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    domain: ActiveDomain,
    n_nodes: int | Sequence[int],
    n_replicates: int = 1,
    n_weight_samples: int = 100_000,
    rng: _random.RngLike = None,
) -> IntegrationResult:
    r"""Integrate a function of the active variables with a rule on the active domain.

    For :math:`g` a function of :math:`y` only,
    :math:`\int g(W_1^\top x) \rho(x)\, dx = \int g(y) \pi(y)\, dy \approx
    \sum_k \omega_k g(y_k)`; no inactive draws and no model evaluations are
    needed.

    Parameters
    ----------
    surrogate : Surrogate or callable
        A surrogate whose ``subspace`` equals ``domain.subspace`` (for example
        from :func:`~active_subspaces.fit_surrogate`, or a ``ReducedModel``), or
        a callable mapping normalized active coordinates ``(K, n)`` to values
        ``(K,)``.
    domain : ActiveDomain
        The active domain, from :func:`~active_subspaces.active_domain`.
    n_nodes : int or sequence of int
        As for :func:`active_quadrature_rule`.
    n_replicates : int, default 1
        Number of independent replicates :math:`R`; only the zonotope rule is
        random, so :math:`R \ge 2` is useful there.
    n_weight_samples : int, default 100000
        :math:`N_X` for the zonotope rule.
    rng : None, int, SeedSequence or Generator, optional
        Random source of the zonotope rule.

    Returns
    -------
    IntegrationResult
        With ``source == "surrogate"``, ``node_means`` the values
        :math:`g(y_k)`, and ``inactive_std_error`` ``None``.

    Raises
    ------
    InputError
        ``input.parameter`` for an invalid argument as in
        :func:`integration_design`, a surrogate whose subspace is not
        ``domain.subspace`` (including a surrogate on all inputs), or an object
        that is neither a surrogate nor callable; ``input.shape`` if the values
        are not a float array of shape ``(Q,)``.
    NumericalError
        ``numerical.nonfinite`` if a value at a node is not finite;
        ``numerical.singular`` as in :func:`active_quadrature_rule`.

    Warns
    -----
    AccuracyWarning
        ``accuracy.weight_deficit`` as in :func:`active_quadrature_rule`.

    See Also
    --------
    active_quadrature_rule : The rule used.
    integrate : Integrate a model of all the inputs.

    Notes
    -----
    The estimate is :math:`\hat\mu = R^{-1} \sum_r \sum_k \omega_{k,r} g(y_{k,r})`.
    It is the exact integral of :math:`g(W_1^\top x)` up to the quadrature
    error of the rule; whether :math:`g(W_1^\top x)` approximates :math:`f(x)`
    well is a property of the surrogate.

    Examples
    --------
    The two-node rule integrates :math:`y^2` exactly; here
    :math:`E[y^2] = 1/3`.

    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0], [1.0, 2.0])
    >>> sub = ac.subspace_from_basis([[1.0], [0.5]], space=space)
    >>> res = ac.integrate_surrogate(
    ...     lambda Y: Y[:, 0] ** 2, domain=ac.active_domain(sub), n_nodes=2
    ... )
    >>> print(res.source, f"{res.estimate:.6f}")
    surrogate 0.333333
    """
    domain = _check_domain(domain)
    g, obj = _check_surrogate(surrogate, domain)
    design = integration_design(
        domain,
        n_nodes=n_nodes,
        n_mc=0,
        n_replicates=n_replicates,
        n_weight_samples=n_weight_samples,
        rng=rng,
    )
    Q = design.nodes.shape[0]
    raw = g(np.array(design.nodes))
    try:
        vals = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputError(
            f"surrogate: returned {type(raw).__name__}, not a float array of shape ({Q},)",
            code="input.shape",
            details={"param": "surrogate", "expected": [Q]},
        ) from exc
    if vals.shape != (Q,):
        raise InputError(
            f"surrogate: expected values of shape ({Q},), got {vals.shape}",
            code="input.shape",
            details={"param": "surrogate", "expected": [Q], "got": list(vals.shape)},
        )
    if not np.isfinite(vals).all():
        raise NumericalError(
            "surrogate: returned non-finite values at the rule nodes",
            code="numerical.nonfinite",
            details={
                "what": "surrogate values",
                "n_nonfinite": int(np.count_nonzero(~np.isfinite(vals))),
            },
        )
    mu_r, mu, total = _combine(design, vals)
    params = dict(design.provenance.parameters)
    params.pop("n_mc", None)
    params["surrogate"] = surrogate
    prov = _make_provenance(
        "integrate_surrogate", params, space=domain.subspace.space, rng_state=design.provenance.rng
    )
    return IntegrationResult(
        source="surrogate",
        design=design,
        surrogate=_builtin_surrogate(obj),
        f=np.zeros(0),
        node_means=vals,
        node_variances=np.full(Q, np.nan),
        estimate=mu,
        inactive_std_error=None,
        total_std_error=total,
        replicate_estimates=mu_r,
        autocorrelation_time=None,
        qoi=domain.subspace.qoi,
        provenance=prov,
    )
