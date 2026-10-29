r"""The one-object reduced-model workflow.

:func:`build_reduced_model` estimates an active subspace from data or from a
model, builds its active domain and a surrogate on the active variables, and
returns a frozen :class:`ReducedModel`. The operations are functions of this
module, with delegating methods of the same names on :class:`ReducedModel`:

- :func:`predict` and :func:`predict_gradient` (the functions of
  :mod:`active_subspaces.surrogates`, through the ``Surrogate`` protocol);
- :func:`average`, the mean of :math:`f` by active-variable integration;
- :func:`probability`, plain Monte Carlo on the surrogate with a Wilson score
  interval;
- :func:`minimum`, minimization of the surrogate over the active domain and
  the lift of the minimizer.

The reduced model approximates :math:`f(\xi) \approx g(W_1^\top A^{-1}(\xi - b))`,
where :math:`\xi = A x + b` maps normalized inputs :math:`x` to physical inputs
:math:`\xi`, and :math:`g` is a surrogate on the active variables
:math:`y = W_1^\top x`.

Model evaluations of :func:`build_reduced_model` on the model path: ``M``
values; ``M`` gradients, or ``M m`` extra values for forward finite
differences when the model has no gradient; plus ``n_design * n_mc`` values
for the surrogate and the evaluations at the vertices of the active domain.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
import scipy.special
from numpy.typing import ArrayLike, NDArray

from active_subspaces import _random
from active_subspaces import _validation as _v
from active_subspaces import subspaces as _subspaces
from active_subspaces.errors import AccuracyWarning, InputError, _warn
from active_subspaces.geometry import ActiveDomain, active_domain
from active_subspaces.gradients import (
    GradientResult,
    _step,
    finite_difference_gradients,
    local_linear_gradients,
)
from active_subspaces.integration import (
    IntegrationResult,
    integrate,
    integrate_surrogate,
)
from active_subspaces.models import (
    Model,
    ValueFn,
    as_model,
    evaluate,
    evaluate_with_grads,
)
from active_subspaces.optimization import MinimizationResult, minimize
from active_subspaces.results import (
    QoIResult,
    _invariant,
    _make_provenance,
    _register_result,
)
from active_subspaces.spaces import InputSpace
from active_subspaces.subspaces import SubspaceResult, active_subspace, with_dimension
from active_subspaces.surrogates import (
    PolynomialSurrogate,
    RBFSurrogate,
    fit_surrogate,
    fit_surrogate_from_model,
    predict,
    predict_gradient,
)

__all__ = (
    "ProbabilityResult",
    "ReducedModel",
    "average",
    "build_reduced_model",
    "minimum",
    "predict",
    "predict_gradient",
    "probability",
)

# Re-exported at the top level; the generic operation names stay module-qualified [D-148].
_toplevel = ("ProbabilityResult", "ReducedModel", "build_reduced_model")

_log = logging.getLogger(__name__)

_ALPHA = 6  # oversampling of the default M = floor(alpha (m + 1) ln m) [D-115]
_N_MC = 50  # [D-171]
_NBOOT = 100
_FD_STEP = 1e-6
_ENVELOPE = {"uniform": 4, "gaussian": 6}  # [D-63], [D-121]
_WEIGHT_TOL = 1e-10
_CHUNK_ELEMENTS = 1 << 20  # entries per probability draw chunk (~8 MB)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@_register_result
class ReducedModel(QoIResult):
    r"""A subspace, its active domain and a surrogate on the active variables.

    Built by :func:`build_reduced_model`; the constructor is used by
    deserialization (:func:`~active_subspaces.from_dict`,
    :func:`~active_subspaces.load`). A reduced model satisfies the
    :class:`~active_subspaces.surrogates.Surrogate` protocol by delegating
    :meth:`evaluate` and :meth:`gradient` to ``surrogate``, so it can be
    passed wherever a surrogate is accepted.

    Parameters
    ----------
    space : Uniform or Gaussian
        The input space; it maps physical inputs :math:`\xi` to normalized
        inputs :math:`x`.
    subspace : SubspaceResult
        The active subspace, with ``1 <= subspace.n <= m - 1``; it carries
        ``n``, ``W``, the eigenvalues and any bootstrap replicates.
    domain : ActiveDomain
        The active domain of ``subspace`` (normalized active coordinates).
    surrogate : PolynomialSurrogate or RBFSurrogate
        The surrogate :math:`g` on the normalized active variables
        :math:`y = W_1^\top x`.
    X : ndarray, shape (M, m), float64
        The samples behind the subspace, in **normalized** coordinates
        (:attr:`X_physical` gives them in physical coordinates).
    f : ndarray, shape (M,), float64
        The function values at the samples.
    grads : ndarray, shape (M, m), float64, or None
        The gradients used to estimate the subspace, in **normalized**
        coordinates; ``None`` for an estimator that uses values only.
    quad_weights : ndarray, shape (M,), float64
        Non-negative quadrature weights summing to 1 (``1/M`` on the model
        path).
    source : {"data", "model"}
        The build path.
    gradient_source : {"given", "local_linear", "model", "finite_difference", "none"}
        Where ``grads`` came from: passed by the caller, estimated by local
        linear models, evaluated by the model, finite differences of the
        model, or not used.
    r_squared : float or None
        Out-of-sample coefficient of determination of the surrogate (see
        Notes); ``None`` when the weighted variance of ``f`` is 0 or the
        leave-one-out residuals are undefined.
    r_squared_kind : {"holdout", "loo"}
        ``"holdout"`` on the model path, ``"loo"`` (leave-one-out) on the
        data path.
    provenance : Provenance
        How the result was computed.
    qoi : str
        Name of the quantity of interest (default ``"f"``).

    See Also
    --------
    build_reduced_model : Build a reduced model from data or a model.

    Notes
    -----
    The reduced model approximates

    .. math::

        f(\xi) \approx g\bigl(W_1^\top A^{-1}(\xi - b)\bigr),

    where :math:`\xi = A x + b` is the map from normalized to physical
    inputs. With weights :math:`\omega_i` (:math:`\sum_i \omega_i = 1`) and
    out-of-sample predictions :math:`\hat g_i`, the fit quality is

    .. math::

        R^2 = 1 - \frac{\sum_{i=1}^M \omega_i (f_i - \hat g_i)^2}
                       {\sum_{i=1}^M \omega_i (f_i - \bar f)^2},
        \qquad \bar f = \sum_{i=1}^M \omega_i f_i .

    On the model path :math:`\hat g_i = g(W_1^\top x_i)` at the ``M``
    gradient samples, which the surrogate (fitted on its own design) never
    saw, and :math:`\omega_i = 1/M`. On the data path :math:`\hat g_i` are
    the leave-one-out predictions of the surrogate, with the hyperparameters
    held fixed, and :math:`\omega_i` are ``quad_weights``.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
    >>> X = space.sample(40, rng=0)
    >>> f, G = ac.evaluate_with_grads(model, X=X)
    >>> rm = ac.build_reduced_model(
    ...     space=space, X=X, f=f, grads=G, n=1, surrogate_kind="poly", rng=0
    ... )
    >>> print(rm.m, rm.n, rm.source, rm.gradient_source, rm.r_squared_kind)
    3 1 data given loo
    >>> print(np.round(rm.predict(X[:3]), 3))
    [0.564 0.343 0.364]
    """

    TYPE = "ReducedModel"
    SCHEMA_VERSION = 1

    space: InputSpace
    subspace: SubspaceResult
    domain: ActiveDomain
    surrogate: PolynomialSurrogate | RBFSurrogate
    X: NDArray[np.float64]
    f: NDArray[np.float64]
    grads: NDArray[np.float64] | None
    quad_weights: NDArray[np.float64]
    source: Literal["data", "model"]
    gradient_source: Literal["given", "local_linear", "model", "finite_difference", "none"]
    r_squared: float | None
    r_squared_kind: Literal["holdout", "loo"]

    # --- derived ---

    @property
    def m(self) -> int:
        """Number of inputs ``m`` (``space.dim``).

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
        >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
        >>> rm = ac.build_reduced_model(model=model, n=1, surrogate_kind="poly", rng=0)
        >>> rm.m
        3
        """
        return self.space.dim

    @property
    def n(self) -> int:
        """Active dimension ``n`` (``subspace.n``).

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
        >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
        >>> rm = ac.build_reduced_model(model=model, n=1, surrogate_kind="poly", rng=0)
        >>> rm.n
        1
        """
        return self.subspace.n

    @property
    def active_dim(self) -> int:
        """Active dimension ``n``, the name the ``Surrogate`` protocol requires.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
        >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
        >>> rm = ac.build_reduced_model(model=model, n=1, surrogate_kind="poly", rng=0)
        >>> rm.active_dim == rm.n
        True
        """
        return self.subspace.n

    @property
    def W1(self) -> NDArray[np.float64]:
        """Active directions, shape ``(m, n)``, in normalized coordinates (``subspace.W1``).

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
        >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
        >>> rm = ac.build_reduced_model(model=model, n=1, surrogate_kind="poly", rng=0)
        >>> rm.W1.shape
        (3, 1)
        """
        return self.subspace.W1

    @property
    def eigenvalues(self) -> NDArray[np.float64]:
        """Eigenvalues of the subspace, shape ``(m,)`` (NaN for a ``"basis"`` subspace).

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
        >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
        >>> X = space.sample(40, rng=0)
        >>> f, G = ac.evaluate_with_grads(model, X=X)
        >>> rm = ac.build_reduced_model(
        ...     space=space, X=X, f=f, grads=G, n=1, surrogate_kind="poly", rng=0
        ... )
        >>> print(np.round(rm.eigenvalues, 3))
        [0.944 0.124 0.057]
        """
        return self.subspace.eigenvalues

    @property
    def X_physical(self) -> NDArray[np.float64]:
        """The samples ``X`` in physical coordinates, shape ``(M, m)``, read-only.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
        >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
        >>> X = space.sample(40, rng=0)
        >>> f, G = ac.evaluate_with_grads(model, X=X)
        >>> rm = ac.build_reduced_model(
        ...     space=space, X=X, f=f, grads=G, n=1, surrogate_kind="poly", rng=0
        ... )
        >>> bool(np.allclose(rm.X_physical, X))
        True
        """
        return _v.freeze_array(self.space.from_normalized(self.X))

    # --- Surrogate protocol [D-104] ---

    def evaluate(self, Y: ArrayLike) -> NDArray[np.float64]:
        r"""Evaluate the surrogate ``g`` at normalized active coordinates.

        Delegates to ``surrogate.evaluate``.

        Parameters
        ----------
        Y : array_like, shape (K, n)
            Points in **normalized active** coordinates :math:`y = W_1^\top x`.

        Returns
        -------
        ndarray, shape (K,), float64
            The values :math:`g(y)`.

        Raises
        ------
        InputError
            ``input.shape`` if `Y` is not ``(K, n)``; ``input.nonfinite`` if
            it is not finite.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
        >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
        >>> X = space.sample(40, rng=0)
        >>> f, G = ac.evaluate_with_grads(model, X=X)
        >>> rm = ac.build_reduced_model(
        ...     space=space, X=X, f=f, grads=G, n=1, surrogate_kind="poly", rng=0
        ... )
        >>> print(np.round(rm.evaluate([[-1.0], [0.0], [1.0]]), 3))
        [0.902 0.226 0.969]
        """
        return self.surrogate.evaluate(Y)

    def gradient(self, Y: ArrayLike) -> NDArray[np.float64]:
        r"""Evaluate the gradient of ``g`` at normalized active coordinates.

        Delegates to ``surrogate.gradient``.

        Parameters
        ----------
        Y : array_like, shape (K, n)
            Points in **normalized active** coordinates :math:`y = W_1^\top x`.

        Returns
        -------
        ndarray, shape (K, n), float64
            The gradients :math:`\nabla_y g(y)`, in normalized active
            coordinates.

        Raises
        ------
        InputError
            ``input.shape`` if `Y` is not ``(K, n)``; ``input.nonfinite`` if
            it is not finite.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
        >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
        >>> X = space.sample(40, rng=0)
        >>> f, G = ac.evaluate_with_grads(model, X=X)
        >>> rm = ac.build_reduced_model(
        ...     space=space, X=X, f=f, grads=G, n=1, surrogate_kind="poly", rng=0
        ... )
        >>> print(np.round(rm.gradient([[1.0]]), 3))
        [[1.453]]
        """
        return self.surrogate.gradient(Y)

    # --- delegating methods [D-22] ---

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Predict ``f`` at physical points; same as :func:`predict` ``(self, X)``.

        Parameters
        ----------
        X : array_like, shape (K, m)
            Points in **physical** coordinates, inside the support of
            ``space``.

        Returns
        -------
        ndarray, shape (K,), float64
            :math:`g(W_1^\top A^{-1}(\xi - b))` for each row :math:`\xi`.

        Raises
        ------
        InputError
            ``input.shape``, ``input.nonfinite`` or ``input.out_of_space``
            for invalid `X`.

        See Also
        --------
        predict : The module-level function.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
        >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
        >>> X = space.sample(40, rng=0)
        >>> f, G = ac.evaluate_with_grads(model, X=X)
        >>> rm = ac.build_reduced_model(
        ...     space=space, X=X, f=f, grads=G, n=1, surrogate_kind="poly", rng=0
        ... )
        >>> print(np.round(rm.predict(X[:3]), 3))
        [0.564 0.343 0.364]
        """
        return predict(self, X)

    def predict_gradient(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Predict the physical gradient of ``f``; same as :func:`predict_gradient` ``(self, X)``.

        Parameters
        ----------
        X : array_like, shape (K, m)
            Points in **physical** coordinates, inside the support of
            ``space``.

        Returns
        -------
        ndarray, shape (K, m), float64
            :math:`A^{-\top} W_1 \nabla_y g(y)`, the gradient with respect to
            the **physical** inputs.

        Raises
        ------
        InputError
            ``input.shape``, ``input.nonfinite`` or ``input.out_of_space``
            for invalid `X`.

        See Also
        --------
        predict_gradient : The module-level function.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
        >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
        >>> X = space.sample(40, rng=0)
        >>> f, G = ac.evaluate_with_grads(model, X=X)
        >>> rm = ac.build_reduced_model(
        ...     space=space, X=X, f=f, grads=G, n=1, surrogate_kind="poly", rng=0
        ... )
        >>> rm.predict_gradient(X[:5]).shape
        (5, 3)
        """
        return predict_gradient(self, X)

    def average(self, **kw: Any) -> IntegrationResult:
        """Estimate the mean of ``f``; same as :func:`average` ``(self, **kw)``.

        Parameters
        ----------
        **kw
            Keyword arguments of :func:`average` (``n_nodes`` is required).

        Returns
        -------
        IntegrationResult
            The estimate and its error components.

        See Also
        --------
        average : The module-level function, with the full parameter list.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
        >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
        >>> X = space.sample(40, rng=0)
        >>> f, G = ac.evaluate_with_grads(model, X=X)
        >>> rm = ac.build_reduced_model(
        ...     space=space, X=X, f=f, grads=G, n=1, surrogate_kind="poly", rng=0
        ... )
        >>> print(f"{rm.average(n_nodes=5).estimate:.3f}")
        0.463
        """
        return average(self, **kw)

    def probability(self, lower: float, upper: float, **kw: Any) -> ProbabilityResult:
        """Estimate ``P[lower <= g <= upper]``; same as :func:`probability`.

        Parameters
        ----------
        lower, upper : float
            Bounds of the closed interval, ``lower < upper``; ``-inf`` and
            ``inf`` are allowed.
        **kw
            Keyword arguments of :func:`probability` (``n_samples``, ``rng``).

        Returns
        -------
        ProbabilityResult
            The estimate, its standard error and the Wilson interval.

        See Also
        --------
        probability : The module-level function, with the full parameter list.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
        >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
        >>> X = space.sample(40, rng=0)
        >>> f, G = ac.evaluate_with_grads(model, X=X)
        >>> rm = ac.build_reduced_model(
        ...     space=space, X=X, f=f, grads=G, n=1, surrogate_kind="poly", rng=0
        ... )
        >>> print(rm.probability(0.3, 0.6, n_samples=2000, rng=0).p)
        0.391
        """
        return probability(self, lower, upper, **kw)

    def minimum(self, **kw: Any) -> MinimizationResult:
        """Approximate ``min f``; same as :func:`minimum` ``(self, **kw)``.

        Parameters
        ----------
        **kw
            Keyword arguments of :func:`minimum` (``model``, ``n_starts``,
            ``executor``, ``batch_size``, ``rng``).

        Returns
        -------
        MinimizationResult
            The active minimizer, its lift and the values there.

        See Also
        --------
        minimum : The module-level function, with the full parameter list.

        Examples
        --------
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
        >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
        >>> X = space.sample(40, rng=0)
        >>> f, G = ac.evaluate_with_grads(model, X=X)
        >>> rm = ac.build_reduced_model(
        ...     space=space, X=X, f=f, grads=G, n=1, surrogate_kind="poly", rng=0
        ... )
        >>> print(f"{rm.minimum(rng=0).f_star:.3f}")
        0.226
        """
        return minimum(self, **kw)

    def _check(self) -> None:
        sp, sub = self.space, self.subspace
        _invariant(_same_space(sub.space, sp), "subspace.space must equal space")
        _invariant(_same_space(self.surrogate.space, sp), "surrogate.space must equal space")
        _invariant(self.domain.subspace == sub, "domain.subspace must equal subspace")
        _invariant(
            self.surrogate.subspace is not None and self.surrogate.subspace == sub,
            "surrogate.subspace must equal subspace",
        )
        _invariant(
            self.qoi == sub.qoi == self.surrogate.qoi,
            "qoi must equal subspace.qoi and surrogate.qoi",
        )
        m = sp.dim
        _invariant(m >= 2 and 1 <= sub.n <= m - 1, "need m >= 2 and 1 <= n <= m - 1")
        X = self.X
        _invariant(X.ndim == 2 and X.shape[1] == m and X.shape[0] >= 1, "X must be (M, m)")
        M = X.shape[0]
        _invariant(self.f.shape == (M,), "f must be (M,)")
        _invariant(
            self.grads is None or self.grads.shape == (M, m), "grads must be None or (M, m)"
        )
        w = self.quad_weights
        _invariant(w.shape == (M,), "quad_weights must be (M,)")
        for name in ("X", "f", "grads", "quad_weights"):
            a = getattr(self, name)
            _invariant(a is None or bool(np.isfinite(a).all()), f"{name} must be finite")
        _invariant(
            bool(np.all(w >= 0)) and abs(math.fsum(w.tolist()) - 1.0) <= _WEIGHT_TOL,
            "quad_weights must be >= 0 and sum to 1",
        )
        if self.source == "data":
            _invariant(
                self.gradient_source in ("given", "local_linear", "none"),
                "bad gradient_source for source 'data'",
            )
            _invariant(
                self.r_squared_kind == "loo", "r_squared_kind must be 'loo' for source 'data'"
            )
        else:
            _invariant(
                self.gradient_source in ("model", "finite_difference", "none"),
                "bad gradient_source for source 'model'",
            )
            _invariant(
                self.r_squared_kind == "holdout",
                "r_squared_kind must be 'holdout' for source 'model'",
            )
        _invariant(
            (self.gradient_source == "none") == (self.grads is None),
            "gradient_source 'none' iff grads is None",
        )
        r2 = self.r_squared
        _invariant(
            r2 is None or (math.isfinite(r2) and r2 <= 1.0),
            "r_squared must be None or a finite float <= 1",
        )


@_register_result
class ProbabilityResult(QoIResult):
    r"""Monte Carlo estimate of ``P[lower <= g(W1^T x) <= upper]`` under the surrogate.

    Built by :func:`probability`; the constructor is used by deserialization.
    The estimate is the probability **under the surrogate** :math:`g`, not
    under :math:`f`, and :attr:`std_error` and :meth:`confidence_interval`
    cover only its Monte Carlo error.

    Parameters
    ----------
    reduced_model : ReducedModel
        The reduced model whose surrogate was sampled.
    lower, upper : float
        Bounds of the closed interval, ``lower < upper``; ``-inf`` and ``inf``
        are allowed.
    p : float
        The estimate :math:`\hat p \in [0, 1]`.
    std_error : float
        :math:`\sqrt{\hat p (1 - \hat p) / N}`.
    n_samples : int
        The number :math:`N` of Monte Carlo draws.
    provenance : Provenance
        How the result was computed.
    qoi : str
        Name of the quantity of interest, ``reduced_model.qoi``.

    See Also
    --------
    probability : Compute this result.

    Notes
    -----
    With :math:`x_i` drawn independently from the input density :math:`\rho`,

    .. math::

        \hat p = \frac{1}{N} \sum_{i=1}^N
        \mathbf 1\bigl[\ell \le g(W_1^\top x_i) \le u\bigr],
        \qquad
        \widehat{\mathrm{se}} = \sqrt{\hat p (1 - \hat p) / N},

    where :math:`\ell` is ``lower`` and :math:`u` is ``upper``. Averaging
    over the inactive variables smooths :math:`f`, so :math:`\hat p` can
    differ from :math:`\mathbb P[\ell \le f \le u]` by more than its Monte
    Carlo error.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
    >>> X = space.sample(40, rng=0)
    >>> f, G = ac.evaluate_with_grads(model, X=X)
    >>> rm = ac.build_reduced_model(
    ...     space=space, X=X, f=f, grads=G, n=1, surrogate_kind="poly", rng=0
    ... )
    >>> res = ac.reduced_model.probability(rm, 0.3, 0.6, n_samples=2000, rng=0)
    >>> print(res.p, f"{res.std_error:.4f}", res.n_samples)
    0.391 0.0109 2000
    """

    TYPE = "ProbabilityResult"
    SCHEMA_VERSION = 1

    reduced_model: ReducedModel
    lower: float
    upper: float
    p: float
    std_error: float
    n_samples: int

    def confidence_interval(self, level: float = 0.99) -> tuple[float, float]:
        r"""Return the Wilson score interval for :attr:`p` at a confidence level.

        Parameters
        ----------
        level : float, default 0.99
            Confidence level :math:`c`, ``0 < level < 1``. The default keeps
            the 2016 level.

        Returns
        -------
        lo : float
            Lower bound, ``0 <= lo <= p``.
        hi : float
            Upper bound, ``p <= hi <= 1``.

        Raises
        ------
        InputError
            ``input.parameter`` if `level` is not a real number in (0, 1).

        Notes
        -----
        With :math:`q = \Phi^{-1}\bigl((1 + c)/2\bigr)` the Wilson score
        interval [1]_ is :math:`[\tilde p - h, \tilde p + h]`, where

        .. math::

            \tilde p = \frac{\hat p + q^2/(2N)}{1 + q^2/N},
            \qquad
            h = \frac{q}{1 + q^2/N}
                \sqrt{\frac{\hat p (1 - \hat p)}{N} + \frac{q^2}{4N^2}} .

        Unlike the normal interval :math:`\hat p \pm q\,\widehat{\mathrm{se}}`
        used in 2016, it lies in :math:`[0, 1]` and does not collapse to a
        point at :math:`\hat p \in \lbrace 0, 1 \rbrace`. The bounds are
        clipped to :math:`[0, \hat p]` and :math:`[\hat p, 1]` against
        rounding.

        References
        ----------
        .. [1] E. B. Wilson. Probable inference, the law of succession, and
           statistical inference. *J. Amer. Statist. Assoc.*
           22(158):209–212, 1927.

        Examples
        --------
        >>> import numpy as np
        >>> import active_subspaces as ac
        >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
        >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
        >>> X = space.sample(40, rng=0)
        >>> f, G = ac.evaluate_with_grads(model, X=X)
        >>> rm = ac.build_reduced_model(
        ...     space=space, X=X, f=f, grads=G, n=1, surrogate_kind="poly", rng=0
        ... )
        >>> res = rm.probability(0.3, 0.6, n_samples=2000, rng=0)
        >>> print(np.round(res.confidence_interval(0.95), 3))
        [0.37  0.413]
        """
        c = _v.real(level, "level", lo=0.0, hi=1.0, lo_open=True, hi_open=True)
        return _wilson(self.p, self.n_samples, c)

    def _check(self) -> None:
        _invariant(
            not (math.isnan(self.lower) or math.isnan(self.upper)),
            "lower and upper must not be NaN",
        )
        _invariant(self.lower < self.upper, "lower < upper required")
        _invariant(self.n_samples >= 1, "n_samples must be >= 1")
        _invariant(0.0 <= self.p <= 1.0, "p must lie in [0, 1]")
        _invariant(
            math.isfinite(self.std_error) and self.std_error >= 0.0,
            "std_error must be finite and >= 0",
        )
        _invariant(self.qoi == self.reduced_model.qoi, "qoi must equal reduced_model.qoi")


def _wilson(p: float, N: int, c: float) -> tuple[float, float]:
    q = float(scipy.special.ndtri((1.0 + c) / 2.0))
    q2 = q * q
    denom = 1.0 + q2 / N
    center = (p + q2 / (2.0 * N)) / denom
    half = q / denom * math.sqrt(p * (1.0 - p) / N + q2 / (4.0 * N * N))
    lo = min(max(center - half, 0.0), p)
    hi = max(min(center + half, 1.0), p)
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _same_space(a: Any, b: Any) -> bool:
    return type(a) is type(b) and a == b


def _param(message: str, param: str, **details: Any) -> InputError:
    return InputError(message, code="input.parameter", details={"param": param, **details})


def _space_mismatch(param: str, what: str) -> InputError:
    return _param(f"{param}: its space differs from {what}", param, reason="space_mismatch")


def _default_n_samples(m: int) -> int:
    """``floor(6 (m + 1) ln m)`` [D-115].

    0 at ``m = 1`` [C-08], which is why ``m >= 2`` is required.
    """
    return math.floor(_ALPHA * (m + 1) * math.log(m))


def _r_squared(f: np.ndarray, ghat: np.ndarray, w: np.ndarray) -> float | None:
    """Weighted out-of-sample ``R^2`` of spec 11 §3.3 [D-120].

    ``None`` when the weighted variance of ``f`` is 0.
    """
    f = np.asarray(f, dtype=np.float64)
    ghat = np.asarray(ghat, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    pos = w > 0
    if not pos.any() or np.ptp(f[pos]) == 0:
        return None
    w = w / w.sum()
    fbar = float(np.sum(w * f))
    den = float(np.sum(w * (f - fbar) ** 2))
    if not den > 0:
        return None
    r2 = 1.0 - float(np.sum(w * (f - ghat) ** 2)) / den
    return r2 if math.isfinite(r2) else None


def _check_executor(executor: Any) -> None:
    if executor is not None and not callable(getattr(executor, "submit", None)):
        raise _param("executor: expected a concurrent.futures.Executor", "executor")


def _opt_int(value: Any, name: str, lo: int) -> int | None:
    return None if value is None else _v.integer(value, name, lo=lo)


def _check_rm(rm: Any) -> ReducedModel:
    if not isinstance(rm, ReducedModel):
        raise _param(f"rm: expected a ReducedModel, got {type(rm).__name__}", "rm")
    return rm


def _name_of(obj: Any) -> Any:
    """Name and options of an estimator or rule instance, for provenance.

    A string passes through.
    """
    if obj is None or isinstance(obj, str):
        return obj
    name = getattr(obj, "name", None)
    options = getattr(obj, "options", None)
    if isinstance(name, str) and callable(options):
        return {"name": name, "options": options()}
    return repr(obj)


# ---------------------------------------------------------------------------
# build_reduced_model (§5.1)
# ---------------------------------------------------------------------------


def build_reduced_model(
    *,
    space: InputSpace | None = None,
    X: ArrayLike | None = None,
    f: ArrayLike | None = None,
    grads: ArrayLike | GradientResult | None = None,
    quad_weights: ArrayLike | None = None,
    model: Model | ValueFn | None = None,
    subspace: SubspaceResult | None = None,
    n: int | None = None,
    estimator: str | _subspaces.Estimator | None = None,
    dimension_rule: str | _subspaces.DimensionRule | None = None,
    nboot: int | None = None,
    surrogate_kind: Literal["rbf", "poly"] = "rbf",
    degree: int = 2,
    n_samples: int | None = None,
    n_design: int | None = None,
    n_mc: int | None = None,
    fd_step: float | None = None,
    executor: concurrent.futures.Executor | None = None,
    batch_size: int | None = None,
    qoi: str | None = None,
    rng: _random.RngLike = None,
) -> ReducedModel:
    r"""Build a reduced model ``f(xi) ~ g(W1^T A^{-1}(xi - b))`` from data or from a model.

    The builder estimates an active subspace (or reuses ``subspace``), builds
    its active domain, fits a surrogate on the active variables and measures
    its out-of-sample fit. Give exactly one source: data (``X`` and ``f``,
    optionally ``grads`` and ``quad_weights``) or ``model``.

    - **Data path.** The subspace is estimated from the samples; gradients
      are ``grads`` if given, else local linear gradients for an estimator
      that needs them. The surrogate is fitted to the pairs
      :math:`(W_1^\top x_i, f_i)`.
    - **Model path.** ``n_samples`` points are drawn from the input density
      and the model is evaluated there (with its gradient, or forward finite
      differences when it has none). The surrogate is fitted to conditional
      means :math:`\hat G(y_k)` of ``n_mc`` inactive draws at each of
      ``n_design`` design points on the active domain.

    Parameters
    ----------
    space : Uniform or Gaussian, optional
        The input space. Required with data unless ``subspace`` is given;
        otherwise taken from ``model`` or ``subspace``. Every space given or
        bound must be equal.
    X : array_like, shape (M, m), optional
        Data path: samples in **physical** coordinates, inside the support of
        the space.
    f : array_like, shape (M,) or (M, 1), optional
        Data path: function values at `X`.
    grads : array_like, shape (M, m), or GradientResult, optional
        Data path: gradients with respect to the **physical** inputs at `X`
        (the library applies the chain rule), or a
        :class:`~active_subspaces.GradientResult` for the same space and
        ``M`` rows.
    quad_weights : array_like, shape (M,), optional
        Data path: non-negative quadrature weights summing to 1; default
        ``1/M``.
    model : Model or callable, optional
        Model path: a :class:`~active_subspaces.Model` or a function of
        **physical** points ``(K, m) -> (K,)``; a bare callable needs ``space``
        or ``subspace``.
    subspace : SubspaceResult, optional
        A computed subspace to reuse; estimation is skipped and ``estimator``,
        ``dimension_rule`` and ``nboot`` must be ``None``. With ``n`` given,
        :func:`~active_subspaces.with_dimension` is applied first.
    n : int, optional
        Active dimension, ``1 <= n <= m - 1``; default: chosen by the
        dimension rule (or ``subspace.n``).
    estimator : str or Estimator, optional
        Subspace estimator name or instance (see
        :data:`active_subspaces.subspaces.ESTIMATORS`); default ``"as"``.
    dimension_rule : str or DimensionRule, optional
        Rule that chooses ``n`` when ``n`` is ``None``; default
        ``EVG(gap="linear")``.
    nboot : int, optional
        Bootstrap replicates of the subspace estimate, ``>= 0``; default 100.
    surrogate_kind : {"rbf", "poly"}, default "rbf"
        Radial-basis or polynomial surrogate.
    degree : int, default 2
        Total degree of the polynomial, or of the RBF's polynomial tail.
    n_samples : int, optional
        Model path: the number ``M`` of samples; default
        :math:`\lfloor 6 (m + 1) \ln m \rfloor`.
    n_design : int, optional
        Model path: number of surrogate design points; default :math:`5^n`.
    n_mc : int, optional
        Model path: inactive draws per design point, ``>= 2``; default 50.
    fd_step : float, optional
        Model path, model without gradient: relative forward-difference step
        in normalized coordinates; default ``1e-6``.
    executor : concurrent.futures.Executor, optional
        Model path: executor for every model evaluation.
    batch_size : int, optional
        Model path: rows per model call.
    qoi : str, optional
        Name of the quantity of interest; default ``model.qoi``, else
        ``subspace.qoi``, else ``"f"``.
    rng : None, int, SeedSequence or Generator, optional
        Seed or generator; spawned into independent streams for the samples,
        the subspace bootstrap and the surrogate design.

    Returns
    -------
    ReducedModel
        The subspace, active domain, surrogate, samples (normalized) and the
        out-of-sample ``r_squared``.

    Raises
    ------
    InputError
        ``input.parameter`` for a bad or inconsistent argument (both or
        neither source, a model-path parameter with data, mismatched spaces
        or ``qoi``, ``m < 2``, ``n`` outside ``[1, m - 1]``); ``input.shape``,
        ``input.dtype``, ``input.nonfinite``, ``input.out_of_space``,
        ``input.size_mismatch`` or ``input.weights`` for invalid data;
        ``input.too_few_samples`` if there are too few points for the
        surrogate.
    NumericalError
        ``numerical.singular`` or ``numerical.nonfinite`` if the subspace or
        surrogate fit fails numerically.
    ModelError
        Model path: ``model.raised``, ``model.shape``, ``model.dtype``,
        ``model.nonfinite`` or ``model.executor`` if a model evaluation
        fails.

    Warns
    -----
    AccuracyWarning
        ``accuracy.large_dimension`` if the dimension rule chose ``n`` above
        the supported envelope (4 for a uniform space, 6 for a Gaussian one);
        and the accuracy warnings of the subspace estimate and the surrogate
        fit (for example ``accuracy.loo_undefined``).

    See Also
    --------
    ReducedModel : The result and its operations.
    active_subspaces.active_subspace : The subspace estimate.
    active_subspaces.fit_surrogate : The data-path surrogate fit.
    active_subspaces.fit_surrogate_from_model : The model-path surrogate fit.

    Notes
    -----
    For fixed :math:`W_1` the best :math:`L^2(\rho)` approximation of the form
    :math:`g(W_1^\top x)` is the conditional mean
    :math:`G(y) = \int f(W_1 y + W_2 z)\, \pi(z \mid y)\, dz`, with

    .. math::

        \lVert f - G(W_1^\top \cdot) \rVert_{L^2(\rho)}
        \le C_P\, (\lambda_{n+1} + \dots + \lambda_m)^{1/2},

    where :math:`C_P` is a Poincaré constant of :math:`\rho` [1]_, [2]_.
    The default model-path sample size is the heuristic
    :math:`M = \lfloor \alpha (m + 1) \ln m \rfloor` with :math:`\alpha = 6`
    [2]_, [3]_; it needs :math:`m \ge 2`, and a reduced model needs
    :math:`1 \le n \le m - 1`. The default number of inactive draws per design
    point is 50 (the 2016 code used 10). The fit quality ``r_squared`` is
    defined in :class:`ReducedModel`.

    References
    ----------
    .. [1] P. G. Constantine, E. Dow, Q. Wang. Active subspace methods in
       theory and practice: applications to kriging surfaces. *SIAM J. Sci.
       Comput.* 36(4):A1500–A1524, 2014.
    .. [2] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.
    .. [3] P. G. Constantine, D. F. Gleich. Computing active subspaces with
       Monte Carlo. arXiv:1408.0545, 2014 (details to verify).

    Examples
    --------
    From data, with physical samples and gradients:

    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
    >>> X = space.sample(40, rng=0)
    >>> f, G = ac.evaluate_with_grads(model, X=X)
    >>> rm = ac.build_reduced_model(
    ...     space=space, X=X, f=f, grads=G, n=1, surrogate_kind="poly", rng=0
    ... )
    >>> print(rm.n, rm.r_squared_kind, f"{rm.r_squared:.3f}")
    1 loo 0.791

    From a model, which the builder samples itself:

    >>> rm = ac.build_reduced_model(model=model, n=1, surrogate_kind="poly", rng=0)
    >>> print(rm.X.shape, rm.gradient_source, rm.r_squared_kind)
    (26, 3) model holdout
    """
    # 1. validate the call (§4.1)
    data_given = X is not None or f is not None
    source: Literal["data", "model"]
    gsrc: Literal["given", "local_linear", "model", "finite_difference", "none"]
    r2_kind: Literal["holdout", "loo"]
    if model is not None:
        extra = [
            k
            for k, v in (("X", X), ("f", f), ("grads", grads), ("quad_weights", quad_weights))
            if v is not None
        ]
        if extra:
            raise _param(
                f"give either model or data, not both (got model and {', '.join(extra)})",
                extra[0],
                reason="two_sources",
            )
        source = "model"
    elif data_given:
        if X is None or f is None:
            missing = "f" if f is None else "X"
            raise _param(
                f"{missing}: X and f must be given together", missing, reason="missing_data"
            )
        source = "data"
    else:
        if grads is not None or quad_weights is not None:
            raise _param(
                "X and f are required with grads or quad_weights", "X", reason="missing_data"
            )
        raise _param("no source: give X and f, or model", "model", reason="no_source")
    if source == "data":
        given_model_only = [
            k
            for k, v in (
                ("n_samples", n_samples),
                ("n_design", n_design),
                ("n_mc", n_mc),
                ("fd_step", fd_step),
                ("executor", executor),
                ("batch_size", batch_size),
            )
            if v is not None
        ]
        if given_model_only:
            p = given_model_only[0]
            raise _param(
                f"{p}: a model-path parameter, not used with data", p, reason="model_path_only"
            )

    if subspace is not None:
        if not isinstance(subspace, SubspaceResult):
            raise _param(
                f"subspace: expected a SubspaceResult, got {type(subspace).__name__}", "subspace"
            )
        for name, value in (
            ("estimator", estimator),
            ("dimension_rule", dimension_rule),
            ("nboot", nboot),
        ):
            if value is not None:
                raise _param(
                    f"{name}: not used when subspace is given", name, reason="subspace_given"
                )

    # spaces [D-76]
    if space is not None:
        _v.check_space(space)
    bound = space
    if subspace is not None:
        if bound is not None and not _same_space(bound, subspace.space):
            raise _space_mismatch("subspace", "space")
        bound = subspace.space
    mdl: Model | None = None
    if model is not None:
        if isinstance(model, Model):
            if bound is not None and not _same_space(bound, model.space):
                raise _space_mismatch("model", "space" if space is not None else "subspace.space")
            mdl = model
        else:
            mdl = as_model(model, space=bound)
        bound = mdl.space
    if bound is None:
        raise _param("space: required with data unless subspace is given", "space")
    sp = bound
    m = sp.dim
    if m < 2:
        raise _param(
            f"space: a reduced model needs m >= 2 inputs, got m = {m}; fit a spec 09 surrogate "
            "for a function of one input",
            "space",
            reason="one_input",
        )

    # qoi
    if qoi is not None and (not isinstance(qoi, str) or not qoi):
        raise _param("qoi: expected a non-empty string", "qoi")
    if qoi is None:
        qoi = mdl.qoi if mdl is not None else (subspace.qoi if subspace is not None else "f")
    if subspace is not None and subspace.qoi != qoi:
        raise _param(
            f"qoi: {qoi!r} differs from subspace.qoi {subspace.qoi!r}",
            "qoi",
            reason="qoi_mismatch",
        )

    # scalars
    if n is not None:
        n = _v.integer(n, "n", lo=1, hi=m - 1)
    if subspace is not None and n is None and subspace.n >= m:
        raise _param(
            f"subspace: n = {subspace.n} = m leaves no inactive variables; pass n <= m - 1",
            "subspace",
            reason="n_equals_m",
        )
    nboot = _opt_int(nboot, "nboot", 0)
    surrogate_kind = cast(
        Literal["rbf", "poly"], _v.choice(surrogate_kind, "surrogate_kind", ("rbf", "poly"))
    )
    degree = _v.integer(degree, "degree", lo=0)
    n_samples = _opt_int(n_samples, "n_samples", 1)
    n_design = _opt_int(n_design, "n_design", 1)
    n_mc = _opt_int(n_mc, "n_mc", 2)  # >= 2, before any model evaluation [D-115 amended]
    batch_size = _opt_int(batch_size, "batch_size", 1)
    if fd_step is not None:
        _step(fd_step, m)
    _check_executor(executor)
    est_name: str | None = None
    est_options: dict[str, Any] | None = None
    if subspace is None:
        est = _subspaces._resolve(
            "as" if estimator is None else estimator, _subspaces.ESTIMATORS, "estimator"
        )
        need = bool(est.needs_gradients)
        est_name, est_options = est.name, est.options()
    else:
        need = False

    # 2. streams
    gen, rng_desc = _random.generator(rng)
    s_sample, s_subspace, s_surrogate = _random.spawn(gen, 3)
    _log.info("build_reduced_model: %s path, m = %d", source, m)

    # 3-4. samples, values and gradients
    digests: dict[str, Any] = {}
    Gn: np.ndarray | None = None
    grads_arg: Any = None
    if source == "data":
        gr = grads if isinstance(grads, GradientResult) else None
        d = _v.data(
            sp,
            X=X,
            f=f,
            grads=None if gr is not None else grads,
            quad_weights=quad_weights,
            require=("X", "f"),
            use_quad_weights=True,
        )
        M = d.M
        xn, fv, w = d.X, d.f, d.quad_weights
        assert xn is not None  # required above
        assert fv is not None
        assert w is not None
        digests.update(d.digests)
        if gr is not None:
            if not _same_space(gr.space, sp):
                raise _space_mismatch("grads", "space")
            if gr.grads.shape[0] != M:
                raise InputError(
                    f"grads: the GradientResult has {gr.grads.shape[0]} rows, X has {M}",
                    code="input.size_mismatch",
                    details={"param": "grads", "expected": M, "got": int(gr.grads.shape[0])},
                )
            Gn = np.array(gr.grads, dtype=np.float64)
            digests["grads"] = _v.digest(gr.grads_physical)
            grads_arg, gsrc = gr, "given"
        elif grads is not None:
            Gn, grads_arg, gsrc = d.grads, grads, "given"
        elif need:
            _log.info("build_reduced_model: local linear gradients at %d samples", M)
            lg = local_linear_gradients(X=X, f=f, space=sp, qoi=qoi)
            Gn, grads_arg, gsrc = np.array(lg.grads, dtype=np.float64), lg, "local_linear"
        else:
            gsrc = "none"
        X_sub, f_sub, w_sub = X, f, quad_weights
    else:
        assert mdl is not None
        M = _default_n_samples(m) if n_samples is None else n_samples
        xn = sp.sample(M, rng=s_sample, normalized=True)
        xi = sp.from_normalized(xn)
        w = np.full(M, 1.0 / M)
        if need and mdl.has_grad:
            _log.info("build_reduced_model: %d model values and gradients", M)
            fv, Gp = evaluate_with_grads(mdl, X=xi, executor=executor, batch_size=batch_size)
            fv = np.asarray(fv, dtype=np.float64)
            grads_arg = np.asarray(Gp, dtype=np.float64)
            Gn = sp.grads_to_normalized(grads_arg)
            gsrc = "model"
        elif need:
            _log.info("build_reduced_model: %d model values and finite-difference gradients", M)
            fd = finite_difference_gradients(
                mdl,
                X=xi,
                h=_FD_STEP if fd_step is None else fd_step,
                executor=executor,
                batch_size=batch_size,
            )
            fv = np.array(fd.values, dtype=np.float64)
            Gn = np.array(fd.grads, dtype=np.float64)
            grads_arg, gsrc = fd, "finite_difference"
        else:
            _log.info("build_reduced_model: %d model values", M)
            fv = np.asarray(
                evaluate(mdl, X=xi, executor=executor, batch_size=batch_size), dtype=np.float64
            )
            gsrc = "none"
        X_sub, f_sub, w_sub = xi, fv, None

    # 5. subspace
    rule_chose = False
    if subspace is not None:
        sub = subspace if n is None else with_dimension(subspace, n)
    else:
        sub = active_subspace(
            space=sp,
            X=X_sub,
            f=f_sub,
            grads=grads_arg,
            quad_weights=w_sub,
            estimator="as" if estimator is None else estimator,
            n=n,
            dimension_rule=dimension_rule,
            nboot=_NBOOT if nboot is None else nboot,
            rng=s_subspace,
            qoi=qoi,
        )
        rule_chose = n is None
    if sub.n >= m:
        raise _param(
            f"the active dimension n = {sub.n} equals m; a reduced model needs n <= m - 1",
            "n",
            reason="n_equals_m",
        )
    n_act = sub.n
    _log.info(
        "build_reduced_model: active dimension n = %d (%s)",
        n_act,
        "rule " + sub.dimension_rule if rule_chose else "fixed",
    )

    # 6. envelope check [D-121]
    if rule_chose:
        envelope = _ENVELOPE[sp.kind]
        if n_act > envelope:
            _warn(
                AccuracyWarning,
                f"the dimension rule chose n = {n_act}, above the documented envelope n <= "
                f"{envelope}; "
                f"geometry and the default design grow as 5**n = {5**n_act}",
                code="accuracy.large_dimension",
                details={"n": n_act, "envelope": envelope, "n_design": 5**n_act},
            )

    # 7. domain
    domain = active_domain(sub)
    _log.info("build_reduced_model: active domain of kind %s", domain.kind)

    # 8. surrogate
    if source == "data":
        sur = fit_surrogate(
            X=X, f=f, space=sp, subspace=sub, kind=surrogate_kind, degree=degree, qoi=qoi
        )
    else:
        assert mdl is not None
        sur = fit_surrogate_from_model(
            domain=domain,
            model=mdl,
            kind=surrogate_kind,
            degree=degree,
            n_design=5**n_act if n_design is None else n_design,
            n_mc=_N_MC if n_mc is None else n_mc,
            executor=executor,
            batch_size=batch_size,
            qoi=qoi,
            rng=s_surrogate,
        )
    _log.info("build_reduced_model: %s surrogate on %d points", surrogate_kind, sur.n_samples)

    # 9. fit quality [D-120]
    if source == "model":
        ghat = np.asarray(sur.evaluate(xn @ sub.W1), dtype=np.float64)
        r2 = _r_squared(fv, ghat, w)
        r2_kind = "holdout"
    else:
        e = sur.loo_residuals
        r2 = None if e is None else _r_squared(fv, fv - np.asarray(e), w)
        r2_kind = "loo"
    _log.info("build_reduced_model: %s R^2 = %s", r2_kind, r2)

    # 10. result
    params: dict[str, Any] = {
        "source": source,
        "model": None if mdl is None else mdl.name,
        "subspace": subspace,
        "n": n,
        "estimator": est_name,
        "estimator_options": est_options,
        "dimension_rule": _name_of(dimension_rule),
        "nboot": nboot,
        "surrogate_kind": surrogate_kind,
        "degree": degree,
        "n_samples": M if source == "model" else None,
        "n_design": n_design,
        "n_mc": n_mc,
        "fd_step": fd_step,
        "batch_size": batch_size,
        "gradient_source": gsrc,
        "qoi": qoi,
    }
    prov = _make_provenance(
        "build_reduced_model", params, space=sp, rng_state=rng_desc, digests=digests
    )
    return ReducedModel(
        space=sp,
        subspace=sub,
        domain=domain,
        surrogate=sur,
        X=xn,
        f=fv,
        grads=Gn,
        quad_weights=w,
        source=source,
        gradient_source=gsrc,
        r_squared=r2,
        r_squared_kind=r2_kind,
        qoi=qoi,
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# Operations (§4.3, §5.2)
# ---------------------------------------------------------------------------


def average(
    rm: ReducedModel,
    *,
    n_nodes: int | Sequence[int],
    model: Model | ValueFn | None = None,
    n_mc: int | None = None,
    n_replicates: int = 1,
    executor: concurrent.futures.Executor | None = None,
    batch_size: int | None = None,
    rng: _random.RngLike = None,
) -> IntegrationResult:
    r"""Estimate the mean of ``f`` over the input density.

    With ``model=None`` the surrogate is integrated by the active-variable
    quadrature rule (:func:`~active_subspaces.integrate_surrogate` on
    ``rm.surrogate``); no model evaluations and no inactive Monte Carlo are
    involved. With a model, ``f`` itself is integrated
    (:func:`~active_subspaces.integrate`, with inactive Monte Carlo at each
    node). The integration result is returned unchanged.

    Parameters
    ----------
    rm : ReducedModel
        The reduced model; its active domain defines the quadrature rule.
    n_nodes : int or sequence of int
        Size of the active-variable rule, as in
        :func:`~active_subspaces.active_quadrature_rule`: the number of
        nodes on an interval, of interior design points on a zonotope, or the
        target total (or per-dimension counts) for a Gaussian space.
    model : Model or callable, optional
        Integrate ``f`` itself; a function of **physical** points, and a
        ``Model`` must have ``model.space == rm.space``.
    n_mc : int, optional
        With `model` only: inactive draws per node; default 50.
    n_replicates : int, default 1
        Number :math:`R` of independent replicates; ``R >= 2`` gives
        ``total_std_error``.
    executor : concurrent.futures.Executor, optional
        With `model` only: executor for the model evaluations.
    batch_size : int, optional
        With `model` only: rows per model call.
    rng : None, int, SeedSequence or Generator, optional
        Seed or generator of the rule (zonotope) and the inactive draws.

    Returns
    -------
    IntegrationResult
        ``estimate`` and its error components; ``source`` is ``"surrogate"``
        or ``"model"``.

    Raises
    ------
    InputError
        ``input.parameter`` if `rm` is not a ``ReducedModel``, if `n_mc`,
        `executor` or `batch_size` is given without `model`, or for an
        invalid count; the input errors of
        :func:`~active_subspaces.integrate` for a bad `model`.
    NumericalError
        ``numerical.singular`` if the quadrature rule cannot be computed.
    ModelError
        With `model`: ``model.raised``, ``model.shape``, ``model.dtype``,
        ``model.nonfinite`` or ``model.executor`` if an evaluation fails.

    Warns
    -----
    AccuracyWarning
        The accuracy warnings of the integration functions, for example
        ``accuracy.correlated_draws`` when ``n_mc < 20`` hit-and-run steps
        are too few for a reliable autocorrelation time.

    See Also
    --------
    active_subspaces.integrate_surrogate : The surrogate integration.
    active_subspaces.integrate : The integration of ``f``.

    Notes
    -----
    Since :math:`g` depends on :math:`y = W_1^\top x` only,

    .. math::

        \int g(W_1^\top x)\, \rho(x)\, dx = \int g(y)\, \pi(y)\, dy
        \approx \sum_{k=1}^{Q} \omega_k\, g(y_k),

    with the nodes :math:`y_k` and weights :math:`\omega_k` of the
    active-variable rule [1]_. With a model,
    :math:`\mu \approx \sum_{k=1}^{Q} \omega_k \hat G(y_k)`, where
    :math:`\hat G(y_k)` is the mean of ``f`` over ``n_mc`` inactive draws at
    node :math:`y_k`.

    References
    ----------
    .. [1] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.

    Examples
    --------
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
    >>> X = space.sample(40, rng=0)
    >>> f, G = ac.evaluate_with_grads(model, X=X)
    >>> rm = ac.build_reduced_model(
    ...     space=space, X=X, f=f, grads=G, n=1, surrogate_kind="poly", rng=0
    ... )
    >>> res = ac.reduced_model.average(rm, n_nodes=5)
    >>> print(res.source, f"{res.estimate:.3f}")
    surrogate 0.463
    >>> res = ac.reduced_model.average(rm, n_nodes=5, model=model, n_mc=20, rng=0)
    >>> print(res.source, f"{res.estimate:.3f}")
    model 0.430
    >>> print(f"{model.reference_results.mean:.3f}")
    0.445
    """
    rm = _check_rm(rm)
    if model is None:
        for name, value in (("n_mc", n_mc), ("executor", executor), ("batch_size", batch_size)):
            if value is not None:
                raise _param(f"{name}: used only with model", name, reason="model_only")
        return integrate_surrogate(
            rm.surrogate, domain=rm.domain, n_nodes=n_nodes, n_replicates=n_replicates, rng=rng
        )
    mdl = as_model(model, space=rm.space)
    return integrate(
        mdl,
        domain=rm.domain,
        n_nodes=n_nodes,
        n_mc=_N_MC if n_mc is None else n_mc,
        n_replicates=n_replicates,
        executor=executor,
        batch_size=batch_size,
        rng=rng,
    )


def probability(
    rm: ReducedModel,
    lower: float,
    upper: float,
    *,
    n_samples: int = 10_000,
    rng: _random.RngLike = None,
) -> ProbabilityResult:
    r"""Estimate ``P[lower <= g(W1^T x) <= upper]`` under the surrogate by plain Monte Carlo.

    The interval is closed; ``-inf`` or ``inf`` give one-sided
    probabilities. The result's
    :meth:`~ProbabilityResult.confidence_interval` gives the Wilson score
    interval. The estimate is for the surrogate: averaging over the inactive
    variables smooths ``f``, so it can differ from
    ``P[lower <= f <= upper]`` by more than the Monte Carlo error.

    Parameters
    ----------
    rm : ReducedModel
        The reduced model whose surrogate is sampled.
    lower, upper : float
        Bounds on the value of the surrogate, ``lower < upper``; ``-inf`` and
        ``inf`` are allowed, NaN is rejected.
    n_samples : int, default 10000
        Number :math:`N` of draws from the input density.
    rng : None, int, SeedSequence or Generator, optional
        Seed or generator of the draws.

    Returns
    -------
    ProbabilityResult
        The estimate ``p``, its ``std_error`` and ``n_samples``.

    Raises
    ------
    InputError
        ``input.parameter`` if `rm` is not a ``ReducedModel``, if a bound is
        not a real number or NaN, if ``lower >= upper``, or if `n_samples` is
        not a positive integer.

    See Also
    --------
    ProbabilityResult : The result, with the formulas.

    Notes
    -----
    With :math:`x_i` drawn independently from :math:`\rho` (normalized
    coordinates),

    .. math::

        \hat p = \frac{1}{N} \sum_{i=1}^N
        \mathbf 1\bigl[\ell \le g(W_1^\top x_i) \le u\bigr],
        \qquad
        \widehat{\mathrm{se}} = \sqrt{\hat p (1 - \hat p) / N}.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
    >>> X = space.sample(40, rng=0)
    >>> f, G = ac.evaluate_with_grads(model, X=X)
    >>> rm = ac.build_reduced_model(
    ...     space=space, X=X, f=f, grads=G, n=1, surrogate_kind="poly", rng=0
    ... )
    >>> res = ac.reduced_model.probability(rm, -np.inf, 0.5, n_samples=2000, rng=0)
    >>> print(res.p, np.round(res.confidence_interval(), 3))
    0.7135 [0.687 0.739]
    """
    rm = _check_rm(rm)
    lo = _v.real(lower, "lower", finite=False)
    hi = _v.real(upper, "upper", finite=False)
    if not lo < hi:
        raise _param(
            f"lower < upper required, got lower = {lo!r}, upper = {hi!r}",
            "upper",
            reason="empty_interval",
        )
    N = _v.integer(n_samples, "n_samples", lo=1)
    gen, rng_desc = _random.generator(rng)
    m = rm.m
    W1 = np.asarray(rm.W1)
    chunk = max(1, _CHUNK_ELEMENTS // m)
    count = 0
    for start in range(0, N, chunk):
        rows = min(chunk, N - start)
        x = rm.space.sample(rows, rng=gen, normalized=True)
        g = np.asarray(rm.surrogate.evaluate(x @ W1), dtype=np.float64)
        count += int(np.count_nonzero((g >= lo) & (g <= hi)))
    p = count / N
    se = math.sqrt(p * (1.0 - p) / N)
    params = {"reduced_model": rm, "lower": lo, "upper": hi, "n_samples": N}
    prov = _make_provenance(
        "reduced_model.probability", params, space=rm.space, rng_state=rng_desc
    )
    return ProbabilityResult(
        reduced_model=rm,
        lower=lo,
        upper=hi,
        p=p,
        std_error=se,
        n_samples=N,
        qoi=rm.qoi,
        provenance=prov,
    )


def minimum(
    rm: ReducedModel,
    *,
    model: Model | ValueFn | None = None,
    n_starts: int = 10,
    executor: concurrent.futures.Executor | None = None,
    batch_size: int | None = None,
    rng: _random.RngLike = None,
) -> MinimizationResult:
    r"""Approximate ``min f``: minimize the surrogate over the active domain and lift ``y*``.

    Calls :func:`~active_subspaces.minimize` on ``rm.surrogate`` over
    ``rm.domain``, with the stored samples for the lift; the result is
    returned unchanged.

    Parameters
    ----------
    rm : ReducedModel
        The reduced model.
    model : Model or callable, optional
        If given, ``f`` is evaluated at the lifted minimizer
        (``f_model``); a function of **physical** points.
    n_starts : int, default 10
        Number of starting points of the multistart minimization of the
        surrogate.
    executor : concurrent.futures.Executor, optional
        Executor for the model evaluation.
    batch_size : int, optional
        Rows per model call.
    rng : None, int, SeedSequence or Generator, optional
        Seed or generator of the starting points.

    Returns
    -------
    MinimizationResult
        Named fields ``f_star`` (the surrogate minimum), ``f_model`` and the
        properties ``x_star`` (normalized), ``x_star_physical`` and
        ``y_star``.

    Raises
    ------
    InputError
        ``input.parameter`` if `rm` is not a ``ReducedModel`` or for an
        invalid argument; the input errors of
        :func:`~active_subspaces.minimize`.
    SolverError
        ``solver.failed`` or ``solver.infeasible`` if the lift's quadratic
        program fails.
    ModelError
        With `model`: ``model.raised``, ``model.shape``, ``model.dtype`` or
        ``model.nonfinite`` if the evaluation fails.

    Warns
    -----
    AccuracyWarning
        ``accuracy.curvature_clipped`` if the quadratic model of the lift has
        curvature below the floor in the inactive variables (as for a ridge
        function), and the other warnings of
        :func:`~active_subspaces.minimize`.
    ConvergenceWarning
        ``convergence.failed_starts`` or ``convergence.max_iter`` from the
        multistart minimization.

    See Also
    --------
    active_subspaces.minimize : The minimization and the lift.

    Notes
    -----
    The minimum :math:`\min_\xi f(\xi)` is approximated by
    :math:`y^\ast = \arg\min_{y} g(y)` over the active domain, followed by a
    lift of :math:`y^\ast` to a full point :math:`x^\ast` with
    :math:`W_1^\top x^\ast = y^\ast`, chosen by a quadratic model of ``f``
    fitted on the stored samples [1]_.

    References
    ----------
    .. [1] P. G. Constantine. *Active Subspaces: Emerging Ideas for Dimension
       Reduction in Parameter Studies.* SIAM, 2015.

    Examples
    --------
    >>> import numpy as np
    >>> import active_subspaces as ac
    >>> space = ac.Uniform([0.0, 0.0, 0.0], [1.0, 2.0, 4.0])
    >>> model = ac.test_functions.quadratic([1.0, 0.1, 0.05], space=space)
    >>> X = space.sample(40, rng=0)
    >>> f, G = ac.evaluate_with_grads(model, X=X)
    >>> rm = ac.build_reduced_model(
    ...     space=space, X=X, f=f, grads=G, n=1, surrogate_kind="poly", rng=0
    ... )
    >>> res = ac.reduced_model.minimum(rm, model=model, rng=0)
    >>> print(f"{res.f_star:.3f}", f"{res.f_model:.3f}")
    0.226 0.001
    >>> print(np.round(res.x_star_physical, 3))
    [0.492 0.988 1.974]
    """
    rm = _check_rm(rm)
    mdl = None if model is None else as_model(model, space=rm.space)
    return minimize(
        rm.surrogate,
        domain=rm.domain,
        X=rm.X_physical,
        f=rm.f,
        space=rm.space,
        model=mdl,
        n_starts=n_starts,
        executor=executor,
        batch_size=batch_size,
        rng=rng,
    )
