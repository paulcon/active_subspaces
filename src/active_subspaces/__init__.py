"""Discover and exploit active subspaces of functions of many inputs.

The package estimates the directions in input space along which a function of
many inputs varies most (:func:`active_subspace` and other sufficient
dimension reduction estimators), describes the geometry of the active
variables (:func:`active_domain`, :func:`forward_map`, :func:`inverse_map`),
and exploits them for surrogates, integration, optimization and reduced
models. Inputs are described by an input space (:class:`Uniform` or
:class:`Gaussian`); data are passed in physical coordinates and every result
is stored in normalized coordinates.

The core names are available at the top level, ``import active_subspaces as
ac``. ``ac.plotting`` and ``ac.test_functions`` are imported on first access;
``active_subspaces.experimental`` only by an explicit import.

Examples
--------
>>> import numpy as np
>>> import active_subspaces as ac
>>> space = ac.Uniform.reference(3)
>>> X = space.sample(100, rng=0)
>>> a = np.array([1.0, 0.5, 0.0])
>>> sub = ac.active_subspace(grads=np.outer(np.cos(X @ a), a), space=space)
>>> sub.n, sub.W1.round(3).ravel()
(1, array([0.894, 0.447, 0.   ]))
"""

import logging as _logging
import typing as _typing

from active_subspaces import (
    designs,
    errors,
    geometry,
    gradients,
    integration,
    models,
    optimization,
    reduced_model,
    registry,
    results,
    solvers,
    spaces,
    subspaces,
    surrogates,
)
from active_subspaces._version import __version__
from active_subspaces.designs import (
    ActiveDesign,
    ConditionalSamples,
    active_design,
    conditional_samples,
)
from active_subspaces.errors import (
    AccuracyWarning,
    ActiveSubspacesDeprecationWarning,
    ActiveSubspacesError,
    ActiveSubspacesWarning,
    ConvergenceWarning,
    ExperimentalWarning,
    InputError,
    ModelError,
    NumericalError,
    SchemaError,
    SolverError,
    UnsupportedError,
    ValidityWarning,
)
from active_subspaces.geometry import (
    ActiveCoordinates,
    ActiveDomain,
    InverseMapResult,
    active_domain,
    forward_map,
    in_active_domain,
    inverse_map,
    zonotope_vertex_count,
)
from active_subspaces.gradients import (
    GradientResult,
    finite_difference_gradients,
    finite_difference_gradients_multi,
    local_linear_gradients,
)
from active_subspaces.integration import (
    ActiveQuadratureRule,
    IntegrationDesign,
    IntegrationResult,
    active_quadrature_rule,
    estimate_integral,
    integrate,
    integrate_surrogate,
    integration_design,
)
from active_subspaces.models import (
    GradFn,
    Model,
    ReferenceResults,
    ValueAndGradFn,
    ValueFn,
    as_model,
    evaluate,
    evaluate_grads,
    evaluate_multi,
    evaluate_with_grads,
    pointwise,
)
from active_subspaces.optimization import (
    ActiveMinimizationResult,
    LiftResult,
    MinimizationResult,
    lift,
    minimize,
    minimize_active,
)
from active_subspaces.reduced_model import (
    ProbabilityResult,
    ReducedModel,
    build_reduced_model,
)
from active_subspaces.registry import ModelRegistry
from active_subspaces.results import Provenance, QoIResult, Result, from_dict, load
from active_subspaces.solvers import (
    ScipyBackend,
    SolverBackend,
    SolverOutput,
    get_backend,
    solve_lp,
    solve_qp,
)
from active_subspaces.spaces import Gaussian, InputSpace, Uniform
from active_subspaces.subspaces import (
    SubspaceResult,
    active_subspace,
    activity_scores,
    bootstrap_interval,
    dimension_criterion,
    mean_distance,
    orient,
    subspace_from_basis,
    subspace_from_matrix,
    with_dimension,
)
from active_subspaces.surrogates import (
    PolynomialSurrogate,
    RBFSurrogate,
    Surrogate,
    fit_surrogate,
    fit_surrogate_from_model,
    predict,
    predict_gradient,
    quadratic_terms,
)

_logging.getLogger("active_subspaces").addHandler(_logging.NullHandler())

# Spec 15 §4.3: each core module's top-level names (its `_toplevel`, else its
# whole `__all__`), the core module names, the lazy submodules and `__version__`;
# a sorted tuple.
__all__ = (
    "AccuracyWarning",
    "ActiveCoordinates",
    "ActiveDesign",
    "ActiveDomain",
    "ActiveMinimizationResult",
    "ActiveQuadratureRule",
    "ActiveSubspacesDeprecationWarning",
    "ActiveSubspacesError",
    "ActiveSubspacesWarning",
    "ConditionalSamples",
    "ConvergenceWarning",
    "ExperimentalWarning",
    "Gaussian",
    "GradFn",
    "GradientResult",
    "InputError",
    "InputSpace",
    "IntegrationDesign",
    "IntegrationResult",
    "InverseMapResult",
    "LiftResult",
    "MinimizationResult",
    "Model",
    "ModelError",
    "ModelRegistry",
    "NumericalError",
    "PolynomialSurrogate",
    "ProbabilityResult",
    "Provenance",
    "QoIResult",
    "RBFSurrogate",
    "ReducedModel",
    "ReferenceResults",
    "Result",
    "SchemaError",
    "ScipyBackend",
    "SolverBackend",
    "SolverError",
    "SolverOutput",
    "SubspaceResult",
    "Surrogate",
    "Uniform",
    "UnsupportedError",
    "ValidityWarning",
    "ValueAndGradFn",
    "ValueFn",
    "__version__",
    "active_design",
    "active_domain",
    "active_quadrature_rule",
    "active_subspace",
    "activity_scores",
    "as_model",
    "bootstrap_interval",
    "build_reduced_model",
    "conditional_samples",
    "designs",
    "dimension_criterion",
    "errors",
    "estimate_integral",
    "evaluate",
    "evaluate_grads",
    "evaluate_multi",
    "evaluate_with_grads",
    "finite_difference_gradients",
    "finite_difference_gradients_multi",
    "fit_surrogate",
    "fit_surrogate_from_model",
    "forward_map",
    "from_dict",
    "geometry",
    "get_backend",
    "gradients",
    "in_active_domain",
    "integrate",
    "integrate_surrogate",
    "integration",
    "integration_design",
    "inverse_map",
    "lift",
    "load",
    "local_linear_gradients",
    "mean_distance",
    "minimize",
    "minimize_active",
    "models",
    "optimization",
    "orient",
    "plotting",
    "pointwise",
    "predict",
    "predict_gradient",
    "quadratic_terms",
    "reduced_model",
    "registry",
    "results",
    "solve_lp",
    "solve_qp",
    "solvers",
    "spaces",
    "subspace_from_basis",
    "subspace_from_matrix",
    "subspaces",
    "surrogates",
    "test_functions",
    "with_dimension",
    "zonotope_vertex_count",
)

# PEP 562 lazy submodules: imported on first attribute access, never at package import.
_LAZY_SUBMODULES: tuple[str, ...] = ("plotting", "test_functions")

# Removed 2016 top-level attributes and their replacements (spec 15 §4.3, spec 18 §5.4)
# [D-168]. Only an error message: the keys stay AttributeErrors, nothing is imported.
_MIGRATION_URL = "https://active-subspaces.readthedocs.io/en/stable/migration.html"
_REMOVED_2016: dict[str, str] = {
    "utils": (
        "Its modules moved: ac.plotting (plotters), ac.designs, ac.solvers (qp_solver), "
        "ac.models (simrunners), ac.surrogates (response_surfaces), "
        "ac.Uniform and ac.Gaussian (normalizers); est_noise (ECNoise) was dropped."
    ),
    "domains": "Use ac.active_domain, ac.forward_map, ac.inverse_map.",
    "integrals": (
        "Use ac.integrate, ac.integrate_surrogate, ac.integration_design, "
        "ac.active_quadrature_rule."
    ),
    "optimizers": "Use ac.minimize, ac.minimize_active, ac.lift.",
    "response_surfaces": (
        "Use ac.fit_surrogate, ac.fit_surrogate_from_model, ac.predict, ac.predict_gradient."
    ),
    "base": "Use ac.build_reduced_model and ac.ReducedModel (module ac.reduced_model).",
    "sdr": 'Use ac.active_subspace(..., estimator="ols") or estimator="qphd".',
    "gradfree": "Use `import active_subspaces.experimental` (experimental.gradfree).",
    "ActiveSubspaceReducedModel": "Use ac.build_reduced_model, which returns an ac.ReducedModel.",
}


if _typing.TYPE_CHECKING:  # the lazy submodules, for type checkers only
    from active_subspaces import plotting, test_functions


def __getattr__(name: str) -> _typing.Any:
    if name in _LAZY_SUBMODULES:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    msg = f"module {__name__!r} has no attribute {name!r}"
    hint = _REMOVED_2016.get(name)
    if hint is not None:
        msg += (
            ". It was part of the 2016 API (v0.1), which v1.0 replaces without a compatibility "
            f"layer. {hint}\nSee the migration guide: {_MIGRATION_URL}"
        )
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_SUBMODULES))
