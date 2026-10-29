"""Factories for every registered result type (spec 15 §5.4, spec 04 §8).

Each factory builds a small, valid instance through the public API with a
fixed seed, quickly (well under a second each), so round-trip, immutability and
equality tests can be parametrized over ``FACTORIES``::

    @pytest.mark.parametrize("name", sorted(FACTORIES))
    def test_roundtrip(name, assert_roundtrip):
        assert_roundtrip(FACTORIES[name]())

Unlike the other helpers, this module imports ``active_subspaces``: it builds
the objects under test, it does not measure them.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Mapping
from types import MappingProxyType

import numpy as np

import active_subspaces as asub

__all__ = ("FACTORIES", "make")

_SEED = 20261028


@functools.cache
def _space() -> asub.Uniform:
    return asub.Uniform([0.0, -2.0, 10.0], [1.0, 2.0, 30.0], names=("a", "b", "c"))


def _phi(X: np.ndarray) -> np.ndarray:
    x = _space().to_normalized(X)
    s = x[:, 0] + 0.5 * x[:, 1]
    return s**2 + s


def _phi_grad(X: np.ndarray) -> np.ndarray:
    x = _space().to_normalized(X)
    s = x[:, 0] + 0.5 * x[:, 1]
    gx = np.outer(2 * s + 1, [1.0, 0.5, 0.0])
    return _space().grads_to_physical(gx)


@functools.cache
def _model() -> asub.Model:
    return asub.Model(space=_space(), value=_phi, grad=_phi_grad, name="factory_ridge")


@functools.cache
def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = _space().sample(40, rng=_SEED)
    f, G = asub.evaluate_with_grads(_model(), X=X)
    return X, f, G


@functools.cache
def subspace_result() -> asub.SubspaceResult:
    _, _, G = _data()
    return asub.active_subspace(grads=G, space=_space(), nboot=10, rng=_SEED)


@functools.cache
def gradient_result() -> asub.GradientResult:
    X, f, _ = _data()
    return asub.local_linear_gradients(X=X, f=f, space=_space())


@functools.cache
def reference_results() -> asub.ReferenceResults:
    from active_subspaces import test_functions

    return test_functions.quadratic([1.0, 0.5, 0.1], name="factory_quadratic").reference_results


@functools.cache
def active_domain() -> asub.ActiveDomain:
    return asub.active_domain(subspace_result())


@functools.cache
def inverse_map_result() -> asub.InverseMapResult:
    return asub.inverse_map(subspace_result(), Y=np.array([[0.1], [-0.2]]), n_draws=3, rng=_SEED)


@functools.cache
def active_design() -> asub.ActiveDesign:
    return asub.active_design(domain=active_domain(), n_points=4, rng=_SEED)


@functools.cache
def conditional_samples() -> asub.ConditionalSamples:
    return asub.conditional_samples(
        domain=active_domain(), model=_model(), n_design=4, n_mc=4, rng=_SEED
    )


@functools.cache
def active_quadrature_rule() -> asub.ActiveQuadratureRule:
    return asub.active_quadrature_rule(active_domain(), n_nodes=3, rng=_SEED)


@functools.cache
def integration_design() -> asub.IntegrationDesign:
    return asub.integration_design(active_domain(), n_nodes=3, n_mc=4, rng=_SEED)


@functools.cache
def integration_result() -> asub.IntegrationResult:
    with (
        warnings.catch_warnings()
    ):  # hit-and-run chains of n_mc = 4 < 20 warn correlated_draws [D-114]
        warnings.filterwarnings(
            "ignore", message=r".*accuracy\.correlated_draws", category=asub.AccuracyWarning
        )
        return asub.integrate(_model(), domain=active_domain(), n_nodes=3, n_mc=4, rng=_SEED)


@functools.cache
def polynomial_surrogate() -> asub.PolynomialSurrogate:
    X, f, _ = _data()
    return asub.fit_surrogate(X=X, f=f, subspace=subspace_result(), kind="poly", degree=2)


@functools.cache
def rbf_surrogate() -> asub.RBFSurrogate:
    X, f, _ = _data()
    return asub.fit_surrogate(X=X, f=f, subspace=subspace_result(), kind="rbf", degree=2)


@functools.cache
def active_minimization_result() -> asub.ActiveMinimizationResult:
    return asub.minimize_active(
        polynomial_surrogate(), domain=active_domain(), n_starts=2, rng=_SEED
    )


def _quiet(fn: Callable[[], asub.Result]) -> asub.Result:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", asub.AccuracyWarning)
        return fn()


@functools.cache
def lift_result() -> asub.LiftResult:
    X, f, _ = _data()
    return _quiet(lambda: asub.lift(np.array([0.1]), domain=active_domain(), X=X, f=f))


@functools.cache
def minimization_result() -> asub.MinimizationResult:
    X, f, _ = _data()
    return _quiet(
        lambda: asub.minimize(
            polynomial_surrogate(), domain=active_domain(), X=X, f=f, n_starts=2, rng=_SEED
        )
    )


@functools.cache
def reduced_model() -> asub.ReducedModel:
    X, f, G = _data()
    return asub.build_reduced_model(
        X=X, f=f, grads=G, space=_space(), surrogate_kind="poly", rng=_SEED
    )


@functools.cache
def probability_result() -> asub.ProbabilityResult:
    return reduced_model().probability(-np.inf, 0.5, n_samples=200, rng=_SEED)


@functools.cache
def sketched_subspace():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", asub.ExperimentalWarning)
        from active_subspaces.experimental import sketched_active_subspace

        return sketched_active_subspace(_model(), k=2, M=10, rng=_SEED)


FACTORIES: Mapping[str, Callable[[], asub.Result]] = MappingProxyType(
    {
        "ActiveDesign": active_design,
        "ActiveDomain": active_domain,
        "ActiveMinimizationResult": active_minimization_result,
        "ActiveQuadratureRule": active_quadrature_rule,
        "ConditionalSamples": conditional_samples,
        "GradientResult": gradient_result,
        "IntegrationDesign": integration_design,
        "IntegrationResult": integration_result,
        "InverseMapResult": inverse_map_result,
        "LiftResult": lift_result,
        "MinimizationResult": minimization_result,
        "PolynomialSurrogate": polynomial_surrogate,
        "ProbabilityResult": probability_result,
        "RBFSurrogate": rbf_surrogate,
        "ReducedModel": reduced_model,
        "ReferenceResults": reference_results,
        "SketchedSubspace": sketched_subspace,
        "SubspaceResult": subspace_result,
    }
)


def make(type_name: str) -> asub.Result:
    """The factory instance of the result class named ``type_name``."""
    return FACTORIES[type_name]()
