"""Shared setup for the acceptance tests (spec 01 §3, §5.1).

The spaces of spec 01 §5.1, the known-answer problems of §3.2, and the
hand-written coordinate maps that turn a normalized problem ``φ(x)`` into a
physical model ``f(ξ) = φ(x(ξ))``. The maps and the chain rule are written
here by hand and never use the library's maps to define the truth (§5.1).
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

import active_subspaces as asub

__all__ = (
    "HandMap",
    "allow_warnings",
    "expect_warning",
    "ka1",
    "physical_model",
    "space",
)

# Spec 01 §5.1 (bounds; Gaussian means and variances [D-67]).
U5 = {
    "lower": [0.0, -5.0, 10.0, 1e-3, 100.0],
    "upper": [1.0, 5.0, 30.0, 2e-3, 500.0],
    "names": ("a", "b", "c", "d", "e"),
}
G5 = {"mean": [1.0, -2.0, 0.0, 5.0, 10.0], "var": [0.25, 4.0, 1.0, 1e-2, 100.0]}
R7 = {
    "lower": [16.3, 13.7, 0.001, -0.36, -0.3, -0.3, -0.3],
    "upper": [24.3, 17.3, 0.002, 0.36, 0.3, 0.3, 0.3],
}
U6 = {"lower": [-1.0, 0.0, 5.0, -100.0, 1e-3, 0.0], "upper": [1.0, 2.0, 6.0, 100.0, 3e-3, 10.0]}
G6 = {"mean": [0.0] * 6, "var": [1.0, 4.0, 9.0, 0.01, 1.0, 25.0]}
U3 = {"lower": [0.0, -10.0, 100.0], "upper": [1.0, 10.0, 200.0]}
_DEFS = {"U5": U5, "G5": G5, "R7": R7, "U6": U6, "G6": G6, "U3": U3}


def space(name: str) -> asub.InputSpace:
    """The named space of spec 01 §5.1."""
    d = _DEFS[name]
    if "lower" in d:
        return asub.Uniform(
            d["lower"], d["upper"], **({"names": d["names"]} if "names" in d else {})
        )
    return asub.Gaussian(d["mean"], d["var"])


@dataclass(frozen=True)
class HandMap:
    """``x = (ξ − shift)/scale`` per coordinate; ``∂_ξ f = ∂_x φ / scale`` (spec 01 §5.1)."""

    shift: np.ndarray
    scale: np.ndarray

    @classmethod
    def of(cls, name: str) -> HandMap:
        d = _DEFS[name]
        if "lower" in d:
            lo, up = np.asarray(d["lower"], float), np.asarray(d["upper"], float)
            return cls(shift=(lo + up) / 2, scale=(up - lo) / 2)
        return cls(shift=np.asarray(d["mean"], float), scale=np.sqrt(np.asarray(d["var"], float)))

    @property
    def A(self) -> np.ndarray:
        """``ξ = shift + A x`` with ``A = diag(scale)``."""
        return np.diag(self.scale)

    def x(self, xi: np.ndarray) -> np.ndarray:
        return (np.asarray(xi, float) - self.shift) / self.scale

    def xi(self, x: np.ndarray) -> np.ndarray:
        return self.shift + np.asarray(x, float) * self.scale

    def grad(self, gx: np.ndarray) -> np.ndarray:
        return np.asarray(gx, float) / self.scale


def physical_model(
    name: str,
    phi: Callable[[np.ndarray], np.ndarray],
    dphi: Callable[[np.ndarray], np.ndarray] | None = None,
    *,
    model_name: str | None = None,
) -> tuple[asub.Model, Callable[[np.ndarray], np.ndarray]]:
    """``(Model, f)`` for ``f(ξ) = φ(x(ξ))`` on the named space; ``f`` is the hand-built truth."""
    hm = HandMap.of(name)

    def fun(Xi: np.ndarray) -> np.ndarray:
        return phi(hm.x(Xi))

    kw: dict = {"value": fun}
    if dphi is not None:
        kw["grad"] = lambda Xi: hm.grad(dphi(hm.x(Xi)))
    return asub.Model(space=space(name), name=model_name, **kw), fun


def ka1(m: int, kappa) -> tuple[np.ndarray, np.ndarray]:
    """KA-1: ``(Q, K)`` with ``Q = I − 2vv^T/(v^Tv)``, ``v = (1..m)``, ``K = Q diag(κ) Q``."""
    v = np.arange(1.0, m + 1)
    Q = np.eye(m) - 2.0 * np.outer(v, v) / (v @ v)
    return Q, Q @ np.diag(np.asarray(kappa, float)) @ Q


@contextlib.contextmanager
def allow_warnings(*codes: str) -> Iterator[list]:
    """Allow (not require) package warnings with the given codes; any other warning fails the
    test.
    """
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            yield rec
        finally:
            _check_allowed(rec, codes)


def _check_allowed(rec: list, codes: tuple[str, ...]) -> None:
    for w in rec:
        code = getattr(w.message, "code", None)
        why = f"unexpected warning {w.category.__name__}({code!r}): {w.message}"
        assert isinstance(w.message, asub.ActiveSubspacesWarning), why
        assert code in codes, why


_UNPICKLED: list[int] = []


def mark_unpickled() -> None:
    """Side effect of a malicious pickle payload (UC-12 'no pickle')."""
    _UNPICKLED.append(1)


def was_unpickled() -> bool:
    return bool(_UNPICKLED)


class Evil:
    """An object whose unpickling calls :func:`mark_unpickled`."""

    def __reduce__(self):
        return (mark_unpickled, ())


@contextlib.contextmanager
def expect_warning(category: type, code: str) -> Iterator[None]:
    """Require a package warning of ``category`` with ``code`` (and allow no other warning)."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        yield
    codes = [(w.category, getattr(w.message, "code", None)) for w in rec]
    assert (category, code) in codes, f"expected {category.__name__}({code!r}); got {codes}"
    others = [c for c in codes if c != (category, code)]
    assert not others, f"unexpected warnings {others}"


# UC-01 setup, shared with UC-05 and UC-12: KA-1 on m = 5.
KAPPA = np.array([2.0, 1.0, 0.1, 0.05, 0.01])


def uc01_model(name: str = "U5") -> asub.Model:
    _, K = ka1(5, KAPPA)
    return physical_model(
        name,
        lambda x: 0.5 * np.einsum("ij,jk,ik->i", x, K, x),
        lambda x: x @ K,
        model_name="quad5",
    )[0]


def uc01_mc(stat_rng) -> tuple[np.ndarray, np.ndarray, np.ndarray, asub.SubspaceResult]:
    """``(X, f, G, sub)`` of the UC-01 Monte Carlo run: M = 1000 on U5, ``nboot=200``."""
    sp = space("U5")
    X = sp.sample(1000, rng=stat_rng("UC-01/sample"))
    f, G = asub.evaluate_with_grads(uc01_model("U5"), X=X)
    return (
        X,
        f,
        G,
        asub.active_subspace(grads=G, space=sp, nboot=200, rng=stat_rng("UC-01/bootstrap")),
    )
