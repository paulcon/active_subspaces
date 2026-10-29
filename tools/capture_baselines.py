"""Capture the D-37 regression baselines from the Phase 1 reference ports (spec 15 §5.13, spec 17 §5.5).

Each case runs a 2016 call on the Python 3 port under ``legacy/port-<branch>/``
(built by ``design-specs/reproducers/port/make_port.sh``) with inputs drawn from
``np.random.SeedSequence((37, <case index>))`` and writes
``tests/data/baselines/<area>/<case>.npz`` (inputs ``in/<name>``, outputs
``out/<name>``) plus ``<case>.json`` (environment, legacy call, per-output
status). The legacy global RNG is seeded only where the legacy call draws, and
the seed is recorded. The files are byte-identical from run to run.

Usage (from the repository root)::

    python tools/capture_baselines.py                  # both branches, one process each
    python tools/capture_baselines.py --branch master  # one branch (the port is imported, v1 never)
    python tools/capture_baselines.py --v1             # add the v1_reference outputs [D-163]
    python tools/capture_baselines.py --check          # recapture into a temporary directory and compare bytes

Legacy capture never imports v1, and ``--v1`` never imports the port: ``--v1``
reads the stored inputs and runs ``src/active_subspaces``. A legacy capture
keeps the ``v1_reference`` outputs already in a case file.

Before a legacy capture, ``design-specs/reproducers/repro.py`` (searched in the
repository and next to it) must print PASS for every check on the port
(spec 17 §5.5 step 1); ``--skip-repro`` bypasses this for development only.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import filecmp
import hashlib
import io
import itertools
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import types
import warnings
import zipfile
from typing import Any, Callable

for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"  # one BLAS thread: summation order, and so the bytes, do not depend on scheduling

import numpy as np  # noqa: E402
import scipy  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tests", "data", "baselines")
LEGACY = {b: os.path.join(ROOT, "legacy", f"port-{b}") for b in ("master", "dev")}
LEGACY_COMMITS = {"master": "64817a19db250e4b01bcd51055ad0f7d2a5665b8", "dev": "361ba2c3052d8346c581f9a11fd0ca755a351d25"}
SEED_ROOT = 37
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# Status records (spec 17 §5.5 step 4)
# ---------------------------------------------------------------------------


def reference(tier: str, scale: str = "max", compare: str = "array", **extra: Any) -> dict:
    """``{"status": "reference"}``; ``compare`` is ``array``, ``eigvecs`` (sign-free, by cluster), ``set`` or ``exact``."""
    return {"status": "reference", "tolerance": {"tier": tier, "scale": scale}, "compare": compare, **extra}


def known_wrong(finding: str, relation: dict, **extra: Any) -> dict:
    return {"status": "known_wrong", "finding": finding, "relation": relation, **extra}


def summary(stats: dict, **extra: Any) -> dict:
    return {"status": "summary", "stats": stats, **extra}


def v1_reference(tier: str, scale: str = "max", **extra: Any) -> dict:
    return {"status": "v1_reference", "tolerance": {"tier": tier, "scale": scale}, **extra}


def _summary_stats(values: np.ndarray, **extra: Any) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {
        "R": int(v.size),
        "mean": float(np.mean(v)),
        "sd": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "median": float(np.median(v)),
        "q05": float(np.quantile(v, 0.05)),
        "q95": float(np.quantile(v, 0.95)),
        **extra,
    }


@dataclasses.dataclass
class Capture:
    """What one case produces: arrays, per-output records and the legacy call as text."""

    inputs: dict[str, Any]
    outputs: dict[str, tuple[Any, dict]]
    call: str
    inputs_from: str = "SeedSequence"
    legacy_seeds: dict[str, Any] = dataclasses.field(default_factory=dict)
    notes: list[str] = dataclasses.field(default_factory=list)
    not_stored: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Case:
    index: int
    area: str
    name: str
    branch: str
    run: Callable[[], Capture]
    v1: Callable[[dict[str, np.ndarray], dict], dict[str, tuple[Any, dict]]] | None = None


CASES: list[Case] = []


def case(index: int, area: str, name: str, branch: str = "master"):
    def deco(fn: Callable[[], Capture]) -> Callable[[], Capture]:
        CASES.append(Case(index, area, name, branch, fn))
        return fn

    return deco


def v1_capture(area: str, name: str):
    def deco(fn):
        for i, c in enumerate(CASES):
            if (c.area, c.name) == (area, name):
                CASES[i] = dataclasses.replace(c, v1=fn)
        return fn

    return deco


def _gen(index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence((SEED_ROOT, index)))


# ---------------------------------------------------------------------------
# Legacy-side helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _legacy_context(seed: int | None = None):
    """Silence the port (prints, warnings); seed the legacy global RNG if asked and restore it afterwards."""
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            yield
    finally:
        np.random.set_state(state)


class _Proxy(types.SimpleNamespace):
    """Attribute proxy: overrides first, then the wrapped module."""

    def __init__(self, wrapped: Any, **overrides: Any) -> None:
        super().__init__(**overrides)
        object.__setattr__(self, "_wrapped", wrapped)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_wrapped"), name)


class _Py2Amin:
    """``np.amin`` under Python 2 ordering: a number is smaller than any function, so ``np.amin < x`` is False [C-01]."""

    def __call__(self, *a, **k):
        return np.amin(*a, **k)

    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return True


@contextlib.contextmanager
def _patched(module: Any, name: str, value: Any):
    old = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, old)


def _raises(fn: Callable[[], Any]) -> tuple[str, str] | None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - the legacy exception is the recorded output
        return type(exc).__name__, str(exc)
    return None


def _ka1(m: int = 5, kappa=(2.0, 1.0, 0.1, 0.05, 0.01)) -> np.ndarray:
    """KA-1 (spec 01 §3.2): ``K = Q diag(kappa) Q`` with the Householder ``Q`` of ``v = (1..m)``."""
    v = np.arange(1, m + 1, dtype=np.float64)
    Q = np.eye(m) - 2.0 * np.outer(v, v) / (v @ v)
    return Q @ np.diag(np.asarray(kappa, dtype=np.float64)) @ Q


def _eig_outputs(e: np.ndarray, W: np.ndarray, e_record: dict | None = None, W_record: dict | None = None) -> dict:
    return {
        "e": (np.asarray(e, dtype=np.float64).ravel(), e_record or reference("F12", "lambda1")),
        "W": (np.asarray(W, dtype=np.float64), W_record or reference("F10", compare="eigvecs")),
    }


# ---------------------------------------------------------------------------
# subspaces
# ---------------------------------------------------------------------------


@case(0, "subspaces", "as_exact_rule")
def _as_exact_rule() -> Capture:
    import active_subspaces.subspaces as ss

    m = 5
    K = _ka1(m)
    X = np.array(list(itertools.product((-1.0, 1.0), repeat=m))) / math.sqrt(3.0)
    G = X @ K
    w = np.full(X.shape[0], 1.0 / X.shape[0])
    with _legacy_context():
        s = ss.Subspaces()
        s.compute(df=G, weights=w[:, None])
    out = _eig_outputs(s.eigenvals, s.eigenvecs)
    out["n"] = (np.int64(s.W1.shape[1]), reference("X", compare="exact"))
    return Capture(
        inputs={"X": X, "grads": G, "weights": w, "K": K},
        outputs=out,
        call="Subspaces().compute(df=grads, weights=weights[:, None])  # sstype='AS', ptype='EVG'",
        inputs_from="KA-1 (spec 01 §3.2): 2-point Gauss-Legendre tensor nodes {±1/sqrt(3)}^5, weights 1/32",
    )


@case(1, "subspaces", "as_mc")
def _as_mc() -> Capture:
    import active_subspaces.subspaces as ss

    m, M = 5, 500
    K = _ka1(m)
    X = _gen(1).uniform(-1.0, 1.0, (M, m))
    G = X @ K
    with _legacy_context():
        s = ss.Subspaces()
        s.compute(df=G)
    out = _eig_outputs(s.eigenvals, s.eigenvecs)
    out["n"] = (np.int64(s.W1.shape[1]), reference("X", compare="exact"))
    return Capture(inputs={"grads": G, "K": K}, outputs=out, call="Subspaces().compute(df=grads)  # weights 1/M, EVG", notes=["grads = X K with X ~ U[-1, 1]^5 (X not stored)."])


def _ols_inputs(index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gen = _gen(index)
    M, m = 100, 5
    X = gen.standard_normal((M, m))
    g = np.array([1.0, -0.5, 0.25, 0.0, 2.0])
    f = 1.0 + X @ g + 0.01 * gen.standard_normal(M)
    return X, f, g


@case(2, "subspaces", "ols")
def _ols() -> Capture:
    import active_subspaces.subspaces as ss

    X, f, g = _ols_inputs(2)
    w = np.full((X.shape[0], 1), 1.0 / X.shape[0])
    with _legacy_context():
        e, W = ss.ols_subspace(X, f[:, None], w)
    return Capture(
        inputs={"X": X, "f": f, "g": g},
        outputs=_eig_outputs(e, W, reference("F10", "lambda1"), reference("F10", compare="eigvecs")),
        call="ols_subspace(X, f[:, None], np.full((M, 1), 1/M))",
        notes=["Gaussian X (gamma = 1 equals the declared Sigma = I); f linear plus 0.01 N(0, 1) noise."],
    )


@case(3, "subspaces", "qphd_gaussian")
def _qphd_gaussian() -> Capture:
    import active_subspaces.subspaces as ss

    gen = _gen(3)
    M, m = 200, 4
    X = gen.standard_normal((M, m))
    B = gen.standard_normal((m, m))
    H = (B + B.T) / 2.0
    g = np.array([1.0, 0.0, -1.0, 0.5])
    f = 0.3 + X @ g + 0.5 * np.einsum("ij,jk,ik->i", X, H, X)
    w = np.full((M, 1), 1.0 / M)
    with _legacy_context():
        e, W = ss.qphd_subspace(X, f[:, None], w)
    return Capture(
        inputs={"X": X, "f": f, "g": g, "H": H},
        outputs=_eig_outputs(e, W, reference("F10", "lambda1"), reference("F10", compare="eigvecs")),
        call="qphd_subspace(X, f[:, None], np.full((M, 1), 1/M))",
        notes=["Exact quadratic f = c0 + g^T x + x^T H x / 2 on Gaussian X; max X > 1, so the 2016 gamma is 1 (= declared Sigma)."],
    )


@case(4, "subspaces", "qphd_bounded")
def _qphd_bounded() -> Capture:
    import active_subspaces.subspaces as ss

    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (200, 4))
    c = np.array([1.0, 0.5, 0.1, 0.0])
    f = (X @ c) ** 2
    with _legacy_context():
        exc = _raises(lambda: ss.qphd_subspace(X, f[:, None], np.ones((200, 1)) / 200))
    assert exc is not None and exc[0] == "TypeError", exc
    return Capture(
        inputs={"X": X, "f": f, "c": c},
        outputs={"C": (np.zeros(0), known_wrong("C-01", {"kind": "raises", "exception": exc[0], "message": exc[1]}))},
        call="qphd_subspace(X, f[:, None], np.ones((200, 1)) / 200)",
        inputs_from="repro:C-01 (np.random.default_rng(0))",
        notes=["v1 returns C = g g^T + H Sigma H with Sigma = I/3 (KA-4 with g = 0, H = 2 c c^T) [D-40]."],
    )


@case(5, "subspaces", "li_ridge")
def _li_ridge() -> Capture:
    import active_subspaces.subspaces as ss

    m, M, nboot, seed = 5, 300, 50, 0
    a = np.array([2.0, 1.0, 0.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    X = _gen(5).uniform(-1.0, 1.0, (M, m))
    G = np.outer(np.exp(X @ a), a)
    with _legacy_context(seed):
        s = ss.Subspaces()
        s.compute(df=G, ptype="LI", nboot=nboot)
    n = np.int64(s.W1.shape[1])
    return Capture(
        inputs={"grads": G, "a": a, "nboot": np.int64(nboot)},
        outputs={"n": (n, known_wrong("C-17", {"kind": "shift", "by": -1}))},
        call="Subspaces().compute(df=grads, ptype='LI', nboot=50)",
        inputs_from="SeedSequence; the repro:C-17 ridge f = exp(a^T x), a = (2, 1, 0, 0, 0)/sqrt(5)",
        legacy_seeds={"np.random.seed": seed},
        notes=["Every bootstrap replicate has the same W1 (KA-2), so n is deterministic; the 2016 ladle returns argmin + 1."],
    )


def _sdr_inputs(index: int, M: int = 200, m: int = 4) -> tuple[np.ndarray, np.ndarray]:
    gen = _gen(index)
    X = gen.standard_normal((M, m))
    f = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.25 * X[:, 2] + 0.01 * gen.standard_normal(M)
    assert np.unique(f).size == M
    return X, f


@case(6, "subspaces", "sir", branch="dev")
def _sir() -> Capture:
    import active_subspaces.subspaces as ss

    X, f = _sdr_inputs(6)
    M = X.shape[0]
    with _legacy_context():
        e, W = ss.sir_subspace(X, f[:, None], np.full((M, 1), 1.0 / M))
    return Capture(
        inputs={"X": X, "f": f, "n_slices": np.int64(10)},
        outputs=_eig_outputs(e, W, reference("F10", "lambda1"), reference("F10", compare="eigvecs")),
        call="sir_subspace(X, f[:, None], np.full((M, 1), 1/M))  # H = 10 percentile slices",
        notes=["Gaussian X, distinct f, M = 200 divisible by S = 10: the 2016 percentile slices equal v1's equal-count slices."],
    )


@case(7, "subspaces", "save", branch="dev")
def _save() -> Capture:
    import active_subspaces.subspaces as ss

    X, f = _sdr_inputs(7)
    M = X.shape[0]
    with _legacy_context():
        e, W = ss.save_subspace(X, f[:, None], np.full((M, 1), 1.0 / M))
    rel = {"kind": "scale", "factor": "1/S", "value": 0.1, "tier": "F10"}
    return Capture(
        inputs={"X": X, "f": f, "n_slices": np.int64(10)},
        outputs=_eig_outputs(e, W, known_wrong("N-01", rel, decision="D-42"), reference("F10", compare="eigvecs")),
        call="save_subspace(X, f[:, None], np.full((M, 1), 1/M))  # H = 10",
        notes=["2016 sums V_h^2 without the slice proportions p_h = 1/S: C_v1 = C_2016 / S."],
    )


def _phd_f(X: np.ndarray, Q: np.ndarray, d: np.ndarray) -> np.ndarray:
    Y = X @ Q
    return 0.5 * np.sum(d * Y * Y, axis=1) + 0.1 * X[:, 0]


@case(8, "subspaces", "phd", branch="dev")
def _phd() -> Capture:
    import active_subspaces.subspaces as ss

    gen = _gen(8)
    M, m = 300, 4
    Q = np.linalg.qr(gen.standard_normal((m, m)))[0]
    d = np.array([2.0, 1.0, 0.4, 0.1])
    du = np.array([2.0, 1.0, 0.6, 0.4])
    Xg = gen.standard_normal((M, m))
    Xu = gen.uniform(-1.0, 1.0, (M, m))
    fg, fu = _phd_f(Xg, Q, d), _phd_f(Xu, np.eye(m), du)
    w = np.full((M, 1), 1.0 / M)
    Zu = Xu - Xu.mean(axis=0)
    assert np.all(np.linalg.eigvalsh(Zu.T @ ((fu - fu.mean())[:, None] * Zu)) > 0)  # PSD, so 2016's |e| keeps the signs
    with _legacy_context():
        eg, Wg = ss.phd_subspace(Xg, fg[:, None], w)
        with _patched(ss, "np", _Proxy(np, amin=_Py2Amin())):
            eu, Wu = ss.phd_subspace(Xu, fu[:, None], w)
    out = {f"gaussian_{k}": v for k, v in _eig_outputs(eg, Wg, reference("F10", "lambda1"), reference("F10", compare="eigvecs")).items()}
    rel = {"kind": "scale", "factor": "gamma**-0.5", "value": math.sqrt(3.0), "tier": "F10"}
    out.update({f"uniform_{k}": v for k, v in _eig_outputs(eu, Wu, known_wrong("N-02", rel, decision="D-43"), reference("F10", compare="eigvecs")).items()})
    return Capture(
        inputs={"X_gaussian": Xg, "f_gaussian": fg, "X_uniform": Xu, "f_uniform": fu, "Q": Q, "d": d, "d_uniform": du},
        outputs=out,
        call="phd_subspace(X, f[:, None], np.full((M, 1), 1/M)) on Gaussian and on uniform X",
        notes=[
            "Gaussian: f = sum_i d_i (q_i^T x)^2 / 2 + 0.1 x_1 (convex: by Stein's lemma the pHd matrix is PSD, so 2016's |e| keeps the signs).",
            "Uniform: f = sum_i d_i x_i^2 / 2 + 0.1 x_1 (axis-aligned, so E[(f - Ef) x x^T] = diag(2 d_i / 45) is PSD; checked on the sample).",
            "Uniform X: run with Python 2 semantics for `np.amin < -1.0` (False, so gamma = 1/3); the Python 3 port raises TypeError there [C-01].",
            "2016 scales by gamma^{-1/2}; v1 whitens with the declared Sigma = gamma I, a factor gamma^{-1}: C_v1 = gamma^{-1/2} C_2016.",
        ],
    )


@case(9, "subspaces", "phd_signs", branch="dev")
def _phd_signs() -> Capture:
    import active_subspaces.subspaces as ss

    gen = _gen(9)
    M, m = 300, 4
    Q = np.linalg.qr(gen.standard_normal((m, m)))[0]
    d = np.array([2.0, -1.2, 0.5, -0.1])
    X = gen.standard_normal((M, m))
    f = _phd_f(X, Q, d)
    with _legacy_context():
        e, W = ss.phd_subspace(X, f[:, None], np.full((M, 1), 1.0 / M))
    return Capture(
        inputs={"X": X, "f": f, "Q": Q, "d": d},
        outputs=_eig_outputs(e, W, known_wrong("N-07", {"kind": "abs_of", "tier": "F10"}, decision="D-31"), reference("F10", compare="eigvecs")),
        call="phd_subspace(X, f[:, None], np.full((M, 1), 1/M))  # indefinite quadratic",
    )


@case(10, "subspaces", "bootstrap_minmax")
def _bootstrap_minmax() -> Capture:
    import active_subspaces.subspaces as ss

    m, M, seed = 5, 500, 0
    X = _gen(10).uniform(-1.0, 1.0, (M, m))
    G = X @ _ka1(m)
    with _legacy_context(seed):
        s = ss.Subspaces()
        s.compute(df=G, nboot=100)
    return Capture(
        inputs={},
        outputs={},
        call="Subspaces().compute(df=X K, nboot=100)  # KA-1, M = 500 uniform (inputs not stored: nothing is compared)",
        legacy_seeds={"np.random.seed": seed},
        not_stored={
            "e_br": f"stochastic min/max bootstrap ranges, shape {list(s.e_br.shape)}; replaced by replicates [N-08], [D-30]",
            "sub_br": f"stochastic min/mean/max subspace errors, shape {list(s.sub_br.shape)}; replaced by replicates [N-08], [D-30]",
        },
    )


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------


def fd_function(X: np.ndarray) -> np.ndarray:
    """f = exp(0.7 x1 - 0.3 x2) + x3 sin(x4), batch (M, 4) -> (M,)."""
    X = np.atleast_2d(X)
    return np.exp(0.7 * X[:, 0] - 0.3 * X[:, 1]) + X[:, 2] * np.sin(X[:, 3])


@case(11, "gradients", "finite_difference")
def _finite_difference() -> Capture:
    import active_subspaces.gradients as gr

    X = _gen(11).uniform(-0.9, 0.9, (20, 4))
    with _legacy_context():
        df = gr.finite_difference_gradients(X, lambda x: fd_function(x)[0], h=1e-6)
    return Capture(
        inputs={"X": X, "h": np.float64(1e-6)},
        outputs={"grads": (np.asarray(df, dtype=np.float64), reference("F8"))},
        call="finite_difference_gradients(X, lambda x: fd_function(x)[0], h=1e-6)",
        notes=["fd_function: f = exp(0.7 x1 - 0.3 x2) + x3 sin(x4); interior points only, so v1's backward rule at x_j = 1 is not exercised."],
    )


@case(12, "gradients", "local_linear_small")
def _local_linear_small() -> Capture:
    import active_subspaces.gradients as gr

    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (15, 10))
    f = rng.normal(size=(15, 1))
    with _legacy_context(0):
        exc = _raises(lambda: gr.local_linear_gradients(X, f))
    assert exc is not None and exc[0] == "ValueError", exc
    return Capture(
        inputs={"X": X, "f": f[:, 0]},
        outputs={"grads": (np.zeros(0), known_wrong("C-19", {"kind": "raises", "exception": exc[0], "message": exc[1]}))},
        call="local_linear_gradients(X, f[:, None])  # default p = floor(1.7 m) = 17 > M = 15",
        inputs_from="repro:C-19 (np.random.default_rng(0))",
        legacy_seeds={"np.random.seed": 0},
    )


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


@case(13, "geometry", "interval")
def _interval() -> Capture:
    import active_subspaces.domains as dom
    import active_subspaces.subspaces as ss

    gen = _gen(13)
    G = gen.standard_normal((20, 4)) * np.array([3.0, 1.0, 0.5, 0.1])
    with _legacy_context():
        s = ss.Subspaces()
        s.compute(df=G)
        s.partition(1)
        d = dom.BoundedActiveVariableDomain(s)
    return Capture(
        inputs={"grads": G, "W1": np.asarray(s.W1, dtype=np.float64)},
        outputs={
            "endpoints": (np.asarray(d.vertY, dtype=np.float64).ravel(), reference("F14", compare="set")),
            "preimages": (np.asarray(d.vertX, dtype=np.float64), reference("X", compare="set")),
        },
        call="s = Subspaces(); s.compute(df=grads); s.partition(1); BoundedActiveVariableDomain(s).vertY, .vertX",
    )


@case(14, "geometry", "zonotope_m3n2")
def _zonotope_m3n2() -> Capture:
    import active_subspaces.domains as dom
    import active_subspaces.subspaces as ss

    G = _gen(14).standard_normal((10, 3))
    with _legacy_context(42):
        s = ss.Subspaces()
        s.compute(df=G)
        s.partition(2)
        d = dom.BoundedActiveVariableDomain(s)
    Y = np.asarray(d.vertY, dtype=np.float64)
    assert Y.shape[0] == dom.nzv(3, 2)
    return Capture(
        inputs={"grads": G, "W1": np.asarray(s.W1, dtype=np.float64)},
        outputs={"vertices": (Y, reference("F12", compare="set")), "preimages": (np.asarray(d.vertX, dtype=np.float64), reference("X", compare="set"))},
        call="s = Subspaces(); s.compute(df=grads); s.partition(2); BoundedActiveVariableDomain(s).vertY  # zonotope_vertices, random search",
        inputs_from="SeedSequence; the 2016 test fixture shape (10 N(0, 1) gradients, m = 3, n = 2)",
        legacy_seeds={"np.random.seed": 42},
        notes=["The search is random but the vertex set is complete (nzv(3, 2) = 6 found), hence deterministic; the 2016 row order is not."],
    )


@case(15, "geometry", "zonotope_inert")
def _zonotope_inert() -> Capture:
    import active_subspaces.domains as dom

    rng = np.random.default_rng(0)
    W1 = np.linalg.qr(rng.normal(size=(4, 2)))[0]
    W1 = np.vstack((W1, np.zeros((1, 2))))
    W1 /= np.linalg.norm(W1, axis=0)
    with _legacy_context(0):
        Y, X = dom.zonotope_vertices(W1, Nsamples=2000, maxcount=20)
    return Capture(
        inputs={"W1": W1},
        outputs={"rows": (np.asarray(Y, dtype=np.float64), known_wrong("C-20", {"kind": "hull_of", "of": "rows", "tier": "F12"}))},
        call="zonotope_vertices(W1, Nsamples=2000, maxcount=20)  # m = 5, n = 2, last row of W1 zero",
        inputs_from="repro:C-20 (np.random.default_rng(0))",
        legacy_seeds={"np.random.seed": 0},
        notes=[f"2016 returns {np.asarray(Y).shape[0]} rows (searching for nzv(5, 2) = 10); the zonotope has nzv(4, 2) = 8 vertices."],
    )


# ---------------------------------------------------------------------------
# quadrature, surrogates, integration
# ---------------------------------------------------------------------------


@case(16, "quadrature", "gauss")
def _gauss() -> Capture:
    import active_subspaces.utils.quadrature as gq

    ks = np.arange(1, 11, dtype=np.int64)
    out = {}
    for label, fn in (("hermite", gq.gauss_hermite), ("legendre", gq.gauss_legendre)):
        with _legacy_context():
            rules = [fn(int(k)) for k in ks]
        out[f"{label}_nodes"] = (np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x, _ in rules]), reference("F14"))
        out[f"{label}_weights"] = (np.concatenate([np.asarray(w, dtype=np.float64).ravel() for _, w in rules]), reference("F14"))
    return Capture(
        inputs={"k": ks},
        outputs=out,
        call="gauss_hermite(k), gauss_legendre(k) for k = 1..10 (port with the [P-08] eigh fix); concatenated in k order",
        inputs_from="deterministic (k = 1..10)",
        notes=["Both 2016 rules are probability rules (weights sum to 1; probabilists' Hermite), as in v1."],
    )


def poly_function(X: np.ndarray) -> np.ndarray:
    """f = exp(0.5 x1) (+ sin(2 x2) when k = 2)."""
    f = np.exp(0.5 * X[:, 0])
    if X.shape[1] > 1:
        f = f + np.sin(2.0 * X[:, 1])
    return f


@case(17, "surrogates", "poly")
def _poly() -> Capture:
    from active_subspaces.utils.response_surfaces import PolynomialApproximation

    gen = _gen(17)
    inputs: dict[str, Any] = {}
    out: dict[str, tuple[Any, dict]] = {}
    for k in (1, 2):
        X = gen.uniform(-1.0, 1.0, (30, k))
        Xt = gen.uniform(-1.0, 1.0, (20, k))
        inputs[f"X_k{k}"], inputs[f"Xtest_k{k}"] = X, Xt
        for d in (2, 3):
            with _legacy_context():
                pr = PolynomialApproximation(d)
                pr.train(X, poly_function(X)[:, None])
                pred = pr.predict(Xt)[0].ravel()
                from active_subspaces.utils.response_surfaces import index_set

                idx = index_set(d, k).astype(np.int64)
            inputs[f"indices_k{k}_d{d}"] = idx
            out[f"coef_k{k}_d{d}"] = (np.asarray(pr.poly_weights, dtype=np.float64).ravel(), reference("F10"))
            out[f"pred_k{k}_d{d}"] = (pred, reference("F10"))
    return Capture(
        inputs=inputs,
        outputs=out,
        call="PolynomialApproximation(d).train(X, f[:, None]); .poly_weights; .predict(Xtest)  # uniform weights",
        notes=["f = poly_function(X): exp(0.5 x1) (+ sin(2 x2) for k = 2); coefficients are in the order of the stored 2016 index set."],
    )


@case(18, "surrogates", "poly_weighted")
def _poly_weighted() -> Capture:
    from active_subspaces.utils.response_surfaces import PolynomialApproximation, index_set

    gen = _gen(18)
    X = gen.uniform(-1.0, 1.0, (30, 2))
    w = gen.integers(1, 5, 30).astype(np.float64)
    Xt = gen.uniform(-1.0, 1.0, (20, 2))
    with _legacy_context():
        pr = PolynomialApproximation(2)
        pr.train(X, poly_function(X)[:, None], weights=w[:, None])
        pred = pr.predict(Xt)[0].ravel()
    rel = {"kind": "reweighted", "weights": "w**2", "tier": "F10"}
    return Capture(
        inputs={"X": X, "weights": w, "Xtest": Xt, "indices": index_set(2, 2).astype(np.int64)},
        outputs={
            "coef": (np.asarray(pr.poly_weights, dtype=np.float64).ravel(), known_wrong("C-04", rel)),
            "pred": (pred, known_wrong("C-04", rel)),
            "Rsqr": (np.float64(pr.Rsqr), known_wrong("C-04", {"kind": "not_compared", "reason": "computed on the weighted f"})),
        },
        call="PolynomialApproximation(2).train(X, f[:, None], weights=w[:, None])  # rows scaled by w, not sqrt(w)",
    )


@case(19, "surrogates", "rbf_scale")
def _rbf_scale() -> Capture:
    from active_subspaces.utils.response_surfaces import RadialBasisApproximation

    Y = np.linspace(-1, 1, 25)[:, None]
    Yt = np.linspace(-1, 1, 200)[:, None]
    errs = {}
    for c in (1.0, 100.0):
        with _legacy_context():
            r = RadialBasisApproximation(2)
            r.train(Y, c * np.sin(3 * Y), v=(c**2) * 1e-2 * np.ones((25, 1)))
            pred = r.predict(Yt)[0]
        truth = c * np.sin(3 * Yt)
        errs[c] = float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))
    rel = {"kind": "improves", "metric": "relative prediction error at scale 100", "direction": "lower", "legacy_at_scale_1": float(f"{errs[1.0]:.6g}")}  # context only; its last digits vary from run to run
    return Capture(
        inputs={"Y": Y[:, 0], "Ytest": Yt[:, 0], "scales": np.array([1.0, 100.0])},
        outputs={"rel_error_scale100": (np.float64(errs[100.0]), known_wrong("C-18", rel, also="N-05"))},
        call="RadialBasisApproximation(2).train(Y, c sin(3Y), v=c^2 1e-2) for c in (1, 100); relative 2-norm error at Ytest",
        inputs_from="repro:C-18 (deterministic)",
        notes=[f"The 2016 error at scale 1 ({errs[1.0]:.4g}) is not known wrong and is kept only in the relation, for context; the defect is the loss of accuracy under scaling."],
    )


@case(20, "integration", "av_unbounded")
def _av_unbounded() -> Capture:
    import active_subspaces.domains as dom
    import active_subspaces.integrals as ints
    import active_subspaces.subspaces as ss
    from active_subspaces.utils.response_surfaces import PolynomialApproximation

    gen = _gen(20)
    M, m = 60, 4
    a = gen.standard_normal(m)
    a /= np.linalg.norm(a)
    X = gen.standard_normal((M, m))
    f = np.exp(0.7 * (X @ a)) + 0.1 * X[:, 1]
    G = 0.7 * np.exp(0.7 * (X @ a))[:, None] * a + 0.1 * np.eye(m)[1]
    with _legacy_context():
        s = ss.Subspaces()
        s.compute(df=G)
        s.partition(1)
        W1 = np.asarray(s.W1, dtype=np.float64)
        pr = PolynomialApproximation(3)
        pr.train(X @ W1, f[:, None])
        avmap = dom.UnboundedActiveVariableMap(dom.UnboundedActiveVariableDomain(s))
        mu = ints.av_integrate(lambda y: pr.predict(np.atleast_2d(y))[0], avmap, 10)
    return Capture(
        inputs={"X": X, "f": f, "grads": G, "W1": W1, "n_nodes": np.int64(10)},
        outputs={"estimate": (np.float64(mu), reference("F12"))},
        call="av_integrate(poly3 fitted on (X W1, f), UnboundedActiveVariableMap(...), 10)  # Gaussian, n = 1",
    )


@case(21, "integration", "bounded_weights")
def _bounded_weights() -> Capture:
    return Capture(
        inputs={},
        outputs={},
        call="interval_quadrature_rule(avmap, N, NX=10000)  # n = 1, bounded",
        inputs_from="none",
        not_stored={"weights": "Monte Carlo histogram weights (stochastic); replaced by deterministic interval-rule weights [N-10], [D-47]"},
    )


# ---------------------------------------------------------------------------
# optimization, designs
# ---------------------------------------------------------------------------


def _c16_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    m = 4
    Q, _ = np.linalg.qr(rng.normal(size=(m, m)))
    A = Q @ np.diag([10.0, 1.0, 0.5, 0.2]) @ Q.T
    x0 = np.array([0.3, -0.2, 0.1, 0.25])
    X = rng.uniform(-1, 1, (400, m))
    return A, x0, X, 2 * (X - x0) @ A


def c16_function(X: np.ndarray, A: np.ndarray, x0: np.ndarray) -> np.ndarray:
    D = np.atleast_2d(X) - x0
    return np.einsum("ij,jk,ik->i", D, A, D)


@case(22, "optimization", "lift")
def _lift() -> Capture:
    import active_subspaces.domains as dom
    import active_subspaces.optimizers as opt
    import active_subspaces.subspaces as ss

    A, x0, X, G = _c16_case()
    f = c16_function(X, A, x0)
    out = {}
    with _legacy_context():
        s = ss.Subspaces()
        s.compute(df=G)
        s.partition(1)
        y = np.array([[float(s.W1[:, 0] @ x0)]])
        for Dom, Map, name in ((dom.UnboundedActiveVariableDomain, opt.UnboundedMinVariableMap, "gaussian"), (dom.BoundedActiveVariableDomain, opt.BoundedMinVariableMap, "uniform")):
            mv = Map(Dom(s))
            mv.train(X, f[:, None])
            xs = np.asarray(mv.inverse(y)[0], dtype=np.float64)
            out[f"f_xstar_{name}"] = (np.float64(c16_function(xs, A, x0)[0]), known_wrong("C-16", {"kind": "improves", "metric": "f(x*)", "direction": "lower"}, decision="D-49"))
    return Capture(
        inputs={"X": X, "f": f, "grads": G, "A": A, "x0": x0},
        outputs=out,
        call="s.compute(df=grads); s.partition(1); {Unbounded,Bounded}MinVariableMap(...).train(X, f); f(inverse(W1^T x0))",
        inputs_from="repro:C-16 (np.random.default_rng(0))",
        notes=["2016 lift_dim p = n + 2 = 3 (m - n > 2), the v1 default [D-107]; the true minimum is f(x0) = 0."],
    )


@v1_capture("optimization", "lift")
def _lift_v1(inp: dict[str, np.ndarray], meta: dict) -> dict[str, tuple[Any, dict]]:
    import active_subspaces as asub

    A, x0 = inp["A"], inp["x0"]
    out = {}
    for name, space in (("gaussian", asub.Gaussian.reference(4)), ("uniform", asub.Uniform.reference(4))):
        sub = asub.active_subspace(grads=inp["grads"], space=space, n=1)
        d = asub.active_domain(sub)
        y = np.asarray(sub.W1).T @ x0
        res = asub.lift(y, domain=d, X=inp["X"], f=inp["f"])
        out[f"v1_f_xstar_{name}"] = (
            np.float64(c16_function(np.asarray(res.x)[None, :], A, x0)[0]),
            v1_reference("F8", decision="D-163", call=f"lift(W1^T x0, domain=active_domain(active_subspace(grads, {type(space).__name__}.reference(4), n=1)), X, f)  # default lift_dim = 3"),
        )
    return out


def maximin_metric(Y: np.ndarray, vert: np.ndarray) -> float:
    """Smallest distance among the design points and from them to the vertices."""
    from scipy.spatial.distance import cdist

    N = Y.shape[0]
    d0 = (cdist(Y, Y) + 1e5 * np.eye(N)).min()
    return float(min(d0, cdist(Y, vert).min()))


@case(23, "designs", "maximin")
def _maximin() -> Capture:
    import active_subspaces.utils.designs as des

    vert = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    R = 20
    out = {}
    for N in (5, 10):
        vals, failed = [], 0
        for seed in range(R):
            shim = _Proxy(np, random=_Proxy(np.random, seed=lambda _s, seed=seed: np.random.seed(seed)))
            with _legacy_context(), _patched(des, "np", shim):
                try:
                    Y = des.maximin_design(vert, N)
                except Exception:  # noqa: BLE001 - a failed SLSQP start, as 2016 reported it
                    failed += 1
                    continue
            vals.append(maximin_metric(np.asarray(Y), vert))
        out[f"min_distance_N{N}"] = (np.zeros(0), summary(_summary_stats(np.array(vals), n_failed=failed, seeds=f"0..{R - 1}"), direction="higher", spec="09 §8"))
    return Capture(
        inputs={"vertices": vert, "W1": np.eye(3)[:, :2], "N": np.array([5, 10], dtype=np.int64)},
        outputs=out,
        call="maximin_design(vertices, N) for N in (5, 10), seeds 0..19",
        inputs_from="deterministic: square zonotope of W1 = [e1 e2], m = 3",
        legacy_seeds={"np.random.seed": "0..19, in place of the fixed reseed with 42 inside maximin_design [R-03]"},
        notes=["min_distance = smallest distance among the design points and to the vertices (v1 ActiveDesign.min_distance)."],
    )


# ---------------------------------------------------------------------------
# test functions (2016 tutorials, master)
# ---------------------------------------------------------------------------

_TUTORIALS = os.path.join(LEGACY["master"], "tutorials", "test_functions")
N_TEST_FUNCTION_POINTS = 250  # spec 17 §5.5 says 1000; 250 keeps the five files near 150 KB in total
TEST_FUNCTIONS = {
    # name: (case index, tutorial file, value function, gradient function, m)
    "otl_circuit": (24, "otl_circuit/otlcircuit_functions.py", "circuit", "circuit_grad", 6),
    "piston": (25, "piston/piston_functions.py", "piston", "piston_grad", 7),
    "robot_arm": (26, "robot/robot_functions.py", "robot", "robot_grad", 8),
    "wing_weight": (27, "wing_weight/wing_functions.py", "wing", "wing_grad", 10),
    "borehole": (28, "borehole/borehole_functions.py", "borehole", "borehole_grad", 8),
}


def _load_tutorial(relpath: str) -> dict:
    path = os.path.join(_TUTORIALS, relpath)
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    namespace: dict = {"__name__": "legacy_tutorial"}
    exec(compile(source, path, "exec"), namespace)  # noqa: S102 - the 2016 tutorial sources of the legacy commit
    return namespace


def _test_function_case(name: str) -> Callable[[], Capture]:
    index, relpath, fname, gname, m = TEST_FUNCTIONS[name]

    def run() -> Capture:
        gen = _gen(index)
        if name == "borehole":
            x = np.column_stack((gen.standard_normal((N_TEST_FUNCTION_POINTS, 2)), gen.uniform(-1.0, 1.0, (N_TEST_FUNCTION_POINTS, m - 2))))
        else:
            x = gen.uniform(-1.0, 1.0, (N_TEST_FUNCTION_POINTS, m))
        ns = _load_tutorial(relpath)
        with _legacy_context():
            f = np.asarray(ns[fname](x), dtype=np.float64)[:, 0]
            g = np.asarray(ns[gname](x), dtype=np.float64)
        notes = []
        if name == "robot_arm":
            keep = f > 1e-3
            x, f, g = x[keep], f[keep], g[keep]
            notes.append(f"points with f <= 1e-3 (near u = v = 0) dropped: {int((~keep).sum())} of {N_TEST_FUNCTION_POINTS}")
        if name == "borehole":
            notes.append("2016 inputs: x1, x2 standard normal (rw = 0.0161812 x1 + 0.1, r = exp(1.0056 x2 + 7.71)), x3..x8 uniform; v1 migration recipe of spec 12 §3.5")
        return Capture(
            inputs={"X": x},
            outputs={"values": (f, reference("F14", "max|f|", spec="12: 1e-13 relative")), "grads": (g, reference("F12", "row max"))},
            call=f"tutorials/test_functions/{relpath}: {fname}(X), {gname}(X)  # normalized inputs",
            notes=notes,
        )

    return run


for _name in TEST_FUNCTIONS:
    case(TEST_FUNCTIONS[_name][0], "test_functions", _name)(_test_function_case(_name))


# ---------------------------------------------------------------------------
# reduced model (dev)
# ---------------------------------------------------------------------------


@case(30, "reduced_model", "probability", branch="dev")
def _probability() -> Capture:
    import active_subspaces.base as base

    a = np.array([1.0, 0.5, -0.2])
    lb, ub, M, seed = -0.5, 1.0, 10_000, 0
    rm = base.ActiveSubspaceReducedModel(3, False)
    rm.as_respsurf = lambda X: (X @ a)[:, None]
    with _legacy_context(seed):
        p, plb, pub = rm.probability(lb, ub, M=M)
    rel = {"kind": "swap", "p": float(p), "n": M, "level": 0.99, "atol": 2e-3}
    return Capture(
        inputs={"a": a, "bounds": np.array([lb, ub]), "n_samples": np.int64(M)},
        outputs={"p": (np.float64(p), summary({"R": 1, "mean": float(p), "sd": math.sqrt(p * (1 - p) / M)}, note="one Monte Carlo estimate; sd is its standard error")), "interval": (np.array([plb, pub]), known_wrong("C-06", rel))},
        call="ActiveSubspaceReducedModel(3, False) with as_respsurf(X) = X a; probability(lb, ub, M=10000) -> (p, plb, pub)",
        inputs_from="deterministic surrogate g(x) = a^T x on Gaussian inputs",
        legacy_seeds={"np.random.seed": seed},
        notes=["2016 plb = p + 2.58 se and pub = p - 2.58 se: the pair is reversed."],
    )


# ---------------------------------------------------------------------------
# Writing (deterministic bytes)
# ---------------------------------------------------------------------------


def _write_npz(path: str, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asarray(arrays[name], order="C"), allow_pickle=False)  # keeps 0-d arrays 0-d
            info = zipfile.ZipInfo(name + ".npy", date_time=_FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, buf.getvalue())


def _read_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n")


def _blas() -> str:
    try:
        info = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        return f"{info.get('name')} {info.get('version')}"
    except Exception:  # noqa: BLE001
        return "unknown"


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "blas": _blas(),
    }


def _port_digest(branch: str) -> str:
    h = hashlib.sha256()
    base = LEGACY[branch]
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                p = os.path.join(dirpath, fn)
                h.update(os.path.relpath(p, base).replace(os.sep, "/").encode())
                with open(p, "rb") as fh:
                    h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _array_meta(arr: np.ndarray) -> dict:
    return {"shape": list(arr.shape), "dtype": str(arr.dtype)}


def _paths(out: str, c: Case) -> tuple[str, str]:
    d = os.path.join(out, c.area)
    return os.path.join(d, c.name + ".npz"), os.path.join(d, c.name + ".json")


def _v1_part(json_path: str, npz_path: str) -> tuple[dict, dict, dict | None]:
    """Existing v1_reference outputs (records, arrays) and v1 environment, kept by a legacy recapture."""
    if not (os.path.exists(json_path) and os.path.exists(npz_path)):
        return {}, {}, None
    with open(json_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    recs = {k: v for k, v in meta.get("outputs", {}).items() if v.get("status") == "v1_reference"}
    arrays = _read_npz(npz_path)
    return recs, {k: arrays["out/" + k] for k in recs}, meta.get("v1_environment")


def write_case(out: str, c: Case, cap: Capture) -> None:
    npz_path, json_path = _paths(out, c)
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    v1_recs, v1_arrays, v1_env = _v1_part(json_path, npz_path)
    arrays: dict[str, np.ndarray] = {}
    meta_inputs, meta_outputs = {}, {}
    for k, v in cap.inputs.items():
        arr = np.asarray(v)
        arrays["in/" + k] = arr
        meta_inputs[k] = _array_meta(arr)
    for k, (v, rec) in cap.outputs.items():
        arr = np.asarray(v)
        arrays["out/" + k] = arr
        meta_outputs[k] = {**_plain(rec), **_array_meta(arr)}
    for k, rec in v1_recs.items():
        arrays["out/" + k] = v1_arrays[k]
        meta_outputs[k] = rec
    meta = {
        "area": c.area,
        "case": c.name,
        "case_index": c.index,
        "seed_sequence": [SEED_ROOT, c.index],
        "inputs_from": cap.inputs_from,
        "environment": {**_environment(), "legacy_branch": c.branch, "legacy_commit": LEGACY_COMMITS[c.branch], "port_digest": _port_digest(c.branch)},
        "legacy_call": cap.call,
        "legacy_seeds": cap.legacy_seeds,
        "inputs": meta_inputs,
        "outputs": meta_outputs,
        "not_stored": cap.not_stored,
        "notes": cap.notes,
    }
    if v1_env is not None:
        meta["v1_environment"] = v1_env
    _write_npz(npz_path, arrays)
    _write_json(json_path, meta)


def write_v1(out: str, c: Case) -> None:
    npz_path, json_path = _paths(out, c)
    with open(json_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    arrays = _read_npz(npz_path)
    inputs = {k[3:]: v for k, v in arrays.items() if k.startswith("in/")}
    import active_subspaces

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        produced = c.v1(inputs, meta)
    for k, (v, rec) in produced.items():
        arr = np.asarray(v)
        arrays["out/" + k] = arr
        meta["outputs"][k] = {**rec, **_array_meta(arr)}
    meta["v1_environment"] = {**_environment(), "active_subspaces": active_subspaces.__version__}
    _write_npz(npz_path, arrays)
    _write_json(json_path, meta)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _find_repro() -> str | None:
    for base in (ROOT, os.path.dirname(ROOT)):
        p = os.path.join(base, "design-specs", "reproducers", "repro.py")
        if os.path.exists(p):
            return p
    return None


def _check_reproducers(branch: str) -> None:
    repro = _find_repro()
    if repro is None:
        raise SystemExit("capture_baselines: design-specs/reproducers/repro.py not found; use --skip-repro only for development")
    env = {**os.environ, "MPLBACKEND": "agg"}
    proc = subprocess.run([sys.executable, "-I", repro, LEGACY[branch]], capture_output=True, text=True, env=env, check=False)
    lines = [ln for ln in proc.stdout.splitlines() if ln[:2] in ("C-", "N-", "P-", "R-", "A-")]
    if proc.returncode != 0 or not lines or not all("PASS" in ln for ln in lines):
        raise SystemExit(f"capture_baselines: a reproducer does not reproduce on legacy/port-{branch}:\n{proc.stdout}\n{proc.stderr}")


def _selected(names: list[str]) -> list[Case]:
    if not names:
        return sorted(CASES, key=lambda c: c.index)
    keep = [c for c in CASES if f"{c.area}/{c.name}" in names or c.area in names]
    missing = set(names) - {f"{c.area}/{c.name}" for c in keep} - {c.area for c in keep}
    if missing:
        raise SystemExit(f"capture_baselines: unknown case(s) {sorted(missing)}")
    return sorted(keep, key=lambda c: c.index)


def capture_branch(branch: str, out: str, names: list[str], skip_repro: bool) -> None:
    if not skip_repro:
        _check_reproducers(branch)
    sys.path.insert(0, LEGACY[branch])
    if "active_subspaces" in sys.modules:
        raise SystemExit("capture_baselines: active_subspaces already imported; the port must be the only one in this process")
    import active_subspaces

    if not os.path.abspath(active_subspaces.__file__).startswith(LEGACY[branch] + os.sep):
        raise SystemExit(f"capture_baselines: imported {active_subspaces.__file__}, not the {branch} port")
    for c in _selected(names):
        if c.branch == branch:
            write_case(out, c, c.run())
            print(f"{branch}: {c.area}/{c.name}")


def capture_v1(out: str, names: list[str]) -> None:
    sys.path.insert(0, os.path.join(ROOT, "src"))
    for c in _selected(names):
        if c.v1 is not None:
            write_v1(out, c)
            print(f"v1: {c.area}/{c.name}")


def _run_child(args: list[str]) -> None:
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), *args], check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cases", nargs="*", help="area or area/case names (default: all)")
    ap.add_argument("--branch", choices=("master", "dev", "all"), default="all")
    ap.add_argument("--v1", action="store_true", help="capture the v1_reference outputs from src/ [D-163]")
    ap.add_argument("--check", action="store_true", help="recapture into a temporary directory and compare bytes with the stored files")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--skip-repro", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    extra = ["--skip-repro"] if a.skip_repro else []
    if a.check:
        with tempfile.TemporaryDirectory() as tmp:
            if os.path.isdir(a.out):
                shutil.copytree(a.out, tmp, dirs_exist_ok=True)  # keeps v1_reference outputs, which a legacy run preserves
            _run_child(["--branch", "all", "--out", tmp, *extra, *a.cases])
            if a.v1:
                _run_child(["--v1", "--out", tmp, *a.cases])
            bad = []
            for c in _selected(a.cases):
                for p_new, p_old in zip(_paths(tmp, c), _paths(a.out, c)):
                    if not (os.path.exists(p_old) and filecmp.cmp(p_new, p_old, shallow=False)):
                        bad.append(os.path.relpath(p_old, ROOT))
            for p in bad:
                print(f"differs: {p}")
            print("baselines identical" if not bad else f"{len(bad)} file(s) differ")
            return 1 if bad else 0
    if a.v1:
        capture_v1(a.out, a.cases)
    elif a.branch == "all":
        for b in ("master", "dev"):
            _run_child(["--branch", b, "--out", a.out, *extra, *a.cases])
    else:
        capture_branch(a.branch, a.out, a.cases, a.skip_repro)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
