"""D-37 regression baselines: loading and the status/relation comparison (spec 17 §4, §5.5).
NumPy/SciPy only.

A baseline is ``tests/data/baselines/<area>/<case>.npz`` (``in/<name>``,
``out/<name>``) plus ``<case>.json``, written by ``tools/capture_baselines.py``.
``assert_matches_baseline`` dispatches on the output's status:

- ``reference`` / ``v1_reference``: agreement at the stored tier, never bitwise.
  ``compare`` selects arrays (norm-wise), ``eigvecs`` (sign-free, per cluster of
  equal eigenvalues, tolerance scaled by the eigenvalue gap), ``set`` (rows as
  a set) or ``exact``.
- ``known_wrong``: the stored relation holds **and** the value differs from the
  legacy value, so a reintroduced defect fails.
- ``summary``: a two-sample comparison of means (one-sided when ``direction``
  is given).
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull

from helpers.metrics import subspace_distance
from helpers.stats import z_bound
from helpers.tolerances import TIERS, U, assert_close

__all__ = ("BASELINE_DIR", "Baseline", "assert_matches_baseline", "load_baseline")

BASELINE_DIR = Path(__file__).resolve().parents[1] / "data" / "baselines"
_EQUAL_EIG = 1e-8  # eigenvalues closer than this times |lambda_1| form one invariant subspace


@dataclass(frozen=True)
class Baseline:
    """One ``<case>.npz`` + ``<case>.json`` (spec 17 §5.5)."""

    area: str
    case: str
    inputs: Mapping[str, NDArray]
    outputs: Mapping[str, NDArray]
    meta: Mapping[str, Any]

    def record(self, name: str) -> Mapping[str, Any]:
        """The JSON record of output ``name`` (status, tolerance or relation)."""
        return self.meta["outputs"][name]


def load_baseline(area: str, case: str) -> Baseline:
    """Load ``tests/data/baselines/<area>/<case>``; arrays are read-only."""
    base = BASELINE_DIR / area / case
    with open(base.with_suffix(".json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    inputs, outputs = {}, {}
    with np.load(base.with_suffix(".npz"), allow_pickle=False) as data:
        for key in data.files:
            arr = data[key]
            arr.flags.writeable = False
            kind, _, name = key.partition("/")
            (inputs if kind == "in" else outputs)[name] = arr
    return Baseline(area=area, case=case, inputs=inputs, outputs=outputs, meta=meta)


def _rtol(rec: Mapping[str, Any]) -> float:
    tier = rec.get("tolerance", rec).get("tier", "X")
    return 0.0 if tier == "X" else TIERS[tier]


def _close(a: Any, d: Any, rtol: float) -> bool:
    try:
        assert_close(np.asarray(a, dtype=np.float64), d, rtol=rtol)
    except AssertionError:
        return False
    return True


def _compare_rows_max(value: NDArray, legacy: NDArray, rtol: float, msg: str) -> None:
    v, d = np.asarray(value, dtype=np.float64), np.asarray(legacy, dtype=np.float64)
    assert v.shape == d.shape, f"{msg}: shape {v.shape} != {d.shape}"
    scale = np.max(np.abs(d), axis=1)
    err = np.max(np.abs(v - d), axis=1) / scale
    assert float(err.max()) <= rtol, f"{msg}: max row-scaled error {err.max():.3e} > {rtol:.1e}"


def _match_sets(value: NDArray, rows: NDArray, tol: float, msg: str) -> None:
    v, d = (
        np.atleast_2d(np.asarray(value, dtype=np.float64)),
        np.atleast_2d(np.asarray(rows, dtype=np.float64)),
    )
    if v.shape[0] == 1 and d.shape[0] > 1 and v.size == d.shape[0]:  # endpoints stored flat
        v = v.reshape(-1, 1)
    if d.shape[0] == 1 and v.shape[0] > 1 and d.size == v.shape[0]:
        d = d.reshape(-1, 1)
    assert v.shape == d.shape, f"{msg}: {v.shape[0]} rows, legacy {d.shape[0]}"
    D = np.linalg.norm(v[:, None, :] - d[None, :, :], axis=2)
    nearest = D.argmin(axis=1)
    assert np.unique(nearest).size == v.shape[0], f"{msg}: rows do not match one to one"
    err = float(D[np.arange(v.shape[0]), nearest].max())
    assert err <= tol, f"{msg}: set distance {err:.3e} > {tol:.3e}"


def _sets_equal(value: NDArray, rows: NDArray, tol: float) -> bool:
    try:
        _match_sets(value, rows, tol, "")
    except AssertionError:
        return False
    return True


def _eigvecs(value: NDArray, legacy: NDArray, e: NDArray, tier_tol: float, msg: str) -> None:
    """Sign-free comparison per cluster of (numerically) equal eigenvalues, tolerance max(tier,
    100 m u λ1 / gap).
    """
    V, W = np.asarray(value, dtype=np.float64), np.asarray(legacy, dtype=np.float64)
    assert V.shape == W.shape, f"{msg}: shape {V.shape} != {W.shape}"
    lam = np.abs(np.asarray(e, dtype=np.float64))
    m = lam.size
    lam1 = float(lam.max())
    bounds = [0] + [i + 1 for i in range(m - 1) if lam[i] - lam[i + 1] > _EQUAL_EIG * lam1] + [m]
    for a, b in itertools.pairwise(bounds):
        if b - a == m:
            continue
        others = np.concatenate([lam[:a], lam[b:]])
        gap = float(np.min(np.abs(lam[a:b, None] - others[None, :])))
        tol = max(tier_tol, 100.0 * m * U * lam1 / gap)
        dist = subspace_distance(V[:, a:b], W[:, a:b])
        assert dist <= tol, (
            f"{msg}: columns {a}..{b - 1}: distance {dist:.3e} > {tol:.3e} (gap {gap:.3e})"
        )


def _sibling_eigenvalues(baseline: Baseline, name: str) -> NDArray:
    key = name[:-1] + "e" if name.endswith("W") else "e"
    return baseline.outputs[key]


def _reference(
    baseline: Baseline, name: str, value: Any, rec: Mapping[str, Any], legacy: NDArray
) -> None:
    msg = f"{baseline.area}/{baseline.case}:{name}"
    rtol = _rtol(rec)
    compare = rec.get("compare", "array")
    scale = rec.get("tolerance", {}).get("scale", "max")
    if compare == "exact":
        assert np.array_equal(np.asarray(value), legacy), f"{msg}: {value!r} != {legacy!r}"
    elif compare == "eigvecs":
        _eigvecs(value, legacy, _sibling_eigenvalues(baseline, name), rtol, msg)
    elif compare == "set":
        if rtol == 0.0:
            a = {tuple(r) for r in np.atleast_2d(np.asarray(value, dtype=np.float64)).tolist()}
            b = {tuple(r) for r in np.atleast_2d(legacy).tolist()}
            assert a == b, f"{msg}: row sets differ"
        else:
            _match_sets(value, legacy, rtol * float(np.max(np.abs(legacy))), msg)
    elif scale == "row max":
        _compare_rows_max(value, legacy, rtol, msg)
    else:
        assert_close(value, legacy, rtol=rtol, err_msg=msg)


def _differs(value: Any, legacy: NDArray, rtol: float, msg: str) -> None:
    v = np.asarray(value)
    if v.shape != legacy.shape:
        return
    if v.dtype.kind in "biu" and legacy.dtype.kind in "biu":
        assert not np.array_equal(v, legacy), f"{msg}: equals the known-wrong legacy value"
        return
    assert not _close(v, legacy, max(rtol, TIERS["F14"])), (
        f"{msg}: equals the known-wrong legacy value within tolerance"
    )


def _outcome_differs(got: Mapping[str, Any], legacy_status: str, legacy_sigma: float) -> bool:
    if got["status"] != legacy_status:
        return True
    if got["sigma"] is None or math.isnan(legacy_sigma):
        return False
    return abs(got["sigma"] - legacy_sigma) > 1e-6 * abs(legacy_sigma)


def _known_wrong(
    baseline: Baseline,
    name: str,
    value: Any,
    rec: Mapping[str, Any],
    legacy: NDArray,
    relation_value: Any,
) -> None:
    rel = rec["relation"]
    kind = rel["kind"]
    msg = f"{baseline.area}/{baseline.case}:{name} [{rec['finding']} {kind}]"
    rtol = TIERS[rel["tier"]] if "tier" in rel else TIERS["F12"]
    if kind == "shift":
        assert np.array_equal(np.asarray(value), legacy + rel["by"]), (
            f"{msg}: {value!r} != legacy {legacy!r} + {rel['by']}"
        )
        _differs(value, legacy, 0.0, msg)
    elif kind == "scale":
        assert_close(
            value,
            rel["value"] * legacy.astype(np.float64),
            rtol=rtol,
            err_msg=f"{msg}: v1 != {rel['factor']} x legacy",
        )
        _differs(value, legacy, rtol, msg)
    elif kind == "abs_of":
        assert_close(
            np.abs(np.asarray(value, dtype=np.float64)),
            legacy,
            rtol=rtol,
            err_msg=f"{msg}: legacy != |v1|",
        )
        _differs(value, legacy, rtol, msg)
    elif kind == "reweighted":
        assert relation_value is not None, (
            f"{msg}: pass relation_value = v1 called with weights {rel['weights']}"
        )
        assert_close(
            relation_value,
            legacy,
            rtol=rtol,
            err_msg=f"{msg}: legacy != v1 with weights {rel['weights']}",
        )
        _differs(value, legacy, rtol, msg)
    elif kind == "raises":
        why = f"{msg}: v1 must return where 2016 raised {rel['exception']}"
        assert value is not None, why
        assert not isinstance(value, BaseException), why
    elif kind == "hull_of":
        rows = np.asarray(legacy, dtype=np.float64)
        hull = rows[ConvexHull(rows).vertices]
        tol = rtol * float(np.max(np.abs(rows)))
        _match_sets(value, hull, tol, f"{msg}: v1 vertices vs hull of the legacy rows")
        assert not _sets_equal(value, rows, tol), f"{msg}: v1 vertices equal the legacy rows"
    elif kind == "swap":
        v = np.asarray(value, dtype=np.float64)
        atol = float(rel["atol"])
        assert legacy[0] > legacy[1], f"{msg}: the stored legacy pair is not reversed"
        assert v[0] <= v[1], f"{msg}: v1 pair {v} is not ordered"
        assert float(np.max(np.abs(v - legacy[::-1]))) <= atol, (
            f"{msg}: v1 {v} is not the reversed legacy pair {legacy[::-1]} (atol {atol})"
        )
        assert float(np.max(np.abs(v - legacy))) > atol, (
            f"{msg}: v1 equals the reversed legacy pair"
        )
    elif kind == "improves":
        v, d = float(np.asarray(value)), float(legacy)
        better = v < d if rel["direction"] == "lower" else v > d
        assert better, (
            f"{msg}: v1 {rel['metric']} = {v!r} is not {rel['direction']} than legacy {d!r}"
        )
    elif kind == "differs":
        expected = rel["expected_v1"]
        got = list(value)
        assert len(got) == len(expected), f"{msg}: {len(got)} outcomes, expected {len(expected)}"
        statuses = baseline.outputs["status"]
        sigmas = baseline.outputs["sigma"]
        for i, (g, e) in enumerate(zip(got, expected, strict=False)):
            why = f"{msg}: row {i}: {g} != expected {e}"
            assert g["status"] == e["status"], why
            assert g.get("order") == e["order"], why
            if e["sigma"] is None:
                assert g["sigma"] is None, f"{msg}: row {i}: sigma {g['sigma']} != None"
            else:
                assert abs(g["sigma"] - e["sigma"]) <= rtol * abs(e["sigma"]), (
                    f"{msg}: row {i}: sigma {g['sigma']!r} != {e['sigma']!r}"
                )
            assert _outcome_differs(g, str(statuses[i]), float(sigmas[i])), (
                f"{msg}: row {i}: v1 {g} equals the 2016 outcome"
            )
    elif kind == "not_compared":
        raise AssertionError(f"{msg}: this legacy value is not compared ({rel.get('reason', '')})")
    else:
        raise AssertionError(f"{msg}: unknown relation kind {kind!r}")


def _summary(baseline: Baseline, name: str, value: Any, rec: Mapping[str, Any], k: int) -> None:
    s = rec["stats"]
    if isinstance(value, Mapping):
        m1, s1, r1 = float(value["mean"]), float(value["sd"]), int(value["R"])
    else:
        v = np.asarray(value, dtype=np.float64).ravel()
        m1, s1, r1 = float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size)
    se = math.sqrt(s1**2 / r1 + float(s["sd"]) ** 2 / int(s["R"]))
    z = z_bound(k)
    diff = m1 - float(s["mean"])
    msg = (
        f"{baseline.area}/{baseline.case}:{name}: v1 mean {m1!r} vs 2016 mean {s['mean']!r} "
        f"(se {se:.3e}, z {z})"
    )
    direction = rec.get("direction")
    if direction == "higher":
        assert diff >= -z * se, msg
    elif direction == "lower":
        assert diff <= z * se, msg
    else:
        assert abs(diff) <= z * se, msg


def assert_matches_baseline(
    baseline: Baseline, name: str, value: Any, *, relation_value: Any = None, k: int = 1
) -> None:
    """Check v1's ``value`` for output ``name`` against the baseline's status (spec 17 §5.5).

    ``relation_value`` is v1 called as the relation states (``reweighted``);
    ``k`` is the Bonferroni count for a ``summary`` comparison inside a larger check.
    """
    rec = baseline.record(name)
    status = rec["status"]
    legacy = baseline.outputs[name]
    if status in ("reference", "v1_reference"):
        _reference(baseline, name, value, rec, legacy)
    elif status == "known_wrong":
        _known_wrong(baseline, name, value, rec, legacy, relation_value)
    elif status == "summary":
        _summary(baseline, name, value, rec, k)
    else:
        raise AssertionError(f"{baseline.area}/{baseline.case}:{name}: unknown status {status!r}")
