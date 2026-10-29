"""``helpers.baselines.assert_matches_baseline`` on synthetic baselines (spec 17 §8 "Baselines").

Every status and relation kind passes on a value that satisfies it and fails on one that does
not; in particular a known-wrong value that equals the v1 value must fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from helpers.baselines import BASELINE_DIR, Baseline, assert_matches_baseline, load_baseline


def _b(outputs: dict, records: dict, inputs: dict | None = None) -> Baseline:
    return Baseline(
        area="synthetic",
        case="case",
        inputs=inputs or {},
        outputs={k: np.asarray(v) for k, v in outputs.items()},
        meta={"outputs": records},
    )


def _kw(relation: dict) -> dict:
    return {"status": "known_wrong", "finding": "X-00", "relation": relation}


def _fails(*args, **kwargs) -> None:
    with pytest.raises(AssertionError):
        assert_matches_baseline(*args, **kwargs)


def test_reference_array_and_exact():
    b = _b(
        {"v": [1.0, 2.0], "n": 3},
        {
            "v": {"status": "reference", "tolerance": {"tier": "F12"}},
            "n": {"status": "reference", "tolerance": {"tier": "X"}, "compare": "exact"},
        },
    )
    assert_matches_baseline(b, "v", np.array([1.0, 2.0 + 1e-13]))
    _fails(b, "v", np.array([1.0, 2.0 + 1e-10]))
    assert_matches_baseline(b, "n", np.int64(3))
    _fails(b, "n", np.int64(2))


def test_reference_eigvecs_sign_free_and_clusters():
    W = np.linalg.qr(np.random.default_rng(1).standard_normal((4, 4)))[0]
    e = np.array([3.0, 1.0, 0.0, 0.0])  # a 2-dimensional null cluster: only its span is compared
    b = _b(
        {"W": W, "e": e},
        {"W": {"status": "reference", "tolerance": {"tier": "F10"}, "compare": "eigvecs"}},
    )
    R = np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]])
    V = W * np.array([-1.0, 1.0, 1.0, 1.0])
    V[:, 2:] = W[:, 2:] @ R
    assert_matches_baseline(b, "W", V)
    V[:, [0, 1]] = V[:, [1, 0]]
    _fails(b, "W", V)


def test_reference_sets():
    rows = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    b = _b(
        {"Y": rows, "S": rows},
        {
            "Y": {"status": "reference", "tolerance": {"tier": "F12"}, "compare": "set"},
            "S": {"status": "reference", "tolerance": {"tier": "X"}, "compare": "set"},
        },
    )
    assert_matches_baseline(b, "Y", rows[::-1] + 1e-14)
    assert_matches_baseline(b, "S", rows[[2, 0, 1]])
    _fails(b, "Y", rows[:2])
    _fails(b, "S", rows + 1e-3)


def test_v1_reference():
    b = _b({"x": 0.25}, {"x": {"status": "v1_reference", "tolerance": {"tier": "F8"}}})
    assert_matches_baseline(b, "x", np.float64(0.25 * (1 + 1e-10)))
    _fails(b, "x", np.float64(0.2501))


def test_known_wrong_shift():
    b = _b({"n": 2}, {"n": _kw({"kind": "shift", "by": -1})})
    assert_matches_baseline(b, "n", np.int64(1))
    _fails(b, "n", np.int64(2))  # the known-wrong value itself
    _fails(b, "n", np.int64(3))


def test_known_wrong_scale_and_abs():
    b = _b(
        {"e": [2.0, 1.0]},
        {"e": _kw({"kind": "scale", "factor": "1/S", "value": 0.5, "tier": "F12"})},
    )
    assert_matches_baseline(b, "e", np.array([1.0, 0.5]))
    _fails(b, "e", np.array([2.0, 1.0]))
    b = _b({"e": [2.0, 1.0]}, {"e": _kw({"kind": "abs_of", "tier": "F12"})})
    assert_matches_baseline(b, "e", np.array([2.0, -1.0]))
    _fails(b, "e", np.array([2.0, 1.0]))


def test_known_wrong_scale_by_one_cannot_pass():
    b = _b(
        {"e": [2.0, 1.0]},
        {"e": _kw({"kind": "scale", "factor": "1", "value": 1.0, "tier": "F12"})},
    )
    _fails(b, "e", np.array([2.0, 1.0]))


def test_known_wrong_reweighted():
    b = _b({"c": [1.0, 2.0]}, {"c": _kw({"kind": "reweighted", "weights": "w**2", "tier": "F12"})})
    assert_matches_baseline(b, "c", np.array([1.1, 2.0]), relation_value=np.array([1.0, 2.0]))
    _fails(b, "c", np.array([1.0, 2.0]), relation_value=np.array([1.0, 2.0]))
    _fails(b, "c", np.array([1.1, 2.0]), relation_value=np.array([1.1, 2.0]))
    _fails(b, "c", np.array([1.1, 2.0]))


def test_known_wrong_raises():
    b = _b({"C": np.zeros(0)}, {"C": _kw({"kind": "raises", "exception": "TypeError"})})
    assert_matches_baseline(b, "C", object())
    _fails(b, "C", TypeError("x"))


def test_known_wrong_hull_of():
    square = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
    rows = np.vstack([square, [[0.0, 1.0], [1.0, 0.0]]])  # two boundary rows that are not vertices
    b = _b({"rows": rows}, {"rows": _kw({"kind": "hull_of", "of": "rows", "tier": "F12"})})
    assert_matches_baseline(b, "rows", square[::-1])
    _fails(b, "rows", rows)
    _fails(b, "rows", square[:3])


def test_known_wrong_swap():
    b = _b({"i": [0.6, 0.4]}, {"i": _kw({"kind": "swap", "atol": 1e-3})})
    assert_matches_baseline(b, "i", np.array([0.4, 0.6 + 1e-4]))
    _fails(b, "i", np.array([0.6, 0.4]))
    _fails(b, "i", np.array([0.3, 0.7]))


def test_known_wrong_improves():
    b = _b(
        {"err": 0.5}, {"err": _kw({"kind": "improves", "metric": "error", "direction": "lower"})}
    )
    assert_matches_baseline(b, "err", np.float64(0.1))
    _fails(b, "err", np.float64(0.5))
    _fails(b, "err", np.float64(0.6))


def test_known_wrong_differs():
    expected = [
        {"status": "detected", "sigma": 1e-3, "order": 2},
        {"status": "h_too_large", "sigma": None, "order": None},
    ]
    rec = _kw({"kind": "differs", "expected_v1": expected, "tier": "F14"})
    b = _b(
        {"status": np.array(["detected", "detected"]), "sigma": np.array([2e-3, 5e-4])},
        {"status": rec, "sigma": rec},
    )
    assert_matches_baseline(b, "status", [dict(e) for e in expected])
    _fails(b, "status", [{"status": "detected", "sigma": 2e-3, "order": 2}, expected[1]])
    same = _b(
        {"status": np.array(["detected", "h_too_large"]), "sigma": np.array([1e-3, np.nan])},
        {"status": rec, "sigma": rec},
    )
    _fails(same, "status", [dict(e) for e in expected])


def test_known_wrong_not_compared():
    b = _b({"R": 0.9}, {"R": _kw({"kind": "not_compared"})})
    _fails(b, "R", np.float64(0.9))


def test_summary_two_and_one_sided():
    stats = {"R": 20, "mean": 1.0, "sd": 0.1}
    b = _b(
        {"s": np.zeros(0), "h": np.zeros(0)},
        {
            "s": {"status": "summary", "stats": stats},
            "h": {"status": "summary", "stats": stats, "direction": "higher"},
        },
    )
    v = 1.0 + 0.1 * np.linspace(-1.5, 1.5, 20)
    assert_matches_baseline(b, "s", v)
    _fails(b, "s", v + 1.0)
    assert_matches_baseline(b, "h", v + 1.0)
    _fails(b, "h", v - 1.0)
    assert_matches_baseline(b, "s", {"mean": 1.01, "sd": 0.01, "R": 1})


def test_every_stored_case_loads_with_known_statuses():
    """Every stored output has one of the four statuses; known-wrong ones name a finding and a
    relation.
    """
    cases = sorted(
        p.relative_to(BASELINE_DIR).with_suffix("") for p in BASELINE_DIR.glob("*/*.json")
    )
    assert len(cases) >= 26
    for rel in cases:
        b = load_baseline(*rel.parts)
        assert set(b.outputs) == set(b.meta["outputs"]), rel
        for name, rec in b.meta["outputs"].items():
            assert rec["status"] in ("reference", "known_wrong", "summary", "v1_reference"), (
                rel,
                name,
            )
            if rec["status"] == "known_wrong":
                assert rec["finding"], (rel, name)
                assert rec["relation"]["kind"], (rel, name)
        assert b.meta["environment"]["legacy_commit"] in (
            "64817a19db250e4b01bcd51055ad0f7d2a5665b8",
            "361ba2c3052d8346c581f9a11fd0ca755a351d25",
        )
