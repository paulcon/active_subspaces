"""Tests for spec 05: rng normalization, provenance description, spawning and replay."""

from __future__ import annotations

import json

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import _random
from active_subspaces.results import _make_provenance as make_provenance


def _draws(gen, k=5):
    return gen.random(k)


def test_none_gives_fresh_recorded_entropy():
    g1, d1 = _random.generator(None)
    _g2, d2 = _random.generator(None)
    assert d1["kind"] == "seed_sequence"
    assert d1["bit_generator"] == "PCG64"
    assert d1["entropy"] != d2["entropy"]
    assert isinstance(d1["entropy"], str)
    assert d1["entropy"].isdigit()
    assert d1["spawn_key"] == []
    assert d1["n_children_spawned"] == 0
    a = _draws(g1)
    np.testing.assert_array_equal(_draws(_random.replay(d1)), a)


@pytest.mark.parametrize("seed", [0, 7, 2**70, np.int64(12), np.uint32(5)])
def test_int_seed_matches_default_rng(seed):
    g, d = _random.generator(seed)
    np.testing.assert_array_equal(_draws(g), _draws(np.random.default_rng(int(seed))))
    assert d == {
        "kind": "seed_sequence",
        "bit_generator": "PCG64",
        "entropy": str(int(seed)),
        "spawn_key": [],
        "n_children_spawned": 0,
    }
    np.testing.assert_array_equal(
        _draws(_random.replay(d)), _draws(np.random.default_rng(int(seed)))
    )


def test_seed_sequence_used_as_given():
    ss = np.random.SeedSequence(99)
    ss.spawn(3)
    child = ss.spawn(1)[0]
    g, d = _random.generator(child)
    assert d["spawn_key"] == [3]
    assert d["n_children_spawned"] == 0
    np.testing.assert_array_equal(_draws(g), _draws(np.random.default_rng(child)))
    g2, d2 = _random.generator(ss)
    assert d2["n_children_spawned"] == 4
    r = _random.replay(d2)
    np.testing.assert_array_equal(_draws(r.spawn(1)[0]), _draws(g2.spawn(1)[0]))


def test_seed_sequence_list_entropy_and_pool_size():
    ss = np.random.SeedSequence([1, 2, 2**80], pool_size=8)
    g, d = _random.generator(ss)
    assert d["entropy"] == ["1", "2", str(2**80)]
    assert d["pool_size"] == 8
    np.testing.assert_array_equal(_draws(_random.replay(json.loads(json.dumps(d)))), _draws(g))


def test_generator_used_as_given_and_state_recorded_before_draws():
    caller = np.random.default_rng(5)
    caller.random(3)
    caller.spawn(2)
    g, d = _random.generator(caller)
    assert g is caller
    assert d["kind"] == "generator"
    assert d["bit_generator"] == "PCG64"
    assert d["seed_sequence"]["n_children_spawned"] == 2
    json.dumps(d, allow_nan=False)
    replayed = _random.replay(json.loads(json.dumps(d)))
    kids_a = g.spawn(2)
    a = _draws(g)
    kids_b = replayed.spawn(2)
    np.testing.assert_array_equal(_draws(replayed), a)
    for ka, kb in zip(kids_a, kids_b, strict=False):
        np.testing.assert_array_equal(_draws(ka), _draws(kb))


def test_generator_with_other_bit_generator():
    caller = np.random.Generator(np.random.MT19937(3))
    caller.random(10)
    _, d = _random.generator(caller)
    assert d["bit_generator"] == "MT19937"
    a = _draws(caller)
    np.testing.assert_array_equal(_draws(_random.replay(json.loads(json.dumps(d)))), a)


def test_generator_with_pending_uint32():
    caller = np.random.default_rng(8)
    caller.integers(0, 2**31, dtype=np.uint32)  # leaves a buffered 32-bit half
    _, d = _random.generator(caller)
    a = caller.integers(0, 2**31, size=4, dtype=np.uint32)
    np.testing.assert_array_equal(_random.replay(d).integers(0, 2**31, size=4, dtype=np.uint32), a)


@pytest.mark.parametrize(
    "bad",
    [
        True,
        False,
        np.bool_(True),
        -1,
        np.int64(-3),
        1.0,
        "1",
        [1, 2],
        np.random.RandomState(0),
        np.random,
        np.random.PCG64(1),
    ],
)
def test_rejected_rng_values(bad):
    with pytest.raises(asub.InputError) as e:
        _random.generator(bad)
    assert e.value.code == "input.parameter"
    assert e.value.details["param"] == "rng"


def test_spawn_children_match_numpy_and_ignore_workers():
    g, _ = _random.generator(17)
    kids = _random.spawn(g, 4)
    ref = np.random.default_rng(17).spawn(4)
    assert len(kids) == 4
    for a, b in zip(kids, ref, strict=False):
        np.testing.assert_array_equal(_draws(a), _draws(b))
    # parent draws after the spawn come from the parent stream
    ref_parent = np.random.default_rng(17)
    ref_parent.spawn(4)
    np.testing.assert_array_equal(_draws(g), _draws(ref_parent))


def test_replay_from_provenance_and_result():
    from active_subspaces.results import Result
    from active_subspaces.results import _register_result as register_result

    @register_result
    class _TRng(Result):
        TYPE = "_TestRngResult"
        SCHEMA_VERSION = 1
        draws: tuple[float, ...]

    def stochastic(rng=None):
        gen, desc = _random.generator(rng)
        return _TRng(
            draws=tuple(gen.random(3)),
            provenance=make_provenance("stochastic", {}, rng_state=desc),
        )

    for rng in (None, 4, np.random.SeedSequence(4), np.random.default_rng(4)):
        r = stochastic(rng)
        again = stochastic(_random.replay(r))
        assert again == r
        assert stochastic(_random.replay(r.provenance)) == r
        assert stochastic(_random.replay(asub.from_dict(json.loads(json.dumps(r.to_dict()))))) == r
    assert stochastic(9) == stochastic(9) == stochastic(np.random.default_rng(9))


def test_replay_errors():
    with pytest.raises(ValueError, match="no rng description"):
        _random.replay(None)
    with pytest.raises(ValueError, match="unknown rng description kind"):
        _random.replay({"kind": "other", "bit_generator": "PCG64"})


def test_no_global_state_touched():
    original = np.random.get_state()  # noqa: NPY002 (checks the legacy global state, spec 05 §8)
    try:
        before = np.random.get_state()  # noqa: NPY002 (checks the legacy global state, spec 05 §8)
        _random.generator(None)
        _random.generator(5)
        asub.Uniform([0], [1]).sample(10)
        asub.Gaussian([0], [1]).sample(10, rng=3)
        after = np.random.get_state()  # noqa: NPY002 (checks the legacy global state, spec 05 §8)
        assert np.array_equal(before[1], after[1])
        assert before[2:] == after[2:]
    finally:
        np.random.set_state(original)  # noqa: NPY002 (checks the legacy global state, spec 05 §8)
