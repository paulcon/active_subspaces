"""UC-12: save, load and provenance replay (spec 01 §5; spec 17 V01.61–V01.65).

``G`` is the UC-01 Monte Carlo gradient sample; the run under review uses ``rng=None`` (fresh
entropy [D-72]).
``_random.replay`` is the private test helper spec 01 §4 names for the replay recipe of spec 05
[D-71].
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import _random
from helpers.usecases import Evil, physical_model, space, uc01_mc, was_unpickled


@pytest.fixture
def run(stat_rng):
    _, _, G, _ = uc01_mc(stat_rng)
    return G, asub.active_subspace(grads=G, space=space("U5"), nboot=50)


def test_round_trips(run, tmp_path, assert_readonly):
    """V01.61 [D-19]: ``.npz`` and JSON (list and ``binary=True``) round trips give == results;
    arrays read-only.
    """
    _, sub = run
    sub.save(tmp_path / "sub.npz")
    back = asub.load(tmp_path / "sub.npz")
    assert back == sub
    assert type(back) is type(sub)
    assert back.provenance.input_space == sub.provenance.input_space == sub.space.to_dict()
    assert_readonly(back)
    for binary in (False, True):
        text = json.dumps(sub.to_dict(binary=binary), allow_nan=False)
        again = asub.from_dict(json.loads(text))
        assert again == sub
        assert again.provenance.input_space == sub.space.to_dict()


def test_provenance_and_digest(run):
    """V01.62 [D-20], [D-65]: function name; sha256 of the physical ``G``; the space; ``rng`` kind
    and entropy string;
    a one-ulp change in ``G`` gives a new digest."""
    G, sub = run
    prov = sub.provenance
    assert prov.function == "active_subspaces.active_subspace"
    digest = hashlib.sha256(np.ascontiguousarray(G, dtype=np.float64).tobytes()).hexdigest()
    assert prov.parameters["grads"]["sha256"] == digest
    assert prov.input_space == space("U5").to_dict()
    assert prov.to_dict()["input_space"] == space("U5").to_dict()
    assert prov.rng["kind"] == "seed_sequence"
    ent = prov.rng["entropy"]
    assert isinstance(ent, str)
    assert ent.isdigit()
    G2 = np.array(G)
    G2[3, 2] = np.nextafter(G2[3, 2], np.inf)
    sub2 = asub.active_subspace(grads=G2, space=space("U5"))
    assert sub2.provenance.parameters["grads"]["sha256"] != digest


def test_replay(run, tmp_path):
    """V01.63 [D-71], [D-72]: replaying the recorded entropy of an ``rng=None`` run reproduces
    it bitwise.
    """
    G, sub = run
    sub.save(tmp_path / "sub.npz")
    back = asub.load(tmp_path / "sub.npz")
    again = asub.active_subspace(
        grads=G, space=space("U5"), nboot=50, rng=_random.replay(back.provenance)
    )
    assert again == sub


W = np.array([1, 0.5, 0, -0.3, 0, 0.1, 0]) / np.linalg.norm([1, 0.5, 0, -0.3, 0, 0.1, 0])


def _ka6():
    def phi(x):
        s = x[:, 0] + x[:, 1]
        return s**2 + s

    def dphi(x):
        s = x[:, 0] + x[:, 1]
        return np.outer(2 * s + 1, [1.0, 1.0, 0, 0, 0])

    return physical_model("U5", phi, dphi)[0]


def test_nested_and_reduced_model(tmp_path, stat_rng):
    """V01.64 (spec 04 §5.3): a saved UC-06 surrogate loads with its embedded subspace; a saved
    UC-09 reduced
    model loads equal and predicts bitwise the same with no model in memory."""
    model06, _ = physical_model(
        "R7", lambda x: 1 + 2 * (x @ W) - 3 * (x @ W) ** 2, lambda x: np.outer(2 - 6 * (x @ W), W)
    )
    X = space("R7").sample(50, rng=stat_rng("UC-06/sample"))
    f, G = asub.evaluate_with_grads(model06, X=X)
    sub_06 = asub.active_subspace(grads=G, space=space("R7"))
    surr = asub.fit_surrogate(subspace=sub_06, X=X, f=f, kind="poly", degree=2)
    surr.save(tmp_path / "surr.npz")
    surr2 = asub.load(tmp_path / "surr.npz")
    assert surr2 == surr
    assert surr2.subspace == sub_06

    rm = asub.build_reduced_model(model=_ka6(), rng=stat_rng("UC-09/build"))
    X_test = space("U5").sample(100, rng=stat_rng("UC-09/test_points"))
    expected = rm.predict(X_test)
    rm.save(tmp_path / "rm.npz")
    del rm
    rm2 = asub.load(tmp_path / "rm.npz")
    assert rm2 == asub.build_reduced_model(model=_ka6(), rng=stat_rng("UC-09/build"))
    assert np.array_equal(rm2.predict(X_test), expected)


def test_no_pickle(run, tmp_path):
    """V01.65 (spec 04 §8 safety): a valid saved result with one array replaced by an object array
    whose unpickling
    has a side effect -> ``SchemaError`` ``schema.invalid``, and nothing is unpickled."""
    _, sub = run
    sub.save(tmp_path / "sub.npz")
    with np.load(tmp_path / "sub.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    victim = min(k for k in arrays if k != "__meta__")
    arrays[victim] = np.array([Evil()], dtype=object)
    np.savez(tmp_path / "evil.npz", **arrays)
    with pytest.raises(asub.SchemaError) as e:
        asub.load(tmp_path / "evil.npz")
    assert e.value.code == "schema.invalid"
    assert not was_unpickled()
