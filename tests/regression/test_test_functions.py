"""Agreement of the v1 test functions with the 2016 tutorial functions (spec 12 §3.5, §8; V12.3,
V12.4) [D-37].

The 2016 functions are run from the Python 3 reference port's copy of the 2016
``tutorials/test_functions`` sources (``legacy/port-master/``), with the one library call they
make (``BoundedNormalizer.unnormalize``) supplied by a stub with the 2016 formula. ``legacy/``
was removed before v1.0 and is kept at the git tag ``legacy-py3``, so these tests skip outside
that checkout; the baselines in ``tests/data/baselines/test_functions/`` carry the comparison.
"""

from __future__ import annotations

import pathlib
import types

import numpy as np
import pytest

from active_subspaces import test_functions as tf

_TUTORIALS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "legacy"
    / "port-master"
    / "tutorials"
    / "test_functions"
)


class _BoundedNormalizer:
    """The 2016 ``utils.misc.BoundedNormalizer`` (unnormalize only)."""

    def __init__(self, lb, ub):
        m = lb.size
        self.lb = lb.reshape((1, m))
        self.ub = ub.reshape((1, m))

    def unnormalize(self, X):
        return (self.ub - self.lb) * (X + 1.0) / 2.0 + self.lb


def _load_2016(relpath: str) -> dict:
    path = _TUTORIALS / relpath
    if not path.exists():
        pytest.skip(f"2016 tutorial source {path} not available")
    source = path.read_text().replace("import active_subspaces as ac", "")
    stub = types.SimpleNamespace(
        utils=types.SimpleNamespace(
            misc=types.SimpleNamespace(BoundedNormalizer=_BoundedNormalizer)
        )
    )
    namespace: dict = {"np": np, "ac": stub, "__name__": "legacy_" + path.stem}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace


CASES = {
    "otl_circuit": (
        "otl_circuit/otlcircuit_functions.py",
        "circuit",
        "circuit_grad",
        tf.OTL_CIRCUIT,
    ),
    "piston": ("piston/piston_functions.py", "piston", "piston_grad", tf.PISTON),
    "robot_arm": ("robot/robot_functions.py", "robot", "robot_grad", tf.ROBOT_ARM),
    "wing_weight": ("wing_weight/wing_functions.py", "wing", "wing_grad", tf.WING_WEIGHT),
}


def _rel_rows(a: np.ndarray, b: np.ndarray) -> float:
    scale = np.max(np.abs(b), axis=1, keepdims=True)
    return float(np.max(np.abs(a - b) / scale))


@pytest.mark.parametrize("name", sorted(CASES))
def test_vs_2016(name):
    relpath, fname, gname, model = CASES[name]
    legacy = _load_2016(relpath)
    space = model.space
    x = np.random.default_rng(2016).uniform(-1.0, 1.0, (1000, space.dim))
    f16 = legacy[fname](x)[:, 0]
    g16 = legacy[gname](x)
    if name == "robot_arm":  # away from u = v = 0
        keep = f16 > 1e-3
        x, f16, g16 = x[keep], f16[keep], g16[keep]
    X = space.from_normalized(x)
    f = model.value(X)
    g = space.grads_to_normalized(model.grad(X))
    assert np.max(np.abs(f - f16) / np.abs(f16)) <= 1e-13
    assert _rel_rows(g, g16) <= 1e-12


def test_borehole_migration():
    legacy = _load_2016("borehole/borehole_functions.py")
    gen = np.random.default_rng(2016)
    x = np.column_stack((gen.standard_normal((1000, 2)), gen.uniform(-1.0, 1.0, (1000, 6))))
    f16 = legacy["borehole"](x)[:, 0]
    g16 = legacy["borehole_grad"](x)
    lo, up = tf.BOREHOLE.space.lower, tf.BOREHOLE.space.upper
    half = 0.5 * (up - lo)
    X = np.empty_like(x)
    X[:, 0] = 0.0161812 * x[:, 0] + 0.1
    X[:, 1] = np.exp(1.0056 * x[:, 1] + 7.71)
    X[:, 2:] = lo[2:] + half[2:] * (x[:, 2:] + 1.0)
    f = tf.borehole(X)
    G = tf.borehole_grad(X)
    g = np.empty_like(G)
    g[:, 0] = 0.0161812 * G[:, 0]
    g[:, 1] = 1.0056 * X[:, 1] * G[:, 1]
    g[:, 2:] = half[2:] * G[:, 2:]
    assert np.max(np.abs(f - f16) / np.abs(f16)) <= 1e-12
    assert _rel_rows(g, g16) <= 1e-12
