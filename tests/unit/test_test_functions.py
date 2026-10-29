"""Unit tests of the built-in test problems (spec 12 §4.6-4.7; V12.2, V12.11)."""

from __future__ import annotations

import numpy as np
import pytest

import active_subspaces as asub
from active_subspaces import test_functions as tf
from active_subspaces.errors import InputError

ENGINEERING = {
    "borehole": (tf.BOREHOLE, tf.borehole, tf.borehole_grad, 8),
    "otl_circuit": (tf.OTL_CIRCUIT, tf.otl_circuit, tf.otl_circuit_grad, 6),
    "piston": (tf.PISTON, tf.piston, tf.piston_grad, 7),
    "robot_arm": (tf.ROBOT_ARM, tf.robot_arm, tf.robot_arm_grad, 8),
    "wing_weight": (tf.WING_WEIGHT, tf.wing_weight, tf.wing_weight_grad, 10),
}


def _known_answer_models():
    out = []
    for space in (asub.Uniform.reference(4), asub.Gaussian.reference(4)):
        out.append(tf.quadratic([3.0, 1.0, 0.5, 0.0], space=space))
        for profile in ("linear", "sin", "square"):
            out.append(tf.ridge([1.0, -2.0, 0.5, 3.0], profile=profile, space=space))
    out.append(tf.quadratic([1.0, 0.1, 0.01], space=asub.Uniform([0, 1, 2], [3, 5, 9])))
    gs = asub.Gaussian(
        np.array([1.0, -1.0, 0.0]), np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 0.5]])
    )
    out.append(tf.ridge([1.0, 1.0, 1.0], profile="square", space=gs))
    return out


@pytest.mark.parametrize("name", sorted(ENGINEERING))
def test_engineering_models(name):
    model, value, grad, m = ENGINEERING[name]
    assert model.name == name
    assert model.dim == m
    assert model.value is value
    assert model.grad is grad
    assert model.value_and_grad is None
    assert model.cost == 1e-6
    assert model.batch_size is None
    assert model.references == ("SurjanovicBingham",)
    assert isinstance(model.space, asub.Uniform)
    assert len(model.input_descriptions) == m
    assert model.reference_results.model == name
    assert model.qoi == "f"
    X = model.space.sample(5, rng=0)
    assert value(X).shape == (5,)
    assert grad(X).shape == (5, m)
    for bad in (X[0], X[:, :-1], X[None]):
        for fn in (value, grad):
            with pytest.raises(InputError) as e:
                fn(bad)
            assert e.value.code == "input.shape"


def test_input_tables():
    sp = tf.BOREHOLE.space
    assert sp.names == ("rw", "r", "Tu", "Hu", "Tl", "Hl", "L", "Kw")
    assert np.array_equal(sp.lower, [0.05, 100, 63070, 990, 63.1, 700, 1120, 9855])
    assert np.array_equal(sp.upper, [0.15, 50000, 115600, 1110, 116, 820, 1680, 12045])
    assert tf.OTL_CIRCUIT.space.units[-1] is None
    assert tf.OTL_CIRCUIT.qoi_units == "V"
    assert tf.ROBOT_ARM.qoi_units is None
    assert tf.ROBOT_ARM.space.upper[0] == 2 * np.pi
    assert tf.WING_WEIGHT.space.names[3] == "Lambda"
    assert tf.WING_WEIGHT.space.units[3] == "deg"
    assert tf.PISTON.space.names == ("M", "S", "V0", "k", "P0", "Ta", "T0")


def test_value_and_grad():
    for model in _known_answer_models() + [
        tf.builtin_models()[n] for n in tf.builtin_models() if "_10" in n
    ]:
        X = model.space.sample(17, rng=4)
        f, g = model.value_and_grad(X)
        assert np.array_equal(f, model.value(X))
        assert np.array_equal(g, model.grad(X))
        f2, g2 = asub.evaluate_with_grads(model, X=X)
        assert np.array_equal(f2, f)
        assert np.array_equal(g2, g)


def test_known_answer_reference_results():
    lam = np.array([3.0, 1.0, 0.5, 0.0])
    model = tf.quadratic(lam, name="q4")
    rr = model.reference_results
    assert rr.exact
    assert rr.model == "q4"
    assert rr.n_samples is None
    assert rr.seed is None
    assert np.array_equal(rr.eigenvalues, lam)
    assert np.array_equal(rr.eigenvectors, tf.dct_basis(4))
    assert tf.quadratic(lam).reference_results.model == "<unnamed>"
    assert rr.provenance.function == "active_subspaces.test_functions.quadratic"
    rd = tf.ridge([0.0, 0.0, 2.0], profile="linear").reference_results
    assert np.allclose(rd.eigenvectors[:, 0], [0, 0, 1], rtol=0, atol=1e-15)
    assert np.array_equal(rd.eigenvalues, [1.0, 0.0, 0.0])
    assert rd.variance == pytest.approx(1 / 3)
    assert np.array_equal(
        tf.ridge([1.0, 0.0], profile="linear").reference_results.eigenvectors, np.eye(2)
    )
    assert np.array_equal(tf.ridge([-5.0]).reference_results.eigenvectors, [[-1.0]])


def test_factory_errors():
    cases = [
        lambda: tf.quadratic([1.0, 2.0]),
        lambda: tf.quadratic([1.0, -1.0]),
        lambda: tf.quadratic([0.0, 0.0]),
        lambda: tf.quadratic([1.0, np.nan]),
        lambda: tf.quadratic([]),
        lambda: tf.quadratic([[1.0]]),
        lambda: tf.quadratic([1.0, 0.5], directions=[[1.0, 0.1], [0.0, 1.0]]),
        lambda: tf.quadratic([1.0, 0.5], directions=np.eye(3)),
        lambda: tf.quadratic([1.0, 0.5], space=asub.Uniform.reference(3)),
        lambda: tf.quadratic([1.0, 0.5], space="uniform"),
        lambda: tf.quadratic([1.0, 0.5], name="Bad Name"),
        lambda: tf.ridge([0.0, 0.0]),
        lambda: tf.ridge([1.0, np.inf]),
        lambda: tf.ridge([1.0, 1.0], profile="cubic"),
        lambda: tf.ridge([1.0, 1.0], space=asub.Gaussian.reference(3)),
        lambda: tf.dct_basis(0),
        lambda: tf.dct_basis(2.5),
    ]
    for case in cases:
        with pytest.raises(InputError) as e:
            case()
        assert e.value.code == "input.parameter"


def test_builtin_known_answer_instances():
    reg = tf.builtin_models()
    lam = reg["quadratic_uniform_10"].reference_results.eigenvalues
    assert lam[0] == 1.0
    assert lam[1] == 0.1
    assert lam[2] == 1e-4
    assert lam[9] == pytest.approx(1e-4 * 2.0**-7, rel=1e-15)
    gaps = lam[:-1] - lam[1:]
    log_gaps = np.log(lam[:-1]) - np.log(lam[1:])
    assert np.argmax(gaps) == 0
    assert np.argmax(log_gaps) == 1
    assert isinstance(reg["quadratic_gaussian_10"].space, asub.Gaussian)
    a = reg["ridge_sin_uniform_10"].reference_results.eigenvectors[:, 0]
    assert np.allclose(a, np.arange(1, 11) / np.linalg.norm(np.arange(1, 11)), rtol=0, atol=1e-15)
