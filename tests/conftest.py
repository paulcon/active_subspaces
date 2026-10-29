"""Shared fixtures and markers for the v1 test suite (spec 15 §5.4).

Each test gets exactly one tier marker: the name of its tier directory
(``unit/``, ``known_answer/``, ...; ``api/`` counts as ``unit``), or ``unit``
for a test directly in ``tests/``. ``slow`` is added by hand.

Autouse checks (every test): nothing is written to stdout/stderr, the
legacy global NumPy random state is untouched, the test runs in a fresh empty
working directory and fails if any file appears there [R-02], and pyplot
figures it opened are closed. BLAS is pinned to one thread.

Helper fixtures for component tests:

- ``seed``: a fixed integer seed.
- ``uniform_space(m)`` / ``gaussian_space(m)``: factories for random,
  non-trivial spaces (unequal ranges; correlated covariance).
- ``make_provenance``: a ready ``Provenance`` for hand-built results.
- ``assert_roundtrip(result)``: dict/JSON (list and base64), ``.npz`` via path
  and ``BytesIO``, and pickle round trips all give an equal result.
- ``assert_readonly(result)``: every array field is read-only for good.
- ``stat_rng(label="")``: a fresh ``np.random.SeedSequence`` for a statistical
  check (spec 17 §5.2, ``tests/helpers/seeds.py``). ``--stat-seeds=fixed``
  (default) uses the spec 01 seed for an ``ACCEPTANCE_SEEDS`` key, else
  ``check_seed(nodeid, label)``; ``--stat-seeds=random`` draws one entropy per
  run, records it (``record_property("stat_seed", …)`` and a report section on
  failure); ``--stat-seeds=<int>`` replays that entropy.
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import dataclasses  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import active_subspaces as asub  # noqa: E402
from active_subspaces import results as _results  # noqa: E402
from helpers import seeds as _seeds  # noqa: E402
from helpers.isolation import empty_working_directory, figures_closed  # noqa: E402

_TIERS = ("unit", "known_answer", "property", "statistical", "regression", "acceptance")
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


_STAT_ENTROPY = pytest.StashKey[object]()
_STAT_USED = pytest.StashKey[int]()


def pytest_addoption(parser):
    parser.addoption(
        "--stat-seeds",
        default="fixed",
        help="seeds of statistical checks: 'fixed' (default), 'random', or an integer entropy to "
        "replay (spec 17 §5.2)",
    )


def pytest_configure(config):
    value = str(config.getoption("--stat-seeds", default="fixed"))
    if value == "fixed":
        entropy = None
    elif value == "random":
        entropy = int(np.random.SeedSequence().entropy)
    elif value.isdigit():
        entropy = int(value)
    else:
        raise pytest.UsageError(
            f"--stat-seeds: expected 'fixed', 'random' or a non-negative integer, got {value!r}"
        )
    config.stash[_STAT_ENTROPY] = entropy


def pytest_report_header(config):
    entropy = config.stash.get(_STAT_ENTROPY, None)
    return (
        None
        if entropy is None
        else f"stat-seeds: entropy {entropy} (replay with --stat-seeds={entropy})"
    )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    entropy = item.stash.get(_STAT_USED, None)
    if entropy is not None and report.failed:
        report.sections.append(
            (
                "stat-seeds",
                f"statistical seeds from entropy {entropy}; replay with --stat-seeds={entropy}",
            )
        )
    return report


def pytest_collection_modifyitems(config, items):
    for item in items:
        rel = os.path.relpath(str(item.path), _TESTS_DIR).split(os.sep)
        tier = rel[0] if len(rel) > 1 else "unit"
        tier = "unit" if tier == "api" else tier
        if tier in _TIERS:
            item.add_marker(getattr(pytest.mark, tier))


@pytest.fixture(autouse=True)
def _no_output_and_no_global_rng(capsys):
    state = np.random.get_state()  # noqa: NPY002 (checks the legacy global state, spec 05 §8)
    yield
    after = np.random.get_state()  # noqa: NPY002 (checks the legacy global state, spec 05 §8)
    changed = "the legacy global NumPy random state was changed"
    assert state[0] == after[0], changed
    assert np.array_equal(state[1], after[1]), changed
    assert state[2:] == after[2:], changed
    out, err = capsys.readouterr()
    assert out == "", f"unexpected output: stdout={out!r} stderr={err!r}"
    assert err == "", f"unexpected output: stdout={out!r} stderr={err!r}"


@pytest.fixture(autouse=True)
def _empty_working_directory(tmp_path_factory):
    with empty_working_directory(tmp_path_factory.mktemp("cwd")):
        yield


@pytest.fixture(autouse=True)
def _figures_closed():
    with figures_closed():
        yield


@pytest.fixture
def seed() -> int:
    return 20261028


@pytest.fixture
def stat_rng(request, record_property):
    entropy = request.config.stash.get(_STAT_ENTROPY, None)
    nodeid = request.node.nodeid.split("[")[0]
    if entropy is not None:
        record_property("stat_seed", str(entropy))
        request.node.stash[_STAT_USED] = entropy

    def make(label: str = "") -> np.random.SeedSequence:
        return np.random.SeedSequence(_seeds.stat_seed(nodeid, label, entropy))

    return make


@pytest.fixture
def uniform_space():
    def make(m: int, seed: int = 0) -> asub.Uniform:
        gen = np.random.default_rng(seed)
        lower = gen.uniform(-10, 10, m)
        widths = 10.0 ** gen.uniform(-2, 2, m)
        return asub.Uniform(lower, lower + widths)

    return make


@pytest.fixture
def gaussian_space():
    def make(m: int, seed: int = 0) -> asub.Gaussian:
        gen = np.random.default_rng(seed)
        B = gen.standard_normal((m, m))
        cov = B @ B.T + m * np.eye(m)
        cov = (cov + cov.T) / 2
        return asub.Gaussian(gen.standard_normal(m), cov)

    return make


@pytest.fixture
def make_provenance():
    def make(function: str = "test", **kwargs) -> asub.Provenance:
        return _results._make_provenance(function, kwargs.pop("parameters", {}), **kwargs)

    return make


def _roundtrip(result, tmp_path) -> None:
    for binary in (False, True):
        d = result.to_dict(binary=binary)
        text = json.dumps(d, allow_nan=False)
        back = asub.from_dict(json.loads(text))
        assert back == result
        assert type(back) is type(result)
        assert back.to_dict(binary=binary)["fields"] == d["fields"]
    path = tmp_path / f"{type(result).TYPE}.npz"
    result.save(path)
    assert asub.load(path) == result
    buf = io.BytesIO()
    result.save(buf)
    buf.seek(0)
    assert asub.load(buf) == result
    assert pickle.loads(pickle.dumps(result)) == result


@pytest.fixture
def assert_roundtrip(tmp_path):
    return lambda result: _roundtrip(result, tmp_path)


def _readonly(result) -> None:
    for f in dataclasses.fields(result):
        value = getattr(result, f.name)
        values = value if isinstance(value, tuple) else (value,)
        for v in values:
            if isinstance(v, np.ndarray):
                assert not v.flags.writeable
                with pytest.raises(ValueError, match="WRITEABLE"):
                    v.flags.writeable = True
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result, dataclasses.fields(result)[0].name, None)


@pytest.fixture
def assert_readonly():
    return _readonly
