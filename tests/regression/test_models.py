"""Rebuild of the stored reference results (spec 12 §5.7, §8; V12.14)."""

from __future__ import annotations

import importlib.util
import platform
from pathlib import Path

import numpy as np
import pytest
import scipy

from active_subspaces import _reference_data
from helpers.tolerances import TIERS, U, assert_close, assert_eigenvalues_close

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "make_reference_results.py"


def _tool():
    if not _TOOL.exists():
        pytest.skip("tools/make_reference_results.py not available")
    spec = importlib.util.spec_from_file_location("make_reference_results", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_module_renders_itself():
    tool = _tool()
    text = Path(_reference_data.__file__).read_text(encoding="utf-8")
    assert tool.render(_reference_data.REFERENCE_RESULTS, _reference_data.N_SAMPLES) == text


def _array(field):
    return np.asarray(field["data"], dtype=np.float64).reshape(field["shape"])


def _same_build(prov):
    """Bitwise agreement is promised only on the generating platform and versions [D-15]."""
    return (
        prov["platform"] == platform.platform()
        and prov["python_version"] == platform.python_version()
        and prov["numpy_version"] == np.__version__
        and prov["scipy_version"] == scipy.__version__
    )


def _assert_fields_close(fresh, stored, name):
    """Elsewhere at the spec 17 tiers: eigenvalues and scalars at F10, standard errors at F8,
    and each eigenvector whose eigenvalue gap bounds its rounding error by F8 (spec 17 §3.1).
    """
    for key in ("model", "qoi", "exact", "n_samples", "seed"):
        assert fresh[key] == stored[key], (name, key)
    lam = _array(stored["eigenvalues"])
    assert_eigenvalues_close(_array(fresh["eigenvalues"]), lam, rtol=TIERS["F10"])
    se = _array(stored["eigenvalue_se"])
    assert_close(_array(fresh["eigenvalue_se"]), se, rtol=TIERS["F8"], err_msg=name)
    for key in ("mean", "mean_se", "variance"):
        assert_close(fresh[key], stored[key], rtol=TIERS["F10"], err_msg=f"{name} {key}")
    W, W0 = _array(fresh["eigenvectors"]), _array(stored["eigenvectors"])
    m = lam.size
    for k in range(m):
        gap = np.min(np.abs(np.delete(lam, k) - lam[k]))
        tol = 100.0 * m * U * np.abs(lam[0]) / gap if gap > 0 else np.inf
        if tol <= TIERS["F8"]:
            assert_close(W[:, k], W0[:, k], rtol=0.0, atol=tol, err_msg=f"{name} w_{k + 1}")


@pytest.mark.slow
def test_reference_results_rebuild():
    """Bitwise on the generating build; at tier tolerance elsewhere (spec 17 §3.2). Provenance
    (timestamp, platform) is that of the generating run.
    """
    tool = _tool()
    fresh = tool.compute(_reference_data.N_SAMPLES)
    for name in tool.NAMES:
        stored = _reference_data.REFERENCE_RESULTS[name]
        if _same_build(stored["provenance"]):
            assert fresh[name]["fields"] == stored["fields"], name
        else:
            _assert_fields_close(fresh[name]["fields"], stored["fields"], name)
