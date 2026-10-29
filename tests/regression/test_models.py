"""Rebuild of the stored reference results (spec 12 §5.7, §8; V12.14)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from active_subspaces import _reference_data

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


@pytest.mark.slow
def test_reference_results_rebuild():
    """Fields are compared bitwise; provenance (timestamp, platform) is that of the generating
    run.
    """
    tool = _tool()
    fresh = tool.compute(_reference_data.N_SAMPLES)
    for name in tool.NAMES:
        assert fresh[name]["fields"] == _reference_data.REFERENCE_RESULTS[name]["fields"], name
