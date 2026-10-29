"""The migration guide's name map resolves against the API (spec 18 §8 item 2, spec 15 §5.7)
[D-167].

Every backticked ``ac.`` name in the v1 column of the map tables of ``docs/migration.md``
must resolve on ``active_subspaces`` (importing ``active_subspaces.experimental`` for names
under it). Removing or renaming a public name therefore needs a guide edit in the same PR.
"""

from __future__ import annotations

import importlib
import re
import warnings
from pathlib import Path

import pytest

import active_subspaces as ac
from active_subspaces.errors import ExperimentalWarning

GUIDE = Path(__file__).resolve().parents[2] / "docs" / "migration.md"
_CELL_SPLIT = re.compile(r"(?<!\\)\|")
_CODE = re.compile(r"`([^`]+)`")
_NAME = re.compile(r"\bac\.([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)")


def _tables(text: str) -> list[list[list[str]]]:
    """Markdown tables as lists of rows of cells (header first, separator row dropped)."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in [*text.splitlines(), ""]:
        if line.lstrip().startswith("|"):
            cells = [c.strip() for c in _CELL_SPLIT.split(line.strip().strip("|"))]
            if not all(set(c) <= set("-: ") for c in cells):
                current.append(cells)
        elif current:
            tables.append(current)
            current = []
    return tables


def _v1_names(text: str) -> set[str]:
    names: set[str] = set()
    for table in _tables(text):
        cols = [i for i, h in enumerate(table[0]) if h.lower().startswith("v1")]
        for row in table[1:]:
            for i in cols:
                for span in _CODE.findall(row[i] if i < len(row) else ""):
                    names.update(_NAME.findall(span))
    return names


def _resolve(dotted: str) -> object:
    parts = dotted.split(".")
    if parts[0] == "experimental":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            obj: object = importlib.import_module("active_subspaces.experimental")
        parts = parts[1:]
    else:
        obj = ac
    for part in parts:
        obj = getattr(obj, part)
    return obj


def test_parser_reads_v1_column_only():
    text = (
        "| 2016 | v1 | Notes |\n|---|---|---|\n"
        "| `ac.domains.X` | `ac.active_domain(s)`, `ac.solve_lp(c, A_ub=-A) \\| x` | "
        "`ac.ignored` |\n"
        '\n| 2016 return | v1 field |\n|---|---|\n| `ac.utils` | `ac.subspaces.EVG(gap="log")` |\n'
    )
    assert _v1_names(text) == {"active_domain", "solve_lp", "subspaces.EVG"}


def test_guide_exists_and_has_a_name_map():
    names = _v1_names(GUIDE.read_text(encoding="utf-8"))
    assert len(names) >= 80, sorted(names)  # the map tables were parsed, not skipped


@pytest.mark.parametrize("name", sorted(_v1_names(GUIDE.read_text(encoding="utf-8"))))
def test_v1_name_resolves(name):
    try:
        _resolve(name)
    except AttributeError as exc:
        pytest.fail(f"docs/migration.md maps to ac.{name}, which does not exist: {exc}")
