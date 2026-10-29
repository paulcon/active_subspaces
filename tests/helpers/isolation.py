"""Per-test isolation used by the autouse fixtures of ``conftest.py`` (spec 15 §5.4).

- :func:`empty_working_directory`: run in a fresh, empty working directory and
  fail if any file appears there [R-02].
- :func:`figures_closed`: close the matplotlib pyplot figures a test created.
  It never imports ``matplotlib.pyplot`` itself (the plotting tests check that
  the package does not import it, spec 13).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def empty_working_directory(directory: Path) -> Iterator[Path]:
    """Change into the empty ``directory``; on exit restore the old one and fail if anything
    was written.
    """
    old = os.getcwd()
    os.chdir(directory)
    try:
        yield directory
    finally:
        os.chdir(old)
    left = sorted(p.name for p in directory.iterdir())
    assert not left, f"files written to the working directory: {left}"


def _open_figures() -> set[int]:
    plt = sys.modules.get("matplotlib.pyplot")
    return set() if plt is None else set(plt.get_fignums())


@contextmanager
def figures_closed() -> Iterator[None]:
    """Close every pyplot figure opened inside the block (figures open before it are left
    alone).
    """
    before = _open_figures()
    try:
        yield
    finally:
        plt = sys.modules.get("matplotlib.pyplot")
        if plt is not None:
            for num in sorted(_open_figures() - before):
                plt.close(num)
