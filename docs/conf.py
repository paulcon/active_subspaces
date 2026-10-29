"""Sphinx configuration for active_subspaces (spec 15 §5.7, §5.8).

Environment variables:

- ``AS_DOCS_FAST=1`` skips the API reference and docstring validation, for drafting other pages.
- ``AS_DOCS_EXECUTE`` sets ``nb_execution_mode`` (default ``force``: every notebook page runs).
- Notebook kernels get ``src/`` on ``PYTHONPATH`` and the inline matplotlib backend (below).
- ``AS_DOCS_OFFLINE=1`` drops the intersphinx inventories, for builds without network access
  (an unreachable inventory is a warning, so ``-W`` would fail). Links to the Python, NumPy,
  SciPy and matplotlib docs are then plain text.
"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The docs describe and execute the source tree they are built from, installed or not:
# autodoc imports from src/, and the notebook kernels (subprocesses) inherit PYTHONPATH.
SRC = str(ROOT / "src")
sys.path.insert(0, SRC)
os.environ["PYTHONPATH"] = os.pathsep.join(
    [SRC, *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]
)
sys.path.insert(0, str(Path(__file__).parent / "_ext"))

# The library builds figures without pyplot, so a Figure left as a cell's last expression shows
# as text unless IPython has the inline PNG formatter. The kernels start with a throwaway IPython
# profile that enables it ("%matplotlib inline" in every kernel), so pages need no setup cell.
_IPYTHONDIR = Path(tempfile.mkdtemp(prefix="as-docs-ipython-"))
atexit.register(shutil.rmtree, _IPYTHONDIR, ignore_errors=True)
(_IPYTHONDIR / "profile_default").mkdir()
(_IPYTHONDIR / "profile_default" / "ipython_kernel_config.py").write_text(
    'c.IPKernelApp.matplotlib = "inline"\n', encoding="utf-8"
)
os.environ["IPYTHONDIR"] = str(_IPYTHONDIR)

import as_docs  # noqa: E402

import active_subspaces  # noqa: E402
from active_subspaces.errors import ExperimentalWarning  # noqa: E402

# autodoc imports active_subspaces.experimental, which warns once by design.
warnings.filterwarnings("ignore", category=ExperimentalWarning)

FAST = os.environ.get("AS_DOCS_FAST") == "1"
OFFLINE = os.environ.get("AS_DOCS_OFFLINE") == "1"

project = "active_subspaces"
author = "Paul Constantine and the active_subspaces contributors"
copyright = "2014–2026, " + author
release = active_subspaces.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
    "myst_nb",
    "sphinx_copybutton",
    "as_docs",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "_ext",
    "_templates",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
]
if FAST:
    exclude_patterns += ["api/**"]
    suppress_warnings = ["toc.excluded"]  # index.md still lists api/index
as_docs_fast = FAST

# MyST / myst-nb
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3
nb_execution_mode = os.environ.get("AS_DOCS_EXECUTE", "force")
nb_execution_raise_on_error = True
nb_execution_timeout = 300
nb_merge_streams = True
if nb_execution_mode == "off":
    # unexecuted text notebooks carry no language_info, so myst-nb cannot pick a lexer
    suppress_warnings = [*globals().get("suppress_warnings", []), "myst-nb.lexer"]

# autosummary: one page per public module, generated from its __all__ (spec 15 §5.7)
autosummary_generate = True
autosummary_ignore_module_all = False
autosummary_imported_members = False
# BOREHOLE and borehole would write the same file on case-insensitive file systems.
autosummary_filename_map = {} if FAST else as_docs.case_collisions()

# autodoc: the docstrings state types, shapes and coordinates; signatures stay short.
autodoc_typehints = "none"
autodoc_member_order = "groupwise"
autodoc_inherit_docstrings = True

# numpydoc: validate the docstring of every documented public object (spec 15 §5.7, §8).
# Checks left out, and why:
#   GL01  summary on the line after the opening quotes. The project follows PEP 257 and ruff's
#         numpy convention (summary on the opening line), which GL01 contradicts.
#   ES01  extended summary: optional in the spec's section list.
#   SA01  See Also: optional in the spec's section list.
#   EX01  Examples: required by the spec for public functions and classes, but numpydoc would
#         also require it of every method, property and constant; tracked outside the build.
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_validation_checks = set() if FAST else {"all", "GL01", "ES01", "SA01", "EX01"}
# Type aliases have a read-only __doc__; their attribute docstrings are not visible to numpydoc.
numpydoc_validation_exclude = {r"\.ValueFn$", r"\.GradFn$", r"\.ValueAndGradFn$"}

intersphinx_mapping = (
    {}
    if OFFLINE
    else {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable", None),
        "scipy": ("https://docs.scipy.org/doc/scipy", None),
        "matplotlib": ("https://matplotlib.org/stable", None),
    }
)

html_theme = "pydata_sphinx_theme"
html_title = f"active_subspaces {release}"
html_theme_options = {
    "github_url": "https://github.com/paulcon/active_subspaces",
    "navigation_with_keys": False,
}
