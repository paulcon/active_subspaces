"""Sphinx helpers for the active_subspaces docs (spec 15 §5.7).

Everything here is generated from the package at build time, so the pages cannot drift:

- ``as-codes-table``: the error and warning code table, from ``active_subspaces._codes.CODES``.
- ``as-toplevel-table``: every top-level name of ``active_subspaces.__all__``, linked to the
  module page that documents it.
- A build check that ``api/index.md`` lists exactly the public modules: the modules named in
  ``active_subspaces.__all__`` plus ``active_subspaces.experimental``.

Problems are reported as Sphinx warnings, so ``sphinx-build -W`` fails on them.
"""

from __future__ import annotations

import importlib
import inspect
import re
import warnings
from pathlib import Path
from typing import Any

from docutils import nodes
from sphinx import addnodes
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

LOGGER = logging.getLogger(__name__)

PACKAGE = "active_subspaces"
# Imported explicitly by users, never through the package namespace (spec 15 §4.3).
EXPLICIT_MODULES = ("active_subspaces.experimental",)
# Internal design IDs (D-08, N-10, ...) belong to the design records, not the user pages.
_IDS = re.compile(r"\s*\[(?:[ACDNPR]-\d+[a-z]?(?:,\s*)?)+\]")


def _import(name: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # experimental warns on import; nothing to report here
        return importlib.import_module(name)


def public_modules() -> list[str]:
    """Full names of the public modules: modules in ``__all__``, then the explicit ones."""
    pkg = _import(PACKAGE)
    names = [f"{PACKAGE}.{n}" for n in pkg.__all__ if inspect.ismodule(getattr(pkg, n, None))]
    return [*sorted(names), *EXPLICIT_MODULES]


def case_collisions() -> dict[str, str]:
    """``autosummary_filename_map`` entries for names equal up to case in one module.

    ``BOREHOLE`` and ``borehole`` would write the same file on case-insensitive file
    systems (macOS, Windows); the upper-case name gets a suffix.
    """
    mapping: dict[str, str] = {}
    for modname in public_modules():
        names = list(getattr(_import(modname), "__all__", ()))
        lower = [n.lower() for n in names]
        for n in names:
            if lower.count(n.lower()) > 1 and n != n.lower():
                mapping[f"{modname}.{n}"] = f"{modname}.{n}-data"
    return mapping


def _table(header: list[str], rows: list[list[nodes.Node]], widths: list[int]) -> nodes.table:
    table = nodes.table(classes=["as-generated-table"])
    group = nodes.tgroup(cols=len(header))
    table += group
    for w in widths:
        group += nodes.colspec(colwidth=w)
    head = nodes.thead()
    group += head
    row = nodes.row()
    for h in header:
        row += nodes.entry("", nodes.paragraph(text=h))
    head += row
    body = nodes.tbody()
    group += body
    for cells in rows:
        row = nodes.row()
        for cell in cells:
            row += nodes.entry("", cell)
        body += row
    return table


class _GeneratedTable(SphinxDirective):
    has_content = False

    def _xref(self, role: str, target: str, short: bool = True) -> list[nodes.Node]:
        """A Python-domain cross-reference, independent of the page's markup language."""
        text = target.rsplit(".", 1)[-1] if short else target
        ref = addnodes.pending_xref(
            "",
            nodes.literal(text, text, classes=["xref", "py", f"py-{role}"]),
            refdomain="py",
            reftype=role,
            reftarget=target,
            refexplicit=True,
            refwarn=not self.config.as_docs_fast,  # the API pages are not built then
        )
        ref["py:module"] = None
        ref["py:class"] = None
        return [ref]

    def _cell(self, *children: nodes.Node) -> nodes.paragraph:
        par = nodes.paragraph()
        par.extend(children)
        return par


class CodesTable(_GeneratedTable):
    """The table of error and warning codes, generated from ``_codes.CODES``."""

    def run(self) -> list[nodes.Node]:
        codes = _import(f"{PACKAGE}._codes").CODES
        errors = _import(f"{PACKAGE}.errors")
        rows: list[list[nodes.Node]] = []
        used: set[str] = set()
        for code, (class_name, meaning) in codes.items():
            if class_name not in errors.__all__:
                LOGGER.warning(
                    "code %r names %r, which is not in active_subspaces.errors.__all__",
                    code,
                    class_name,
                    location=self.get_location(),
                )
            used.add(class_name)
            rows.append(
                [
                    self._cell(nodes.literal(text=code)),
                    self._cell(*self._xref("class", f"{PACKAGE}.errors.{class_name}")),
                    self._cell(nodes.Text(_IDS.sub("", meaning))),
                ]
            )
        out: list[nodes.Node] = [_table(["Code", "Class", "Meaning"], rows, [30, 20, 50])]
        bases = {"ActiveSubspacesError", "ActiveSubspacesWarning"}
        unused = [c for c in errors.__all__ if c not in used and c not in bases]
        if unused:
            par = nodes.paragraph(text="Classes with no codes in this release: ")
            for i, name in enumerate(unused):
                if i:
                    par += nodes.Text(", ")
                par.extend(self._xref("class", f"{PACKAGE}.errors.{name}"))
            par += nodes.Text(".")
            out.append(par)
        return out


class TopLevelTable(_GeneratedTable):
    """Every non-module name of ``active_subspaces.__all__`` and where it is documented."""

    def run(self) -> list[nodes.Node]:
        pkg = _import(PACKAGE)
        modules = [m for m in public_modules() if m not in EXPLICIT_MODULES]
        rows: list[list[nodes.Node]] = []
        for name in pkg.__all__:
            obj = getattr(pkg, name)
            if inspect.ismodule(obj) or name == "__version__":
                continue
            home = None
            # prefer the defining module (ac.predict is surrogates.predict, re-exported by
            # reduced_model); otherwise the first public module that lists the same object
            defining = getattr(obj, "__module__", None)
            for modname in sorted(modules, key=lambda m: m != defining):
                mod = _import(modname)
                if name in getattr(mod, "__all__", ()) and getattr(mod, name) is obj:
                    home = modname
                    break
            if home is None:
                LOGGER.warning(
                    "active_subspaces.%s is in __all__ but in no public module's __all__",
                    name,
                    location=self.get_location(),
                )
                continue
            rows.append(
                [
                    self._cell(nodes.literal(text=f"ac.{name}")),
                    self._cell(*self._xref("obj", f"{home}.{name}", short=False)),
                ]
            )
        return [_table(["Name", "Documented as"], rows, [40, 60])]


_AUTOSUMMARY_ENTRY = re.compile(r"^\s{0,8}(active_subspaces(?:\.\w+)*)\s*$", re.MULTILINE)


def check_api_index(app: Any) -> None:
    """Warn unless ``api/index.md`` autosummarizes exactly the public modules."""
    if app.config.as_docs_fast:
        return
    index = Path(app.srcdir) / "api" / "index.md"
    listed = _AUTOSUMMARY_ENTRY.findall(index.read_text(encoding="utf-8"))
    expected = public_modules()
    missing = sorted(set(expected) - set(listed))
    extra = sorted(set(listed) - set(expected))
    if missing or extra:
        LOGGER.warning(
            "api/index.md does not match the public modules: missing %s, not public %s",
            missing,
            extra,
            location="api/index",
        )


def drop_inherited_data_docstring(
    app: Any, what: str, name: str, obj: Any, options: Any, lines: list[str]
) -> None:
    """Show no docstring, rather than its type's, for a constant without its own, and warn.

    autodoc falls back to ``type(obj).__doc__`` for module data, so ``ESTIMATORS`` would read
    "Read-only proxy of a mapping.". numpydoc does not report GL08 for data, so this does.
    """
    if what not in ("data", "attribute"):
        return
    own = type(obj).__doc__
    if lines and isinstance(own, str) and own.strip():
        if inspect.cleandoc("\n".join(lines)) == inspect.cleandoc(own):
            lines[:] = []
    if not any(line.strip() for line in lines) and app.config.numpydoc_validation_checks:
        LOGGER.warning(
            "[numpydoc] Validation warnings while processing docstring for %r:\n"
            "  GL08: The object does not have a docstring (add a #: comment)\n",
            name,
        )


def public_bases(app: Any, name: str, obj: Any, options: Any, bases: list[Any]) -> None:
    """Replace private base classes (``_EstimatorBase``) by their public ancestors."""
    out: list[Any] = []

    def add(base: Any) -> None:
        if getattr(base, "__name__", "").startswith("_"):
            for b in getattr(base, "__bases__", ()):
                add(b)
        elif base not in out:
            out.append(base)

    for base in list(bases):
        add(base)
    bases[:] = out


def setup(app: Any) -> dict[str, Any]:
    app.add_config_value("as_docs_fast", False, "env")
    app.add_directive("as-codes-table", CodesTable)
    app.add_directive("as-toplevel-table", TopLevelTable)
    app.connect("builder-inited", check_api_index)
    # before numpydoc (priority 500), so that its validation sees the cleared docstring
    app.connect("autodoc-process-docstring", drop_inherited_data_docstring, priority=400)
    app.connect("autodoc-process-bases", public_bases)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
