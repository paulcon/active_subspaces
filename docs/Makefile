# Build the docs from the repository root's docs/ directory.
#   make html   full build, warnings are errors (what CI and Read the Docs run)
#   make fast   skip the API reference and docstring validation (AS_DOCS_FAST=1)
#   make clean  remove the build and the generated API stubs
SPHINXBUILD ?= python -m sphinx
SPHINXOPTS  ?= -W --keep-going
BUILDDIR    ?= _build

.PHONY: html fast clean

html:
	MPLBACKEND=agg $(SPHINXBUILD) -b html $(SPHINXOPTS) -d $(BUILDDIR)/doctrees . $(BUILDDIR)/html

fast:
	AS_DOCS_FAST=1 MPLBACKEND=agg $(SPHINXBUILD) -b html $(SPHINXOPTS) -d $(BUILDDIR)/doctrees-fast . $(BUILDDIR)/html-fast

clean:
	rm -rf $(BUILDDIR) api/generated
