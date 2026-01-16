# Minimal makefile for Sphinx documentation at the top level,
# this time supporting the "make mode" option for sphinx-build

# You can set these variables from the command line.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" stops in case first argument is an error.
# If the first argument is a quote-closed string, all spaces within
# the string become single spaces.

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the "make mode" option.
# define print_help
# $(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
# endef
# help:
# 	@echo "Please use \`make <target>' where <target> is one of the following:"
# 	@$(print_help)

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)"

latexpdf:
	$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS)

dirhtml:
	$(SPHINXBUILD) -b dirhtml "$(SOURCEDIR)" "$(BUILDDIR)/dirhtml" $(SPHINXOPTS)
