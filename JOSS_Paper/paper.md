---
title: 'Python Active-subspaces Utility Library'
tags:
  - python
  - active subspaces
  - dimension reduction
  - uncertainty quantification
  - sensitivity analysis
  - surrogate modeling
authors:
 - name: Paul Constantine
   orcid: 0000-0003-3726-6307
   affiliation: 1
 - name: Ryan Howard
   affiliation: 1
 - name: Andrew Glaws
   affiliation: 1
 - name: Zachary Grey
   affiliation: 1
 - name: Paul Diaz
   affiliation: 2
 - name: Leslie Fletcher
affiliations:
 - name: Colorado School of Mines, Golden, CO
   index: 1
 - name: University of Colorado Boulder, Boulder, CO
   index: 2
date: 29 September 2016
bibliography: paper.bib
repository: https://github.com/paulcon/active_subspaces
archive_doi: https://doi.org/10.5281/zenodo.158941
---

# Summary

Active subspaces are part of an emerging set of tools for discovering and exploiting low-dimensional structure in a function of several variables [@Const2015; @CDW2015]. The active subspace for a given function is the span of a set of important directions in the function's domain, where importance is defined by the eigenvalues of a symmetric, positive semidefinite matrix derived from the function's partial derivatives. Perturbing the inputs within the active subspace changes the output more, on average, than perturbing the inputs orthogonally to the active subspace. The functions of interest arise in complex computer simulation models in science and engineering, where the inputs are the model's physical parameters and the output is the scientific or engineering quantity of interest. Identifying an active subspace in a given model enables one to reduce the input dimension for essential parameter studies---such as optimization or uncertainty quantification. When the simulation model is computationally expensive, this dimension reduction may enable otherwise infeasible parameter studies.

The Python Active-subspaces Utility Library [@PAUL] contains Python utilities for working with active subspaces in a given model. Given either (i) a function handle to the model's output and gradients as a function of the model's inputs or (ii) a set of previously generated input/output pairs, the Utility Library provides several methods for estimating the active subspace---along with several diagnostics for the quality of the estimated active subspace. With an active subspace in hand, the Utility Library contains tools for (i) constructing a polynomial or Gaussian radial basis response surface, (ii) estimating the minimum value of the function, and (iii) estimating the average of the function---all of which exploit the low-dimensional active subspace for increased efficiency. The Library also contains several convenience functions for plotting that may reveal simple relationships between the model's active variables (linear combinations of the original variables) and its output. 

# References

