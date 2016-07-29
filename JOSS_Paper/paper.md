---
title: 'Python Active-Subspaces Utility Library (PAUL): Discovering and Using Active Subspaces in Python'
tags:
  - python
  - active subspaces
  - dimension reduction
  - uncertainty quantification
  - sensitivity analysis
  - surrogate modelling
authors:
 - name: Paul Constantine
   orcid: 0000-0003-3726-6307
   affiliation: Colorado School of Mines
date: 28 July 2016
bibliography: paper.bib
---

# Summary

Active Subspaces is a method of dimension reduction that identifies linear combinations of input parameters whose values 
affect a model's output more, on average, than those of orthogonal linear combinations [@Const2015; @CDW2015]. Reducing dimension in this way can 
enable otherwise infeasible parameter studies for expensive high-dimensional models. The PAUL is a python package that implements 
estimation of active subspaces in the python language. It requires Numpy, Scipy, and Matplotlib (and certain functionality uses Gurobi 
by default but reverts to Scipy if Gurobi is absent). The active subspace can be computed in several different ways, taking either input/output 
pairs or gradients of output with respect to parameters, and can be automatically partitioned according to several heuristics or manually 
by the user.

The PAUL also has various functions and classes for diagnostics and exploitation of the active subspace. The primary diagnostic tools are the 
plotting functions that can make plots of eigenvalues, eigenvectors, subspace errors, and summary plots (graphs of model output against the 
first one or two active variables). An important use of active subspaces is surrogate modelling; the PAUL contains classes for polynomial and 
radial-basis response surfaces (of arbitrary order that can be specified by the user) and functions that can take advantage of these constructions 
for efficient integration and optimization (which could be very difficult with high-dimensional input spaces and expensive models).

# References

