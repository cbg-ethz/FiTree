![PyPI](https://img.shields.io/pypi/v/fitree)
![Build Status](https://img.shields.io/github/actions/workflow/status/cbg-ethz/FiTree/test.yaml)
![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Repo Status](https://img.shields.io/badge/status-active-brightgreen)


# FiTree

FiTree is a Python package for Bayesian inference of fitness landscapes via tree-structured branching processes.

## Installation

```
pip install fitree
```

## Getting started

FiTree takes tumor mutation trees as input and learns a matrix representing the fitness effects of individual mutations as well as their pairwise interactions. We provide small examples on how to use FiTree:

1. [Pre-processing of tree input](analysis/AML/script/process_trees.ipynb)

2. [Tree generation and inference](analysis/simulations/script/simulation.ipynb)

For large-scale simulation studies and real data application, we recommend looking into the [snakemake workflows](workflows).


## Preprint

Our paper is available as a [preprint on bioRxiv](https://www.biorxiv.org/content/10.1101/2025.01.24.634649v1). All data and code necessary to reproduce the figures and analyses presented in the paper can be found in the [analysis folder](analysis).

