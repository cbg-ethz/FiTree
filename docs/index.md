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

1. [Pre-processing of tree input](https://github.com/cbg-ethz/FiTree/blob/main/analysis/AML/script/process_trees.ipynb)

2. [Tree generation and inference](https://github.com/cbg-ethz/FiTree/blob/main/analysis/simulations/script/simulation_demo.ipynb)

For large-scale simulation studies and real data application, we recommend looking into the [snakemake workflows](https://github.com/cbg-ethz/FiTree/tree/main/workflows).


## Preprint

The preprint of the paper is provided [here](https://www.biorxiv.org/content/10.1101/2025.01.24.634649v1). We provide the data and script to reproduce the figures in the paper [here](https://github.com/cbg-ethz/FiTree/tree/main/analysis).

