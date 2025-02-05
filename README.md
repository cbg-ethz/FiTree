# FiTree

FiTree is a Python package for Bayesian inference of fitness landscapes via tree-structured branching processes.

## Installation

```
pip install fitree
```

## Getting started

FiTree takes tumor mutation trees as input and learns a matrix representing the fitness effects of individual mutations as well as their pairwise interactions. We provide small examples on how to use FiTree:

1. [Pre-processing of tree input](demo/AML/process_trees.ipynb)

2. [Tree generation and inference](demo/simulations/simulation.ipynb)

For large-scale simulation studies and real data application, we recommend looking into the [snakemake workflows](workflows).


## Preprint

The preprint of the paper is provided [here](https://www.biorxiv.org/content/10.1101/2025.01.24.634649v1).

