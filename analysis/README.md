# FiTree Analysis

This folder contains the analysis code to reproduce the results presented in the paper ["Bayesian inference of fitness landscapes via tree-structured branching processes"](https://doi.org/10.1101/2025.01.24.634649).

## Overview

The analysis code is organized to facilitate reproduction of the computational results and figures from the paper. 

## Structure

The analysis folder is organized into two main subfolders:

### 1. simulations

This subfolder contains data and scripts related to the simulation studies in the paper. Note that this does not contain the full workflows to generate the simulated data.

### 2. AML

This subfolder contains data, pre-processing code, and plotting scripts for the Acute Myeloid Leukemia (AML) analysis presented in the paper. This includes data visualization code but not the complete inference pipeline.

## Main Workflows

The actual workflows for generating simulated data and inferring fitness landscapes are located in the workflows folder under the main package directory:

[FiTree/workflows](../workflows)

These workflows implement the complete pipeline for both simulation studies and real data analysis described in the paper.

## Data Availability

All simulated data and raw results used in this analysis are available on Zenodo:
[https://doi.org/10.1101/2025.01.24.634649](https://doi.org/10.1101/2025.01.24.634649)

