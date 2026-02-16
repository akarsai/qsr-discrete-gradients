# Code for the paper *"A discrete gradient scheme for preserving QSR-dissipativity"*

This repository contains the code for the numerical experiments in the paper.

## Reproducing the results

1. Install [`uv`](https://github.com/astral-sh/uv) by following the [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone the repository and switch folder: 
```shell
git clone https://github.com/akarsai/qsr-discrete-gradients.git && cd qsr-discrete-gradients
``` 
3. Run experiments (the plots will be placed in the `results/figures` directory):
```shell
uv run python plots/dg_qsr.py
```



## Some hints
- Throughout the codebase, the time index is always at position `0`. The state `z` thus is stored in an array with shape `z.shape == (number_of_timepoints, dimension)`.
- Since the implementation uses the algorithmic differentiation capabilities of JAX, the implementations of all functions need to be written in a JAX-compatible fashion. The provided examples should be a good starting point.
- In case of questions, feel free to reach out.