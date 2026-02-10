![tests workflow status](https://github.com/hchau630/niarb/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/hchau630/niarb/graph/badge.svg?token=5F1D4MSYGH)](https://codecov.io/gh/hchau630/niarb)

# About
Package for fitting firing rate models of the cortex, in particular for mouse V1.

# Install
After cloning/forking this repository, create a new conda environment with python version >=3.11, then do
`pip install -r requirements.txt`.

# Usage
This package comes with a command line interface (CLI) for fitting models and plotting them with only a configuration file. To fit models, simply write a configuration file, e.g. `fit.toml` (the configuration can be written in JSON or TOML), then do `niarb fit fit.toml -o fits` to fit models according to the specificaation of your configuration file and outputs the results to a directory called `fits`. Similarly, you can use the command `niarb plot {YOUR_CONFIG}` to create various plots, such as distribution of fitted model parameters and the perturbation response of fitted models. For more details on how to write the configuration files, please refer to the examples located in the directory `examples/`.

While there are tons of configuration options available, its primary aim is to provide a quick and simple way to jumpstart your own project rather than cover all possible use cases. For more advanced use cases, it is recommended to use the command line interface code under `src/niarb/cli` as a starting template for your own code.

# Testing
To run tests, simply run the command `pytest` in the project root directory.

# Benchmark
This package is benchmarked with the `asv` package, with the benchmarks located at `benchmarks/`. Benchmark results can be viewed at https://hchau630.github.io/niarb.