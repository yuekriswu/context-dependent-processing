# Examples
| File/Directory    | Description                                                                             |
|-------------------| ----------------------------------------------------------------------------------------|
| `basic.py`        | Basic components of model fitting                                                       |
| `gaussian.py`     | Example of running and visualizing a model with Gaussian spatial connectivity           |
| `linear_ring/`    | Linear model on a ring. See `linear_ring/README.md` for details                         |
| `SSN_ring/`       | SSN model on a ring. See `SSN_ring/README.md` for details                               |
| `ricciardi_ring/` | Model on a ring with ricciardi nonlinearity. See `ricciardi_ring/README.md` for details |

# Writing configuration files
Some of the examples involve fitting or plotting models with a configuration file. The configuration file may be provided as either a JSON file or a TOML file. Available configuration key-value pairs are in direct correspondence with the arguments of the `run` functions in `src/niarb/cli/*.py`. Special parsing rules are applied to values associated with dunder keys, e.g. `__matrix__`, `__indices__`, `__call__`, etc. For details of how configuration files are parsed, see `src/niarb/parsing.py` and the function `load_config` in `src/niarb/io.py`.

# Differences between fitting a linear, SSN, vs ricciardi model
When fitting a linear model, only connectivity parameters need to be optimized. When fitting an SSN model, one also needs to optimize the baseline voltage. When fitting a ricciardi model, one also needs to optimize both the baseline voltage as well as perturbation strength. For nonlinear models, double precision is needed for highest quality model fitting, although single precision should also work ok.

# Tips
By default, no output is printed when the `niarb` commands are run. You can change that by using the flag `--linfo` which logs useful information to the command line. To get more information for debugging, you may use the flag `--ldebug [module]` for logging debug messages from a particular module (e.g. `--ldebug niarb.cli` logs all debug messages from scripts in the `niarb/cli` directory). You may also want to use the flag `--progress` to display a progress bar.