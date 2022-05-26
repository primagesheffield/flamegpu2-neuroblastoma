# PRIMAGE USFD Agent Based Neuroblastoma model

## Prerequisites

The dependencies below are required for building the project, it's advised to that you use the newest version of any dependencies.

* [CMake](https://cmake.org/) >= 3.18
  * CMake 3.16 is known to have issues on certain platforms
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) >= 11.0
* [git](https://git-scm.com/): Required by CMake for downloading dependencies
* C++17 capable C++ compiler (host), compatible with the installed CUDA version
  * [make](https://www.gnu.org/software/make/) and [GCC](https://gcc.gnu.org/) `>= 8.1` (Linux)


## Building

It is necessary to successfully build the model, before it can be executed.

This branch of the repository contains build scripts setup for PLGrid's Prometheus & Ares clusters.

These jobs can be submitted from the root of the repository using `sbatch scripts/build_on_prometheus.sh` or `sbatch scripts/build_on_ares.sh`.

If the job submission fails, it's likely the grant specified in the job submission script will need to be updated (these rotate every 6 months?).
Similarly, Ares is currently in beta and lacks a testing partition for short build jobs. It may speed up jobs if this is updated when Ares leaves beta.

The build scripts will create `build_output.out` and `build_error.err`, containing the `stdout` and `stderr` from the job respectively.

On success, you should find the USFD orchestrator model exists at `build/bin/Release/orchestrator_FGPUNB`.

*Note: Currently this branch is not pinned to a particular release of FLAMEGPU2, instead it follows origin/master, rebuilds may therefore update FLAMEGPU2.*

## Running

This branch of the repository contains test run scripts setup for PLGrid's Prometheus & Ares clusters. These simply pass an input file to the model, to ensure it executes to completion and outputs a correctly formatted output file (no validation of the contents of the output file is carried out).
These jobs can be submitted from the root of the repository using `sbatch scripts/test_run_on_prometheus.sh` or `sbatch scripts/test_run_on_ares.sh`.


*Note: Prometheus has multiple partitions with GPUs `plgrid-gpu` and `plgrid-gpu-v100`. The prior has K80 GPUs, these are older, hence less demand in the job queue but will execute the model more slowly than the latter which has V100s.*


### Running Parameter Sweeps

This branch of the repository contains a python script which can be used to perform parameter sweeps reading inputs from `.csv` and writing them back out to `.csv`.

The python script executes the model via the Orchestrator interface by creating temporary orchestrator input files (and parsing the output).

In an interactive session the batch script can be executed from the root using `python3 scripts/batch_run.py <input csv> <number of GPUs available>`, this will produce an output file with `_out` appended the input file's name.

To execute this as a batch job, a sample job submission scripts and matching input file are available which can be executed using `sbatch scripts/test_batch_on_prometheus.sh` or `sbatch scripts/test_batch_on_ares.sh`.

*Note: You may wish to adjust the number of GPUs and CPU cores in the job script. Only 1 CPU core per GPU is strictly required, however more may provide slight improvements to performance.*