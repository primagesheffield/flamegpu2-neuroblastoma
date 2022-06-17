#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name=compile.a100-tmp.sh

# Must compile for the A100 nodes on the A100 nodes for the correct CPU arch
#SBATCH --partition=gpu-a100-tmp
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

# 12 CPU cores (1/4th of the node) and 1 GPUs worth of memory < 1/4th of the node)
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
## Name of file to which standard output will be redirected
#SBATCH --output="build_output.out"
## Name of file to which the standard error stream will be redirected
#SBATCH --error="build_error.err"

## Default pre-job stuff
## srun /bin/hostname
cd $SLURM_SUBMIT_DIR

## Start job

# Use A100 specific module environment
module unuse /usr/local/modulefiles/live/eb/all
module unuse /usr/local/modulefiles/live/noeb
module use /usr/local/modulefiles/staging/eb-znver3/all/

# Load modules from the A100 specific environment
module load GCC/11.2.0
module load CUDA/11.4.1
module load CMake/3.21.1-GCCcore-11.2.0

mkdir -p build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DVISUALISATION=OFF -DSEATBELTS=OFF -DCUDA_ARCH="80"

cmake --build . --target all --parallel 6