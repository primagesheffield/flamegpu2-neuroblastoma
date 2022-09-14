#!/bin/bash -l
## Job name
#SBATCH -J build_fgpu2_nb
## Number of allocated nodes
#SBATCH -N 1
## Number of tasks per node (by default this corresponds to the number of cores allocated per node)
#SBATCH --ntasks-per-node=4
## Memory allocated per core (default is 5GB)
#SBATCH --mem-per-cpu=5GB
## Max task execution time (format is HH:MM:SS)
#SBATCH --time=00:30:00
## Name of grant to which resource usage will be charged (primage1 or primage1gpu)
#SBATCH -A plgprimage4-gpu-a100
## Name of partition (plgrid-testing or plgrid-gpu)
#SBATCH -p plgrid-gpu-a100
## Specify that we want GPUs (if using plgrid-gpu)
##SBATCH --gres=gpu:0
## Name of file to which standard output will be redirected
#SBATCH --output="build_output.out"
## Name of file to which the standard error stream will be redirected
#SBATCH --error="build_error.err"

## Default pre-job stuff
## srun /bin/hostname
cd $SLURM_SUBMIT_DIR

## Start job

module load GCCcore/11.3.0
module load CMake/3.23.1
module load CUDA/11.7.0

HOME_PATH=`pwd`

mkdir -p build

cmake -DCMAKE_BUILD_TYPE=Release -DVISUALISATION=OFF -DSEATBELTS=OFF -DCUDA_ARCH="80" -S $HOME_PATH/.. -B $HOME_PATH/build

cd build

cmake --build . --target all --parallel 4
