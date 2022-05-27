#!/bin/bash -l
## Job name
#SBATCH -J test_run_fgpu2nb
## Number of allocated nodes
#SBATCH -N 1
## Number of tasks per node (by default this corresponds to the number of cores allocated per node)
#SBATCH --ntasks-per-node=1
## Memory allocated per core (default is 5GB)
#SBATCH --mem-per-cpu=5GB
## Max task execution time (format is HH:MM:SS)
#SBATCH --time=00:10:00
## Name of grant to which resource usage will be charged (primage1 or primage1gpu)
#SBATCH -A plgprimage3
## Name of partition (plgrid-testing or plgrid-gpu)
#SBATCH -p plgrid-gpu-v100
## Specify that we want GPUs (if using plgrid-gpu)
#SBATCH --gres=gpu:1
## Name of file to which standard output will be redirected
#SBATCH --output="test_run_output.out"
## Name of file to which the standard error stream will be redirected
#SBATCH --error="test_run_error.err"

## Default pre-job stuff
## srun /bin/hostname
cd $SLURM_SUBMIT_DIR

## Start job

#module load plgrid/apps/cuda/11.2
module load plgrid/tools/gcc/8.2.0

nvidia-smi

./build/bin/Release/orchestrator_FGPUNB --in inputs/in-2022-03-29.json --primage inputs/out-2022-03-29.json