#!/bin/bash -l
## Job name
#SBATCH -J test_run_fgpu2nb
## Number of allocated nodes
#SBATCH -N 1
## Number of tasks per node (by default this corresponds to the number of cores allocated per node)
#SBATCH --ntasks-per-node=4
## Memory allocated per core (default is 5GB)
#SBATCH --mem-per-cpu=5GB
## Max task execution time (format is HH:MM:SS)
#SBATCH --time=00:30:00
## Name of grant to which resource usage will be charged (primage1 or primage1gpu)
#SBATCH -A plgprimage4-gpu
## Name of partition (plgrid-testing or plgrid-gpu)
#SBATCH -p plgrid-gpu-v100
## Specify that we want GPUs (if using plgrid-gpu)
#SBATCH --gres=gpu:2
## Name of file to which standard output will be redirected
#SBATCH --output="test_batch_output.out"
## Name of file to which the standard error stream will be redirected
#SBATCH --error="test_batch_error.err"

## Default pre-job stuff
## srun /bin/hostname
cd $SLURM_SUBMIT_DIR

## Start job

module load cmake/3.20.1-gcccore-10.3.0
module load cudacore/11.2.2

nvidia-smi

python3 scripts/batch_run.py scripts/test_batch_in.csv $SLURM_GPUS_ON_NODE