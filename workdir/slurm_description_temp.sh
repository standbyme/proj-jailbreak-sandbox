#!/bin/sh -l

#SBATCH --mail-user=hongyu@purdue.edu
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH -A standby
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH --mem=32G

unset PYTHONPATH

module purge
module load cuda/12.1.1.lua
module load cudnn/cuda-12.1_8.9.lua

unset PYTHONPATH

/depot/zcelik/data/hongyu/venv/bin/python /scratch/gilbreth/hongyu/project/sandbox/proj-jailbreak-sandbox/workdir/temp_copy.py