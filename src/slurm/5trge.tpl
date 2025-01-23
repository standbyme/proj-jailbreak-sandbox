#!/bin/sh -l

#SBATCH --mail-user=x@x.x
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH -A standby
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --constraint=A100-80GB
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name={{task_name}}

unset PYTHONPATH

module purge
module load gcc/12.3.0.lua
module load openmpi/4.1.5-gpu-cuda12.lua
module load cuda/12.1.1.lua
module load cudnn/cuda-12.1_8.9.lua

unset PYTHONPATH

./venv/bin/python ../src/core/step_5_target_response_guard_evaluation.py --slurm_unit_index "$SLURM_ARRAY_TASK_ID" --generation_name {{ generation_name }} --dataset_name {{ dataset_name }} --target_model_name {{ target_model_name }} --guard_name {{ guard_name }}