#!/bin/sh -l

#SBATCH --mail-user=hongyu@purdue.edu

#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH -A antoniob
#SBATCH --qos=standby
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=a100-80gb
#SBATCH --job-name={{task_name}}

unset PYTHONPATH

module purge
module load cuda/12.6.0.lua

unset PYTHONPATH

uv run ../src/core/step_4_target_response_ground_truth_guard_evaluation.py --slurm_unit_index "$SLURM_ARRAY_TASK_ID" --generation_name {{ generation_name }} --dataset_name {{ dataset_name }} --target_model_name {{ target_model_name }}