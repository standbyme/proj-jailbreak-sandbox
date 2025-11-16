#!/bin/sh -l

#SBATCH --mail-user=hongyu@purdue.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH -A antoniob
#SBATCH --time=14-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=a100-80gb
#SBATCH --job-name={{task_name}}-{{generation_name}}

unset PYTHONPATH

module purge
module load cuda/12.6.0.lua

unset PYTHONPATH

uv run ../src/core/step_1_jailbreak_generation.py --slurm_unit_index "$SLURM_ARRAY_TASK_ID" --target_model_name {{target_model_name}} --generation_name {{generation_name}} --dataset_name {{dataset_name}}