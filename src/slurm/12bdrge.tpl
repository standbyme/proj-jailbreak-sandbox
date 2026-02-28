#!/bin/sh -l

#SBATCH --mail-user=anonymoush@anonymousu.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH -A standby
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --constraint=A100
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name={{task_name}}

unset PYTHONPATH

module purge
module load gcc/11.5.0.lua
module load openmpi/5.0.5.lua
module load cuda/12.6.0.lua
module load cudnn/9.2.0.82-12.lua

unset PYTHONPATH


./venv/bin/python ../src/core/step_12_benign_draft_response_guard_evaluation.py --slurm_unit_index "$SLURM_ARRAY_TASK_ID" --draft_model_name {{ draft_model_name }} --draft_number {{ draft_number }}