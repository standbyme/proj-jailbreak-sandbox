#!/bin/sh -l

#SBATCH --mail-user=x@x.x

#SBATCH --mail-type=END

#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH -A standby
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name={{task_name}}

unset PYTHONPATH

module purge
module load cuda/12.1.1.lua
module load cudnn/cuda-12.1_8.9.lua

unset PYTHONPATH

# draft_model_name="opt-125m-AWQ"

/depot/anonymousz/data/anonymoush/venv/bin/python /scratch/gilbreth/anonymoush/project/sandbox/proj-jailbreak-sandbox/src/core/step_12_benign_analysis.py --slurm_unit_index "$SLURM_ARRAY_TASK_ID" --draft_model_name opt-125m-AWQ --draft_number {{project_name}}