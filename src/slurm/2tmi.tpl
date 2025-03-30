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
module load gcc/11.5.0.lua
module load openmpi/5.0.5.lua
module load cuda/12.6.0.lua
module load cudnn/9.2.0.82-12.lua

unset PYTHONPATH

./venv/bin/python ../src/core/step_2_target_model_inference.py --slurm_unit_index "$SLURM_ARRAY_TASK_ID" --generation_name {{generation_name}} --dataset_name {{dataset_name}} --target_model_name {{target_model_name}}