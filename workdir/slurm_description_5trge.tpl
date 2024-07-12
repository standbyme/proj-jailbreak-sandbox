#!/bin/sh -l

#SBATCH --mail-user=hongyu@purdue.edu
{%if is_last%}
#SBATCH --mail-type=END
{%endif%}
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH -A standby
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --constraint=A100-80GB
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name={{task}}_{{ slurm_unit_index }}

unset PYTHONPATH

module purge
module load cuda/12.1.1.lua
module load cudnn/cuda-12.1_8.9.lua

unset PYTHONPATH

# guard_name="LlamaGuardPrompt"
guard_name="LlamaGuardPromptResponse"

# target_model_name="Meta-Llama-3-70B-Instruct-AWQ"
# target_model_name="Qwen1.5-72B-Chat-AWQ"
target_model_name="Phi-3-medium-128k-instruct"


/depot/zcelik/data/hongyu/venv/bin/python ../src/core/step_5_target_response_guard_evaluation.py --slurm_unit_index {{ slurm_unit_index }} --generation_name AutoDAN --dataset_name RPAB --target_model_name "$target_model_name" --guard_name "$guard_name"