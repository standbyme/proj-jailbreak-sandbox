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
module load gcc/12.3.0.lua
module load openmpi/4.1.5-gpu-cuda12.lua
module load cuda/12.1.1.lua
module load cudnn/cuda-12.1_8.9.lua

unset PYTHONPATH

./venv/bin/python ../src/core/step_6_draft_response_guard_evaluation.py --slurm_unit_index {{ slurm_unit_index }} --generation_name {{ generation_name }} --dataset_name {{ dataset_name}} --target_model_name {{ target_model_name }} --draft_number 5 --draft_model_name {{ draft_model_name }}

./venv/bin/python /scratch/gilbreth/hongyu/project/sandbox/proj-jailbreak-sandbox/src/core/step_6_draft_response_guard_evaluation.py --slurm_unit_index {{ slurm_unit_index }} --generation_name {{ generation_name }} --dataset_name {{ dataset_name}} --target_model_name {{ target_model_name }} --draft_number 10 --draft_model_name {{ draft_model_name }}

./venv/bin/python /scratch/gilbreth/hongyu/project/sandbox/proj-jailbreak-sandbox/src/core/step_6_draft_response_guard_evaluation.py --slurm_unit_index {{ slurm_unit_index }} --generation_name {{ generation_name }} --dataset_name {{ dataset_name}} --target_model_name {{ target_model_name }} --draft_number 15 --draft_model_name {{ draft_model_name }}

./venv/bin/python /scratch/gilbreth/hongyu/project/sandbox/proj-jailbreak-sandbox/src/core/step_6_draft_response_guard_evaluation.py --slurm_unit_index {{ slurm_unit_index }} --generation_name {{ generation_name }} --dataset_name {{ dataset_name}} --target_model_name {{ target_model_name }} --draft_number 20 --draft_model_name {{ draft_model_name }}

./venv/bin/python /scratch/gilbreth/hongyu/project/sandbox/proj-jailbreak-sandbox/src/core/step_6_draft_response_guard_evaluation.py --slurm_unit_index {{ slurm_unit_index }} --generation_name {{ generation_name }} --dataset_name {{ dataset_name}} --target_model_name {{ target_model_name }} --draft_number 25 --draft_model_name {{ draft_model_name }}

./venv/bin/python /scratch/gilbreth/hongyu/project/sandbox/proj-jailbreak-sandbox/src/core/step_6_draft_response_guard_evaluation.py --slurm_unit_index {{ slurm_unit_index }} --generation_name {{ generation_name }} --dataset_name {{ dataset_name}} --target_model_name {{ target_model_name }} --draft_number 30 --draft_model_name {{ draft_model_name }}

./venv/bin/python /scratch/gilbreth/hongyu/project/sandbox/proj-jailbreak-sandbox/src/core/step_6_draft_response_guard_evaluation.py --slurm_unit_index {{ slurm_unit_index }} --generation_name {{ generation_name }} --dataset_name {{ dataset_name}} --target_model_name {{ target_model_name }} --draft_number 35 --draft_model_name {{ draft_model_name }}