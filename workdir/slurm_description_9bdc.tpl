#!/bin/sh -l

#SBATCH --mail-user=hongyu.cai@oracle.com
{%if is_last%}
#SBATCH --mail-type=END
{%endif%}
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --output=log/{{task}}_{{ slurm_unit_index }}.out
#SBATCH --time=96:00:00
#SBATCH --gpus-per-node=1
#SBATCH --constraint="shape=BM.GPU.B4.8"
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name={{name}}_{{task}}_{{ slurm_unit_index }}

unset PYTHONPATH

./venv/bin/python ../src/core/step_9_benign_dataset_clean.py --slurm_unit_index {{ slurm_unit_index }} --target_model_name Meta-Llama-3-70B-Instruct-AWQ