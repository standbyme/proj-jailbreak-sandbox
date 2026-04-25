#!/usr/bin/env python3
#
# SLURM matrix-submission helper. Run from workdir/.
#
#   python submit.py --task <task_name>
#
# Reads ../src/slurm/<task_name>.yaml for the experimental matrix and
# ../src/slurm/<task_name>.tpl for the SLURM job template, expands the
# Cartesian product of matrix values, renders one sbatch script per
# combination into temp/, and submits each as a SLURM array job of size
# `count`.

import argparse
from configparser import ConfigParser
import itertools
from pathlib import Path
import subprocess
import sys

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined
import yaml


def add_proj_to_PYTHONPATH():
    cwd = Path().cwd()
    assert cwd.name == "workdir"
    sys.path.insert(0, str(cwd.parent))


add_proj_to_PYTHONPATH()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    required=True,
)
args = parser.parse_args()


task_name = args.task

env = Environment(
    loader=FileSystemLoader(searchpath="../src/slurm"), undefined=StrictUndefined
)

template = env.get_template(f"{task_name}.tpl")
project_config = yaml.load(open(f"../src/slurm/project.yaml"), Loader=yaml.FullLoader)
task_config = yaml.load(open(f"../src/slurm/{task_name}.yaml"), Loader=yaml.FullLoader)

project_name = project_config["name"]
count = int(task_config["count"])

matrix = task_config["matrix"]

keys = list(matrix.keys())
combinations = list(itertools.product(*[matrix[key] for key in keys]))
configures = [dict(zip(keys, combination)) for combination in combinations]

# delete temp folder
subprocess.run(["rm", "-rf", "temp"])

# create temp folder
subprocess.run(["mkdir", "temp"])

for i, configure in enumerate(configures):
    print(configure)

    slurm_description_path = f"temp/{i}.sh"
    with open(slurm_description_path, "w") as f:
        data = template.render(
            project_name=project_name,
            task_name=task_name,
            **configure,
        )
        f.write(data)

    subprocess.run(["sbatch", f"--array=0-{count-1}", slurm_description_path])
