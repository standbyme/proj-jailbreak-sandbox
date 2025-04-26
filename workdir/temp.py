import subprocess
import itertools

from tqdm import tqdm
from pathlib import Path

arg_matrix = {
    "generation_name": ["AutoDAN", "GCG"],
    "dataset_name": ["RPAB"],
    "target_model_name": [
        "Meta-Llama-3-70B-Instruct-AWQ",
        "Qwen1.5-72B-Chat-AWQ",
        "Phi-3-medium-128k-instruct",
    ],
    "draft_model_name": [
        "Qwen2.5-0.5B",
        "Llama-3.2-1B",
        "SmolLM2-135M",
        "SmolLM2-360M",
    ],
    "guard_name": ["LlamaGuardResponse"],
    "slurm_unit_index": list(range(50)),
}

# Get all combinations except slurm_unit_index (handled separately)
keys = [
    "generation_name",
    "dataset_name",
    "target_model_name",
    "draft_model_name",
    "guard_name",
]
combinations = list(itertools.product(*(arg_matrix[k] for k in keys)))

print("Total combinations:", len(combinations))
# Generate and run the command for each combination and slurm_unit_index
for combo in tqdm(combinations):
    combo_dict = dict(zip(keys, combo))
    # print("Running for combination:", combo_dict)
    for slurm_unit_index in arg_matrix["slurm_unit_index"]:
        result_path = (
            Path().cwd()
            / "step_6_result"
            / combo_dict["dataset_name"]
            / combo_dict["target_model_name"]
            / combo_dict["generation_name"]
            / combo_dict["draft_model_name"]
            / f"{20}"
        )
        result_path.mkdir(parents=True, exist_ok=True)

        command = [
            "./venv/bin/python",
            "../src/core/step_6_draft_response_guard_evaluation.py",
            "--slurm_unit_index",
            str(slurm_unit_index),
            "--generation_name",
            combo_dict["generation_name"],
            "--dataset_name",
            combo_dict["dataset_name"],
            "--target_model_name",
            combo_dict["target_model_name"],
            "--draft_model_name",
            combo_dict["draft_model_name"],
            "--guard_name",
            combo_dict["guard_name"],
        ]

        v = result_path / f"{slurm_unit_index}.pkl"
        if not v.exists():
            print(" ".join(command))

        # try:
        #     subprocess.run(command)
        # except Exception as e:
        #     print(f"Error running command {command}: {e}")
