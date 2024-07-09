import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm


def add_proj_to_PYTHONPATH():
    cwd = Path().cwd()
    assert cwd.name == "workdir"
    sys.path.insert(0, str(cwd.parent))


add_proj_to_PYTHONPATH()
from src.core.evaluation import MultifacetedResponseEvaluation


def handle_intent(step_2_pickle_data, result_path, response_evaluation):
    intent = step_2_pickle_data["intent"]
    attempts = step_2_pickle_data["attempts"]

    checkpoint = {"intent": intent, "attempts": []}

    for attempt in tqdm(attempts):
        prompt = attempt["prompt"]
        response = attempt["response"]

        assert isinstance(prompt, str)
        assert isinstance(response, str)

        label = response_evaluation.evaluate(response)

        v = {
            "prompt": prompt,
            "response": response,
            "label": label,
        }
        checkpoint["attempts"].append(v)

    with open(result_path / f"{slurm_unit_index}.pkl", "wb") as f:
        pickle.dump(checkpoint, f)


def main():
    response_evaluation = MultifacetedResponseEvaluation()

    step_2_result_path = (
        Path().cwd()
        / "step_2_result"
        / dataset_name
        / target_model_name
        / generation_name
    )

    result_path = (
        Path().cwd()
        / "step_4_result"
        / dataset_name
        / target_model_name
        / generation_name
    )
    result_path.mkdir(parents=True, exist_ok=True)

    with open(
        step_2_result_path / f"{slurm_unit_index}.pkl",
        "rb",
    ) as f:
        step_2_pickle_data = pickle.load(f)

    handle_intent(step_2_pickle_data, result_path, response_evaluation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--slurm_unit_index",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--target_model_name",
        type=str,
        required=True,
        choices=["Meta-Llama-3-70B-Instruct-AWQ", "Qwen1.5-72B-Chat-AWQ", "Phi-3-medium-128k-instruct"],
    )
    parser.add_argument(
        "--generation_name",
        type=str,
        required=True,
        choices=["GCG", "AutoDAN"],
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["RPAB"],
    )

    args = parser.parse_args()
    slurm_unit_index = args.slurm_unit_index
    target_model_name = args.target_model_name
    generation_name = args.generation_name
    dataset_name = args.dataset_name

    main()
