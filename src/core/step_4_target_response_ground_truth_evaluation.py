import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional


def add_proj_to_PYTHONPATH():
    cwd = Path().cwd()
    assert cwd.name == "workdir"
    sys.path.insert(0, str(cwd.parent))


add_proj_to_PYTHONPATH()
from src.core.evaluation import MultifacetedResponseEvaluation


def handle_intent(pickle_data, response_evaluation, result_path):
    attempts = pickle_data["attempts"]
    
    new_attempts: List[Optional[Dict]] = []
    checkpoint: Dict = {"attempts": new_attempts, "intent": pickle_data["intent"]}

    for attempt in attempts:
        evaluation_result = response_evaluation.evaluate(attempt["response"])
        if evaluation_result:
            new_attempts.append(attempt)
        else:
            new_attempts.append(None)

    with open(result_path / f"{slurm_unit_index}.pkl", "wb") as f:
        pickle.dump(checkpoint, f)


def main():
    response_evaluation = MultifacetedResponseEvaluation()

    step_2_result_path = (
        Path().cwd() / "step_2_result" / dataset_name / model_name / generation_name
    )

    result_path = (
        Path().cwd()
        / "step_3_target_response_label_result"
        / dataset_name
        / model_name
        / generation_name
    )
    result_path.mkdir(parents=True, exist_ok=True)

    with open(
        step_2_result_path / f"{slurm_unit_index}.pkl",
        "rb",
    ) as f:
        pickle_data = pickle.load(f)

    handle_intent(pickle_data, response_evaluation, result_path)


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
        choices=["Meta-Llama-3-70B-Instruct-AWQ"],
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
