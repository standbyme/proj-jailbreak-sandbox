import argparse
import pickle
import sys
from pathlib import Path


def add_proj_to_PYTHONPATH():
    cwd = Path().cwd()
    assert cwd.name == "workdir"
    sys.path.insert(0, str(cwd.parent))


add_proj_to_PYTHONPATH()
from src.core.evaluation import LlamaGuardResponseEvaluation


def handle_intent(pickle_data, response_evaluation, result_path):
    attempts = pickle_data["attempts"]

    checkpoint = []

    for attempt in attempts:
        checkpoint.append(response_evaluation.evaluate(attempt["response"]))

    with open(result_path / f"{slurm_unit_index}.pkl", "wb") as f:
        pickle.dump(checkpoint, f)


def main():
    response_evaluation = LlamaGuardResponseEvaluation()

    step_2_result_path = (
        Path().cwd() / "step_2_result" / dataset_name / model_name / generation_name
    )

    result_path = (
        Path().cwd() / "step_3_target_response_llamaguard_result" / dataset_name / model_name / generation_name
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
        "--model_name",
        type=str,
        required=True,
        choices=["Meta-Llama-3-70B-Instruct-AWQ"],
    )
    parser.add_argument(
        "--generation_name",
        type=str,
        required=True,
        choices=["GCG", "GE", "AutoDAN"],
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["RPAB"],
    )

    args = parser.parse_args()
    slurm_unit_index = args.slurm_unit_index
    model_name = args.model_name
    generation_name = args.generation_name
    dataset_name = args.dataset_name

    main()
