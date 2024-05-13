import argparse
import pickle
import sys
from pathlib import Path


def add_proj_to_PYTHONPATH():
    cwd = Path().cwd()
    assert cwd.name == "workdir"
    sys.path.insert(0, str(cwd.parent))


add_proj_to_PYTHONPATH()
from src.core.language_model import HuggingFaceLanguageModel


def handle_intent(draft_model: HuggingFaceLanguageModel, step_2_pickle_data, step_3_pickle_data, result_path):
    intent = step_2_pickle_data["intent"]
    attempts = step_2_pickle_data["attempts"]

    checkpoint = {"intent": intent, "attempts": []}

    for attempt, is_SV in zip(attempts, step_3_pickle_data):
        if not is_SV:
            checkpoint["attempts"].append(None)
        else:
            prompt = attempt["prompt"]
            draft_response = draft_model.inference(prompt)
            # .replace("[INST]", "").replace("[/INST]", "")
            checkpoint["attempts"].append({"prompt": prompt, "response": draft_response})

    with open(result_path / f"{slurm_unit_index}.pkl", "wb") as f:
        pickle.dump(checkpoint, f)


def main():
    step_2_result_path = (
        Path().cwd() / "step_2_result" / dataset_name / model_name / generation_name
    )

    step_3_target_response_llamaguard_result_path = (
        Path().cwd() / "step_3_target_response_llamaguard_result" / dataset_name / model_name / generation_name
    )

    result_path = (
        Path().cwd() / "step_4_draft_response_result" / dataset_name / model_name / generation_name
    )
    result_path.mkdir(parents=True, exist_ok=True)

    with open(
        step_2_result_path / f"{slurm_unit_index}.pkl",
        "rb",
    ) as f:
        step_2_pickle_data = pickle.load(f)

    with open(
        step_3_target_response_llamaguard_result_path / f"{slurm_unit_index}.pkl",
        "rb",
    ) as f:
        step_3_pickle_data = pickle.load(f)

    draft_model_id = "meta-llama/Meta-Llama-3-8B"
    draft_model = HuggingFaceLanguageModel(draft_model_id)

    handle_intent(draft_model, step_2_pickle_data, step_3_pickle_data, result_path)


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
