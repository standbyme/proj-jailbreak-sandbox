import argparse
import pickle
import sys
import time
from pathlib import Path

from tqdm import tqdm


def add_proj_to_PYTHONPATH():
    cwd = Path().cwd()
    assert cwd.name == "workdir"
    sys.path.insert(0, str(cwd.parent))


add_proj_to_PYTHONPATH()
from src.core.language_model import HuggingFaceLanguageModel
from src.core.utils import get_model_id


def handle_intent(
    step_1_pickle_data, result_path, target_model: HuggingFaceLanguageModel
):
    intent = step_1_pickle_data["intent"]
    attempts = step_1_pickle_data["attempts"]

    if generation_name == "GCG":
        attempts = attempts[-100:]

    checkpoint = {"intent": intent, "attempts": []}

    for attempt in tqdm(attempts):
        prompt = attempt["prompt"]

        start_time = time.perf_counter()
        responses = target_model.inference(
            prompt,
            do_sample=True,
            max_new_tokens=128,
            num_return_sequences=draft_number,
        )
        end_time = time.perf_counter()
        inference_time = end_time - start_time

        assert len(responses) == draft_number

        checkpoint["attempts"].append(
            {"prompt": prompt, "responses": responses, "time": inference_time}
        )

    with open(result_path / f"{slurm_unit_index}.pkl", "wb") as f:
        pickle.dump(checkpoint, f)


def main():
    step_3_result_path = (
        Path().cwd()
        / "step_3_result"
        / dataset_name
        / target_model_name
        / generation_name
        / draft_model_name
        / f"{draft_number}"
    )
    result_path.mkdir(parents=True, exist_ok=True)

    with open(
        step_1_jailbreak_generation_result_path / f"{slurm_unit_index}.pkl",
        "rb",
    ) as f:
        step_1_pickle_data = pickle.load(f)

    draft_model_id = get_model_id(draft_model_name)
    draft_model = HuggingFaceLanguageModel(draft_model_id)
    draft_model.warm_up()

    handle_intent(step_1_pickle_data, result_path, draft_model)


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
        choices=[
            "Meta-Llama-3-70B-Instruct-AWQ",
            "Qwen1.5-72B-Chat-AWQ",
            "Phi-3-medium-128k-instruct",
        ],
    )
    parser.add_argument(
        "--draft_model_name",
        type=str,
        required=True,
        choices=[
            "opt-125m-AWQ",
            "SmolLM-135M",
            "Qwen2.5-0.5B",
            "Llama-3.2-1B",
            "SmolLM2-135M",
            "SmolLM2-360M",
        ],
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
    parser.add_argument(
        "--draft_number",
        type=int,
        required=True,
    )

    args = parser.parse_args()
    slurm_unit_index = args.slurm_unit_index
    target_model_name = args.target_model_name
    draft_model_name = args.draft_model_name
    generation_name = args.generation_name
    dataset_name = args.dataset_name
    draft_number = args.draft_number

    main()
