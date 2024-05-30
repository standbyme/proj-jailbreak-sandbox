import argparse
import pickle
import sys
from pathlib import Path
import time


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

    checkpoint = {"intent": intent, "attempts": []}

    for attempt in attempts:
        prompt = attempt["prompt"]
        prompt_list = [prompt]

        start_time = time.perf_counter()
        responses_list = target_model.inference(
            prompt_list, do_sample=True, max_new_tokens=256, num_return_sequences=1
        )
        assert len(responses_list) == 1
        responses = responses_list[0]
        assert len(responses) == 1
        response = responses[0]
        end_time = time.perf_counter()
        inference_time = end_time - start_time

        checkpoint["attempts"].append(
            {"prompt": prompt, "response": response, "time": inference_time}
        )

    with open(result_path / f"{slurm_unit_index}.pkl", "wb") as f:
        pickle.dump(checkpoint, f)


def main():
    step_1_jailbreak_generation_result_path = (
        Path().cwd()
        / "step_1_result"
        / dataset_name
        / target_model_name
        / generation_name
    )

    result_path = (
        Path().cwd()
        / "step_2_result"
        / dataset_name
        / target_model_name
        / generation_name
    )
    result_path.mkdir(parents=True, exist_ok=True)

    with open(
        step_1_jailbreak_generation_result_path / f"{slurm_unit_index}.pkl",
        "rb",
    ) as f:
        step_1_pickle_data = pickle.load(f)

    target_model_id = get_model_id(target_model_name)
    target_model = HuggingFaceLanguageModel(target_model_id)
    target_model.warm_up()

    handle_intent(step_1_pickle_data, result_path, target_model)


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
