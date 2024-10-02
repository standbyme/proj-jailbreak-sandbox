import argparse
import os
import pickle
import sys
from pathlib import Path
import time
import unittest
from tqdm import tqdm
import torch


def add_proj_to_PYTHONPATH():
    cwd = Path().cwd()
    assert cwd.name == "workdir"
    sys.path.insert(0, str(cwd.parent))


add_proj_to_PYTHONPATH()
from src.core.language_model import HuggingFaceLanguageModel
from src.core.utils import get_model_id


def handle_intent(
    step_1_pickle_data, result_dir_path, target_model: HuggingFaceLanguageModel
):
    intent = step_1_pickle_data["intent"]
    attempts = step_1_pickle_data["attempts"]

    checkpoint_path = result_dir_path / f"tmp.{slurm_unit_index}.pkl"
    result_path = result_dir_path / f"{slurm_unit_index}.pkl"
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
            tc.assertEqual(checkpoint["intent"], intent)
    else:
        checkpoint = {
            "intent": intent,
            "attempts": [],
        }

    if generation_name == "GCG":
        attempts = attempts[-100:]

    total_steps = len(attempts)
    done_steps = len(checkpoint["attempts"])
    num_steps = total_steps - done_steps

    for checkpoint_offset in tqdm(list(range(num_steps))):
        attempt = attempts[done_steps + checkpoint_offset]

        prompt = attempt["prompt"]

        start_time = time.perf_counter()
        responses = target_model.inference(
            prompt, do_sample=True, max_new_tokens=128, num_return_sequences=1
        )
        assert len(responses) == 1
        response = responses[0]
        end_time = time.perf_counter()
        inference_time = end_time - start_time

        checkpoint["attempts"].append(
            {"prompt": prompt, "response": response, "time": inference_time}
        )
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)

    os.rename(checkpoint_path, result_path)


def main():
    tc.assertTrue(torch.cuda.is_available())

    step_1_jailbreak_generation_result_dir_path = (
        Path().cwd()
        / "step_1_result"
        / dataset_name
        / target_model_name
        / generation_name
    )

    result_dir_path = (
        Path().cwd()
        / "step_2_result"
        / dataset_name
        / target_model_name
        / generation_name
    )
    result_dir_path.mkdir(parents=True, exist_ok=True)

    with open(
        step_1_jailbreak_generation_result_dir_path / f"{slurm_unit_index}.pkl",
        "rb",
    ) as f:
        step_1_pickle_data = pickle.load(f)

    target_model_id = get_model_id(target_model_name)
    target_model = HuggingFaceLanguageModel(target_model_id)
    target_model.warm_up()

    handle_intent(step_1_pickle_data, result_dir_path, target_model)


if __name__ == "__main__":
    tc = unittest.TestCase()

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
    print(args)

    slurm_unit_index = args.slurm_unit_index
    target_model_name = args.target_model_name
    generation_name = args.generation_name
    dataset_name = args.dataset_name

    main()
