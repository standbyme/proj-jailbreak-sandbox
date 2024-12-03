import argparse
import os
import pickle
import sys
import unittest
from pathlib import Path

from tqdm import tqdm


def add_proj_to_PYTHONPATH():
    cwd = Path().cwd()
    assert cwd.name == "workdir"
    sys.path.insert(0, str(cwd.parent))


add_proj_to_PYTHONPATH()
from src.core.guard import BatchEvaluation, LlamaGuardBatchEvaluation
from src.core.language_model import HuggingFaceLanguageModel
from src.core.utils import get_model_id


def save_checkpoint(checkpoint_file_path, checkpoint):
    with open(checkpoint_file_path, "wb") as f:
        pickle.dump(checkpoint, f)


def handle_dataset(
    dataset_file_path: Path,
    result_dir_path: Path,
    target_model: HuggingFaceLanguageModel,
    guard: BatchEvaluation,
):
    tc.assertEqual(dataset_file_path.suffix, ".pkl")

    checkpoint_file_path = (
        result_dir_path / f"tmp.{slurm_unit_index}_{dataset_file_path.name}"
    )
    result_file_path = result_dir_path / f"{slurm_unit_index}_{dataset_file_path.name}"

    if result_file_path.exists():
        return

    if checkpoint_file_path.exists():
        with open(checkpoint_file_path, "rb") as f:
            checkpoint = pickle.load(f)
    else:
        checkpoint = {
            "state": 0,
            "data": [],
        }

    with open(dataset_file_path, "rb") as f:
        intents = pickle.load(f)

    for intent in tqdm(intents[checkpoint["state"] :]):
        checkpoint["state"] += 1

        guard.set_is_consider_prompt(True)
        guard.set_is_consider_response(False)
        evaluation_results = guard.evaluate(intent, [])
        tc.assertEqual(len(evaluation_results), 1)

        evaluation_result = evaluation_results[0]

        if evaluation_result:
            save_checkpoint(checkpoint_file_path, checkpoint)
            continue

        responses = target_model.inference(
            intent, do_sample=True, max_new_tokens=128, num_return_sequences=1
        )
        assert len(responses) == 1

        guard.set_is_consider_prompt(False)
        guard.set_is_consider_response(True)
        evaluation_results = guard.evaluate(intent, responses)
        tc.assertEqual(len(evaluation_results), 1)

        evaluation_result = evaluation_results[0]

        if evaluation_result:
            save_checkpoint(checkpoint_file_path, checkpoint)
            continue

        checkpoint["data"].append(intent)
        save_checkpoint(checkpoint_file_path, checkpoint)

    tc.assertEqual(len(intents), checkpoint["state"])
    os.rename(checkpoint_file_path, result_file_path)


def main():
    dataset_dir_path = Path("/depot/zcelik/data/hongyu/sandbox/benign/prompts")

    result_dir_path = Path().cwd() / "step_9_result" / target_model_name
    result_dir_path.mkdir(parents=True, exist_ok=True)

    dataset_file_path_list = list(dataset_dir_path.iterdir())
    dataset_file_path_list.sort()

    file_path = dataset_file_path_list[slurm_unit_index]

    guard = LlamaGuardBatchEvaluation()

    target_model_id = get_model_id(target_model_name)
    target_model = HuggingFaceLanguageModel(target_model_id)

    handle_dataset(file_path, result_dir_path, target_model, guard)


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

    args = parser.parse_args()
    print(args)
    
    slurm_unit_index = args.slurm_unit_index
    target_model_name = args.target_model_name

    tc = unittest.TestCase()
    main()
