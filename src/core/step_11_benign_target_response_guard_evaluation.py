import argparse
import pickle
import sys
import time
from pathlib import Path
import unittest

from tqdm import tqdm


def add_proj_to_PYTHONPATH():
    cwd = Path().cwd()
    assert cwd.name == "workdir"
    sys.path.insert(0, str(cwd.parent))


add_proj_to_PYTHONPATH()
from src.core.guard import (
    BatchEvaluation,
    LlamaGuardBatchEvaluation,
    PerplexityBatchEvaluation,
    PromptGuardBatchEvaluation,
)


def handle_dataset(step_9_pickle_data, result_path, guard: BatchEvaluation, dataset_name):
    items = step_9_pickle_data

    checkpoint = []

    for item in tqdm(items):
        intent = item["intent"]
        response = item["response"]

        tc.assertIsInstance(intent, str)
        tc.assertIsInstance(response, str)

        start_time = time.perf_counter()
        evaluation_result = guard.evaluate(intent, [response])
        end_time = time.perf_counter()
        guard_time = end_time - start_time

        assert len(evaluation_result) == 1
        label = evaluation_result[0]

        v = {
            "intent": intent,
            "response": response,
            "label": label,
            "time": guard_time,
        }
        checkpoint.append(v)

    with open(result_path / f"{dataset_name}.pkl", "wb") as f:
        pickle.dump(checkpoint, f)


def get_guard(guard_name) -> BatchEvaluation:
    if guard_name == "LlamaGuardPrompt":
        v = LlamaGuardBatchEvaluation()
        v.set_is_consider_prompt(True)
        v.set_is_consider_response(False)
    elif guard_name == "LlamaGuardResponse":
        v = LlamaGuardBatchEvaluation()
        v.set_is_consider_prompt(False)
        v.set_is_consider_response(True)
    elif guard_name == "LlamaGuardPromptResponse":
        v = LlamaGuardBatchEvaluation()
        v.set_is_consider_prompt(True)
        v.set_is_consider_response(True)
    elif guard_name == "PerplexityGuardPrompt":
        v = PerplexityBatchEvaluation()
    elif guard_name == "PromptGuard":
        v = PromptGuardBatchEvaluation()
    else:
        raise ValueError(f"Unknown guard_name: {guard_name}")

    v.warmup()
    return v


def main():
    guard = get_guard(guard_name)

    step_9_result_path = (
        Path().cwd()
        / "step_9_result"
        / target_model_name
    )

    cleaned_dataset_file_path_list = list(step_9_result_path.iterdir())
    cleaned_dataset_file_path_list.sort()
    file_path = cleaned_dataset_file_path_list[slurm_unit_index]
    dataset_name = file_path.name.split(".")[0]
    print(f"dataset_name: {dataset_name}", flush=True)

    result_path = (
        Path().cwd()
        / "step_11_result"
        / target_model_name
        / guard_name
    )
    result_path.mkdir(parents=True, exist_ok=True)

    with open(
        file_path,
        "rb",
    ) as f:
        step_9_pickle_data = pickle.load(f)

    handle_dataset(step_9_pickle_data, result_path, guard, dataset_name)
    print("11: Done", flush=True)


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
        "--guard_name",
        type=str,
        required=True,
        choices=[
            "LlamaGuardPrompt",
            "LlamaGuardPromptResponse",
            "PerplexityGuardPrompt",
            "PromptGuard",
        ],
    )

    args = parser.parse_args()
    slurm_unit_index = args.slurm_unit_index
    target_model_name = args.target_model_name
    guard_name = args.guard_name

    tc = unittest.TestCase()

    main()
