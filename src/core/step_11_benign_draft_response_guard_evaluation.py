import argparse
import pickle
import sys
import time
import unittest
from pathlib import Path
from typing import List

from tqdm import tqdm


def add_proj_to_PYTHONPATH():
    cwd = Path().cwd()
    assert cwd.name == "workdir"
    sys.path.insert(0, str(cwd.parent))


add_proj_to_PYTHONPATH()
from src.core.guard import BatchEvaluation, LlamaGuardBatchEvaluation


def handle_dataset(
    step_10_pickle_data, result_path, guard: BatchEvaluation, dataset_name: str
):
    items = step_10_pickle_data

    checkpoint = []

    for item in tqdm(items):
        intent = item["intent"]
        responses: List[str] = item["responses"]

        tc.assertIsInstance(intent, str)
        tc.assertIsInstance(responses, list)

        start_time = time.perf_counter()
        evaluation_result = guard.evaluate(intent, responses)
        end_time = time.perf_counter()
        guard_time = end_time - start_time

        tc.assertEqual(len(evaluation_result), len(responses))
        labels = evaluation_result

        v = {
            "intent": intent,
            "responses": responses,
            "labels": labels,
            "time": guard_time,
        }
        checkpoint.append(v)

    with open(result_path / f"{dataset_name}.pkl", "wb") as f:
        pickle.dump(checkpoint, f)


def main():
    guard = LlamaGuardBatchEvaluation()
    guard.set_is_consider_prompt(False)
    guard.set_is_consider_response(True)
    guard.warmup()

    step_10_result_path = (
        Path().cwd() / "step_10_result" / draft_model_name / f"{draft_number}"
    )

    cleaned_dataset_file_path_list = list(step_10_result_path.iterdir())
    cleaned_dataset_file_path_list.sort()
    file_path = cleaned_dataset_file_path_list[slurm_unit_index]
    dataset_name = file_path.name.split(".")[0]
    print(f"dataset_name: {dataset_name}", flush=True)

    result_path = Path().cwd() / "step_11_result" / draft_model_name / f"{draft_number}"
    result_path.mkdir(parents=True, exist_ok=True)

    with open(
        file_path,
        "rb",
    ) as f:
        step_10_pickle_data = pickle.load(f)

    handle_dataset(step_10_pickle_data, result_path, guard, dataset_name)
    print("11: Done", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--slurm_unit_index",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--draft_model_name",
        type=str,
        required=True,
        choices=["opt-125m-AWQ", "SmolLM-135M"],
    )
    parser.add_argument(
        "--draft_number",
        type=int,
        required=True,
    )

    args = parser.parse_args()
    slurm_unit_index = args.slurm_unit_index
    draft_model_name = args.draft_model_name
    draft_number = args.draft_number

    tc = unittest.TestCase()

    main()
