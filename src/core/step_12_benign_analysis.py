import argparse
import pickle
import sys
from pathlib import Path
import time
from typing import Dict, List
import unittest
from tqdm import tqdm

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def add_proj_to_PYTHONPATH():
    cwd = Path().cwd()
    assert cwd.name == "workdir"
    sys.path.insert(0, str(cwd.parent))


add_proj_to_PYTHONPATH()


def figure(DSR_dict: Dict[int, float], accuracy_dict: Dict[int, float]):
    tc.assertEqual(len(DSR_dict), len(accuracy_dict))
    tc.assertEqual(len(DSR_dict), 10)

    sns.lineplot(data=DSR_dict, color="g", marker="o")
    ax2 = plt.twinx()
    sns.lineplot(data=accuracy_dict, color="b", ax=ax2)


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
    step_7_result_path = (
        Path().cwd()
        / "step_7_result"
        / dataset_name
        / target_model_name
        / generation_name
        / draft_model_name
    )

    step_11_result_path = (
        Path().cwd() / "step_11_result" / draft_model_name / f"{draft_number}"
    )

    step_11_result_file_path_list = list(step_11_result_path.iterdir())

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
        "--draft_model_name",
        type=str,
        required=True,
        choices=["opt-125m-AWQ"],
    )
    parser.add_argument(
        "--draft_number",
        type=int,
        required=True,
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
    draft_model_name = args.draft_model_name
    draft_number = args.draft_number
    dataset_name = args.dataset_name

    tc = unittest.TestCase()

    main()
