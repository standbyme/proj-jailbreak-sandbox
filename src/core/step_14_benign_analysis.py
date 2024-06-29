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


def figure(
    DSR_dict: Dict[int, float],
    accuracy_dict: Dict[int, float],
    result_path,
    dataset_name: str,
):
    # keys: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    # set x axis unit to 10
    plt.xticks(range(0, 100, 10))
    # set y axis unit to 0.01
    plt.yticks(list(map(lambda x: x/100, range(90, 101, 1))))


    # tc.assertEqual(list(DSR_dict.keys()), list(accuracy_dict.keys()))


    # tc.assertEqual(len(DSR_dict), len(accuracy_dict))
    # tc.assertEqual(len(DSR_dict), 10)

    # sns.lineplot(data=DSR_dict, color="g", marker="o")
    # ax2 = plt.twinx()
    sns.lineplot(data=accuracy_dict, color="b")
    # save the figure
    plt.savefig(result_path / f"{dataset_name}.png")


def handle_threshold(step_11_pickle_data, threshold: int) -> float:
    checkpoint = []

    items = step_11_pickle_data

    for item in items:
        labels: List[bool] = item["labels"]

        tc.assertIsInstance(labels, list)

        # get the ratio of True in labels
        ratio = sum(labels) / len(labels)

        v = ratio > (threshold / 100)
        v = not v
        checkpoint.append(v)

    accuracy = sum(checkpoint) / len(checkpoint)
    return accuracy


def handle_dataset(step_11_pickle_data) -> Dict[int, float]:
    accuracy_dict = {}

    for threshold in tqdm(range(0, 100, 10)):
        accuracy = handle_threshold(step_11_pickle_data, threshold)
        accuracy_dict[threshold] = accuracy

    return accuracy_dict


def main():
    # step_7_result_path = (
    #     Path().cwd()
    #     / "step_7_result"
    #     / dataset_name
    #     / target_model_name
    #     / generation_name
    #     / draft_model_name
    # )

    step_11_result_path = (
        Path().cwd() / "step_11_result" / draft_model_name / f"{draft_number}"
    )

    cleaned_dataset_file_path_list = list(step_11_result_path.iterdir())
    cleaned_dataset_file_path_list.sort()
    file_path = cleaned_dataset_file_path_list[slurm_unit_index]
    dataset_name = file_path.name.split(".")[0]
    print(f"dataset_name: {dataset_name}", flush=True)

    result_path = Path().cwd() / "step_14_result" / draft_model_name / f"{draft_number}"
    result_path.mkdir(parents=True, exist_ok=True)

    with open(
        file_path,
        "rb",
    ) as f:
        step_11_pickle_data = pickle.load(f)

    accuracy_dict = handle_dataset(step_11_pickle_data)
    print("14: Done", flush=True)

    figure(None, accuracy_dict, result_path, dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--slurm_unit_index",
        type=int,
        required=True,
    )
    # parser.add_argument(
    #     "--target_model_name",
    #     type=str,
    #     required=True,
    #     choices=["Meta-Llama-3-70B-Instruct-AWQ"],
    # )
    # parser.add_argument(
    #     "--generation_name",
    #     type=str,
    #     required=True,
    #     choices=["GCG", "AutoDAN"],
    # )
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
    # parser.add_argument(
    #     "--dataset_name",
    #     type=str,
    #     required=True,
    #     choices=["RPAB"],
    # )

    args = parser.parse_args()
    slurm_unit_index = args.slurm_unit_index
    # target_model_name = args.target_model_name
    # generation_name = args.generation_name
    draft_model_name = args.draft_model_name
    draft_number = args.draft_number
    # dataset_name = args.dataset_name

    tc = unittest.TestCase()

    main()
