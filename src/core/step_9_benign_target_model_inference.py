import argparse
import os
import pickle
import sys
import unittest
from pathlib import Path
import time

from tqdm import tqdm


def add_proj_to_PYTHONPATH():
    cwd = Path().cwd()
    assert cwd.name == "workdir"
    sys.path.insert(0, str(cwd.parent))


add_proj_to_PYTHONPATH()
from src.core.language_model import HuggingFaceLanguageModel
from src.core.utils import get_model_id


def handle_dataset(
    pickle_data,
    result_dir_path,
    target_model: HuggingFaceLanguageModel,
    dataset_name: str,
):
    intents = pickle_data
    checkpoint = []

    for intent in tqdm(intents):
        start_time = time.perf_counter()
        responses = target_model.inference(
            intent,
            do_sample=True,
            max_new_tokens=128,
            num_return_sequences=1,
        )
        end_time = time.perf_counter()
        inference_time = end_time - start_time

        assert len(responses) == 1
        response = responses[0]

        checkpoint.append(
            {"intent": intent, "response": response, "time": inference_time}
        )

    with open(result_dir_path / f"{dataset_name}.pkl", "wb") as f:
        pickle.dump(checkpoint, f)


def main():
    dataset_dir_path = Path("/depot/anonymous/data/anonymoush/sandbox/benign/just-eval")

    dataset_file_path_list = list(dataset_dir_path.iterdir())
    dataset_file_path_list.sort()

    file_path = dataset_file_path_list[slurm_unit_index]

    dataset_name = file_path.name.split(".")[0]
    print(f"dataset_name: {dataset_name}", flush=True)

    result_dir_path = Path().cwd() / "step_9_result" / target_model_name
    result_dir_path.mkdir(parents=True, exist_ok=True)

    with open(
        file_path,
        "rb",
    ) as f:
        pickle_data = pickle.load(f)

    target_model_id = get_model_id(target_model_name)
    target_model = HuggingFaceLanguageModel(target_model_id)
    target_model.warm_up()

    handle_dataset(pickle_data, result_dir_path, target_model, dataset_name)

    print("9: Done", flush=True)


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
