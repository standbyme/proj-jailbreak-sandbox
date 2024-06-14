import argparse
import pickle
import sys
from pathlib import Path
import time
import unittest
from tqdm import tqdm


def add_proj_to_PYTHONPATH():
    cwd = Path().cwd()
    assert cwd.name == "workdir"
    sys.path.insert(0, str(cwd.parent))


add_proj_to_PYTHONPATH()
from src.core.language_model import HuggingFaceLanguageModel
from src.core.utils import get_model_id


def handle_dataset(
    step_9_pickle_data,
    result_path,
    target_model: HuggingFaceLanguageModel,
    dataset_name: str,
):
    intents = step_9_pickle_data
    checkpoint = []

    for intent in tqdm(intents):
        start_time = time.perf_counter()
        responses = target_model.inference(
            intent,
            do_sample=True,
            max_new_tokens=128,
            num_return_sequences=draft_number,
        )
        end_time = time.perf_counter()
        inference_time = end_time - start_time

        assert len(responses) == draft_number

        checkpoint.append(
            {"intent": intent, "responses": responses, "time": inference_time}
        )

    with open(result_path / f"{dataset_name}.pkl", "wb") as f:
        pickle.dump(checkpoint, f)


def main():
    step_9_benign_dataset_clean_result_path = Path().cwd() / "step_9_result"

    cleaned_dataset_file_path_list = list(
        step_9_benign_dataset_clean_result_path.iterdir()
    )
    cleaned_dataset_file_path_list.sort()

    file_path = cleaned_dataset_file_path_list[slurm_unit_index]

    dataset_name = file_path.name.split(".")[0]

    result_path = Path().cwd() / "step_10_result" / draft_model_name / f"{draft_number}"
    result_path.mkdir(parents=True, exist_ok=True)

    with open(
        file_path,
        "rb",
    ) as f:
        step_9_pickle_data = pickle.load(f)

    draft_model_id = get_model_id(draft_model_name)
    draft_model = HuggingFaceLanguageModel(draft_model_id)
    draft_model.warm_up()

    handle_dataset(step_9_pickle_data, result_path, draft_model, dataset_name)


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
        choices=["opt-125m-AWQ"],
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
