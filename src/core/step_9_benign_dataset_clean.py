import argparse
import pickle
import sys
from pathlib import Path
from tqdm import tqdm
import unittest


def add_proj_to_PYTHONPATH():
    cwd = Path().cwd()
    assert cwd.name == "workdir"
    sys.path.insert(0, str(cwd.parent))


add_proj_to_PYTHONPATH()
from src.core.utils import get_model_id
from src.core.language_model import HuggingFaceLanguageModel
from src.core.guard import (
    BatchEvaluation,
    LlamaGuardBatchEvaluation,
)


def handle_dataset(
    dataset_file_path: Path,
    result_path: Path,
    target_model: HuggingFaceLanguageModel,
    guard: BatchEvaluation,
):
    tc.assertEqual(dataset_file_path.suffix, ".pkl")

    result = []

    with open(dataset_file_path, "rb") as f:
        intents = pickle.load(f)
        for intent in tqdm(intents):
            guard.set_is_consider_response(False)
            evaluation_results = guard.evaluate(intent, [])
            tc.assertEqual(len(evaluation_results), 1)

            evaluation_result = evaluation_results[0]

            if evaluation_result:
                continue

            responses = target_model.inference(
                intent, do_sample=True, max_new_tokens=128, num_return_sequences=1
            )
            assert len(responses) == 1

            guard.set_is_consider_response(True)
            evaluation_results = guard.evaluate(intent, responses)
            tc.assertEqual(len(evaluation_results), 1)

            evaluation_result = evaluation_results[0]

            if evaluation_result:
                continue

            result.append(intent)

    result_file_path = result_path / f"{slurm_unit_index}_{dataset_file_path.name}"
    with open(result_file_path, "wb") as f:
        pickle.dump(result, f)


def main():
    dataset_dir_path = Path("/mnt/shared_ad3_mt1/honcai/proj/sandbox/benign/prompts")

    result_path = Path().cwd() / "step_9_result" / target_model_name
    result_path.mkdir(parents=True, exist_ok=True)

    dataset_file_path_list = list(dataset_dir_path.iterdir())
    dataset_file_path_list.sort()

    file_path = dataset_file_path_list[slurm_unit_index]

    guard = LlamaGuardBatchEvaluation()
    guard.set_is_consider_response(False)

    target_model_id = get_model_id(target_model_name)
    target_model = HuggingFaceLanguageModel(target_model_id)

    handle_dataset(file_path, result_path, target_model, guard)


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

    args = parser.parse_args()
    slurm_unit_index = args.slurm_unit_index
    target_model_name = args.target_model_name

    tc = unittest.TestCase()
    main()
