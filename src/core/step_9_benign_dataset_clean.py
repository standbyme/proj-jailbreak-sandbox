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
from src.core.guard import (
    BatchEvaluation,
    LlamaGuardBatchEvaluation,
)


def handle_dataset(
    dataset_file_path: Path, result_path: Path, guard: BatchEvaluation
):
    tc.assertEqual(dataset_file_path.suffix, ".pkl")

    result = []

    with open(dataset_file_path, "rb") as f:
        intents = pickle.load(f)
        for intent in tqdm(intents):
            evaluation_results = guard.evaluate(intent, [])
            tc.assertEqual(len(evaluation_results), 1)

            evaluation_result = evaluation_results[0]

            if evaluation_result:
                result.append(intent)

    result_file_path = result_path / dataset_file_path.name
    with open(result_file_path, "wb") as f:
        pickle.dump(result, f)


def main():
    guard = LlamaGuardBatchEvaluation()
    guard.set_is_consider_response(False)
    guard.warmup()

    dataset_dir_path = Path("/depot/zcelik/data/hongyu/sandbox/benign/prompts")

    result_path = Path().cwd() / "step_9_result"
    result_path.mkdir(parents=True, exist_ok=True)

    dataset_file_path_list = list(dataset_dir_path.iterdir())
    dataset_file_path_list.sort()

    file_path = dataset_file_path_list[slurm_unit_index]

    handle_dataset(file_path, result_path, guard)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--slurm_unit_index",
        type=int,
        required=True,
    )

    args = parser.parse_args()
    slurm_unit_index = args.slurm_unit_index

    tc = unittest.TestCase()
    main()
