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


def handle_benchmark_dataset_prompts_file(
    benchmark_dataset_prompts_file_path: Path, result_path: Path, guard: BatchEvaluation
):
    tc.assertEqual(benchmark_dataset_prompts_file_path.suffix, ".pkl")

    result = []

    with open(benchmark_dataset_prompts_file_path, "rb") as f:
        intents = pickle.load(f)
        for intent in tqdm(intents):
            evaluation_results = guard.evaluate(intent, [])
            tc.assertEqual(len(evaluation_results), 1)

            evaluation_result = evaluation_results[0]

            if evaluation_result:
                result.append(intent)

    result_file_path = result_path / benchmark_dataset_prompts_file_path.name
    with open(result_file_path, "wb") as f:
        pickle.dump(result, f)


def main():
    guard = LlamaGuardBatchEvaluation()
    guard.set_is_consider_response(False)
    guard.warmup()

    benchmark_dataset_path = Path("/depot/zcelik/data/hongyu/sandbox/benign/prompts")

    result_path = Path().cwd() / "step_9_result"
    result_path.mkdir(parents=True, exist_ok=True)

    benchmark_dataset_prompts_file_path_list = list(benchmark_dataset_path.iterdir())

    for file_path in tqdm(benchmark_dataset_prompts_file_path_list[:2]):
        handle_benchmark_dataset_prompts_file(file_path, result_path, guard)


if __name__ == "__main__":
    tc = unittest.TestCase()
    main()
