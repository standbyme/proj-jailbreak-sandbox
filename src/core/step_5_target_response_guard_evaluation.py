import argparse
import pickle
import sys
import time
from pathlib import Path

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
    AllTrueBatchEvaluation,
    GradSafeBatchEvaluation,
)


def handle_intent(step_2_pickle_data, result_path, guard: BatchEvaluation):
    intent = step_2_pickle_data["intent"]
    attempts = step_2_pickle_data["attempts"]

    checkpoint = {"intent": intent, "attempts": []}

    for attempt in tqdm(attempts):
        prompt = attempt["prompt"]
        response = attempt["response"]

        assert isinstance(prompt, str)
        assert isinstance(response, str)

        start_time = time.perf_counter()
        evaluation_result = guard.evaluate(prompt, [response])
        end_time = time.perf_counter()
        guard_time = end_time - start_time

        assert len(evaluation_result) == 1
        label = evaluation_result[0]

        v = {
            "prompt": prompt,
            "response": response,
            "label": label,
            "time": guard_time,
        }
        checkpoint["attempts"].append(v)

    with open(result_path / f"{slurm_unit_index}.pkl", "wb") as f:
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
    elif guard_name == "AllTrueGuard":
        v = AllTrueBatchEvaluation()
    elif guard_name == "GradSafeGuardPrompt":
        v = GradSafeBatchEvaluation()
    else:
        raise ValueError(f"Unknown guard_name: {guard_name}")

    v.warmup()
    return v


def main():
    guard = get_guard(guard_name)

    step_2_result_path = (
        Path().cwd()
        / "step_2_result"
        / dataset_name
        / target_model_name
        / generation_name
    )

    result_path = (
        Path().cwd()
        / "step_5_result"
        / dataset_name
        / target_model_name
        / generation_name
        / guard_name
    )
    result_path.mkdir(parents=True, exist_ok=True)

    with open(
        step_2_result_path / f"{slurm_unit_index}.pkl",
        "rb",
    ) as f:
        step_2_pickle_data = pickle.load(f)

    handle_intent(step_2_pickle_data, result_path, guard)


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
        "--generation_name",
        type=str,
        required=True,
        choices=[
            "GCG",
            "AutoDAN",
            "PAIR",
            "Cipher",
            "DeepInception",
            "GPTFuzzer",
            "ICA",
        ],
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["RPAB"],
    )
    parser.add_argument(
        "--guard_name",
        type=str,
        required=True,
        choices=[
            "LlamaGuardPrompt",
            "LlamaGuardResponse",
            "LlamaGuardPromptResponse",
            "PerplexityGuardPrompt",
            "PromptGuard",
            "AllTrueGuard",
            "GradSafeGuardPrompt"
        ],
    )

    args = parser.parse_args()
    print(args)
    slurm_unit_index = args.slurm_unit_index
    target_model_name = args.target_model_name
    generation_name = args.generation_name
    dataset_name = args.dataset_name
    guard_name = args.guard_name

    main()
