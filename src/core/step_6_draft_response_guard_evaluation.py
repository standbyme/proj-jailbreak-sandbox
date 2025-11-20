import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import List

from tqdm import tqdm


def add_proj_to_PYTHONPATH():
    cwd = Path().cwd()
    assert cwd.name == "workdir"
    sys.path.insert(0, str(cwd.parent))


add_proj_to_PYTHONPATH()
from src.core.guard import BatchEvaluation, LlamaGuardBatchEvaluation


def handle_intent(step_3_pickle_data, result_path, guard: BatchEvaluation):
    intent = step_3_pickle_data["intent"]
    attempts = step_3_pickle_data["attempts"]

    checkpoint = {"intent": intent, "attempts": []}

    for attempt in tqdm(attempts):
        prompt = attempt["prompt"]
        responses: List[str] = attempt["responses"]

        assert isinstance(prompt, str)
        assert isinstance(responses, list)

        start_time = time.perf_counter()
        evaluation_result = guard.evaluate(prompt, responses)
        end_time = time.perf_counter()
        guard_time = end_time - start_time

        assert len(evaluation_result) == len(responses)
        labels = evaluation_result

        v = {
            "prompt": prompt,
            "responses": responses,
            "labels": labels,
            "time": guard_time,
        }
        checkpoint["attempts"].append(v)

    with open(result_path / f"{slurm_unit_index}.pkl", "wb") as f:
        pickle.dump(checkpoint, f)


def get_guard(guard_name) -> BatchEvaluation:
    if guard_name == "LlamaGuardResponse":
        v = LlamaGuardBatchEvaluation()
        v.set_is_consider_prompt(False)
        v.set_is_consider_response(True)
    else:
        raise ValueError(f"Unknown guard_name: {guard_name}")

    v.warmup()
    return v


def main():
    step_3_result_path = (
        Path().cwd()
        / "step_3_result"
        / dataset_name
        / target_model_name
        / generation_name
        / draft_model_name
        / f"{draft_number}"
    )

    result_path = (
        Path().cwd()
        / "step_6_result"
        / dataset_name
        / target_model_name
        / generation_name
        / draft_model_name
        / f"{draft_number}"
    )
    result_path.mkdir(parents=True, exist_ok=True)

    with open(
        step_3_result_path / f"{slurm_unit_index}.pkl",
        "rb",
    ) as f:
        step_3_pickle_data = pickle.load(f)

    handle_intent(step_3_pickle_data, result_path, guard)


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
            "LlamaGuardResponse",
        ],
    )
    parser.add_argument(
        "--draft_model_name",
        type=str,
        required=True,
        choices=[
            "opt-125m-AWQ",
            "SmolLM-135M",
            "Qwen2.5-0.5B",
            "Llama-3.2-1B",
            "SmolLM2-135M",
            "SmolLM2-360M",
        ],
    )

    args = parser.parse_args()
    print(args)
    slurm_unit_index = args.slurm_unit_index
    target_model_name = args.target_model_name
    generation_name = args.generation_name
    dataset_name = args.dataset_name
    guard_name = args.guard_name
    draft_model_name = args.draft_model_name

    assert guard_name == "LlamaGuardResponse"

    guard = get_guard(guard_name)

    # for draft_number in [5, 10, 15, 20, 25, 30, 35]:
    #     main()

    for draft_number in [20]:
        main()
