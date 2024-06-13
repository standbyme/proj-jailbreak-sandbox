import argparse
import pickle
import sys
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
from src.core.guard import (
    BatchEvaluation,
    LlamaGuardBatchEvaluation,
)


def handle_intent(
    intent: str, draft_model: HuggingFaceLanguageModel, guard: BatchEvaluation
):
    prompt = intent
    draft_number = 25
    # use draft_model get 35 responses for intent
    responses = draft_model.inference(
        prompt, do_sample=True, max_new_tokens=128, num_return_sequences=draft_number
    )
    assert len(responses) == draft_number

    # use guard to evaluate the responses
    evaluation_results = guard.evaluate(prompt, responses)
    unsafe_flag = any(evaluation_results)

    if unsafe_flag:
        print(f"Intent: {intent} is unsafe", flush=True)


def main():
    draft_model_name = "opt-125m-AWQ"
    draft_model_id = get_model_id(draft_model_name)
    draft_model = HuggingFaceLanguageModel(draft_model_id)
    draft_model.warm_up()

    guard = LlamaGuardBatchEvaluation()
    guard.set_is_consider_response(True)
    guard.warmup()

    # get the list of files in /depot/antoniob/data/hongyu/benchmark_dataset/MMLU_train/ with the extension .pkl
    benchmark_dataset_path = Path(
        "/depot/antoniob/data/hongyu/benchmark_dataset/MMLU_train"
    )

    for file_name in [
        # "arc_easy.csv_train.pkl",
        # "arc_hard.csv_train.pkl",
        # "aux_law_90s.csv_train.pkl",
        # "mc_test.csv_train.pkl",
        # "race.csv_train.pkl",
        "science_elementary.csv_train.pkl",
        "science_middle.csv_train.pkl",
    ]:
        print(file_name, flush=True)
        file_path = benchmark_dataset_path / file_name
        with open(file_path, "rb") as f:
            intents = pickle.load(f)
            for intent in tqdm(intents):
                handle_intent(intent, draft_model, guard)


if __name__ == "__main__":
    main()
