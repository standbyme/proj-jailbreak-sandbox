import argparse
import pickle
import sys
from pathlib import Path
import time
from typing import List
import unittest
from tqdm import tqdm
import pandas as pd


def add_proj_to_PYTHONPATH():
    cwd = Path().cwd()
    assert cwd.name == "workdir"
    sys.path.insert(0, str(cwd.parent))


add_proj_to_PYTHONPATH()
from src.core.guard import (
    BatchEvaluation,
    # LlamaGuardBatchEvaluation,
)


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

def read_step_2_result(step_2_result_path):
    df_list = []

    for intent_index in range(intent_number):
        with open(
            step_2_result_path / f"{intent_index}.pkl",
            "rb",
        ) as f:
            step_2_pickle_data = pickle.load(f)
            attempts = step_2_pickle_data["attempts"]
            df = pd.DataFrame(attempts)
            df_list.append(df)

    concat_df = pd.concat(df_list, ignore_index=True)
    renamed_concat_df = concat_df.rename(columns={"label": "ground_truth_GPT_label"})

    return renamed_concat_df

def read_step_3_result(step_6_result_path, draft_number: int):
    df_list = []

    for intent_index in range(intent_number):
        with open(
            step_6_result_path / f"{draft_number}" / f"{intent_index}.pkl",
            "rb",
        ) as f:
            step_6_pickle_data = pickle.load(f)
            attempts = step_6_pickle_data["attempts"]
            df = pd.DataFrame(attempts)
            df_list.append(df)

    concat_df = pd.concat(df_list, ignore_index=True)
    renamed_concat_df = concat_df.rename(
        columns={
            "labels": f"guard_sandbox_{draft_number}_label",
            "time": f"guard_sandbox_{draft_number}_time",
        }
    )

    return renamed_concat_df


def read_step_4_result(step_4_result_path):
    df_list = []

    for intent_index in range(intent_number):
        with open(
            step_4_result_path / f"{intent_index}.pkl",
            "rb",
        ) as f:
            step_4_pickle_data = pickle.load(f)
            attempts = step_4_pickle_data["attempts"]
            df = pd.DataFrame(attempts)
            df_list.append(df)

    concat_df = pd.concat(df_list, ignore_index=True)
    renamed_concat_df = concat_df.rename(columns={"label": "ground_truth_GPT_label"})

    return renamed_concat_df


def read_step_5_result(step_5_result_path, guard_name: str):
    df_list = []

    for intent_index in range(intent_number):
        with open(
            step_5_result_path / guard_name / f"{intent_index}.pkl",
            "rb",
        ) as f:
            step_5_pickle_data = pickle.load(f)
            attempts = step_5_pickle_data["attempts"]
            df = pd.DataFrame(attempts)
            df_list.append(df)

    concat_df = pd.concat(df_list, ignore_index=True)
    renamed_concat_df = concat_df.rename(
        columns={
            "label": f"guard_{guard_name}_label",
            "time": f"guard_{guard_name}_time",
        }
    )

    return renamed_concat_df


def read_step_6_result(step_6_result_path, draft_number: int):
    df_list = []

    for intent_index in range(intent_number):
        with open(
            step_6_result_path / f"{draft_number}" / f"{intent_index}.pkl",
            "rb",
        ) as f:
            step_6_pickle_data = pickle.load(f)
            attempts = step_6_pickle_data["attempts"]
            df = pd.DataFrame(attempts)
            df_list.append(df)

    concat_df = pd.concat(df_list, ignore_index=True)
    renamed_concat_df = concat_df.rename(
        columns={
            "labels": f"guard_sandbox_{draft_number}_label",
            "time": f"guard_sandbox_{draft_number}_time",
        }
    )

    return renamed_concat_df

def verify_result(step_2_result: pd.DataFrame, step_x_result: pd.DataFrame):
    assert step_2_result.shape[0] == step_x_result.shape[0]
    assert step_2_result["prompt"].equals(step_x_result["prompt"])
    assert step_2_result["response"].equals(step_x_result["response"])

    return True


def handle(step_4_result_path, step_5_result_path, step_6_result_path, result_path):
    step_4_result = read_step_4_result(step_4_result_path)

    step_5_result_list = []
    for guard_name in ["LlamaGuardPrompt", "LlamaGuardPromptResponse", "PerplexityGuardPrompt"]:
        step_5_result = read_step_5_result(step_5_result_path, guard_name)
        step_5_result_list.append(step_5_result)

    step_6_result_list = []
    for draft_number in [5,10,15,20,25,30,35]:
        step_6_result = read_step_6_result(step_6_result_path, draft_number)
        step_6_result_list.append(step_6_result)



def main():
    for dataset_name in ["RPAB"]:
        for target_model_name in ["Meta-Llama-3-70B-Instruct-AWQ"]:
            for generation_name in ["AutoDAN"]:
                for draft_model_name in ["opt-125m-AWQ"]:
                    step_4_result_path = (
                        Path().cwd()
                        / "step_4_result"
                        / dataset_name
                        / target_model_name
                        / generation_name
                    )

                    step_5_result_path = (
                        Path().cwd()
                        / "step_5_result"
                        / dataset_name
                        / target_model_name
                        / generation_name
                    )

                    step_6_result_path = (
                        Path().cwd()
                        / "step_6_result"
                        / dataset_name
                        / target_model_name
                        / generation_name
                        / draft_model_name
                    )

                    result_path = Path().cwd() / "step_7_result"
                    result_path.mkdir(parents=True, exist_ok=True)
                    handle(
                        step_4_result_path,
                        step_5_result_path,
                        step_6_result_path,
                        result_path,
                    )


if __name__ == "__main__":
    intent_number = 50
    tc = unittest.TestCase()
    main()
