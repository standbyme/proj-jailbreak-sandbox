import argparse
import os
import pickle
import sys
from typing import List
import unittest
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


def add_proj_to_PYTHONPATH():
    cwd = Path().cwd()
    assert cwd.name == "workdir"
    sys.path.insert(0, str(cwd.parent))


add_proj_to_PYTHONPATH()


def handle(intents: List[str], target_model_name: str, generation_name: str):
    source_dir = Path().cwd() / "step_1_result" / "RPAB" / target_model_name
    source_path = source_dir / f"{generation_name}.jsonl"

    target_dir = source_dir / generation_name
    target_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(source_path, lines=True)
    # print(df.head())
    # print(df.columns)

    for intent_index, intent in enumerate(intents):
        attempts = []

        filtered_df = df[df["query"] == intent]
        if filtered_df.empty:
            raise ValueError(f"No records found for intent: {intent}")

        for _, row in filtered_df.iterrows():
            attempts.append({"prompt": row["jailbreak_prompt"]})

        result = {}
        result["intent"] = intent
        result["attempts"] = attempts

        target_path = target_dir / f"{intent_index}.pkl"
        with open(target_path, "wb") as f:
            pickle.dump(result, f)


def main():
    original_dataset_path = Path().cwd() / "dataset" / "RPAB.jsonl"
    original_dataset_df = pd.read_json(original_dataset_path, lines=True)
    intents = original_dataset_df["query"].tolist()

    for target_model_name in [
        # "Meta-Llama-3-70B-Instruct-AWQ",
        # "Qwen1.5-72B-Chat-AWQ",
        "Phi-3-medium-128k-instruct",
    ]:
        for generation_name in [
            # "PAIR",
            "Cipher",
            "DeepInception",
            "GPTFuzzer",
            "ICA",
        ]:
            handle(intents, target_model_name, generation_name)


if __name__ == "__main__":
    tc = unittest.TestCase()

    main()
