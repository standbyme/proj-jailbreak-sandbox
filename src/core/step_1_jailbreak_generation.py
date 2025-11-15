import argparse
import os
import sys
import unittest
from pathlib import Path

from dotenv import load_dotenv
from easyjailbreak.attacker.PAIR_chao_2023 import PAIR
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.models.openai_model import OpenaiModel


def add_proj_to_PYTHONPATH():
    cwd = Path().cwd()
    assert cwd.name == "workdir"
    sys.path.insert(0, str(cwd.parent))


add_proj_to_PYTHONPATH()

from src.core.utils import get_model_id

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


def main(dataset_name: str, generation_name: str, target_model_name: str):
    target_model = from_pretrained(
        model_name_or_path=get_model_id(target_model_name),
        model_name="llama-3",
    )

    dataset = JailbreakDataset.load_csv(path=f"dataset/{dataset_name}.csv")

    if generation_name == "PAIR":
        attack_model = from_pretrained(
            model_name_or_path="lmsys/vicuna-13b-v1.5", model_name="vicuna_v1.1"
        )
        eval_model = OpenaiModel(model_name="gpt-5-nano-2025-08-07", api_keys=api_key)

        attacker = PAIR(
            attack_model=attack_model,
            target_model=target_model,
            eval_model=eval_model,
            jailbreak_datasets=dataset,
        )
    else:
        raise ValueError(f"Unknown method: {generation_name}")

    save_dir = Path().cwd() / "step_1_result" / "RPAB" / target_model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    attacker.attack(save_path=save_dir / f"{generation_name}.jsonl")


if __name__ == "__main__":
    tc = unittest.TestCase()

    parser = argparse.ArgumentParser()

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
        choices=["PAIR"],
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["RPAB"],
    )

    args = parser.parse_args()
    print(args)
