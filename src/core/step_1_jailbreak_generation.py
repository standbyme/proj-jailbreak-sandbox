import argparse
import os
import sys
import unittest
from pathlib import Path

from dotenv import load_dotenv
from easyjailbreak.attacker import GPTFuzzer
from easyjailbreak.attacker.Cipher_Yuan_2023 import Cipher
from easyjailbreak.attacker.DeepInception_Li_2023 import DeepInception
from easyjailbreak.attacker.ICA_wei_2023 import ICA
from easyjailbreak.attacker.PAIR_chao_2023 import PAIR
from easyjailbreak.attacker.TAP_Mehrotra_2023 import TAP
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.huggingface_model import (HuggingfaceModel,
                                                    from_pretrained)
from easyjailbreak.models.openai_model import OpenaiModel
from transformers import RobertaForSequenceClassification, RobertaTokenizer


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

    attack_model = from_pretrained(
        model_name_or_path="lmsys/vicuna-13b-v1.5", model_name="vicuna_v1.1"
    )
    attack_model_gpt = OpenaiModel(
        model_name="gpt-5-nano-2025-08-07", api_keys=api_key, mock=False
    )
    eval_model = OpenaiModel(model_name="gpt-5-nano-2025-08-07", api_keys=api_key, mock=True)

    dataset = JailbreakDataset.load_jsonl(path=f"dataset/{dataset_name}.jsonl")

    if generation_name == "PAIR":
        attacker = PAIR(
            attack_model=attack_model,
            target_model=target_model,
            eval_model=eval_model,
            jailbreak_datasets=dataset,
        )
    elif generation_name == "TAP":
        attacker = TAP(
            attack_model=attack_model,
            target_model=target_model,
            eval_model=eval_model,
            jailbreak_datasets=dataset,
        )
    elif generation_name == "Cipher":
        attacker = Cipher(
            attack_model=None,
            target_model=target_model,
            eval_model=eval_model,
            jailbreak_datasets=dataset,
        )
    elif generation_name == "DeepInception":
        attacker = DeepInception(
            attack_model=None,
            target_model=target_model,
            eval_model=eval_model,
            jailbreak_datasets=dataset,
        )
    elif generation_name == "GPTFuzzer":
        model_path = 'hubert233/GPTFuzz'
        judge_model = RobertaForSequenceClassification.from_pretrained(model_path)
        judge_tokenizer = RobertaTokenizer.from_pretrained(model_path)
        judge_eval_model = HuggingfaceModel(model=judge_model, tokenizer=judge_tokenizer, model_name='zero_shot')
        attacker = GPTFuzzer(
            attack_model=attack_model_gpt,
            target_model=target_model,
            eval_model=judge_eval_model,
            jailbreak_datasets=dataset,
        )
    elif generation_name == "ICA":
        attacker = ICA(
            target_model=target_model,
            jailbreak_datasets=dataset,
            attack_model=None,
            eval_model=None,
        )
    else:
        raise ValueError(f"Unknown method: {generation_name}")

    save_dir = Path().cwd() / "step_1_result" / dataset_name / target_model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{generation_name}.jsonl"

    if generation_name in ["PAIR", "TAP"]:
        attacker.attack(save_path=save_path)
    elif generation_name in ["Cipher", "DeepInception", "ICA"]:
        attacker.attack()
        attacker.attack_results.save_to_jsonl(save_path)
    elif generation_name == "GPTFuzzer":
        attacker.attack()
        attacker.jailbreak_datasets.save_to_jsonl(save_path)
    else:
        raise ValueError(f"Unknown method: {generation_name}")


if __name__ == "__main__":
    tc = unittest.TestCase()

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
        choices=["PAIR", "TAP", "Cipher", "DeepInception", "GPTFuzzer", "ICA"],
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["RPAB", "mini_RPAB"],
    )

    args = parser.parse_args()
    print(args)

    main(
        dataset_name=args.dataset_name,
        generation_name=args.generation_name,
        target_model_name=args.target_model_name,
    )
