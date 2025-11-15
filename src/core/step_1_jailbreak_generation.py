import os

from dotenv import load_dotenv
from easyjailbreak.attacker.PAIR_chao_2023 import PAIR
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.models.openai_model import OpenaiModel

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


def main():
    attack_model = from_pretrained(
        model_name_or_path="lmsys/vicuna-13b-v1.5", model_name="vicuna_v1.1"
    )
    target_model = from_pretrained(
        model_name_or_path="TechxGenus/Meta-Llama-3-70B-Instruct-AWQ",
        model_name="llama-3",
    )
    eval_model = OpenaiModel(model_name="gpt-5-nano-2025-08-07", api_keys=api_key)
    dataset = JailbreakDataset("AdvBench")

    attacker = PAIR(
        attack_model=attack_model,
        target_model=target_model,
        eval_model=eval_model,
        jailbreak_datasets=dataset,
    )

    attacker.attack(save_path="vicuna-13b-v1.5_gpt4_gpt4_AdvBench_result.jsonl")


if __name__ == "__main__":
    main()
