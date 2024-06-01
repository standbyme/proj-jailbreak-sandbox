from abc import ABC, abstractmethod
import os
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
from enum import Enum
import torch

class PromptResponseBatchEvaluation(ABC):
    @abstractmethod
    def evaluate(self, prompt: str, responses: List[str]) -> List[bool]:
        pass

class LlamaGuardPromptResponseBatchEvaluation(PromptResponseBatchEvaluation):
    def __init__(self) -> None:
        model_id = "meta-llama/Meta-Llama-Guard-2-8B"
        assert torch.cuda.is_available()

        self.device = "cuda"

        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map=self.device)

    def warmup(self):
        self.evaluate("Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the g", ["Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the g"]*10)

    def evaluate(self, prompt: str, responses: List[str]) -> List[bool]:
        if len(responses) == 0:
            chats = [
                [
                    {"role": "user", "content": prompt},
                ]
            ]
        else:
            chats = [
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
            for response in responses]

        model_inputs = self.tokenizer.apply_chat_template(
            chats, padding=True, return_tensors="pt", return_dict=True
        )
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=16)
        outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        result = []
        for output in outputs:
            v= output.split('[/INST]')[1].strip().startswith("unsafe")
            result.append(v)

        return result