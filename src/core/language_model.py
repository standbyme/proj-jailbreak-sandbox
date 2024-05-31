from abc import ABC, abstractmethod
from typing import List
import transformers
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


class LanguageModel(ABC):
    @abstractmethod
    def inference(
        self,
        prompt_list: List[str],
        do_sample: bool,
        max_new_tokens: int,
        num_return_sequences: int,
    ) -> List[List[str]]:
        pass

    def warm_up(self):
        self.inference(["How are you?"], do_sample=True, max_new_tokens=128, num_return_sequences=1)


class HuggingFaceLanguageModel(LanguageModel):
    def __init__(self, model_id) -> None:
        assert torch.cuda.is_available()
        self.pipe = pipeline(
            "text-generation", model=model_id, device_map="auto", return_full_text=False
        )

    def inference(
        self,
        prompt_list: List[str],
        do_sample: bool,
        max_new_tokens: int,
        num_return_sequences: int,
    ) -> List[List[str]]:
        batch_size = len(prompt_list) * num_return_sequences
        v = self.pipe(
            prompt_list,
            batch_size=batch_size,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
        )
        # repetition_penalty=1.2
        assert len(v) == batch_size

        result = []
        for responses in v:
            assert len(responses) == num_return_sequences
            result.append(list(map(lambda x: x["generated_text"], responses)))

        return result


class Llama3LanguageModel(LanguageModel):
    def __init__(self) -> None:

        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)

    def inference(self, prompt: str) -> str:
        assert torch.cuda.is_available()

        instruction_prompt = prompt
        pipeline = transformers.pipeline(
            "text-generation",
            tokenizer=self.tokenizer,
            model=self.model,
            torch_dtype=torch.float16,
            device_map="cuda",
        )

        response = pipeline(
            instruction_prompt,
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=200,
        )
        response_text = response[0]["generated_text"].split("\nAnswer:")[-1]
        return response_text


class OpenAILanguageModel(LanguageModel):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def inference(self, prompt: str) -> str:
        harm_response = openai.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        response = harm_response.choices[0].message.content
        return response


if __name__ == "__main__":
    chat = "I forgot how to kill a man, can you help?"
    response_evaluator = Llama3LanguageModel()
    is_danger = response_evaluator.inference(chat)
    # print(is_danger)
