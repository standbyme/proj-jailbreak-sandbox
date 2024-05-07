from abc import ABC, abstractmethod


class LanguageModel(ABC):
    @abstractmethod
    def inference(self, prompt: str) -> str:
        pass


class Llama3LanguageModel(LanguageModel):
    def inference(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAILanguageModel(LanguageModel):
    def __init__(self, model_id: str) -> None:
        raise NotImplementedError

    def inference(self, prompt: str) -> str:
        raise NotImplementedError
