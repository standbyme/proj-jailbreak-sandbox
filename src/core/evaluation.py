from abc import ABC, abstractmethod


class PromptEvaluation(ABC):
    @abstractmethod
    def evaluate(prompt: str) -> bool:
        pass


class ResponseEvaluation(ABC):
    @abstractmethod
    def evaluate(response: str) -> bool:
        pass


class LlamaGuardPromptEvaluation(PromptEvaluation):
    def evaluate(prompt: str) -> bool:
        raise NotImplementedError


class LlamaGuardResponseEvaluation(ResponseEvaluation):
    def evaluate(response: str) -> bool:
        raise NotImplementedError
