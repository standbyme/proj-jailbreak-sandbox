import pickle
import pprint


class PrettyPrinter(pprint.PrettyPrinter):
    def _format(self, object, *args, **kwargs):
        if isinstance(object, list):
            object = [object[0]]
        elif isinstance(object, str):
            object = f"{object[:20]}..."
        return pprint.PrettyPrinter._format(self, object, *args, **kwargs)


def abstract(v):
    printer = PrettyPrinter()
    printer.pprint(v)


class Cache:
    def __init__(self, model_version) -> None:
        path_compatible_model_version = model_version.replace("/", "-")
        self.cache_file_name = f"cache_{path_compatible_model_version}.pkl"
        try:
            self.cache = pickle.load(open(self.cache_file_name, "rb"))
        except FileNotFoundError:
            self.cache = {}

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value

    def save(self):
        pickle.dump(self.cache, open(self.cache_file_name, "wb"))


def get_model_id(model_name: str):
    database = {
        "opt-125m-AWQ": "/scratch/gilbreth/anonymoush/project/sandbox/proj-jailbreak-sandbox/workdir/step_0_result/facebook_opt-125m_awq",
        "Meta-Llama-3-70B-Instruct-AWQ": "TechxGenus/Meta-Llama-3-70B-Instruct-AWQ",
        "opt-125m": "facebook/opt-125m",
        "Qwen1.5-72B-Chat-AWQ": "Qwen/Qwen1.5-72B-Chat-AWQ",
        "Phi-3-medium-128k-instruct": "microsoft/Phi-3-medium-128k-instruct",
        "SmolLM-135M": "HuggingFaceTB/SmolLM-135M",
        "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
        "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
        "SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M",
        "SmolLM2-360M": "HuggingFaceTB/SmolLM2-360M",
    }

    return database[model_name]
