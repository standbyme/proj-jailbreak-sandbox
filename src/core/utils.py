import pickle


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
        "opt-125m-AWQ": "/scratch/gilbreth/hongyu/project/sandbox/proj-jailbreak-sandbox/workdir/step_0_result/facebook_opt-125m_awq",
        "Meta-Llama-3-70B-Instruct-AWQ": "TechxGenus/Meta-Llama-3-70B-Instruct-AWQ",
        "opt-125m": "facebook/opt-125m"
    }

    return database[model_name]