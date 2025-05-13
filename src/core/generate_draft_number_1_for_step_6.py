import argparse
import pickle
import sys
from pathlib import Path

from tqdm import tqdm


def add_proj_to_PYTHONPATH():
    cwd = Path().cwd()
    assert cwd.name == "workdir"
    sys.path.insert(0, str(cwd.parent))


add_proj_to_PYTHONPATH()


def handle_intent(step_6_pickle_data, new_step_6_result_path):
    for attempt in step_6_pickle_data["attempts"]:
        attempt["responses"] = [attempt["responses"][0]]
        attempt["labels"] = [attempt["labels"][0]]

    with open(new_step_6_result_path / f"{slurm_unit_index}.pkl", "wb") as f:
        pickle.dump(step_6_pickle_data, f)


def main():
    original_step_6_result_path = (
        Path().cwd()
        / "step_6_result"
        / dataset_name
        / target_model_name
        / generation_name
        / draft_model_name
        / f"{draft_number}"
    )

    new_step_6_result_path = (
        Path().cwd()
        / "step_6_result"
        / dataset_name
        / target_model_name
        / generation_name
        / draft_model_name
        / "1"
    )
    new_step_6_result_path.mkdir(parents=True, exist_ok=True)

    with open(
        original_step_6_result_path / f"{slurm_unit_index}.pkl",
        "rb",
    ) as f:
        step_6_pickle_data = pickle.load(f)

    handle_intent(step_6_pickle_data, new_step_6_result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--draft_number",
        type=int,
        required=True,
    )

    args = parser.parse_args()
    draft_number = args.draft_number

    for slurm_unit_index in tqdm(range(0, 50)):
        for target_model_name in [
            "Meta-Llama-3-70B-Instruct-AWQ",
            "Qwen1.5-72B-Chat-AWQ",
            "Phi-3-medium-128k-instruct",
        ]:
            for draft_model_name in [
                "opt-125m-AWQ",
                "SmolLM-135M",
                "Qwen2.5-0.5B",
                "Llama-3.2-1B",
                "SmolLM2-135M",
                "SmolLM2-360M",
            ]:
                for generation_name in ["GCG", "AutoDAN"]:
                    for dataset_name in ["RPAB"]:
                        main()
