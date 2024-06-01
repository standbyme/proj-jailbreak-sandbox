import argparse
import pickle
import sys
from pathlib import Path


def add_proj_to_PYTHONPATH():
    cwd = Path().cwd()
    assert cwd.name == "workdir"
    sys.path.insert(0, str(cwd.parent))


add_proj_to_PYTHONPATH()

def calculate_ASR(attempts):
    count = 0

    for attempt in attempts:
        if attempt:
            count += 1
        
    ASR = count / len(attempts)
    return ASR


def main():
    source_path = (
        Path().cwd() / f"{step_name}_result" / dataset_name / model_name / generation_name
    )

    attempts = []

    for slurm_unit_index in range(50):
        try:
            with open(
                source_path / f"{slurm_unit_index}.pkl",
                "rb",
            ) as f:
                pickle_data = pickle.load(f)

            attempts.extend(pickle_data["attempts"])
        except FileNotFoundError:
            print(f"File {source_path / f'{slurm_unit_index}.pkl'} not found.")
    
    ASR = calculate_ASR(attempts)
    print(f"{step_name} ASR: {ASR*100:.4}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--step_name",
        type=str,
        required=True,
        choices=["step_3_target_response_label", "step_4_draft_response", "step_5_draft_response_llamaguard"],
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["Meta-Llama-3-70B-Instruct-AWQ"],
    )
    parser.add_argument(
        "--generation_name",
        type=str,
        required=True,
        choices=["GCG", "AutoDAN"],
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["RPAB"],
    )

    args = parser.parse_args()
    step_name = args.step_name
    model_name = args.model_name
    generation_name = args.generation_name
    dataset_name = args.dataset_name

    main()
