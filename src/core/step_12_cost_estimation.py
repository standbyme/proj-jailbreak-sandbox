import argparse
import pickle
import sys
import time
import unittest
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from calflops import calculate_flops_hf


def add_proj_to_PYTHONPATH():
    cwd = Path().cwd()
    assert cwd.name == "workdir"
    sys.path.insert(0, str(cwd.parent))


add_proj_to_PYTHONPATH()


def main():
    for model_name in tqdm(
        [
            # "HuggingFaceTB/SmolLM-135M",
            # "meta-llama/Meta-Llama-3-70B-InstructGPT",
            # "Qwen/Qwen1.5-72B-Chat",
            # "microsoft/Phi-3-medium-128k-instruct",
            # "meta-llama/Meta-Llama-Guard-2-8B",
            # "meta-llama/Llama-Guard-3-1B",
            # "meta-llama/Llama-Guard-3-1B-INT4",
            # "meta-llama/Llama-2-70b",
            "meta-llama/Llama-2-70b-hf",
        ]
    ):
        try:
            # flops, macs, params = calculate_flops_hf(
            #     model_name=model_name,
            #     access_token=access_token,
            #     input_shape=input_shape,
            #     print_results=True,
            # )
            # print(
            #     "%s FLOPs:%s  MACs:%s  Params:%s \n" % (model_name, flops, macs, params)
            # )

            batch_size, max_seq_length = 1, 256
            model_name = "facebook/opt-125m"

            flops, macs, params = calculate_flops_hf(model_name=model_name, input_shape=(batch_size, max_seq_length))
            print("%s FLOPs:%s  MACs:%s  Params:%s \n" %(model_name, flops, macs, params))
        except Exception as e:
            print(e)


if __name__ == "__main__":
    access_token = "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"
    input_shape = (1, 128)

    main()
