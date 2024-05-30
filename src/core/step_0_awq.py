from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from transformers import AwqConfig


def quantize_and_save(model_id):
    quant_path = "step_1_result/" + model_id.replace("/", "_") + "_awq"

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    model = AutoAWQForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model.quantize(tokenizer, quant_config=quant_config)

    quantization_config = AwqConfig(
        bits=quant_config["w_bit"],
        group_size=quant_config["q_group_size"],
        zero_point=quant_config["zero_point"],
        version=quant_config["version"].lower(),
    ).to_dict()

    model.model.config.quantization_config = quantization_config

    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)


if __name__ == "__main__":
    model_id = "facebook/opt-125m"
    # model_id = "google/gemma-2b"

    quantize_and_save(model_id)
