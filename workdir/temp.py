from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-72B-Chat-AWQ")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-72B-Chat-AWQ",
    torch_dtype="auto",
    device_map="auto"
)
