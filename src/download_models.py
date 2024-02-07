# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

model_list = [
    "meta-llama/Llama-2-70b-chat-hf",
    "TheBloke/Llama-2-70B-chat-AWQ",
    "meta-llama/Llama-2-7b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "TheBloke/Mistral-7B-Instruct-v0.1-AWQ",
    "TheBloke/Llama-2-7b-Chat-AWQ"]

for model_name in model_list:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
