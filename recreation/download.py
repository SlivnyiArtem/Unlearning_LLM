import os

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, AdaLoraModel, AdaLoraConfig
from dotenv import load_dotenv

load_dotenv()


model = AutoModelForCausalLM.from_pretrained(os.getenv("MODEL_NAME"), trust_remote_code=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL_NAME"))
save_path = f'../{os.getenv("ADAPTER_PATH")}'
model.save_pretrained(save_path)