"""Chat with the fine-tuned adapter (v2, Bitext customer support).

pip install transformers peft accelerate bitsandbytes torch
Needs a GPU with ~6 GB free (4-bit loading), e.g. a free Colab T4.
"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER = "Anikate0/mistral-7b-bitext-support-qlora"
SYSTEM = "You are a professional, friendly customer support agent. Reply to the customer's message."

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER).eval()


def respond(message: str, max_new_tokens: int = 200) -> str:
    prompt = f"[INST] {SYSTEM}\n\n{message} [/INST]"   # same template as training
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


if __name__ == "__main__":
    print(respond("I was charged twice for my order, can you help?"))
