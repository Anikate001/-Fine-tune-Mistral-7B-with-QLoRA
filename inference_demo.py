from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

repo_id = "YOUR_HF_USERNAME/mistral-7b-customer-support-qlora"

config = PeftConfig.from_pretrained(repo_id)
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

def respond(review: str) -> str:
    prompt = f"<s>[INST] You are a professional customer support agent. Respond to this review:\n\n{review} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True)
    return tokenizer.decode(out[0], skip_special_tokens=True).split("[/INST]")[-1].strip()

if __name__ == "__main__":
    print(respond("The product broke after 3 uses and support never replied."))
