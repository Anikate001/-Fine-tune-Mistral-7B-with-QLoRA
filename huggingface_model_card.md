---
base_model: mistralai/Mistral-7B-Instruct-v0.2
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
language:
  - en
datasets:
  - bitext/Bitext-customer-support-llm-chatbot-training-dataset
tags:
  - lora
  - qlora
  - peft
  - customer-support
  - mistral
---

# Mistral-7B-Instruct customer-support adapter (QLoRA)

A LoRA adapter for [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), fine-tuned
with 4-bit QLoRA on 2,000 customer-support conversations from the
[Bitext dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).
It learns the dataset's reply style: short, structured answers with placeholders such as `{{Order Number}}`.

Code, notebook and full write-up: [github.com/Anikate001/-Fine-tune-Mistral-7B-with-QLoRA](https://github.com/Anikate001/-Fine-tune-Mistral-7B-with-QLoRA)

## Results

Evaluated on 81 held-out customer messages (3 per intent, never seen in training), against the base model, both
with greedy decoding.

| Metric | Base Mistral-7B-Instruct | This adapter |
|---|---|---|
| ROUGE-L vs the company's reference reply | 0.259 | **0.411** (+59%) |
| LLM-judge preference (`gpt-oss-120b`, answer order randomised) | **56 / 81** | 25 / 81 |

The adapter matches the company's replies much more closely, but a judge rating helpfulness preferred the base
model's fuller answers 69% of the time. Use it when you want replies in this house style, not as a general
improvement in helpfulness.

## How to use

Needs a GPU with about 6 GB free (the base model loads in 4-bit).

```python
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

prompt = f"[INST] {SYSTEM}\n\nI was charged twice for my order, can you help? [/INST]"  # same template as training
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Training

| Setting | Value |
|---|---|
| Quantisation | 4-bit NF4, double quantisation, fp16 compute (bitsandbytes) |
| LoRA | r = 16, alpha = 32, dropout 0.05 |
| Target modules | q, k, v, o, gate, up, down projections |
| Trainable parameters | 41,943,040 of 7.24B (0.58%) |
| Data | 2,000 conversations, stratified across 27 intents; 81 held out |
| Schedule | 1 epoch, 125 steps, effective batch 16, lr 2e-4 cosine, 10 warm-up steps |
| Optimiser | paged AdamW 8-bit |
| Max length | 512 tokens |
| Hardware / time | Google Colab T4 (16 GB), 34 minutes |
| Training loss | 1.69 to 0.57 |
| Libraries | transformers, peft, trl (SFTTrainer), bitsandbytes; tracked with MLflow |

## Limitations

- Trained on templated replies, so answers are short and may contain unfilled placeholders like `{{Order Number}}`.
- Held-out test questions are paraphrases of the same 27 intents seen in training, which flatters overlap metrics.
- One epoch, no validation-based early stopping, loss computed on prompt and reply tokens.
- Not evaluated for safety or factual accuracy; do not use for real customer interactions without review.
