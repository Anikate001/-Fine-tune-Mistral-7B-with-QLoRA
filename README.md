# Mistral-7B Customer Support Fine-Tune (QLoRA)

Fine-tuned [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) 
on a synthetic customer review response dataset using QLoRA (4-bit quantization via BitsAndBytes).

## Results
- Trainable parameters: ~40M / 7B (0.6%) — QLoRA efficiency
- Training loss: 2.1 → 0.82 over 2 epochs
- Hardware: Google Colab T4 (free tier) — ~20 min training time

## Quick Start
pip install transformers peft accelerate bitsandbytes torch

from peft import PeftModel, PeftConfig
# see inference_demo.py for full code

## Technical Details
| Setting | Value |
|---|---|
| Base model | Mistral-7B-Instruct-v0.2 |
| Quantization | 4-bit NF4 (QLoRA) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Dataset size | 500 samples |
| Epochs | 2 |
| Optimizer | paged_adamw_8bit |

## Dataset
Synthetic customer review Q&A pairs generated via `generate_dataset.py`. 
Each sample: a negative/positive review → professional support response.

LIVE AT https://huggingface.co/Anikate0/mistral-7b-customer-support-qlora
