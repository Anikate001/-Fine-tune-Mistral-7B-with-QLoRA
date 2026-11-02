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


---

## v2: real dataset + held-out evaluation (`finetune_v2_bitext_eval.ipynb`)

v1 trained on 500 synthetic samples from 8 templates, so the model mostly learned to repeat those
8 replies. v2 fixes that and adds a proper evaluation:

- **Data:** 2,000 stratified samples from the Bitext customer-support dataset (27 intents, 11 categories),
  plus a separate 100-sample held-out test set the model never sees.
- **Training:** 4-bit NF4 QLoRA (r=16, alpha=32) on all attention and MLP projections, tracked in MLflow.
- **Evaluation:** base vs fine-tuned on the held-out set, scored with ROUGE-L and an LLM-as-judge
  win rate (Llama-3.3-70B, randomised answer order to avoid position bias).

### Results (Colab T4, 81 held-out questions, 3 per intent)

| Metric | Base Mistral-7B-Instruct | Fine-tuned (v2) |
|---|---|---|
| ROUGE-L vs company reference reply | 0.259 | **0.411** (+59%) |
| LLM-judge preference (`gpt-oss-120b`, randomised order) | **56 wins (69%)** | 25 wins (31%) |

Training: 2,000 samples, 1 epoch (125 steps, 34 min), 42M trainable parameters (0.58%), loss 1.69 -> 0.57.
Adapter: [Anikate0/mistral-7b-bitext-support-qlora](https://huggingface.co/Anikate0/mistral-7b-bitext-support-qlora).
Raw numbers: [`results/results.json`](results/results.json).

**What this shows.** Fine-tuning clearly taught the model the company's reply style and content: ROUGE-L
overlap with the reference replies rose 59%. But when an LLM judge compared the two replies for
helpfulness, it preferred the base instruct model 69% of the time. Matching a template is not the
same as being more helpful: the Bitext references are short and templated (with placeholders such as
`{{Order Number}}`), and an already-instruction-tuned base model gives fuller answers. A single
overlap metric would have hidden this, which is why the evaluation uses two.

**Next steps:** train on fuller, higher-quality replies or add a preference stage (DPO) using the judge's
verdicts; test 2-3 epochs and a lower learning rate; review disagreements between ROUGE-L and the judge by hand.
