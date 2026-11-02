"""Resume finetune_v2_bitext_eval.ipynb from step 7 (training) in a live session.

Run in the notebook's namespace after steps 1-6 have run:
    %run -i scripts/continue_from_training.py
Does: QLoRA training (MLflow) -> fine-tuned replies -> ROUGE-L -> judge, results, publish.
"""
import json, os, time

import evaluate
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# ---- 7. QLoRA fine-tuning with MLflow tracking
os.environ["MLFLOW_EXPERIMENT_NAME"] = "mistral-bitext-qlora"
tok.padding_side = "right"
if not isinstance(model, PeftModel):          # step 7 may already have attached the adapter
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
# T4 has no bf16: keep trainable LoRA weights in fp32 so fp16 mixed precision can unscale their grads
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
model.print_trainable_parameters()

args = SFTConfig(
    output_dir="./mistral-bitext-qlora", num_train_epochs=1,
    per_device_train_batch_size=4, gradient_accumulation_steps=4,
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=10,   # warmup_ratio was removed upstream
    bf16=True,                      # same as v1, which trained fine on a T4; fp16 AMP fails on the bf16 base weights
    logging_steps=10, save_strategy="no", optim="paged_adamw_8bit",
    max_length=512, dataset_text_field="text", report_to="mlflow", seed=SEED)
trainer = SFTTrainer(model=model, train_dataset=train_ds, processing_class=tok, args=args)
train_result = trainer.train()
losses = [l["loss"] for l in trainer.state.log_history if "loss" in l]
print(f"train loss: first logged {losses[0]:.3f} -> last logged {losses[-1]:.3f}")

# ---- 8. Fine-tuned replies on the SAME held-out questions
t0 = time.time()
ft_out = generate_batch(model, test_q)
print(f"\nfine-tuned generation took {(time.time()-t0)/60:.1f} min")
json.dump({"q": test_q, "ref": test_ref, "base": base_out, "ft": ft_out}, open("generations.json", "w"), indent=1)
print("Example:\n", test_q[0], "\n->", ft_out[0][:300])

# ---- 9. ROUGE-L
rouge = evaluate.load("rouge")
rl_base = rouge.compute(predictions=base_out, references=test_ref)["rougeL"]
rl_ft = rouge.compute(predictions=ft_out, references=test_ref)["rougeL"]
print(f"ROUGE-L  base={rl_base:.3f}  fine-tuned={rl_ft:.3f}")

# ---- 10-12. Judge, results + MLflow, publish
exec(open(os.path.join(os.path.dirname(os.path.abspath("scripts/judge_and_report.py")), "judge_and_report.py")).read())
