"""Steps 10-12 of finetune_v2_bitext_eval.ipynb as one script.

Run inside the notebook after step 9, in the notebook's own namespace:
    %run -i scripts/judge_and_report.py
It uses the variables the notebook already created (model, tok, test_q, test_ref,
base_out, ft_out, rl_base, rl_ft, losses, train_result, df, train_df, HF_REPO).
"""
import json, os, random, re, time

import mlflow
from google.colab import userdata
from groq import Groq

JUDGE_MODEL = "openai/gpt-oss-120b"   # Groq retired Llama 3.x on 16 Aug 2026
client = Groq(api_key=userdata.get("GROQ_API_KEY"))

JUDGE = """You are evaluating two customer-support replies to the same customer message.
Pick the reply that better resolves the customer's request: correct next steps, helpful, clear and polite.
Placeholders like {{{{Order Number}}}} are allowed template variables, not errors.
A reference reply written by the company is given for guidance.
Finish your answer with a final line of the form VERDICT: A, VERDICT: B, or VERDICT: T (tie).

CUSTOMER: {q}
REFERENCE: {ref}

REPLY A: {a}

REPLY B: {b}"""


def judge(q, ref, a, b):
    for attempt in range(6):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL, temperature=0, max_tokens=600, reasoning_effort="low",
                messages=[{"role": "user", "content": JUDGE.format(q=q, ref=ref, a=a, b=b)}])
            text = r.choices[0].message.content or ""
            m = re.findall(r"VERDICT:\s*([ABT])", text.upper())
            if m:
                return m[-1]
            raise ValueError("no verdict in judge output")
        except Exception as e:
            msg = str(e)
            if "401" in msg or "404" in msg:
                raise RuntimeError(f"Groq error, check key/model: {msg[:200]}") from e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Groq judge kept failing; re-run this script later")


# ---- 10. LLM-as-judge win rate
rng = random.Random(42)
wins = losses_ = ties = 0
for i, (q, ref, b, f) in enumerate(zip(test_q, test_ref, base_out, ft_out)):
    ft_first = rng.random() < 0.5            # randomise order: judges favour position A
    v = judge(q, ref, f, b) if ft_first else judge(q, ref, b, f)
    if v == "T":
        ties += 1
    elif (v == "A") == ft_first:
        wins += 1
    else:
        losses_ += 1
    time.sleep(1.5)
    print(f"  {i+1}/{len(test_q)}  wins={wins} losses={losses_} ties={ties}", end="\r")
win_rate = (wins + 0.5 * ties) / len(test_q)
print(f"\nFine-tuned win rate vs base: {win_rate:.0%}")

# ---- 11. Results + MLflow
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
results = {
    "base_model": BASE_MODEL, "train_samples": len(train_df), "test_samples": len(test_q),
    "intents": int(df.intent.nunique()), "lora_r": 16, "lora_alpha": 32,
    "trainable_pct": round(100 * trainable / total, 2),
    "train_minutes": round(train_result.metrics["train_runtime"] / 60, 1),
    "train_loss_first": round(losses[0], 3), "train_loss_last": round(losses[-1], 3),
    "rougeL_base": round(rl_base, 3), "rougeL_finetuned": round(rl_ft, 3),
    "judge_model": JUDGE_MODEL,
    "judge_wins": wins, "judge_losses": losses_, "judge_ties": ties, "win_rate": round(win_rate, 3),
}
with mlflow.start_run(run_name="heldout-eval"):
    mlflow.log_params({k: v for k, v in results.items() if isinstance(v, str) or k in ("lora_r", "lora_alpha")})
    mlflow.log_metrics({k: float(v) for k, v in results.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    mlflow.log_artifact("generations.json")
json.dump(results, open("results.json", "w"), indent=2)
print("=== SEND THIS TO CLAUDE ===")
print(json.dumps(results, indent=2))

# ---- 12. Publish adapter
model.push_to_hub(HF_REPO, token=os.environ["HF_TOKEN"])
tok.push_to_hub(HF_REPO, token=os.environ["HF_TOKEN"])
print("Adapter live at https://huggingface.co/" + HF_REPO)
