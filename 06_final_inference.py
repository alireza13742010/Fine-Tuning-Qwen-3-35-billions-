"""
STAGE 6 — Final Inference (with user answer selection)
==========================================================
Loads Qwen3-14B (4-bit) + the RLHF-aligned adapter from Stage 5. For
every question, generates several candidate answers and lets you pick
the one you find best — the interactive "choose the best answer"
workflow.

Every choice you make here is also logged to
`./data/preferences/inference_feedback.jsonl`, so you can periodically
feed it back into Stage 5 for continual RLHF improvement.

WHAT CHANGED FROM THE OLD (GLM-10B) VERSION
---------------------------------------------
1. Base model updated to Qwen3-14B, same fix as Stages 3-5.
2. STRUCTURAL CHANGE — single adapter instead of two stacked ones:
   the old script loaded the SFT adapter, merged it into the base
   with merge_and_unload(), then loaded the DPO adapter on top of
   that merged model. That matched Stage 5's old design, where DPO
   trained a *fresh* adapter on a merged base.
   Stage 5 was rewritten to skip merging entirely and continue
   training the *same* SFT adapter under the DPO objective instead
   (see that script's docstring for why — merging blows the 24GB VRAM
   budget for a 14B model). That means Stage 5's final_dpo_adapter
   already contains both the SFT and DPO updates in one adapter.
   So here: load ONE adapter directly onto the raw 4-bit base. Loading
   the SFT adapter first and the DPO adapter second, like the old
   script did, would double-apply the SFT changes and give wrong
   results.
3. Prompting now uses tokenizer.apply_chat_template(...) instead of a
   hand-rolled GLM-style template, with return_dict=True so the result
   unpacks cleanly into model.generate(**inputs) (see Stage 4 for why
   this matters — some transformers versions return a bare tensor,
   others a BatchEncoding, from tokenize=True + return_tensors="pt").
4. trust_remote_code=True removed — not needed for Qwen3.
5. HF_HOME points at the same local ./hf_cache as Stages 3-5.

Run:
    python 06_final_inference.py
"""

import os

# ── Local cache directory setup — MUST happen before importing transformers ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_HF_CACHE = os.path.join(SCRIPT_DIR, "hf_cache")
os.makedirs(LOCAL_HF_CACHE, exist_ok=True)
os.environ.setdefault("HF_HOME", LOCAL_HF_CACHE)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

BASE_MODEL_NAME     = "Qwen/Qwen3-14B"

# Stage 5's adapter already contains the SFT updates plus the DPO
# updates on top of them (single continued adapter, not a fresh one on
# a merged base — see docstring above). This is the adapter you want
# for the fully RLHF-aligned model.
DPO_ADAPTER_PATH    = "./output/dpo_qwen3_14b_medical/final_dpo_adapter"

# Fallback if you haven't run Stage 5 yet: the SFT-only adapter from
# Stage 3.
SFT_ADAPTER_PATH    = "./output/sft_qwen3_14b_medical/final_adapter"

FEEDBACK_LOG_PATH   = "./data/preferences/inference_feedback.jsonl"

NUM_CANDIDATES        = 3
MAX_NEW_TOKENS        = 512
TEMPERATURES          = [0.4, 0.7, 1.0]
TOP_P                 = 0.95

# See Stage 4 for details — Qwen3's chat template supports an explicit
# thinking-mode toggle.
ENABLE_THINKING       = True

SYSTEM_PROMPT = (
    "You are a careful, evidence-based medical reasoning assistant. "
    "Think step by step before answering, and give a clear, safe, "
    "medically sound final response."
)


# ─────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────

def load_final_model():
    print(f"Loading base model (4-bit): {BASE_MODEL_NAME}")
    print(f"Hugging Face cache directory (HF_HOME): {os.environ.get('HF_HOME')}\n")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},   # pin to GPU 0, same as Stages 3-5
    )

    if Path(DPO_ADAPTER_PATH).exists():
        print(f"Loading RLHF-aligned adapter (SFT+DPO): {DPO_ADAPTER_PATH}")
        model = PeftModel.from_pretrained(base_model, DPO_ADAPTER_PATH)
    elif Path(SFT_ADAPTER_PATH).exists():
        print(
            "⚠️  No DPO adapter found yet — falling back to the SFT-only "
            f"adapter: {SFT_ADAPTER_PATH}\n"
            "    Run Stages 4 and 5 for the fully RLHF-aligned model."
        )
        model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    else:
        raise FileNotFoundError(
            "No adapter found at either\n"
            f"  {DPO_ADAPTER_PATH}\n"
            f"  {SFT_ADAPTER_PATH}\n"
            "Run at least Stage 3 (SFT) before running inference."
        )

    model.eval()
    print("✅  Model ready for inference.\n")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────
#  GENERATION
# ─────────────────────────────────────────────────────────────

def build_inputs(tokenizer, question: str):
    """Qwen3's own chat template, with return_dict=True so this reliably
    unpacks into model.generate(**inputs) regardless of transformers
    version quirks (see Stage 4)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
        return_tensors="pt",
        return_dict=True,
    )
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )
    return inputs, prompt_text


def generate_candidates(model, tokenizer, question: str):
    inputs, prompt_text = build_inputs(tokenizer, question)
    inputs = inputs.to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    candidates = []
    for i in range(NUM_CANDIDATES):
        temp = TEMPERATURES[i % len(TEMPERATURES)]
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=temp,
                top_p=TOP_P,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(
            output_ids[0][prompt_len:],
            skip_special_tokens=True,
        ).strip()
        candidates.append({"text": text, "temperature": temp})

    return candidates, prompt_text


# ─────────────────────────────────────────────────────────────
#  FEEDBACK LOGGING
# ─────────────────────────────────────────────────────────────

def log_feedback(prompt: str, chosen: str, rejected_list: list, path: str):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for rejected in rejected_list:
        record = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────
#  MAIN INTERACTIVE LOOP
# ─────────────────────────────────────────────────────────────

def interactive_loop():
    print("=" * 60)
    print("  STAGE 6 — Final Inference")
    print("  (RLHF-aligned Qwen3-14B medical reasoning assistant)")
    print("=" * 60)
    print("Type a medical question, or 'quit' to exit.\n")

    model, tokenizer = load_final_model()

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        print("\nGenerating candidate answers ...\n")
        candidates, full_prompt = generate_candidates(model, tokenizer, question)

        for i, c in enumerate(candidates):
            print(f"--- Answer [{i}]  (temp={c['temperature']}) ---")
            print(c["text"])
            print()

        choice = input(
            f"Which answer do you prefer? (0-{len(candidates)-1}, "
            f"Enter to skip logging): "
        ).strip()

        if choice.isdigit() and 0 <= int(choice) < len(candidates):
            idx = int(choice)
            chosen_text = candidates[idx]["text"]
            rejected_texts = [c["text"] for j, c in enumerate(candidates) if j != idx]

            log_feedback(full_prompt, chosen_text, rejected_texts, FEEDBACK_LOG_PATH)

            print(f"\n✅  FINAL ANSWER (your pick):\n{chosen_text}")
            print(f"\n(Preference logged to {FEEDBACK_LOG_PATH} for future RLHF rounds)")
        else:
            print("No selection logged.")

    print("\nSession ended. Thanks!")


if __name__ == "__main__":
    interactive_loop()