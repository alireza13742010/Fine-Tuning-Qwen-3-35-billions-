"""
STAGE 4 — Human-in-the-loop Preference Collection
=====================================================
This is the "RLHF" data-collection stage: for each medical question,
the fine-tuned SFT model (Qwen3-14B + LoRA adapter from Stage 3) generates
several candidate answers (different sampling temperatures). You, the
user, read them and pick the BEST one (and optionally the WORST one).
Each choice becomes one {prompt, chosen, rejected} preference pair.

These pairs are exactly what DPO (Direct Preference Optimization) needs
to run RLHF-style training WITHOUT needing to train a separate reward
model or run PPO — DPO directly optimizes the policy to prefer your
chosen answers over the rejected ones.

WHAT CHANGED FROM THE OLD (GLM-10B) VERSION
---------------------------------------------
1. Base model is now Qwen/Qwen3-14B, matching Stage 3's SFT run — the
   old script still pointed at THUDM/glm-10b, which doesn't match the
   adapter Stage 3 actually produces.
2. Adapter path now points at Stage 3's real output directory
   (./output/sft_qwen3_14b_medical/final_adapter).
3. Prompting now uses tokenizer.apply_chat_template(...) instead of a
   hand-rolled GLM-style "<|system|>/<|user|>/<|assistant|>" string.
   Qwen3 has its own chat template baked into the tokenizer config —
   using a different one at inference time than the one used during SFT
   silently degrades generation quality.
4. trust_remote_code=True removed — Qwen3 is natively supported in
   `transformers`, no custom modeling code needed.
5. HF_HOME is pointed at the same local ./hf_cache directory Stage 3
   uses, so both stages share one cache instead of redownloading into
   ~/.cache/huggingface.

Run:
    python 04_collect_human_preferences.py
"""

import os

# ── Local cache directory setup — MUST happen before importing transformers ──
# Mirrors Stage 3: keep all downloaded model/tokenizer files inside the
# project directory instead of ~/.cache/huggingface.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_HF_CACHE = os.path.join(SCRIPT_DIR, "hf_cache")
os.makedirs(LOCAL_HF_CACHE, exist_ok=True)
os.environ.setdefault("HF_HOME", LOCAL_HF_CACHE)

# Must be set BEFORE torch initializes any CUDA context — reduces memory
# fragmentation, same as Stage 3.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

BASE_MODEL_NAME     = "Qwen/Qwen3-14B"
SFT_ADAPTER_PATH    = "./output/sft_qwen3_14b_medical/final_adapter"

# Where to pull candidate questions from.
# Uses your held-out validation questions by default; you can also just
# type your own question live during the session (option in the menu).
VAL_DATA_PATH        = "./data/processed/val_processed.jsonl"
NUM_QUESTIONS_TO_ASK = 30          # how many questions to review this session

OUTPUT_PREFERENCES_PATH = "./data/preferences/preferences.jsonl"

# ── Generation settings for candidate answers ─────────────────
NUM_CANDIDATES     = 4             # how many alternative answers to generate per question
MAX_NEW_TOKENS     = 512
TEMPERATURES       = [0.3, 0.6, 0.9, 1.1]   # one temperature per candidate (diversity)
TOP_P              = 0.95

# Qwen3 supports an explicit "thinking" mode via its chat template
# (enable_thinking=True/False). Turn this off if you want short, direct
# answers instead of a visible reasoning trace before the final answer.
ENABLE_THINKING    = True

SYSTEM_PROMPT = (
    "You are a careful, evidence-based medical reasoning assistant. "
    "Think step by step before answering, and give a clear, safe, "
    "medically sound final response."
)


# ─────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────

def load_sft_model():
    print(f"Loading base model (4-bit): {BASE_MODEL_NAME}")
    print(f"Loading SFT LoRA adapter from: {SFT_ADAPTER_PATH}")
    print(f"Hugging Face cache directory (HF_HOME): {os.environ.get('HF_HOME')}\n")

    if not Path(SFT_ADAPTER_PATH).exists():
        raise FileNotFoundError(
            f"Adapter not found at {SFT_ADAPTER_PATH}. "
            "Did Stage 3 (03_sft_train_qwen3_14b.py) finish and save "
            "final_adapter/? Check OUTPUT_DIR in that script if you "
            "changed it."
        )

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
        device_map={"": 0},   # pin to GPU 0, same as Stage 3; use "auto" if multi-GPU
    )

    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    model.eval()

    print("✅  Model + adapter ready.\n")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────
#  GENERATION
# ─────────────────────────────────────────────────────────────

def build_inputs(tokenizer, question: str):
    """Use Qwen3's own chat template rather than a hand-rolled prompt
    string, so inference-time formatting matches what the model saw
    during SFT.

    return_dict=True is important here: depending on the installed
    transformers version, apply_chat_template(tokenize=True,
    return_tensors="pt") can hand back either a bare tensor or a
    BatchEncoding. Forcing return_dict=True guarantees a dict-like
    object with 'input_ids'/'attention_mask' keys that we can safely
    .to(device) and unpack into model.generate(**inputs)."""
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

        generated = tokenizer.decode(
            output_ids[0][prompt_len:],
            skip_special_tokens=True,
        )
        candidates.append({"text": generated.strip(), "temperature": temp})

    return candidates, prompt_text


# ─────────────────────────────────────────────────────────────
#  DATA I/O
# ─────────────────────────────────────────────────────────────

def load_questions():
    if not Path(VAL_DATA_PATH).exists():
        print(f"⚠️  {VAL_DATA_PATH} not found — you'll be asked to type questions manually.")
        return []

    ds = load_dataset("json", data_files=VAL_DATA_PATH, split="train")
    questions = [row["question"] for row in ds.select(range(min(NUM_QUESTIONS_TO_ASK, len(ds))))]
    return questions


def save_preference(prompt: str, chosen: str, rejected: str, out_path: str):
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    record = {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────
#  MAIN SESSION LOOP
# ─────────────────────────────────────────────────────────────

def run_session():
    print("=" * 60)
    print("  STAGE 4 — Human Preference Collection (Qwen3-14B SFT)")
    print("=" * 60)
    print(f"Candidates per question : {NUM_CANDIDATES}")
    print(f"Saving preferences to   : {OUTPUT_PREFERENCES_PATH}")
    print("=" * 60 + "\n")

    model, tokenizer = load_sft_model()
    questions = load_questions()

    collected = 0
    q_idx = 0

    while True:
        # ── Get next question ────────────────────────────────────────
        if q_idx < len(questions):
            question = questions[q_idx]
            q_idx += 1
        else:
            question = input(
                "\nNo more preset questions. Type a medical question "
                "(or press Enter to stop): "
            ).strip()
            if not question:
                break

        print("\n" + "-" * 60)
        print(f"QUESTION [{q_idx}]: {question}")
        print("-" * 60)

        print("Generating candidate answers ...")
        candidates, full_prompt = generate_candidates(model, tokenizer, question)

        for i, c in enumerate(candidates):
            print(f"\n[{i}] (temp={c['temperature']})\n{c['text'][:1200]}")

        print("\n" + "-" * 60)
        best_idx = input(
            f"Which answer is BEST? (0-{len(candidates)-1}, "
            f"'s' to skip this question): "
        ).strip()

        if best_idx.lower() == "s":
            continue

        try:
            best_idx = int(best_idx)
            assert 0 <= best_idx < len(candidates)
        except (ValueError, AssertionError):
            print("Invalid input, skipping this question.")
            continue

        worst_idx = input(
            f"Which answer is WORST? (0-{len(candidates)-1}, "
            f"Enter to auto-pick a random remaining one): "
        ).strip()

        if worst_idx == "":
            import random
            remaining = [i for i in range(len(candidates)) if i != best_idx]
            worst_idx = random.choice(remaining)
        else:
            try:
                worst_idx = int(worst_idx)
                assert 0 <= worst_idx < len(candidates) and worst_idx != best_idx
            except (ValueError, AssertionError):
                print("Invalid input, skipping this question.")
                continue

        chosen_text   = candidates[best_idx]["text"]
        rejected_text = candidates[worst_idx]["text"]

        save_preference(
            prompt=full_prompt,
            chosen=chosen_text,
            rejected=rejected_text,
            out_path=OUTPUT_PREFERENCES_PATH,
        )

        collected += 1
        print(f"✅  Saved preference pair #{collected}")

        cont = input("\nContinue to next question? (Y/n): ").strip().lower()
        if cont == "n":
            break

    print(f"\n{'=' * 60}")
    print(f"  Session complete — {collected} preference pairs collected")
    print(f"  Saved to: {OUTPUT_PREFERENCES_PATH}")
    print(f"{'=' * 60}")
    print("\nStage 4 complete. Proceed to 05_rlhf_dpo_train.py")
    print("(Tip: run this script multiple times/sessions to collect more pairs")
    print(" — new pairs are appended, not overwritten.)")


if __name__ == "__main__":
    run_session()