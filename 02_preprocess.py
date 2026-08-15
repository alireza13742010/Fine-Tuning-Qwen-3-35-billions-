"""
STAGE 2 — Preprocessing
=========================
Converts raw {Question, Complex_CoT, Response} records into a single
formatted training string using a <think> ... </think> reasoning
template, matching how HuatuoGPT-o1 / DeepSeek-R1-style reasoning
datasets are normally consumed.

Final format per example:

    <|system|>
    You are a careful, evidence-based medical reasoning assistant.
    <|user|>
    {Question}
    <|assistant|>
    <think>
    {Complex_CoT}
    </think>
    {Response}

Run:
    python 02_preprocess.py
"""

import json
import re
from pathlib import Path
from datasets import load_dataset, Dataset


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

TRAIN_RAW_PATH   = "./data/raw/train_raw.jsonl"
VAL_RAW_PATH     = "./data/raw/val_raw.jsonl"
OUTPUT_DIR       = "./data/processed"

SYSTEM_PROMPT = (
    "You are a careful, evidence-based medical reasoning assistant. "
    "Think step by step before answering, and give a clear, safe, "
    "medically sound final response."
)

MAX_QUESTION_CHARS   = 4000     # simple length-based filtering
MAX_COT_CHARS        = 12000
MAX_RESPONSE_CHARS   = 4000
MIN_COT_CHARS        = 20       # drop examples with near-empty reasoning


PROMPT_TEMPLATE = """<|system|>
{system}
<|user|>
{question}
<|assistant|>
<think>
{cot}
</think>
{response}"""

# Inference-time prompt (no answer — model must generate it)
INFERENCE_TEMPLATE = """<|system|>
{system}
<|user|>
{question}
<|assistant|>
<think>
"""


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse excess newlines
    return text


def build_example(record: dict) -> dict:
    question = clean_text(record["Question"])
    cot      = clean_text(record["Complex_CoT"])
    response = clean_text(record["Response"])

    full_text = PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        question=question,
        cot=cot,
        response=response,
    )

    return {
        "question": question,
        "cot": cot,
        "response": response,
        "text": full_text,
    }


def is_valid(record: dict) -> bool:
    q, cot, r = record["Question"], record["Complex_CoT"], record["Response"]

    if not q or not cot or not r:
        return False
    if len(q) > MAX_QUESTION_CHARS:
        return False
    if len(cot) < MIN_COT_CHARS or len(cot) > MAX_COT_CHARS:
        return False
    if len(r) > MAX_RESPONSE_CHARS:
        return False
    return True


def process_split(raw_path: str, split_name: str) -> Dataset:
    print(f"\nProcessing split: {split_name}")
    ds = load_dataset("json", data_files=raw_path, split="train")

    before = len(ds)
    ds = ds.filter(is_valid)
    after = len(ds)
    print(f"  Kept {after}/{before} examples after validity filtering")

    ds = ds.map(
        build_example,
        remove_columns=ds.column_names,
        desc=f"Formatting {split_name}",
    )

    return ds


def preprocess():
    print("=" * 60)
    print("  STAGE 2 — Preprocessing")
    print("=" * 60)

    train_ds = process_split(TRAIN_RAW_PATH, "train")
    val_ds   = process_split(VAL_RAW_PATH, "val")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_processed.jsonl"
    val_path   = out_dir / "val_processed.jsonl"

    train_ds.to_json(str(train_path), lines=True)
    val_ds.to_json(str(val_path), lines=True)

    print(f"\n✅  Saved: {train_path}  ({len(train_ds)} examples)")
    print(f"✅  Saved: {val_path}  ({len(val_ds)} examples)")

    print("\n--- Example formatted training text ---")
    print(train_ds[0]["text"][:800], "...\n")

    print("Stage 2 complete. Proceed to 03_sft_train.py")

    return str(train_path), str(val_path)


if __name__ == "__main__":
    preprocess()
