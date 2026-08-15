"""
STAGE 1 — Download the medical reasoning dataset
==================================================
Dataset : FreedomIntelligence/medical-o1-reasoning-SFT
Fields  : Question | Complex_CoT | Response
Configs : "en" (English) | "zh" (Chinese)

This dataset was built with GPT-4o searching solutions to verifiable
medical problems, each validated by a medical verifier — used originally
to train HuatuoGPT-o1.

Run:
    python 01_download_dataset.py
"""

import os
import json
from pathlib import Path
from datasets import load_dataset


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

DATASET_REPO   = "FreedomIntelligence/medical-o1-reasoning-SFT"
LANGUAGE       = "en"                     # "en" or "zh"
OUTPUT_DIR     = "./data/raw"
TRAIN_SPLIT    = "train"                  # dataset only ships a "train" split
VAL_FRACTION   = 0.05                     # carve out a validation set ourselves


def download_and_inspect():
    print("=" * 60)
    print(f"  Downloading: {DATASET_REPO}  (config='{LANGUAGE}')")
    print("=" * 60)

    dataset = load_dataset(DATASET_REPO, LANGUAGE, split=TRAIN_SPLIT)

    print(f"\nTotal examples: {len(dataset)}")
    print(f"Columns       : {dataset.column_names}")

    print("\n--- Sample record ---")
    sample = dataset[0]
    for k, v in sample.items():
        preview = (v[:300] + " ...") if isinstance(v, str) and len(v) > 300 else v
        print(f"\n[{k}]\n{preview}")

    # ── Train / validation split ──────────────────────────────────────────────
    split_dataset = dataset.train_test_split(test_size=VAL_FRACTION, seed=42)
    train_data = split_dataset["train"]
    val_data   = split_dataset["test"]

    print(f"\nTrain examples      : {len(train_data)}")
    print(f"Validation examples : {len(val_data)}")

    # ── Save to disk as JSON (raw, unprocessed) ────────────────────────────────
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_raw.jsonl"
    val_path   = out_dir / "val_raw.jsonl"

    train_data.to_json(str(train_path), lines=True)
    val_data.to_json(str(val_path), lines=True)

    print(f"\n✅  Saved: {train_path}")
    print(f"✅  Saved: {val_path}")
    print("\nStage 1 complete. Proceed to 02_preprocess.py")

    return str(train_path), str(val_path)


if __name__ == "__main__":
    download_and_inspect()
