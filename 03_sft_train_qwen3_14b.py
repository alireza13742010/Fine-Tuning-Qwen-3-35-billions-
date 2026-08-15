"""
STAGE 3 — Supervised Fine-Tuning (SFT) of Qwen3-14B with QLoRA
================================================================
Base model : Qwen/Qwen3-14B
             Dense (non-MoE) causal language model, 14B parameters.
Method     : QLoRA — 4-bit quantized base weights + trainable LoRA adapters
Trainer    : trl.SFTTrainer

WHY THIS IS SIMPLER THAN THE 35B MOE VERSION
---------------------------------------------
1. Dense model → loaded with the standard AutoModelForCausalLM /
   AutoTokenizer, not AutoModelForImageTextToText. No vision head, no
   routed experts, no _no_split_modules device-map complications.

2. ~14B params in 4-bit is roughly ~8-9GB of weights — comfortably inside
   a 24GB GPU alongside activations, optimizer state, and LoRA adapters.
   No need for the aggressive minimization (rank=4, 2 target modules,
   seq_len=256) that the 35B MoE config required. Settings below are
   reasonable defaults, not bare-minimum survival settings.

3. Qwen3-14B has been supported in `transformers` since its April 2025
   release — no bleeding-edge architecture registration risk, no need to
   chase nightly transformers/bitsandbytes builds, and no exposure to the
   transformers-v5 core_model_loading bnb-quantization-during-load bug
   that affected the newly-released Qwen3.6 MoE checkpoint.

LOCAL MODEL STORAGE
--------------------
By default, `transformers`/`huggingface_hub` download and cache all model
files under `~/.cache/huggingface`. This script instead points the entire
Hugging Face cache (models, tokenizers, datasets metadata) at a folder
named `hf_cache/` created next to this script, so everything downloaded
lives inside your project directory rather than your home directory.

This is done by setting `HF_HOME` as an environment variable BEFORE
importing `transformers`/`huggingface_hub` — those libraries read `HF_HOME`
at import time to decide where their cache lives, so the env var must be
set first for it to take effect.

Run:
    python 03_sft_train_qwen3_14b.py --meminfo    # print free/total VRAM, then exit
    python 03_sft_train_qwen3_14b.py --inspect    # print Linear module names, then exit
    python 03_sft_train_qwen3_14b.py              # actual training run
"""

import os

# ── Local cache directory setup — MUST happen before importing transformers ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_HF_CACHE = os.path.join(SCRIPT_DIR, "hf_cache")
os.makedirs(LOCAL_HF_CACHE, exist_ok=True)

# HF_HOME controls where huggingface_hub/transformers/datasets store
# EVERYTHING they download: model weights, tokenizer files, dataset
# metadata caches, etc. Setting this redirects all of it into
# ./hf_cache instead of ~/.cache/huggingface.
os.environ.setdefault("HF_HOME", LOCAL_HF_CACHE)

# Must be set BEFORE torch initializes any CUDA context — reduces memory
# fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

MODEL_NAME        = "Qwen/Qwen3-14B"
TRAIN_DATA_PATH   = "./data/processed/train_processed.jsonl"
VAL_DATA_PATH     = "./data/processed/val_processed.jsonl"
OUTPUT_DIR        = "./output/sft_qwen3_14b_medical"

# 14B in 4-bit leaves much more headroom than the 35B MoE case, so this can
# be raised from the 256 forced on the 35B config. Still conservative —
# raise further once a run completes cleanly and you've checked VRAM
# headroom via --meminfo after a successful load.
MAX_SEQ_LENGTH    = 2048

# ── QLoRA / LoRA hyperparameters ─────────────────────────────────────────
LORA_RANK         = 16
LORA_ALPHA        = 32
LORA_DROPOUT      = 0.05

# Standard Qwen3 dense-model attention + MLP projection names. Confirm
# against --inspect output before a long run — naming is consistent across
# Qwen3 dense sizes, but always verify rather than assume.
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# ── Training hyperparameters ─────────────────────────────────────────────
NUM_EPOCHS         = 1
BATCH_SIZE         = 2
GRAD_ACCUM_STEPS   = 8           # effective batch size = 16
LEARNING_RATE      = 2e-4
WARMUP_RATIO       = 0.03
LOGGING_STEPS      = 10
SAVE_STEPS         = 200
EVAL_STEPS         = 200
LR_SCHEDULER       = "cosine"
SEED               = 42
LOSS_TYPE          = "nll"


# ─────────────────────────────────────────────────────────────
#  GPU MEMORY DIAGNOSTICS
# ─────────────────────────────────────────────────────────────

def print_gpu_meminfo():
    if not torch.cuda.is_available():
        print("No CUDA GPU detected.")
        return

    n = torch.cuda.device_count()
    print(f"\nDetected {n} CUDA device(s):")
    for i in range(n):
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        free_gb = free_bytes / (1024 ** 3)
        total_gb = total_bytes / (1024 ** 3)
        used_gb = total_gb - free_gb
        name = torch.cuda.get_device_name(i)
        print(f"  [{i}] {name}: {used_gb:.1f} GB used / {total_gb:.1f} GB total "
              f"({free_gb:.1f} GB free)")

    print(
        "\nRule of thumb for Qwen3-14B: ~8-9GB for 4-bit weights, "
        "plus activations/optimizer state/LoRA on top. Much more headroom "
        "than the 35B MoE case — if this still OOMs, lower MAX_SEQ_LENGTH "
        "or BATCH_SIZE first."
    )
    print(f"\nHugging Face cache directory (HF_HOME): {os.environ.get('HF_HOME')}")


# ─────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────

def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer():
    print(f"Loading tokenizer: {MODEL_NAME}")
    print(f"Downloads will be cached under: {LOCAL_HF_CACHE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nLoading base model in 4-bit: {MODEL_NAME}")
    print_gpu_meminfo()

    bnb_config = build_bnb_config()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map={"": 0},     # pin everything to GPU 0, no offload
        )
    except torch.cuda.OutOfMemoryError:
        print(
            "\n❌  CUDA OutOfMemoryError while loading fully on GPU.\n"
            "    Next steps, in order of impact:\n"
            "      1. Lower MAX_SEQ_LENGTH (try 512)\n"
            "      2. Lower BATCH_SIZE to 1 and raise GRAD_ACCUM_STEPS\n"
            "      3. Lower LORA_RANK to 8\n"
            "      4. Close any other process holding VRAM (check nvidia-smi)\n"
        )
        raise

    print("\n✅  Model loaded. Post-load GPU memory:")
    print_gpu_meminfo()

    # Standard PEFT helper is fine here — 14B's non-quantized parameters
    # (embeddings, norms, lm_head) are small enough that the fp32 upcast
    # step which OOM'd on the 35B MoE model is a non-issue here.
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    return model, tokenizer


def inspect_linear_modules(model, limit=200):
    """Print every Linear-layer module name so target_modules can be verified
    against the real architecture instead of guessed."""
    print("\nLinear modules found in model (name : in_features x out_features):")
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or module.__class__.__name__ in (
            "Linear4bit", "Linear8bitLt"
        ):
            print(f"  {name}")
            count += 1
            if count >= limit:
                print(f"  ... (truncated after {limit})")
                break
    print(f"\nTotal Linear-like modules: {count}")


def build_lora_model(model):
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train():
    print("=" * 60)
    print("  STAGE 3 — SFT (QLoRA) on Qwen3-14B — single GPU")
    print("=" * 60)

    if "--meminfo" in sys.argv:
        print_gpu_meminfo()
        return

    model, tokenizer = load_model_and_tokenizer()

    if "--inspect" in sys.argv:
        inspect_linear_modules(model)
        print("\nInspect-only run finished. Update LORA_TARGET_MODULES above if needed, then re-run without --inspect.")
        return

    model = build_lora_model(model)

    print("\nLoading processed dataset ...")
    train_ds = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train")
    val_ds   = load_dataset("json", data_files=VAL_DATA_PATH, split="train")
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # ── Build SFTConfig kwargs defensively ──────────────────────────────────
    # trl's SFTConfig/SFTTrainer parameter names have been renamed several
    # times across recent releases (max_seq_length -> max_length,
    # tokenizer -> processing_class, dataset_text_field handling, etc.).
    # Rather than hardcoding names that may not match your installed
    # version, inspect the actual signature and only pass supported kwargs.
    import inspect

    sft_config_params = set(inspect.signature(SFTConfig.__init__).parameters)

    def add_if_supported(kwargs_dict, name, value):
        if name in sft_config_params:
            kwargs_dict[name] = value
        else:
            print(f"  (skipping unsupported SFTConfig param: {name})")

    config_kwargs = dict(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=LOGGING_STEPS,
    eval_steps=EVAL_STEPS,
    eval_strategy="steps",     # keep periodic eval so you can watch loss/metrics
    save_strategy="no",        # <-- disables all intermediate checkpoint saving
    bf16=True,
    optim="paged_adamw_8bit",
    packing=False,
    seed=SEED)

    # Handle the max_seq_length -> max_length rename explicitly.
    if "max_length" in sft_config_params:
        config_kwargs["max_length"] = MAX_SEQ_LENGTH
    elif "max_seq_length" in sft_config_params:
        config_kwargs["max_seq_length"] = MAX_SEQ_LENGTH

    add_if_supported(config_kwargs, "dataset_text_field", "text")
    add_if_supported(config_kwargs, "loss_type", LOSS_TYPE)

    sft_config = SFTConfig(**config_kwargs)

    # ── Build SFTTrainer kwargs defensively ─────────────────────────────────
    trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters)

    trainer_kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        print("  (warning: neither 'processing_class' nor 'tokenizer' found "
              "on SFTTrainer — relying on model's default tokenizer handling)")

    trainer = SFTTrainer(**trainer_kwargs)

    print("\nStarting training ...\n")
    trainer.train()

    print("\nSaving final LoRA adapter + tokenizer ...")
    final_path = os.path.join(OUTPUT_DIR, "final_adapter")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"\n✅  SFT model adapter saved to: {final_path}")
    print("Stage 3 complete. Proceed to 04_collect_human_preferences.py")

    return final_path


if __name__ == "__main__":
    train()
