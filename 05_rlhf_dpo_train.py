"""
STAGE 5 — RLHF via DPO (Direct Preference Optimization)
===========================================================
Takes the human {prompt, chosen, rejected} pairs collected in Stage 4
and further trains the SFT model (Qwen3-14B + LoRA from Stage 3) so it
directly prefers the human-chosen answers.

Why DPO instead of classic PPO-based RLHF:
  Classic RLHF needs THREE models in memory at once (policy, reference,
  reward model) plus a separate reward-model training stage — very
  heavy for a single 24GB GPU with a 14B base model.
  DPO reformulates the same RLHF objective as a single supervised loss
  computed directly from your preference pairs, needing only the policy
  model + a frozen reference copy. It has become the standard practical
  approach for exactly this "user picks best answer" workflow.

WHAT CHANGED FROM THE OLD (GLM-10B) VERSION
---------------------------------------------
1. Base model / adapter path updated to Qwen3-14B + Stage 3's real
   output directory (same fix as Stages 3-4).
2. trust_remote_code=True removed — not needed for Qwen3.
3. LORA_TARGET_MODULES for the merge-time LoRA config were GLM-specific
   ("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h") —
   those module names don't exist on Qwen3, so they've been dropped
   (see point 4, they're no longer needed at all).
4. NO MORE merge_and_unload(). The old script loaded the base model in
   4-bit, then called merge_and_unload() to fold the SFT LoRA into the
   base weights before attaching a *fresh* LoRA adapter for DPO. Merging
   dequantizes the merged weights back to bf16 — for a 14B model that's
   roughly 28GB, which will not fit the 24GB GPU this whole pipeline is
   built around (Stage 3 sizes its defaults around exactly that budget).
   Instead, this version keeps the base model in 4-bit and continues
   training the *same* SFT LoRA adapter directly against the DPO
   objective — the standard pattern for QLoRA-based DPO. With
   ref_model=None, TRL gets reference logits by temporarily disabling
   the adapter on this same model, so there's no second full model
   copy sitting in VRAM.
5. Added prepare_model_for_kbit_training() before attaching the
   adapter, matching Stage 3 — needed for gradients to flow correctly
   through a 4-bit base during backprop.
6. HF_HOME points at the same local ./hf_cache directory as Stages 3-4.
7. DPOConfig/DPOTrainer kwargs are now built defensively (inspecting
   the installed trl version's actual signature) rather than assuming
   fixed argument names — trl and transformers have renamed several of
   these (evaluation_strategy -> eval_strategy, tokenizer ->
   processing_class) across releases, and Stage 4 already hit one such
   version mismatch.

Run:
    python 05_rlhf_dpo_train.py
"""

import os

# ── Local cache directory setup — MUST happen before importing transformers ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_HF_CACHE = os.path.join(SCRIPT_DIR, "hf_cache")
os.makedirs(LOCAL_HF_CACHE, exist_ok=True)
os.environ.setdefault("HF_HOME", LOCAL_HF_CACHE)

# Must be set BEFORE torch initializes any CUDA context.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import inspect
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

BASE_MODEL_NAME       = "Qwen/Qwen3-14B"
SFT_ADAPTER_PATH      = "./output/sft_qwen3_14b_medical/final_adapter"
PREFERENCES_PATH      = "./data/preferences/preferences.jsonl"
OUTPUT_DIR            = "./output/dpo_qwen3_14b_medical"

MAX_PROMPT_LENGTH     = 1024
MAX_LENGTH            = 2048

# ── DPO-specific ────────────────────────────────────────────
DPO_BETA              = 0.1        # KL penalty strength vs reference model
                                    # lower = follow preferences more aggressively
                                    # higher = stay closer to the SFT model

# ── Training hyperparameters ───────────────────────────────
NUM_EPOCHS            = 3
BATCH_SIZE            = 1
GRAD_ACCUM_STEPS      = 8
LEARNING_RATE         = 5e-5      # DPO typically uses a lower LR than SFT
WARMUP_RATIO          = 0.1
LOGGING_STEPS         = 5
SAVE_STEPS            = 50
SEED                  = 42


# ─────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────

def load_policy_model():
    print(f"Loading base model (4-bit): {BASE_MODEL_NAME}")
    print(f"Attaching SFT adapter as the trainable DPO policy: {SFT_ADAPTER_PATH}")
    print(f"Hugging Face cache directory (HF_HOME): {os.environ.get('HF_HOME')}\n")

    if not Path(SFT_ADAPTER_PATH).exists():
        raise FileNotFoundError(
            f"Adapter not found at {SFT_ADAPTER_PATH}. "
            "Did Stage 3 (03_sft_train_qwen3_14b.py) finish and save "
            "final_adapter/?"
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
        device_map={"": 0},   # pin to GPU 0, same as Stages 3-4
    )
    base_model.config.use_cache = False  # required alongside gradient checkpointing

    # Same prep step as Stage 3: sets up gradient checkpointing hooks and
    # input-grad flags so backprop actually works through the 4-bit base.
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    # Attach the Stage 3 SFT adapter as a *trainable* adapter — no merge.
    # This IS the DPO policy: we keep fine-tuning these same LoRA weights
    # further, now against the preference objective instead of NLL loss.
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=True)

    return model, tokenizer


def prepare_preference_dataset():
    print(f"Loading preference pairs from: {PREFERENCES_PATH}")
    ds = load_dataset("json", data_files=PREFERENCES_PATH, split="train")

    print(f"  Total preference pairs: {len(ds)}")
    if len(ds) < 10:
        print(
            "⚠️  Warning: very few preference pairs collected. "
            "Run 04_collect_human_preferences.py more to gather at least "
            "50-100 pairs for meaningful DPO training."
        )

    split = ds.train_test_split(test_size=0.1, seed=SEED)
    return split["train"], split["test"]


# ─────────────────────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────────────────────

def train_dpo():
    print("=" * 60)
    print("  STAGE 5 — RLHF via DPO (Qwen3-14B)")
    print("=" * 60)

    model, tokenizer = load_policy_model()
    train_ds, eval_ds = prepare_preference_dataset()

    # ── Build DPOConfig kwargs defensively ──────────────────────────────
    # Same reasoning as Stage 3's SFTConfig handling: trl's DPOConfig
    # inherits from transformers' TrainingArguments, which has renamed
    # fields across releases (e.g. evaluation_strategy -> eval_strategy).
    # Inspect the installed version's real signature instead of assuming.
    dpo_config_params = set(inspect.signature(DPOConfig.__init__).parameters)

    def add_if_supported(kwargs_dict, name, value):
        if name in dpo_config_params:
            kwargs_dict[name] = value
        else:
            print(f"  (skipping unsupported DPOConfig param: {name})")

    config_kwargs = dict(
        output_dir=OUTPUT_DIR,
        beta=DPO_BETA,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=SAVE_STEPS,
        save_total_limit=3,
        bf16=True,
        optim="paged_adamw_8bit",
        max_prompt_length=MAX_PROMPT_LENGTH,
        report_to="tensorboard",
        seed=SEED,
    )

    # eval_strategy vs the older evaluation_strategy name.
    if "eval_strategy" in dpo_config_params:
        config_kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in dpo_config_params:
        config_kwargs["evaluation_strategy"] = "steps"

    # max_length vs max_seq_length, just in case.
    if "max_length" in dpo_config_params:
        config_kwargs["max_length"] = MAX_LENGTH
    elif "max_seq_length" in dpo_config_params:
        config_kwargs["max_seq_length"] = MAX_LENGTH

    dpo_config = DPOConfig(**{k: v for k, v in config_kwargs.items() if k in dpo_config_params})

    # ── Build DPOTrainer kwargs defensively ─────────────────────────────
    trainer_params = set(inspect.signature(DPOTrainer.__init__).parameters)

    trainer_kwargs = dict(
        model=model,
        ref_model=None,           # None → TRL uses this same model with
                                   # adapters disabled as the reference,
                                   # instead of loading a second full copy.
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        # peft_config intentionally omitted: `model` is already a
        # PeftModel with a trainable adapter attached, so DPOTrainer
        # trains that adapter directly rather than wrapping a new one.
    )

    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        print("  (warning: neither 'processing_class' nor 'tokenizer' found "
              "on DPOTrainer — relying on model's default tokenizer handling)")

    trainer = DPOTrainer(**trainer_kwargs)

    print("\nStarting DPO training ...\n")
    trainer.train()

    print("\nSaving final RLHF-aligned adapter ...")
    final_path = os.path.join(OUTPUT_DIR, "final_dpo_adapter")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"\n✅  DPO/RLHF model adapter saved to: {final_path}")
    print("Stage 5 complete. Proceed to 06_final_inference.py")

    return final_path


if __name__ == "__main__":
    train_dpo()