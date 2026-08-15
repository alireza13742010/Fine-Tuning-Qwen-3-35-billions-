# Medical Reasoning Assistant — Qwen3-14B (SFT + DPO / RLHF)

A single-GPU (24GB-class) end-to-end pipeline that fine-tunes **Qwen3-14B**
into a medical chain-of-thought reasoning assistant using:

1. **Supervised Fine-Tuning (SFT)** on a verified medical reasoning dataset, via **QLoRA**
2. **Human preference collection** — you pick the best of several model answers
3. **Direct Preference Optimization (DPO)** — RLHF-style alignment without a reward model or PPO
4. An **interactive Streamlit app** to chat with the final model and keep collecting feedback

> ⚠️ **Disclaimer:** This project is for educational / research purposes only.
> It is **not** a medical device and must not be used for real diagnosis,
> treatment decisions, or any clinical purpose.

---

## Table of contents

- [Pipeline overview](#pipeline-overview)
- [Hardware requirements](#hardware-requirements)
- [Installation](#installation)
- [Repository structure](#repository-structure)
- [Stage 1 — Download dataset](#stage-1--download-dataset-01_download_datasetpy)
- [Stage 2 — Preprocess](#stage-2--preprocess-02_preprocesspy)
- [Stage 3 — SFT training (QLoRA)](#stage-3--sft-training-qlora-03_sft_train_qwen3_14bpy)
- [Stage 4 — Human preference collection](#stage-4--human-preference-collection-04_collect_human_preferencespy)
- [Stage 5 — DPO / RLHF training](#stage-5--dpo--rlhf-training-05_rlhf_dpo_trainpy)
- [Stage 6 — CLI inference](#stage-6--cli-inference-06_final_inferencepy)
- [Stage 7 — Streamlit app](#stage-7--streamlit-app-07_streamlit_apppy)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Pipeline overview

```
Raw dataset (HF Hub)
        │
        ▼
[1] 01_download_dataset.py     → ./data/raw/{train,val}_raw.jsonl
        │
        ▼
[2] 02_preprocess.py           → ./data/processed/{train,val}_processed.jsonl
        │
        ▼
[3] 03_sft_train_qwen3_14b.py  → ./output/sft_qwen3_14b_medical/final_adapter
        │
        ▼
[4] 04_collect_human_preferences.py → ./data/preferences/preferences.jsonl
        │
        ▼
[5] 05_rlhf_dpo_train.py       → ./output/dpo_qwen3_14b_medical/final_dpo_adapter
        │
        ▼
[6] 06_final_inference.py  (CLI)      ─┐
[7] 07_streamlit_app.py    (Web UI)   ─┴─► chat with the final RLHF-aligned model
```

Every stage after Stage 3 depends on the output of the previous one — run
them in order the first time through.

---

## Hardware requirements

- **1x NVIDIA GPU with ≥24GB VRAM** (e.g. RTX 3090/4090, A10G, L4). The whole
  pipeline (4-bit QLoRA base + LoRA adapters) is deliberately sized around
  this budget — nothing here merges LoRA weights into the base model, because
  dequantizing a 14B model to bf16 for merging would need ~28GB, which
  doesn't fit.
- **CUDA 12.x** and a matching NVIDIA driver.
- **~30-40GB free disk space** for the base model weights, tokenizer, and
  dataset cache (all stored under `./hf_cache` by default, not your home
  directory).
- Linux or WSL2 recommended (bitsandbytes 4-bit kernels are most reliably
  supported there).

---

## Installation

```bash
# 1. Create and activate an isolated environment
conda create -n medical_rlhf python=3.10 -y
conda activate medical_rlhf

# 2. Install PyTorch with CUDA support (match this to your CUDA version —
#    check https://pytorch.org/get-started/locally/ if you're not on CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install the core libraries used across all stages
pip install "transformers>=4.51.0" \
            "datasets>=2.19.0" \
            "peft>=0.11.0" \
            "trl>=0.9.6" \
            "bitsandbytes>=0.43.0" \
            "accelerate>=0.30.0" \
            "safetensors>=0.8.0"

# 4. Streamlit app dependency (Stage 7 only)
pip install streamlit

# 5. (Optional but recommended) TensorBoard, since Stage 5 logs to it
pip install tensorboard
```

### Library roles at a glance

| Library         | Used for |
|------------------|----------|
| `transformers`   | Loading Qwen3-14B, tokenizer, chat template, `generate()` |
| `datasets`       | Loading/saving `.jsonl` at every stage, filtering, splitting |
| `peft`           | LoRA/QLoRA adapters (`LoraConfig`, `PeftModel`, `get_peft_model`) |
| `trl`            | `SFTTrainer`/`SFTConfig` (Stage 3) and `DPOTrainer`/`DPOConfig` (Stage 5) |
| `bitsandbytes`   | 4-bit (`nf4`) quantization of the 14B base model |
| `accelerate`     | Device placement / mixed-precision plumbing under the hood |
| `safetensors`    | Safe weight (de)serialization — **must be ≥0.8.0** or `diffusers`/`transformers` imports fail |
| `streamlit`      | Stage 7's chat UI |

> **Version pinning note:** `trl` and `transformers` rename constructor
> arguments fairly often (`max_seq_length` → `max_length`,
> `evaluation_strategy` → `eval_strategy`, `tokenizer` →
> `processing_class`, etc). Stages 3 and 5 already defend against this by
> inspecting the installed library's real function signature at runtime
> and only passing supported kwargs — so minor version drift shouldn't
> break training, but very old versions (`trl<0.9`) may still lack DPO
> support entirely.

---

## Repository structure

```
.
├── 01_download_dataset.py
├── 02_preprocess.py
├── 03_sft_train_qwen3_14b.py
├── 04_collect_human_preferences.py
├── 05_rlhf_dpo_train.py
├── 06_final_inference.py
├── 07_streamlit_app.py
├── hf_cache/                          # HF_HOME — model/tokenizer downloads (gitignore this)
├── data/
│   ├── raw/                           # Stage 1 output
│   ├── processed/                     # Stage 2 output
│   └── preferences/                   # Stage 4 + Stage 7 feedback output
└── output/
    ├── sft_qwen3_14b_medical/         # Stage 3 output
    └── dpo_qwen3_14b_medical/         # Stage 5 output
```

`hf_cache/`, `data/`, and `output/` are all generated at runtime — add them
to `.gitignore` rather than committing model weights or datasets.

---

## Stage 1 — Download dataset (`01_download_dataset.py`)

Downloads **`FreedomIntelligence/medical-o1-reasoning-SFT`** from the
Hugging Face Hub — a dataset of verifiable medical questions where each
`{Question, Complex_CoT, Response}` triple was generated by GPT-4o and
validated by a medical verifier (originally built to train HuatuoGPT-o1).

- `LANGUAGE = "en"` — switch to `"zh"` for the Chinese config.
- Carves out a 5% validation split (`VAL_FRACTION = 0.05`) with a fixed
  seed for reproducibility.
- Saves raw, unprocessed records to `./data/raw/{train,val}_raw.jsonl`.

```bash
python 01_download_dataset.py
```

---

## Stage 2 — Preprocess (`02_preprocess.py`)

Converts raw records into a single training string using a `<think>...</think>`
template — matching how DeepSeek-R1 / HuatuoGPT-o1-style reasoning models
are normally trained:

```
<|system|>
{system prompt}
<|user|>
{question}
<|assistant|>
<think>
{chain of thought}
</think>
{final response}
```

Also applies basic quality filtering (`is_valid`) — drops examples with
missing fields, excessively long questions/responses, or near-empty
reasoning traces (`MIN_COT_CHARS = 20`) — before writing
`./data/processed/{train,val}_processed.jsonl`.

```bash
python 02_preprocess.py
```

---

## Stage 3 — SFT training (QLoRA) (`03_sft_train_qwen3_14b.py`)

Fine-tunes **Qwen/Qwen3-14B** (dense, non-MoE) with **QLoRA**:
4-bit `nf4` quantized base weights (via `BitsAndBytesConfig`) + trainable
LoRA adapters on all attention and MLP projections
(`q/k/v/o_proj`, `gate/up/down_proj`), trained with `trl.SFTTrainer`.

Key defaults:

| Setting | Value | Notes |
|---|---|---|
| `MAX_SEQ_LENGTH` | 2048 | Raise once a run completes cleanly and VRAM headroom is confirmed |
| `LORA_RANK` / `LORA_ALPHA` | 16 / 32 | |
| `BATCH_SIZE` / `GRAD_ACCUM_STEPS` | 2 / 8 | Effective batch size 16 |
| `LEARNING_RATE` | 2e-4 | |
| Optimizer | `paged_adamw_8bit` | Memory-efficient optimizer for QLoRA |

Utility flags:

```bash
python 03_sft_train_qwen3_14b.py --meminfo   # print GPU memory, then exit
python 03_sft_train_qwen3_14b.py --inspect   # print every Linear module name, then exit
python 03_sft_train_qwen3_14b.py             # actual training run
```

Run `--inspect` at least once on a new environment/model version to confirm
`LORA_TARGET_MODULES` actually match the loaded architecture before
committing to a long training run.

Output: LoRA adapter + tokenizer saved to
`./output/sft_qwen3_14b_medical/final_adapter`.

```bash
python 03_sft_train_qwen3_14b.py
```

---

## Stage 4 — Human preference collection (`04_collect_human_preferences.py`)

Loads the Stage 3 SFT model and, for each question, generates
**4 candidate answers** at different sampling temperatures
(`TEMPERATURES = [0.3, 0.6, 0.9, 1.1]`). You read them in the terminal and
pick the **best** (and optionally the **worst**) — each choice becomes one
`{prompt, chosen, rejected}` preference pair, saved incrementally to
`./data/preferences/preferences.jsonl`.

- Uses `tokenizer.apply_chat_template(...)` with Qwen3's native
  `enable_thinking` flag, so prompting exactly matches Stage 3's training
  format (a hand-rolled prompt string would silently degrade quality).
- Pulls questions from the held-out validation set by default
  (`NUM_QUESTIONS_TO_ASK = 30`); once those run out you can type your own
  questions live.
- Safe to run multiple sessions — new pairs are **appended**, not
  overwritten, so preference data accumulates over time. Aim for at least
  50-100 pairs before running DPO.

```bash
python 04_collect_human_preferences.py
```

---

## Stage 5 — DPO / RLHF training (`05_rlhf_dpo_train.py`)

Runs **Direct Preference Optimization** on the Stage 4 preference pairs,
continuing to train the *same* SFT LoRA adapter (no merge, no fresh
adapter) directly against the preference objective.

**Why DPO instead of classic PPO-based RLHF:** classic RLHF needs three
models in memory simultaneously (policy, frozen reference, reward model)
plus a separate reward-model training stage. DPO reformulates the same
objective as a single supervised loss computed directly from preference
pairs, needing only the policy model + a reference — which this script
gets **for free** by setting `ref_model=None`: TRL temporarily disables
the LoRA adapter on the same model to compute reference log-probabilities,
avoiding a second full model copy in VRAM.

Loss (conceptually):

```
loss = -log σ( β · [ (logπ_policy(chosen) − logπ_ref(chosen))
                    − (logπ_policy(rejected) − logπ_ref(rejected)) ] )
```

- `DPO_BETA = 0.1` — lower values trust the preference data more
  aggressively; higher values stay closer to the SFT reference behavior.
- `LEARNING_RATE = 5e-5` — deliberately lower than SFT's `2e-4`, since this
  is a smaller alignment nudge on an already fine-tuned model.
- 90/10 train/eval split of the collected preference pairs.

Output: a **single** adapter containing both the SFT and DPO updates,
saved to `./output/dpo_qwen3_14b_medical/final_dpo_adapter`.

```bash
python 05_rlhf_dpo_train.py
```

---

## Stage 6 — CLI inference (`06_final_inference.py`)

Terminal chat loop against the final model: loads the 4-bit base +
**one** adapter (the combined DPO adapter if present, otherwise falls back
to the SFT-only adapter with a warning). Generates 3 candidate answers per
question, lets you pick the best one, and logs that choice to
`./data/preferences/inference_feedback.jsonl` in the same schema Stage 5
expects — so real usage keeps generating fresh DPO training data.

```bash
python 06_final_inference.py
```

---

## Stage 7 — Streamlit app (`07_streamlit_app.py`)

A proper web chat UI over the same model-loading logic as Stage 6:

- Loads the 4-bit base + DPO adapter (SFT fallback) **once**, cached across
  reruns via `st.cache_resource`.
- Streams the response token-by-token (`TextIteratorStreamer` on a
  background thread) instead of waiting for full generation.
- Splits out the `<think>...</think>` reasoning trace into a collapsible
  "💭 Reasoning trace" expander.
- Sidebar controls: temperature, top-p, max new tokens, thinking-mode
  toggle, and a "clear conversation" button.
- 👍/👎 buttons per assistant reply, logged to
  `./data/preferences/streamlit_feedback.jsonl`, so real chat sessions
  can feed back into future DPO rounds alongside Stage 4/6's data.

```bash
streamlit run 07_streamlit_app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

---

## Configuration reference

All tunables live as module-level constants near the top of each script —
no config files or CLI flags to hunt through. The most commonly adjusted
ones:

| Script | Constant | Purpose |
|---|---|---|
| `01_download_dataset.py` | `LANGUAGE` | `"en"` or `"zh"` dataset config |
| `02_preprocess.py` | `MAX_*_CHARS`, `MIN_COT_CHARS` | Quality filtering thresholds |
| `03_sft_train_qwen3_14b.py` | `MAX_SEQ_LENGTH`, `LORA_RANK`, `BATCH_SIZE` | Memory/quality trade-offs |
| `04_collect_human_preferences.py` | `NUM_CANDIDATES`, `TEMPERATURES` | Diversity of candidates to compare |
| `05_rlhf_dpo_train.py` | `DPO_BETA`, `LEARNING_RATE` | Alignment strength / stability |
| `06_final_inference.py` / `07_streamlit_app.py` | `ENABLE_THINKING`, `TEMPERATURES` | Inference behavior |

All HF downloads across every stage are cached under a project-local
`./hf_cache` directory (via `HF_HOME`) rather than `~/.cache/huggingface`,
so the whole project is self-contained and easy to relocate or clean up.

---

## Troubleshooting

- **`No package metadata was found for safetensors>=0.8.0`** — upgrade in
  the active environment: `pip install -U "safetensors>=0.8.0"`, then
  **restart the kernel/terminal** (package metadata is cached at import
  time).
- **CUDA OOM during Stage 3/5** — in order of impact: lower
  `MAX_SEQ_LENGTH`/`MAX_LENGTH`, drop `BATCH_SIZE` to 1 and raise
  `GRAD_ACCUM_STEPS`, lower `LORA_RANK`, close other GPU processes
  (`nvidia-smi`).
- **`trl`/`transformers` kwarg errors** (e.g. `unexpected keyword argument
  'evaluation_strategy'`) — Stages 3 and 5 already inspect installed
  signatures defensively, but if you still hit this, run
  `pip install -U trl transformers` to get current argument names.
- **Adapter not found errors in Stages 4/5/6/7** — confirm the previous
  stage actually completed and check `OUTPUT_DIR` wasn't changed between
  scripts; paths must match exactly.

---

## License
To access the saved weight please call us at: alirezatavakolianart@gmail.com
