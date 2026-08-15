# Fine-Tuning-Qwen-3-35-billions-
This pipeline fine-tunes Qwen3-14B into a medical reasoning assistant by first teaching it step-by-step reasoning via supervised fine-tuning (QLoRA) on a medical chain-of-thought dataset, then aligning it to human preferences using DPO on preference pairs collected from a human picking the best of several model-generated answers
