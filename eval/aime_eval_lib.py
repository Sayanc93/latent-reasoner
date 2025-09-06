#!/usr/bin/env python3
"""
AIME-24 Avg@64 evaluator that works with either:
  - a full HF CausalLM checkpoint, or
  - a LoRA/PEFT adapter directory (requires --base_model).

Usage (standalone):
  python -m aime_eval_lib \
      --model_path PATH/TO/CHECKPOINT \
      --out_dir OUT/RESULTS \
      --base_model Qwen/Qwen2.5-3B-Instruct   # needed if model_path is a LoRA adapter
      --n_seeds 64

Notes
- Sampling matches AceReason settings: T=0.6, top_p=0.95, max_new_tokens=32768.
- Metric is Avg@64 on AIME 2024 (30 problems).
"""

import argparse
import os
import re
import json
import csv
import numpy as np
from typing import List, Optional
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Optional

# ---------- AIME answer parsing ----------
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
LAST_INT_RE = re.compile(r"(-?\d+)")
THREE_DIGIT_RE = re.compile(r"\b(\d{1,3})\b")

def _norm_aime(x: str) -> Optional[str]:
    if x is None: return None
    s = str(x)
    m = BOXED_RE.search(s)
    if m:
        cand = m.group(1)
    else:
        ints = LAST_INT_RE.findall(s)
        cand = ints[-1] if ints else (THREE_DIGIT_RE.search(s).group(1) if THREE_DIGIT_RE.search(s) else None)
    if cand is None: return None
    try:
        v = int(cand)
        return f"{v:03d}" if 0 <= v <= 999 else None
    except ValueError:
        return None

def _match(pred: str, gold: str) -> bool:
    p = _norm_aime(pred); g = _norm_aime(gold)
    return p is not None and g is not None and p == g

# ---------- Model loading (full or LoRA) ----------
def _load_hf_generator(model_path: str, base_model: Optional[str]):
    """
    Returns generate_batch(prompts: List[str], seed: int) -> List[str]
    Detects if model_path is a LoRA adapter (presence of adapter_config.json).
    """
    import os
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if is_lora and base_model is None:
        raise ValueError("LoRA adapter detected; please pass --base_model (e.g., Qwen/Qwen2.5-Math-7B).")

    tok_model_path = base_model if is_lora else model_path
    tok = AutoTokenizer.from_pretrained(tok_model_path, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if is_lora:
        from peft import PeftModel, PeftConfig
        base = AutoModelForCausalLM.from_pretrained(
            base_model, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
            trust_remote_code=True, 
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base, model_path)
        # Optionally merge for a tiny generation speed boost:
        # model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            device_map="auto"
        )

    def generate_batch(prompts: List[str], seed: int, temperature=0.6, top_p=0.95, max_new_tokens=32768):
        g = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
        batch = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **batch,
            do_sample=True, 
            temperature=temperature, 
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            eos_token_id=tok.eos_token_id, 
            pad_token_id=tok.pad_token_id,
            generator=g, 
            use_cache=True
        )
        gens = []
        prompt_len = batch["input_ids"].shape[1]
        for i in range(out.shape[0]):
            gen_ids = out[i, prompt_len:]
            gens.append(tok.decode(gen_ids, skip_special_tokens=True))
        return gens

    return generate_batch

# ---------- AIME24 Avg@N ----------
def run_aime24_avgN(model_path: str, out_dir: str, n_seeds: int = 64, base_model: Optional[str] = None) -> float:
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset("HuggingFaceH4/aime_2024")["train"]  # 30 problems, fields include "problem", "answer"
    problems = [r["problem"] for r in ds]
    answers  = [f'{int(r["answer"]):03d}' for r in ds]

    gen = _load_hf_generator(model_path, base_model)

    per_seed_acc = []
    per_seed_matrix = []
    for seed in range(n_seeds):
        preds = gen(problems, seed=seed, temperature=0.6, top_p=0.95, max_new_tokens=32768)
        correct = [1 if _match(p, g) else 0 for p, g in zip(preds, answers)]
        per_seed_matrix.append(correct)
        per_seed_acc.append(100.0 * sum(correct) / len(correct))

    avgN = float(np.mean(per_seed_acc))
    # Persist results
    with open(os.path.join(out_dir, "aime24_summary.json"), "w") as f:
        json.dump({"avg_at_n": avgN, "n": n_seeds, "per_seed": per_seed_acc}, f, indent=2)
    with open(os.path.join(out_dir, "aime24_correct_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["problem_idx"] + [f"seed_{i}" for i in range(n_seeds)])
        for i in range(len(problems)):
            w.writerow([i] + [per_seed_matrix[s][i] for s in range(n_seeds)])

    print(f"[AIME24] Avg@{n_seeds}: {avgN:.2f}%  →  {out_dir}")
    return avgN

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_seeds", type=int, default=64)
    ap.add_argument("--base_model", default=None, help="Required if model_path is a LoRA adapter dir.")
    args = ap.parse_args()
    run_aime24_avgN(args.model_path, args.out_dir, n_seeds=args.n_seeds, base_model=args.base_model)
