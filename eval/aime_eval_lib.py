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
try:
    # When executed as a package: python -m eval.aime_eval_lib
    from .grader import math_equal
except Exception:
    # When executed as a standalone script: python eval/aime_eval_lib.py
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from grader import math_equal

# Prompting for instruct models
SYSTEM_PROMPT = (
    "You are a helpful math assistant. You should think step-by-step. "
    "Respond with only the final numeric answer in \\boxed{NNN}."
)

# ---------- AIME answer parsing ----------
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
LAST_INT_RE = re.compile(r"(-?\d+)")
THREE_DIGIT_RE = re.compile(r"\b(\d{1,3})\b")

CACHE_DIR = f"{os.getcwd()}/cache"

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

def _load_vllm_generator(model_path: str, base_model: Optional[str]):
    """
    Returns generate_batch(prompts: List[str], seed: int) -> List[str]
    Detects if model_path is a LoRA adapter (presence of adapter_config.json) and uses vLLM.
    """
    import os
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if is_lora and base_model is None:
        raise ValueError("LoRA adapter detected; please pass --base_model (e.g., Qwen/Qwen2.5-Math-7B).")

    tok_model_path = base_model if is_lora else model_path
    tok = AutoTokenizer.from_pretrained(tok_model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if is_lora:
        llm = LLM(model=base_model, tokenizer=tok, dtype=base_model.dtype, enable_lora=True)
        try:
            llm.load_lora_adapter("adapter", model_path)
        except AttributeError:
            try:
                llm.add_lora_adapter("adapter", model_path)
            except Exception as e:
                raise RuntimeError(f"Unable to load LoRA adapter with vLLM: {e}")
        try:
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest("adapter", adapter_id=1)
        except Exception as e:
            raise RuntimeError(f"vLLM LoRARequest import failed. Please update vLLM. Error: {e}")
    else:
        llm = LLM(model=model_path, tokenizer=tok, dtype=base_model.dtype, seed=seed)
        lora_request = None

    def generate_batch(
        prompts: List[str],
        seed: int,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_new_tokens: int = 1024,
        gen_bs: int = 8,
    ) -> List[str]:
        # Build chat-formatted prompts
        texts: List[str] = []
        for p in prompts:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{p}\n\nReturn only the final answer as \\boxed{{NNN}}."},
            ]
            texts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            n=1,
            seed=seed,
        )

        outputs: List[str] = []

        for i in tqdm(range(0, len(texts), gen_bs), desc="prompts", leave=False):
            chunk = texts[i : i + gen_bs]
            if lora_request is not None:
                results = llm.generate(chunk, sampling_params, lora_request=lora_request)
            else:
                results = llm.generate(chunk, sampling_params)
            for r in results:
                outputs.append(r.outputs[0].text)

        return outputs

    return generate_batch

# ---------- Reference-style extraction and matching helpers ----------
def is_completely_wrapped_by_text(input_string: str) -> Optional[str]:
    pattern = r'^\\text{(.*)}$'
    match = re.match(pattern, input_string)
    if match:
        extracted_content = match.group(1)
        extracted_content = extracted_content.replace("(", "").replace(")", "").replace(",", "")
        return extracted_content
    return None


def math_answer_cleaning(answer: str) -> str:
    extracted_content = is_completely_wrapped_by_text(answer)
    answer = extracted_content if extracted_content else answer

    answer = answer.replace(",\\!", "").replace("{,}", "").replace("\\$", "")
    answer = answer.replace("dfrac{", "frac{").replace("tfrac{", "frac{")
    answer = answer.replace("^\\circ", "").replace("^{\\circ}", "")
    answer = answer.replace("\\quad", "")
    answer = re.sub(r'\\,\\text\{.*?\}', '', answer)
    answer = re.sub(r'\\text\{.*?\}', '', answer)
    answer = re.sub(r'(\s\^\{-\d+\})', '', answer)
    answer = answer.replace(" ", "")
    answer = answer.replace("\n", "").replace("\\n", "")
    answer = re.sub(r'([+-]?\d*\.?\d+)[\\]times10\^\{([+-]?\d+)\}', r'\1e\2', answer)
    answer = re.sub(r'([+-]?\d*\.?\d+)[\\]times10\^([+-]?\d+)', r'\1e\2', answer)
    answer = re.sub(r'(\d+)\^\{(\d+)\}', r'\1^\2', answer)
    answer = re.sub(r"10\^\{(-?\d+)\}", r"1e\1", answer)
    answer = answer.replace(",", "").lower()
    if answer.endswith("\\"):
        answer = answer[:-1]
    func_pattern = r'^[a-zA-Z_]\w*\([a-zA-Z_]\w*\)$'
    if "=" in answer and (re.match(func_pattern, answer.split("=")[0]) or len(answer.split("=")[0]) <= 3):
        answer = answer.split("=", 1)[1]
    return answer


def round_number(answer: str) -> str:
    def _is_float(string: str) -> bool:
        try:
            float(string)
            return True
        except Exception:
            return False

    if _is_float(answer) and float(answer) < 1:
        return f"{float(answer):.2g}"
    return answer


def calculate_numbers(input_string: str):
    try:
        return eval(input_string)
    except Exception:
        return None


def is_equal_after_calculation(extracted_answer: str, gold: str) -> bool:
    gold = re.sub(r'\\frac{(.*?)}{(.*?)}', r'(\1/\2)', gold)
    extracted_answer = re.sub(r'\\frac{(.*?)}{(.*?)}', r'(\1/\2)', extracted_answer)
    gold_result = calculate_numbers(gold)
    extracted_result = calculate_numbers(extracted_answer)
    return (gold_result is not None) and (extracted_result is not None) and (gold_result == extracted_result)


def check_after_fraction_mapping(extracted_answer: str, gold: str) -> bool:
    return re.sub(r'\\frac{(.*?)}{(.*?)}', r'\1/\2', extracted_answer) == re.sub(r'\\frac{(.*?)}{(.*?)}', r'\1/\2', gold)


# ---------- AIME24 Avg@N ----------
def run_aime24_avgN(model_path: str, out_dir: str, n_seeds: List[int] = [121, 131, 141, 151, 161, 171, 181, 191], base_model: Optional[str] = None) -> float:
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train", cache_dir=CACHE_DIR)
    problems = [r["problem"] for r in ds]
    answers  = [f'{int(r["answer"]):03d}' for r in ds]

    gen = _load_vllm_generator(model_path, base_model)

    # reference-style patterns
    pattern1_re = re.compile(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", re.DOTALL)
    pattern2_re = re.compile(r"\*\*(.*?)\*\*", re.DOTALL)
    pattern3_re = re.compile(r"\\\[\n(.*?)\n\\\]", re.DOTALL)
    pattern4_re = re.compile(r'is \\\((.*?)\\\)')
    pattern5_re = re.compile(r"\\\[\\n(.*?)\\n\\\]", re.DOTALL)

    per_seed_acc: List[float] = []
    per_seed_matrix: List[List[int]] = []
    for seed in tqdm(n_seeds, desc="seeds"):
        preds = gen(problems, seed=seed, temperature=0.6, top_p=0.95, max_new_tokens=1024, gen_bs=8)
        correct_vec: List[int] = []
        for p, g in zip(preds, answers):
            m1 = pattern1_re.findall(p)
            m2 = pattern2_re.findall(p)
            m3 = pattern3_re.findall(p)
            m4 = pattern4_re.findall(p)
            m5 = pattern5_re.findall(p)
            if len(m1) >= 1:
                extracted = m1[-1]
            elif len(m2) >= 1:
                extracted = m2[-1]
            elif len(m3) >= 1:
                extracted = m3[-1]
            elif len(m4) >= 1:
                extracted = m4[-1]
            elif len(m5) >= 1:
                extracted = m5[-1]
            else:
                extracted = None

            if extracted is None:
                correct_vec.append(0)
                continue

            pred_clean = math_answer_cleaning(str(extracted))
            gold_clean = math_answer_cleaning(str(g))

            is_correct = False
            if math_equal(pred_clean, gold_clean):
                is_correct = True
            elif round_number(pred_clean) == round_number(gold_clean):
                is_correct = True
            elif is_equal_after_calculation(pred_clean, gold_clean):
                is_correct = True
            elif check_after_fraction_mapping(pred_clean, gold_clean):
                is_correct = True

            correct_vec.append(1 if is_correct else 0)

        per_seed_matrix.append(correct_vec)
        per_seed_acc.append(100.0 * sum(correct_vec) / len(correct_vec))

    avgN = float(np.mean(per_seed_acc))
    # Persist results
    with open(os.path.join(out_dir, "aime24_summary.json"), "w") as f:
        json.dump({"avg_at_n": avgN, "n": len(n_seeds), "per_seed": per_seed_acc}, f, indent=2)
    with open(os.path.join(out_dir, "aime24_correct_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["problem_idx"] + [f"seed_{i}" for i in range(n_seeds)])
        for i in range(len(problems)):
            w.writerow([i] + [per_seed_matrix[s][i] for s in range(len(n_seeds))])

    print(f"[AIME24] Avg@{len(n_seeds)}: {avgN:.2f}%  →  {out_dir}")
    return avgN

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", type=str, default="121 131 141 151 161 171 181 191")
    ap.add_argument("--base_model", default=None, help="Required if model_path is a LoRA adapter dir.")
    args = ap.parse_args()

    n_seeds = [int(seed) for seed in args.seeds.split(" ")]
    run_aime24_avgN(args.model_path, args.out_dir, n_seeds=n_seeds, base_model=args.base_model)
