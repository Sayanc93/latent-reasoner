#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GSPO/GRPO + Coconut-style Latent Steps on nvidia/AceReason-Math dataset
-------------------------------------------------------------------
"""

import argparse
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from trl import GRPOTrainer, GRPOConfig  # swap with GSPO trainer if available

from sampling import sample_from_logits
from eval.evaluate_aime import math_answer_cleaning, round_number, is_equal_after_calculation, check_after_fraction_mapping
from eval.tools.grader import math_equal

CACHE_DIR = f"{os.getcwd()}/cache"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---------------------------- Special tokens ---------------------------- #

THINK_OPEN   = "<think>"
THINK_CLOSE  = "</think>"
LATENT_OPEN  = "<latent>"
LATENT_CLOSE = "</latent>"

SYSTEM_PROMPT = """<|im_start|>system
You are a helpful and harmless assistant. You should think step-by-step.
<|im_end|>
<|im_start|>user
{question}

Please place your final answer inside \\boxed{{}}.
<|im_end|>
<|im_start|>assistant
<think>"""


# ---------------------------- Tokenizer helpers ---------------------------- #

def add_special_tokens(tokenizer, model: PreTrainedModel) -> Dict[str, int]:
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [THINK_OPEN, THINK_CLOSE, LATENT_OPEN, LATENT_CLOSE]}
    )
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
    ids = {
        "latent_open": tokenizer.convert_tokens_to_ids(LATENT_OPEN),
        "latent_close": tokenizer.convert_tokens_to_ids(LATENT_CLOSE),
        "think_open": tokenizer.convert_tokens_to_ids(THINK_OPEN),
        "think_close": tokenizer.convert_tokens_to_ids(THINK_CLOSE)
    }
    for k, v in ids.items():
        assert v is not None and v >= 0, f"Missing special token id for {k}"
    return ids

# ---------------------------- Latent Generator ---------------------------- #

class LatentReasoningGenerator:
    """
    Coconut-style latent pondering:
      - Opening a latent block is done by *emitting* `<latent>` visibly.
      - Inside the block, we take silent latent steps by feeding the last hidden state as `inputs_embeds`.
      - At each latent step, we gate CONTINUE vs CLOSE by looking only at logits over {<latent>, </latent>}.
        CONTINUE does another silent step; CLOSE emits `</latent>` visibly (+ appends to input_ids) and exits block.
      - No visible tokens are emitted between the tags.
    We log visible and latent events and count both for compute-aware rewards.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        special_ids: Dict[str, int],
        temperature: float = 1.0,
        top_p: float = 0.0,
        latent_max_per_block: int = 8,
        max_visible_tokens: int = 256,
    ):
        self.model = model
        self.tok = tokenizer
        self.ids = special_ids
        self.temp = temperature
        self.top_p = top_p
        self.latent_cap = int(latent_max_per_block)
        self.max_vis = int(max_visible_tokens)

    @torch.no_grad()
    def generate_one(self, prompt: str) -> Dict[str, Any]:
        self.model.eval()
        device = self.model.device
        enc = self.tok(prompt, return_tensors="pt").to(device)

        input_ids = enc["input_ids"]
        eos_id = self.tok.eos_token_id

        text_out: List[int] = []
        trace: List[Dict[str, Any]] = []
        n_visible = 0

        past = None
        last_h = None

        while n_visible < self.max_vis:
            # Forward for next visible token
            out = self.model(
                input_ids=input_ids,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past,
            )
            past = out.past_key_values
            last_h = out.hidden_states[-1][:, -1:, :]
            logits = out.logits[:, -1, :]

            # Sample next visible token
            token_id, logp_token = sample_from_logits(logits[0], temperature=self.temp, top_p=self.top_p)

            # Record + append visible token
            text_out.append(token_id)
            trace.append({"type": "visible", "token_id": token_id, "logp": logp_token})
            n_visible += 1

            # Append to ids/mask
            token_tensor = torch.tensor([[token_id]], device=device)
            input_ids = torch.cat([input_ids, token_tensor], dim=1)

            # If we opened a latent block, run silent latent steps
            if token_id == self.ids["latent_open"]:
                steps = 0
                while steps < self.latent_cap:
                    # One silent latent step (no visible token)
                    out_lat = self.model(
                        inputs_embeds=last_h,  # feed last hidden as embedding
                        past_key_values=past,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    past = out_lat.past_key_values
                    last_h = out_lat.hidden_states[-1][:, -1:, :]
                    logits_lat = out_lat.logits[:, -1, :]

                    token_id, logp_token = sample_from_logits(
                        logits_lat[0], temperature=self.temp, top_p=0.0
                    )

                    if token_id == self.ids["latent_close"] or (eos_id is not None and token_id == eos_id):
                        # CLOSE latent: emit </latent> visibly and append to input_ids
                        trace.append({"type": "visible", "action": "close", "logp": logp_token})
                        text_out.append(self.ids["latent_close"])
                        n_visible += 1
                        token_tensor = torch.tensor([[self.ids["latent_close"]]], device=device)
                        input_ids = torch.cat([input_ids, token_tensor], dim=1)
                        # Exit latent block
                        break

                    trace.append({"type": "latent", "action": "continue", "logp": logp_token})


            # Stop on EOS
            if eos_id is not None and token_id == eos_id:
                break

        text = self.tok.decode(text_out, skip_special_tokens=False)
        n_latent = sum(1 for ev in trace if ev["type"] == "latent")
        n_visible = sum(1 for ev in trace if ev["type"] == "visible")
        return {"text": text, "trace": trace, "stats": {"n_visible": n_visible, "n_latent": n_latent}}

    @torch.no_grad()
    def generate_batch(self, prompts: List[str]) -> Tuple[List[str], List[List[int]], List[Dict[str,int]]]:
        outs, ids, stats = [], [], []
        for p in prompts:
            system_prompt_prefix_text = SYSTEM_PROMPT.format(question=p)
            out = self.generate_one(system_prompt_prefix_text)
            txt, tr = out["text"], out["trace"]
            outs.append(txt)
            ids.append(self.tok(txt, add_special_tokens=False).input_ids)
            stats.append(out["stats"])
        return outs, ids, stats

# ---------------------------- TRL GRPO Trainer wrapper ---------------------------- #
class LatentReasoningTrainer(GRPOTrainer):
    """
    Wrap TRL's GRPOTrainer to:
      - use Coconut latent generator for completions
      - compute compute-aware, correctness-based rewards
    """
    def __init__(
            self,
            *args,
            latent_generator: LatentReasoningGenerator,
            lam_visible: float = 1e-4,
            lam_latent: float = 2e-4,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.latent_gen = latent_generator
        self.lam_visible = float(lam_visible)
        self.lam_latent = float(lam_latent)
        self._latest_latent_stats: List[Dict[str, int]] = []

     # 1) Use the latent generator for on-policy sampling
    def generate_completions(self, prompts, **gen_kwargs):
        # TRL expects (completions, completion_ids)
        completions, completion_ids, stats = self.latent_gen.generate_batch(prompts)
        self._latest_latent_stats = stats
        return completions, completion_ids

# ---------------------------- Reward functions ---------------------------- #
# Format check regex for <think> ... </think>
THINK_BLOCK = re.compile(r"(?is)<think>\s*(.*?)\s*</think>")

# Extraction patterns for final boxed answer (matches AIME-style formats)
_BOXED_P1 = re.compile(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", re.DOTALL)
_BOXED_P2 = re.compile(r"\*\*(.*?)\*\*", re.DOTALL)
_BOXED_P3 = re.compile(r"\\\[\n(.*?)\n\\\]", re.DOTALL)
_BOXED_P4 = re.compile(r"is \\\((.*?)\\\)")
_BOXED_P5 = re.compile(r"\\\[\\n(.*?)\\n\\\]")

def extract_final_answer(text: str) -> Optional[str]:
    for pattern in (_BOXED_P1, _BOXED_P2, _BOXED_P3, _BOXED_P4, _BOXED_P5):
        match = pattern.search(text)
        if match:
            return match[-1]
    return None

def reward_format(*, completions=None, **kwargs):
    # +0.2 if both <think>...</think> and pattern matched answer blocks exist and are non-empty
    scores = []
    for txt in completions:
        has_think_block = THINK_BLOCK.search(txt) is not None
        has_boxed_block = extract_final_answer(txt) is not None
        scores.append(0.5 if has_think_block and has_boxed_block else 0.0)
    return scores

def reward_accuracy(*, completions=None, answer=None, **kwargs):
    # +2.0 if numeric answer matches gold (within tolerance)
    golds = answer or [None] * len(completions)
    scores = []
    for txt, gold in zip(completions, golds):
        pred = extract_final_answer(txt)
        if pred is None or gold is None:
            scores.append(0.0)
            continue
        pred = math_answer_cleaning(pred)
        gold = math_answer_cleaning(gold)
       
        ok = (
            math_equal(pred, gold) or
            round_number(pred) == round_number(gold) or
            is_equal_after_calculation(pred, gold) or
            check_after_fraction_mapping(pred, gold)
        )
        scores.append(2.0 if ok else 0.0)
    return scores

def reward_compute_penalty(*, completions=None, latent_stats=None, lam_visible=1e-4, lam_latent=2e-4, **kwargs):
    # -lam_visible * n_visible - lam_latent * n_latent
    stats = latent_stats or [{"n_visible": 0, "n_latent": 0} for _ in completions]
    scores = []
    for st in stats:
        r = -lam_visible * float(st.get("n_visible", 0)) - lam_latent * float(st.get("n_latent", 0))
        scores.append(r)
    return scores

# ---------------------------- CLI / main ---------------------------- #
@dataclass
class Args:
    model_id: str
    output_dir: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: int
    num_generations: int
    latent_max_per_block: int
    max_visible_tokens: int
    kl_beta: float
    clip_eps: float
    temperature: float
    top_p: float
    seed: int
    algo: str  # "grpo" or "gspo"

def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--output_dir", type=str, default="output")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-6)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--num_generations", type=int, default=16)
    p.add_argument("--latent_max_per_block", type=int, default=8)
    p.add_argument("--max_visible_tokens", type=int, default=256)
    p.add_argument("--kl_beta", type=float, default=0.02)
    p.add_argument("--clip_eps", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--algo", type=str, default="grpo", choices=["grpo", "gspo"])
    a = p.parse_args()
    return Args(**vars(a))

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_id, cache_dir=CACHE_DIR)
    # ensure left padding for TRL GRPO (recommended in docs)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        cache_dir=CACHE_DIR, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto"
    )

    special_ids = add_special_tokens(tok, model)
    train_ds = build_gsm8k_split(tok, split="train")

    # Optional KL ref (frozen snapshot)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        cache_dir=CACHE_DIR, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto"
    )

    assert model.device == ref_model.device
    assert DEVICE == model.device
    assert DEVICE == ref_model.device

    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    latent_gen = LatentReasoningGenerator(
        model=model,
        tokenizer=tok,
        special_ids=special_ids,
        temperature=args.temperature,
        top_p=args.top_p,
        latent_max_per_block=args.latent_max_per_block,
        max_visible_tokens=args.max_visible_tokens,
    )

    cfg = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        num_generations=args.num_generations,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epsilon=args.clip_eps,
        epsilon_high=0.28,
        beta=args.kl_beta,             # KL to ref (optional in TRL; 0.0 by default per docs)
        scale_rewards=None,
        loss_type="dr_grpo",
        mask_truncated_completions=True,
        importance_sampling_level="token" if args.algo == "grpo" else "sequence",
    )

    trainer = LatentReasoningTrainer(
        model=model,
        processing_class=tok,
        args=cfg,
        train_dataset=train_ds,
        latent_generator=latent_gen,
        lam_visible=1e-4,
        lam_latent=2e-4,
        # NOTE: We’re using GRPO’s internal ref/kl if beta>0 (per TRL docs).
        # If you want an explicit ref model snapshot, set cfg.sync_ref_model=True and related knobs.
        reward_funcs=[
            reward_format,
            reward_accuracy,
            reward_compute_penalty,
        ]
    )

    # Train (use whichever entrypoint your TRL exposes)
    trainer.train()
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
