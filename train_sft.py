#!/usr/bin/env python3
"""
SFT on your v7-math (multi-response) subset with *in-between-epoch* AIME24 evals.

Key ideas:
- We pick a target number of "points per epoch" (e.g., 8). If your epoch has 2,400 optimizer steps,
  then we save/eval every 2,400/8 = 300 steps → 8 dots per epoch.
- We trigger evaluation from a TrainerCallback.on_save hook so each save gets an AIME Avg@64 score.

Usage:
  accelerate launch train_sft.py \
    --dataset_name shockroborty/acereason_v7_math_multiresponse \
    --base_model Qwen/Qwen2.5-Math-7B \
    --out_dir sft_output \
    --epochs 6 \
    --per_device_bs 1 \
    --grad_accum 16 \
    --points_per_epoch 8 \
    --bf16 \
    --lora_r 0

Outputs:
  - Checkpoints: out_dir/checkpoints/checkpoint-STEP
  - Evals:       out_dir/evals/step_{STEP}/aime24_summary.json
  - CSV of (fractional_epoch, step, avg@64): out_dir/fig6_points.csv
"""

import argparse
import os
import math
import csv
from datasets import load_dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
                          TrainerCallback)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
import torch
import time
import wandb
import numpy as np
from collections import defaultdict
from accelerate import Accelerator

CACHE_DIR = f"{os.getcwd()}/cache"

# ----------- CLI -----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default="shockroborty/acereason_v7_math_multiresponse")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-Math-7B")
    ap.add_argument("--out_dir", default=f"{os.getcwd()}/sft_output")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--per_device_bs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--lora_r", type=int, default=0, help="0 disables LoRA (full finetune).")
    ap.add_argument("--points_per_epoch", type=int, default=8, help="How many mid-epoch eval points.")
    # performance & MFU
    ap.add_argument("--torch_compile", action="store_true", help="Enable torch.compile (single GPU only).")
    ap.add_argument("--gpu_peak_tflops", type=float, default=1979.0, help="Per-GPU BF16 peak TFLOPS (H100≈989).")
    ap.add_argument("--seq_len_for_mfu", type=int, default=None, help="Override seq len used for MFU calc.")
    return ap.parse_args()
class PerfMonitor(TrainerCallback):
    """Tracks throughput and approximates MFU (Nanogpt-style: flops/token≈6*n_params).

    Notes:
    - Uses max_len (or --seq_len_for_mfu) as seq length proxy.
    - Tokens/sec computed from optimizer steps and gradient accumulation.
    - MFU is per-GPU TFLOPS / peak (default peak for H100 BF16 ~989 TFLOPS).
    """
    def __init__(self, per_device_bs: int, grad_accum: int, world_size: int, seq_len: int, gpu_peak_tflops: float):
        self.per_device_bs = per_device_bs
        self.grad_accum = max(1, grad_accum)
        self.world_size = max(1, world_size)
        self.seq_len = seq_len
        self.gpu_peak_tflops = gpu_peak_tflops
        self._t0 = None
        self._last_step = None
        self._win_time = 0.0
        self._win_tokens = 0
        self._n_params = None

    def _tokens_per_optimizer_step(self):
        return self.per_device_bs * self.grad_accum * self.world_size * self.seq_len

    def on_train_begin(self, args, state, control, **kwargs):
        self._t0 = time.perf_counter()
        self._last_step = 0
        self._win_time = 0.0
        self._win_tokens = 0

    def on_step_end(self, args, state, control, **kwargs):
        now = time.perf_counter()
        step = state.global_step
        if self._last_step is None:
            self._last_step = step
            self._t0 = now
            return
        # steps progressed since last call (normally 1)
        dstep = max(0, step - self._last_step)
        dt = max(1e-9, now - self._t0)
        self._win_time += dt
        self._win_tokens += dstep * self._tokens_per_optimizer_step()
        self._t0 = now
        self._last_step = step

    def on_log(self, args, state, control, logs=None, **kwargs):
        # initialize params lazily
        if self._n_params is None and "model" in kwargs and kwargs["model"] is not None:
            try:
                self._n_params = sum(p.numel() for p in kwargs["model"].parameters())
            except Exception:
                self._n_params = None

        # compute deltas since last log
        dt = max(1e-9, self._win_time)
        tokens = max(0, self._win_tokens)
        toks_per_s = tokens / dt if dt > 0 else 0.0

        # FLOPs per token (NanoGPT heuristic): ~6 × number of parameters
        has_params = self._n_params is not None
        flops_per_token = (6.0 * self._n_params) if has_params else 0.0

        # Cluster TFLOPS, then per-GPU and MFU
        tflops_total = (toks_per_s * flops_per_token) / 1e12
        tflops_per_gpu = tflops_total / self.world_size  # world_size is clamped ≥ 1
        mfu = tflops_per_gpu / self.gpu_peak_tflops if self.gpu_peak_tflops > 0 else 0.0


        if logs is not None:
            logs["tokens_per_sec"] = round(toks_per_s, 2)
            logs["tflops_per_gpu"] = round(tflops_per_gpu, 2)
            logs["mfu"] = round(mfu, 4)

        # reset window
        self._win_tokens = 0
        self._win_time = 0.0

def compute_steps_per_epoch(num_rows: int, per_device_bs: int, grad_accum: int, world_size: int) -> int:
    effective_bsz = per_device_bs * max(1, world_size) * grad_accum
    return math.ceil(num_rows / effective_bsz)

# ----------- Main -----------
if __name__ == "__main__":
    args = parse_args()
    print(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    acc = Accelerator()
    
    if acc.is_main_process:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "latent-reasoner-sft"), 
            name=f"sft_{args.base_model}", 
            dir=args.out_dir,
            config=vars(args),
            group="latent-reasoner-sft"
        )

    # Enable TF32 on H100 for faster matmuls (safe with BF16 training)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    orig_ds = load_dataset(args.dataset_name, split="train", cache_dir=CACHE_DIR, num_proc=os.cpu_count())  # columns: input, output, prompt_hash, resp_len

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    # --- SFT hygiene: explicit pad + right padding for causal LM ---
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    # --- Minimal, transparent chat template: do NOT hide <thinking>/<answer> ---
    tok.chat_template = """{% for m in messages -%}
<|im_start|>{{ m['role'] }}
{{ m['content'] }}<|im_end|>
{%- endfor %}"""

    # Model (full or LoRA)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        cache_dir=CACHE_DIR,
        low_cpu_mem_usage=True,
        use_cache=False,
        max_position_embeddings=128000
    )

    # Long-context enablement
    if hasattr(model.config, "rope_theta"):
        model.config.rope_theta = 1_000_000          # ~128k effective
    if hasattr(model.config, "max_position_embeddings"):
        model.config.max_position_embeddings = max(getattr(model.config, "max_position_embeddings", 0), 131072)
    
    # Keep tokenizer/model in sync
    tok.model_max_length = max(tok.model_max_length, model.config.max_position_embeddings)
    max_len = tok.model_max_length 
    assert tok.model_max_length == model.config.max_position_embeddings

    print(f"Model config: {model.config}")
    
    # Only compile on single GPU to avoid incompatibilities with DDP/FSDP and device sharding
    world_size = acc.num_processes

    if args.lora_r and args.lora_r > 0:
        lora_cfg = LoraConfig(
            r=args.lora_r, 
            lora_alpha=args.lora_r*2, 
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]
        )
        model = get_peft_model(model, lora_cfg)

    # Static blend per AceReason: no per-epoch rotation, just shuffle
    dataset = orig_ds.shuffle(seed=args.seed)
    dataset = dataset.rename_columns({"input": "prompt", "output": "completion"})
    num_rows_epoch = len(dataset)
    steps_per_epoch = compute_steps_per_epoch(num_rows_epoch, args.per_device_bs, args.grad_accum, world_size)
    save_steps = max(1, steps_per_epoch // max(1, args.points_per_epoch))
    print(f"[CONFIG] steps_per_epoch≈{steps_per_epoch}  save_steps={save_steps}  (points_per_epoch={args.points_per_epoch})")

    # Prefer fused AdamW on Hopper; fall back to paged AdamW 8bit if LoRA requested
    use_bnb = bool(args.lora_r and args.lora_r > 0)
    chosen_optim = "paged_adamw_8bit" if use_bnb else "adamw_torch_fused"

    training_args = SFTConfig(
        output_dir=ckpt_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        bf16=args.bf16,
        fp16=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        torch_compile=args.torch_compile, # only single GPU
        logging_steps=10,
        save_strategy="steps",
        save_steps=save_steps,
        report_to=["wandb"],
        gradient_checkpointing=True,
        optim=chosen_optim,
        max_grad_norm=1.0,
        weight_decay=0.03,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        seed=args.seed,
        data_seed=args.seed,
        save_safetensors=True,
        packing=False,
        torch_empty_cache_steps=100,
        remove_unused_columns=False,
        run_name=f"sft_{args.base_model}_{acc.process_index}",
        use_liger_kernel=True,
        eos_token=tok.eos_token,
        max_seq_length=max_len
    )

    # Perf monitor (tokens/s, TFLOPS, MFU)
    avg_len = int(np.mean(dataset["resp_len"])) if "resp_len" in dataset.column_names else max_len
    seq_len_for_mfu = args.seq_len_for_mfu or avg_len
    perf_cb = PerfMonitor(
        per_device_bs=args.per_device_bs,
        grad_accum=args.grad_accum,
        world_size=world_size,
        seq_len=seq_len_for_mfu,
        gpu_peak_tflops=args.gpu_peak_tflops,
    )

    # --- SFTTrainer with formatting_func: render chat exactly, no hidden tags ---
    def formatting_func(examples):
        texts = []
        for p, c in zip(examples["prompt"], examples["completion"]):
            msgs = [{"role":"user","content": p},
                    {"role":"assistant","content": c}]
            txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            texts.append(txt)
        return texts


    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
        dataset_num_proc=os.cpu_count(),
        callbacks=[perf_cb],
    )

    trainer.train(resume_from_checkpoint=True)
