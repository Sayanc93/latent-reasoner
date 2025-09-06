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
    --base_model Qwen/Qwen2.5-3B-Instruct \
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
import json
import csv
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
                          default_data_collator, TrainerCallback)
from peft import LoraConfig, get_peft_model
import torch
import time

CACHE_DIR = f"{os.getcwd()}/cache"

# ----------- CLI -----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default="shockroborty/acereason_v7_math_multiresponse")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out_dir", default=f"{os.getcwd()}/sft_output")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--per_device_bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--lora_r", type=int, default=0, help="0 disables LoRA (full finetune).")
    ap.add_argument("--points_per_epoch", type=int, default=8, help="How many mid-epoch eval points.")
    ap.add_argument("--n_seeds", type=int, default=64, help="Avg@N seed number")
    # performance & MFU
    ap.add_argument("--torch_compile", action="store_true", help="Enable torch.compile (single GPU only).")
    ap.add_argument("--gpu_peak_tflops", type=float, default=989.0, help="Per-GPU BF16 peak TFLOPS (H100≈989).")
    ap.add_argument("--seq_len_for_mfu", type=int, default=None, help="Override seq len used for MFU calc.")
    return ap.parse_args()

# ----------- Data → chat format -----------
SYSTEM_HINT = None  # keep None; your SFT rows already include reasoning style in outputs
def build_chat(tokenizer, user_text: str, assistant_text: str, max_len: int):
    messages = []
    if SYSTEM_HINT:
        messages.append({"role":"system","content": SYSTEM_HINT})
    messages.append({"role":"user","content": user_text})
    messages.append({"role":"assistant","content": assistant_text})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # toks = tokenizer(text, truncation=True, max_length=max_len)
    toks = tokenizer(text, truncation=True)
    input_ids = toks["input_ids"]

    # Prefix up to the start of assistant (includes assistant preamble for clean boundary)
    messages_u = []
    if SYSTEM_HINT:
        messages_u.append({"role":"system","content": SYSTEM_HINT})
    messages_u.append({"role":"user","content": user_text})
    user_only = tokenizer.apply_chat_template(messages_u, tokenize=False, add_generation_prompt=True)
    # user_prefix = tokenizer(user_only, truncation=True, max_length=max_len)["input_ids"]
    user_prefix = tokenizer(user_only, truncation=True)["input_ids"]

    labels = [-100] * len(input_ids)
    labels[len(user_prefix):] = input_ids[len(user_prefix):]

    return {"input_ids": input_ids, "attention_mask": toks["attention_mask"], "labels": labels}

def make_collate_build_chat(tokenizer, max_len: int):
    def collate_fn(batch):
        feats = [build_chat(tokenizer, ex["input"], ex["output"], max_len=max_len) for ex in batch]
        base = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in feats]
        padded = tokenizer.pad(base, padding=True, return_tensors="pt")
        max_L = padded["input_ids"].shape[1]
        lab_batch = []
        for f in feats:
            lab = f["labels"]
            if len(lab) < max_L:
                lab = lab + [-100] * (max_L - len(lab))
            else:
                lab = lab[:max_L]
            lab_batch.append(lab)
        padded["labels"] = torch.tensor(lab_batch, dtype=torch.long)
        return padded
    return collate_fn

# ----------- Eval-on-save callback -----------
class EvalOnSave(TrainerCallback):
    """
    On each save (every save_steps), we:
      1) detect the just-written checkpoint dir
      2) run AIME24 Avg@N using aime_eval_lib.run_aime24_avgN
      3) append (global_step, fractional_epoch, avg@N) into a CSV
    """
    def __init__(self, run_dir: str, base_model: str, n_seeds: int, steps_per_epoch: int):
        super().__init__()
        self.run_dir = run_dir
        self.base_model = base_model
        self.n_seeds = n_seeds
        self.steps_per_epoch = max(1, steps_per_epoch)
        self.csv_path = os.path.join(run_dir, "aime24_points.csv")
        # init CSV header
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["global_step", "fractional_epoch", "avg_at_n"])

    def on_save(self, args, state, control, **kwargs):
        last_ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.exists(last_ckpt):
            return

        # Eval
        eval_out = os.path.join(self.run_dir, "evals", f"step_{state.global_step}")
        os.makedirs(eval_out, exist_ok=True)

        # Lazy import to avoid circular import at module load
        from eval.aime_eval_lib import run_aime24_avgN

        # If this is LoRA, last_ckpt contains adapters; pass base_model
        avgN = run_aime24_avgN(last_ckpt, eval_out, n_seeds=1, base_model=self.base_model)

        frac_epoch = state.global_step / self.steps_per_epoch
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([state.global_step, f"{frac_epoch:.3f}", f"{avgN:.2f}"])
        print(f"[FIG6] step={state.global_step}  epoch~{frac_epoch:.3f}  AIME24 Avg@{self.n_seeds}={avgN:.2f}%")


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
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Enable TF32 on H100 for faster matmuls (safe with BF16 training)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    ds_raw = load_dataset(args.dataset_name, cache_dir=CACHE_DIR)  # columns: input, output, prompt_hash, resp_len
    # Pick the train split if a DatasetDict is returned
    if isinstance(ds_raw, DatasetDict):
        dataset = ds_raw["train"]
    else:
        dataset = ds_raw
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    max_len = tok.model_max_length * 4 # for yarn rope scaling
    # def _map(ex):
    #     return build_chat(tok, ex["input"], ex["output"], max_len=max_len)

    # dataset = dataset.map(_map, remove_columns=dataset.column_names, desc="Tokenizing", num_proc=os.cpu_count(), cache_dir=CACHE_DIR)

    # Model (full or LoRA)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        cache_dir=CACHE_DIR,
        low_cpu_mem_usage=True,
    )
    # Only compile on single GPU to avoid incompatibilities with DDP/FSDP and device sharding
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.torch_compile and world_size == 1:
        try:
            model = torch.compile(model, mode="max-autotune")
            print("[PERF] torch.compile enabled (mode=max-autotune)")
        except Exception as e:
            print(f"[PERF] torch.compile disabled due to error: {e}")
    
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

    steps_per_epoch = compute_steps_per_epoch(
        num_rows=len(dataset),
        per_device_bs=args.per_device_bs,
        grad_accum=args.grad_accum,
        world_size=world_size,
    )
    save_steps = max(1, steps_per_epoch // max(1, args.points_per_epoch))
    print(f"[CONFIG] steps_per_epoch≈{steps_per_epoch}  save_steps={save_steps}  (points_per_epoch={args.points_per_epoch})")

    # Prefer fused AdamW on Hopper; fall back to paged AdamW 8bit if LoRA requested
    use_bnb = bool(args.lora_r and args.lora_r > 0)
    chosen_optim = "paged_adamw_8bit" if use_bnb else "adamw_torch_fused"

    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        bf16=args.bf16,
        fp16=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=min(50, save_steps),
        save_strategy="steps",
        save_steps=2,
        save_total_limit=1000,  # keep many mid-epoch ckpts; prune later if needed
        report_to=[],
        gradient_checkpointing=True,
        optim=chosen_optim,
        max_grad_norm=1.0,
        dataloader_num_workers=os.cpu_count(),
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    # Perf monitor (tokens/s, TFLOPS, MFU)
    seq_len_for_mfu = args.seq_len_for_mfu or max_len
    perf_cb = PerfMonitor(
        per_device_bs=args.per_device_bs,
        grad_accum=args.grad_accum,
        world_size=world_size,
        seq_len=seq_len_for_mfu,
        gpu_peak_tflops=args.gpu_peak_tflops,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tok,
        data_collator=make_collate_build_chat(tok, max_len),
        callbacks=[
            perf_cb,
            EvalOnSave(run_dir=args.out_dir,
                       base_model=args.base_model,
                       n_seeds=args.n_seeds,
                       steps_per_epoch=steps_per_epoch)
        ]
    )

    trainer.train()
