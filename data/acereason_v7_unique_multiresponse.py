
#!/usr/bin/env python3
"""
Build a v7-style SFT mixture:
  1) Maximize UNIQUE PROMPTS first (dominant driver of SFT gains in AceReason).
  2) Then add a few extra responses per selected prompt.
  3) Match a response-length histogram (bins) to keep difficulty balance.

Inputs: a HF dataset with columns:
  - 'input' (prompt), 'output' (assistant), 'prompt_hash' (stable id), 'resp_len' (optional)

Usage:
  uv run data/acereason_v7_unique_multiresponse.py \
    --dataset_name https://huggingface.co/datasets/shockroborty/acereason_v7_math_precomputed \
    --out_dir data/v7_lenmix \
    --target_prompts 200000 \
    --max_responses_per_prompt 3
"""
import os
import math
import argparse
import random
from typing import Dict, List, Tuple
from collections import defaultdict
from datasets import load_dataset, Dataset, DatasetDict

CACHE_DIR = f"{os.getcwd()}/cache"

# --- guards up top ---
def _check_bins(bins, target_frac):
    assert len(bins) == len(target_frac), "bins and target_frac must align"
    assert abs(sum(target_frac) - 1.0) < 1e-6, "target_frac must sum to 1.0"
    for i in range(1, len(bins)):
        assert bins[i-1][1] <= bins[i][0], f"bins overlap or are out of order: {bins[i-1]} -> {bins[i]}"

def _bin_id(n: int, bins):
    # handle underflow
    if n < bins[0][0]:
        return 0
    for i, (lo, hi) in enumerate(bins):
        if lo <= n < hi:
            return i
    return len(bins) - 1  # overflow → last bin

def build_lenmix(
    ds,
    target_prompts: int = 2200000,
    max_responses_per_prompt: int = 6,
    bins: List[Tuple[int,int]] = ((0,2000),(2000,4000),(4000,8000),(8000,20000), (20000,40000), (40000, 10**9)),
    target_frac: List[float] = (0.10, 0.10, 0.20, 0.20, 0.35, 0.05),
    seed: int = 42,
):
    rnd = random.Random(seed)
    _check_bins(bins, target_frac)
    # ensure required columns
    required = ["input","output","prompt_hash"]
    for k in required:
        if k not in ds.column_names:
            raise ValueError(f"Missing column '{k}' in dataset. Found: {ds.column_names}")

    # precompute lengths & bins
    lens = [ int(r["resp_len"]) for r in ds ]
    bin_ids = [ _bin_id(n, bins) for n in lens ]

    # index rows by prompt
    by_prompt: Dict[str, List[int]] = defaultdict(list)
    for i,ph in enumerate(ds["prompt_hash"]):
        by_prompt[ph].append(i)

    # 1) maximize unique prompts up to target_prompts while matching target_frac on response lengths
    # build candidate prompts per length bin using their *longest* response as proxy
    prompt_best: Dict[str, Tuple[int,int]] = {}  # prompt_hash -> (best_row_id, best_len)
    for ph, rows in by_prompt.items():
        # pick the longest response row to represent this prompt for binning
        best = max(rows, key=lambda j: lens[j])
        prompt_best[ph] = (best, lens[best])

    # group prompts by the bin of their representative row
    prompts_by_bin: Dict[int, List[str]] = defaultdict(list)
    for ph,(best_row, L) in prompt_best.items():
        b = _bin_id(L, bins)
        prompts_by_bin[b].append(ph)
    for b in prompts_by_bin:
        rnd.shuffle(prompts_by_bin[b])

    chosen_prompts = []
    selected = set()
    for b, frac in enumerate(target_frac):
        want = int(target_prompts * frac)
        pool = prompts_by_bin.get(b, [])
        take = min(want, len(pool))
        chosen_prompts.extend(pool[:take])
        selected.update(pool[:take])

    # do per-bin top-ups first if any bin underfilled
    for b, frac in enumerate(target_frac):
        want = int(target_prompts * frac)
        pool = [ph for ph in prompts_by_bin.get(b, []) if ph not in selected]
        need = max(0, want - sum(1 for ph in chosen_prompts if _bin_id(lens[prompt_best[ph][0]], bins) == b))
        add = pool[:need]
        chosen_prompts.extend(add)
        selected.update(add)

    # global top-up if still short
    if len(chosen_prompts) < target_prompts:
        leftovers = [ph for ph in prompt_best.keys() if ph not in selected]
        rnd.shuffle(leftovers)
        chosen_prompts.extend(leftovers[:(target_prompts - len(chosen_prompts))])

    chosen_prompts = chosen_prompts[:target_prompts]

    # 2) select up to K responses per chosen prompt, preferring longer ones
    row_ids = []
    for ph in chosen_prompts:
        rows = by_prompt[ph]
        rows = sorted(rows, key=lambda j: lens[j], reverse=True)
        row_ids.extend(rows[:max_responses_per_prompt])
    row_ids = list(dict.fromkeys(row_ids))  # keep order, dedup safety

    # --- log composition ---
    from collections import Counter
    row_bins = Counter(_bin_id(lens[j], bins) for j in row_ids)
    print("[lenmix] unique_prompts:", len(set(chosen_prompts)))
    print("[lenmix] rows:", len(set(row_ids)))
    print("[lenmix] rows per bin:", dict(row_bins))

    return ds.select(row_ids)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", type=str, default="shockroborty/acereason_v7_math_precomputed")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--out_dir", type=str, default="data/v7_lenmix")
    ap.add_argument("--target_prompts", type=int, default=2200000)
    ap.add_argument("--max_responses_per_prompt", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ds = load_dataset(args.dataset_name, split=args.split, cache_dir=CACHE_DIR, num_proc=os.cpu_count())
    ds = ds.remove_columns([c for c in ds.column_names if c.startswith("__index_level")])  # HF parquet hygiene

    out = build_lenmix(
        ds,
        target_prompts=args.target_prompts,
        max_responses_per_prompt=args.max_responses_per_prompt,
        seed=args.seed,
    )
    # Save in a HF-friendly format
    DatasetDict({"train": out}).save_to_disk(args.out_dir)
    print(f"[lenmix] wrote {len(out)} rows to {args.out_dir}")
