
#!/usr/bin/env python3
"""
    Build a **priority-based** SFT mixture:
    1) Unique prompts (maximize coverage - include at least 1 response from each prompt)
    2) Long responses (prefer longer responses)
    3) Multiple responses per prompt (up to max_responses_per_prompt)

    Inputs: HF dataset columns
    - 'input' (prompt), 'output' (assistant), 'prompt_hash' (stable id), 'resp_len' (token count)

    Usage:
    uv run data/acereason_v7_unique_multiresponse.py \
        --dataset_name shockroborty/acereason_v7_math_precomputed \
        --out_dir cache/acereason_v7_math_unique_multiresponse \
        --max_responses_per_prompt 6
"""
import os
import argparse
import random
import numpy as np
from typing import Dict, List
from collections import defaultdict, Counter
from datasets import load_dataset, DatasetDict

CACHE_DIR = f"{os.getcwd()}/cache"

def build_lenmix_length_first(
    ds,
    max_responses_per_prompt: int = 6,
    seed: int = 42,
):
    """
    Priority-based selection (ignores target_rows):
      1. Unique prompts (maximize coverage)
      2. Long responses (prefer longer)
      3. Multiple responses per prompt (up to max_responses_per_prompt)
    """
    rnd = random.Random(seed)

    lens = np.array(ds["resp_len"], dtype=int)

    # 1) Group responses by prompt and sort each by length (descending)
    by_prompt: Dict[str, List[int]] = defaultdict(list)
    for i, ph in enumerate(ds["prompt_hash"]):
        by_prompt[ph].append(i)

    for ph, rows in by_prompt.items():
        rows.sort(key=lambda j: lens[j], reverse=True)

    # 2) Phase 1: Select longest response from each unique prompt
    selected_rows: List[int] = []
    
    # Shuffle prompts for tie-breaking, then sort by their longest response
    prompt_list = list(by_prompt.keys())
    rnd.shuffle(prompt_list)
    prompt_list.sort(key=lambda ph: lens[by_prompt[ph][0]], reverse=True)
    
    for ph in prompt_list:
        selected_rows.append(by_prompt[ph][0])

    # 3) Phase 2: Add more responses (up to max_responses_per_prompt), prioritizing by length
    pool = []
    for ph in prompt_list:
        rows = by_prompt[ph]
        # Add remaining responses (indices 1 to max_responses_per_prompt-1)
        for j in rows[1:max_responses_per_prompt]:
            pool.append((lens[j], j))
    
    # Sort pool by length descending and add all
    pool.sort(key=lambda x: x[0], reverse=True)
    selected_rows.extend([row_idx for _, row_idx in pool])

    # ---- Stats ----
    unique_prompts = len({ds[i]["prompt_hash"] for i in selected_rows})
    print("[lenmix] unique_prompts:", unique_prompts)
    print("[lenmix] rows:", len(selected_rows))

    # distribution by responses/prompt
    cnt: Counter[int] = Counter()
    seen: Dict[str, int] = defaultdict(int)
    for i in selected_rows:
        ph = ds[i]["prompt_hash"]
        seen[ph] += 1
    for ph, k in seen.items():
        cnt[k] += 1
    print("[lenmix] prompts by #responses (selected):", dict(sorted(cnt.items())))

    return ds.select(selected_rows)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", type=str, default="shockroborty/acereason_v7_math_precomputed")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--out_dir", type=str, default="data/v7_lenmix")
    ap.add_argument("--max_responses_per_prompt", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ds = load_dataset(args.dataset_name, split=args.split, cache_dir=CACHE_DIR, num_proc=os.cpu_count())
    ds = ds.remove_columns([c for c in ds.column_names if c.startswith("__index_level")])  # HF parquet hygiene

    out = build_lenmix_length_first(
        ds,
        max_responses_per_prompt=args.max_responses_per_prompt,
        seed=args.seed,
    )
    # Save in a HF-friendly format
    DatasetDict({"train": out}).save_to_disk(args.out_dir)
    print(f"[lenmix] wrote {len(out)} rows to {args.out_dir}")
