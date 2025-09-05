#!/usr/bin/env python3
"""
Fast filter to keep only prompts (by prompt_hash) that have >= N responses,
for the precomputed AceReason math dataset.

Input (HF repo_id or local dataset dir prepared by your step-1):
  Columns required: input, output, prompt_hash, resp_len

Output:
  - Saved Arrow dataset containing only rows whose prompt_hash appears at least
    --min-responses times.

Usage:
  python v7_math_multiresp_fast.py \
    --dataset-id shockroborty/acereason_v7_math_precomputed \
    --out-dir /path/to/cache/acereason_v7_math_multiresp \
    --min-responses 9
"""

from datasets import load_dataset, Dataset  # type: ignore
import pyarrow as pa  # type: ignore
import pyarrow.compute as pc  # type: ignore
import numpy as np
import argparse
import os

DEFAULT_BINS = [0, 2000, 4000, 8000, 12000, 16000, 20000, 32768]
CACHE_DIR = f"{os.getcwd()}/cache"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", default="shockroborty/acereason_v7_math_precomputed",
                    help="HF repo id OR local directory created by step-1 (contains Arrow data)")
    ap.add_argument("--out-dir", default=f"{os.getcwd()}/cache/acereason_v7_math_multiresp",
                    help="Where to save the filtered dataset")
    ap.add_argument("--min-responses", type=int, default=9,
                    help="Keep prompts with at least this many responses")
    return ap.parse_args()

def main():
    args = parse_args()

    ds = load_dataset(args.dataset_id, split="train", cache_dir=CACHE_DIR)

    required = {"input", "output", "prompt_hash", "resp_len"}
    missing = required.difference(ds.column_names)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    tbl: pa.Table = ds.data  # underlying Arrow table (zero-copy)
    n_in = tbl.num_rows

    # value_counts over prompt_hash -> StructArray with fields 'values' and 'counts'
    vc = pc.value_counts(tbl["prompt_hash"])
    values = pc.struct_field(vc, "values")
    counts = pc.struct_field(vc, "counts")

    # Select prompts with count >= min_responses
    keep_mask_counts = pc.greater_equal(counts, pa.scalar(args.min_responses))
    kept_prompt_values = pc.filter(values, keep_mask_counts)        # unique prompt_hashes to keep
    kept_counts = pc.filter(counts, keep_mask_counts)               # their counts

    # Filter the original table by membership
    keep_mask_full = pc.is_in(tbl["prompt_hash"], value_set=kept_prompt_values)
    filtered_tbl = tbl.filter(keep_mask_full)
    n_out = filtered_tbl.num_rows

    # Build output Dataset and save
    ds_out = Dataset(filtered_tbl)
    os.makedirs(args.out_dir, exist_ok=True)
    ds_out.save_to_disk(args.out_dir)

    # ---- Sanity report ----
    n_prompts_in = len(vc)                       # number of unique prompts in input
    n_prompts_out = len(kept_prompt_values)      # number of unique prompts kept
    sum_kept = pc.sum(kept_counts).as_py() if n_prompts_out > 0 else 0
    avg_per_prompt = (sum_kept / max(n_prompts_out, 1)) if n_prompts_out > 0 else 0.0

    # Responses-per-prompt histogram (only for kept prompts)
    kept_counts_np = kept_counts.to_numpy() if n_prompts_out > 0 else np.array([], dtype=np.int64)
    rvals, rfreqs = (np.unique(kept_counts_np, return_counts=True)
                     if kept_counts_np.size > 0 else (np.array([], dtype=np.int64), np.array([], dtype=np.int64)))

    # Response-length histogram on filtered rows
    resp_len_np = filtered_tbl["resp_len"].to_numpy() if n_out > 0 else np.array([], dtype=np.int32)
    len_counts, _ = np.histogram(resp_len_np, bins=DEFAULT_BINS)
    len_props = (len_counts / len_counts.sum()) if len_counts.sum() else np.zeros_like(len_counts, dtype=float)

    print("\n=== Sanity Report: multi-response math subset (FAST) ===")
    print(f"Input rows:             {n_in:,}")
    print(f"Input unique prompts:   {n_prompts_in:,}")
    print(f"Min responses kept:     {args.min_responses}")
    print(f"Output rows:            {n_out:,}")
    print(f"Output unique prompts:  {n_prompts_out:,}")
    print(f"Avg responses/prompt:   {avg_per_prompt:.3f}")

    print("\nResponses-per-prompt (kept prompts):")
    max_lines = 10
    for k, c in zip(rvals[:max_lines], rfreqs[:max_lines]):
        print(f"  {int(k)} responses : {int(c):,} prompts")
    if rvals.size > max_lines:
        print("  ...")

    print("\nResponse-length bins:", DEFAULT_BINS)
    print("Counts per bin:      ", [int(x) for x in len_counts])
    print("Share per bin (%):   ", [round(float(p)*100, 2) for p in len_props])
    print("\nSaved to:", args.out_dir)
    print("=========================================================\n")

if __name__ == "__main__":
    main()
