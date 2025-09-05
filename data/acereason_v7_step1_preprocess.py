from datasets import load_dataset  # type: ignore
from transformers import AutoTokenizer  # type: ignore
import hashlib
import os
import numpy as np

DATASET_ID = "nvidia/AceReason-1.1-SFT"
CACHE_DIR   = f"{os.getcwd()}/cache"
NUM_PROC    = os.cpu_count()

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B", trust_remote_code=True, cache_dir=CACHE_DIR, use_fast=True)

# 1) Download to local cache and keep only math (keep as Arrow Dataset so we can save)
ds = load_dataset(DATASET_ID, split="train", cache_dir=CACHE_DIR, num_proc=NUM_PROC)
ds = ds.filter(lambda x: x["category"] == "math", num_proc=NUM_PROC)

# 2) Add prompt hash + response token length (batched + multiprocess)
def add_cols(batch):
    phash = [hashlib.sha1(prompt.encode("utf-8")).hexdigest() for prompt in batch["input"]]
    lens  = [len(tok.encode(o, add_special_tokens=False)) for o in batch["output"]]
    return {"prompt_hash": phash, "resp_len": lens}

ds = ds.map(add_cols, batched=True, num_proc=NUM_PROC)

# 3) Persist the processed view for instant reuse later
ds.save_to_disk(f"{CACHE_DIR}/acereason_v7_math_precomputed", num_proc=NUM_PROC)

# 4) Quick sanity peek (histogram you’ll target when assembling v7)
BINS = [0,2000,4000,8000,12000,16000,20000,32768]
counts, _ = np.histogram(ds["resp_len"], bins=BINS)
props = counts / counts.sum()
print("Resp-length bin counts:", counts)
print("Resp-length bin %:", np.round(props*100, 2))
print("Unique prompts:", len(set(ds["prompt_hash"])))