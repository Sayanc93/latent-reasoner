## INSTRUCTIONS

### Installation

Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies

```bash
uv pip install -r requirements.txt
```

Install flash attention

```bash
uv pip install flash-attn --no-build-isolation
```

### Train SFT

```bash
uv run train_sft.py --bf16
```

```bash
accelerate launch --config_file ./accelerate_config.yaml train_latent_cot_rl.py
```

[CONFIG] steps_per_epoch≈43182  save_steps=5397  (points_per_epoch=8)