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
accelerate launch --config_file ./accelerate_config.yaml train_sft.py
```

```bash
accelerate launch --config_file ./accelerate_config.yaml train_latent_cot_rl.py
```

on 2H100 GPU:
[CONFIG] num_rows_epoch=345453
[CONFIG] steps_per_epoch≈21591  save_steps=5397  (points_per_epoch=4)