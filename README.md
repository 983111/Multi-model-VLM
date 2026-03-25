# Multimodal Foundation Model — VLM Fine-tuning + API

## Quick start (GitHub Codespaces)

```bash
# 1. Fix dependencies first (numpy<2 required by torch 2.2)
bash setup.sh

# 2. Generate synthetic dataset (works offline, no HF login needed)
python data/prepare_dataset.py --mode synthetic --max_samples 300

# 3. Dry-run fine-tune (microsoft/git-base, CPU-safe, ~5 min)
python models/finetune.py --dry_run

# 4. Evaluate
python eval/run_eval.py --dry_run

# 5. Start API (opens on port 8000 in Codespaces)
python api/main.py
```

In a second terminal:
```bash
python api/test_api.py
# or open http://localhost:8000/docs
```

---

## Dataset modes

| Mode | Requires | What it generates |
|---|---|---|
| `synthetic` | nothing (offline) | Coloured geometric shapes with descriptions |
| `medical`   | HF access | ROCO radiology images + captions |
| `product`   | nothing (aliases synthetic) | Same as synthetic |

```bash
# Synthetic (always works)
python data/prepare_dataset.py --mode synthetic --max_samples 500

# Medical (needs HuggingFace — may need: huggingface-cli login)
python data/prepare_dataset.py --mode medical --max_samples 500
```

---

## Models

| Model | Size | Use case |
|---|---|---|
| `microsoft/git-base` | ~700 MB | Default dry-run, CPU-safe |
| `Salesforce/blip2-opt-2.7b` | ~6 GB fp16 | Better quality, needs GPU |
| `llava-hf/llava-1.5-7b-hf` | ~14 GB | Best quality, needs GPU |

Change the full-training model in `models/finetune.py` → `FinetuneConfig.model_name`.

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Base64 image + prompt → text |
| POST | `/predict/file` | Upload image file |
| POST | `/predict/batch` | Up to 16 items |
| GET | `/eval` | Latest eval metrics |

**Interactive docs:** `http://localhost:8000/docs`

---

## If you see "NumPy 1.x compiled module" warning
This is harmless — torch still works. But to silence it:
```bash
pip install "numpy<2.0" --force-reinstall
```
