"""
models/finetune.py
------------------
Fine-tunes a vision-language model on the prepared dataset using LoRA.

Dry-run model : microsoft/git-base  (700 MB, CPU-safe, no tokenizer issues)
Full model    : llava-hf/llava-1.5-7b-hf  (needs ~14 GB VRAM)

Usage:
  python models/finetune.py --dry_run                    # Codespaces CPU safe
  python models/finetune.py --data_dir data/processed    # full training (GPU)
"""

import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class FinetuneConfig:
    # Dry-run: small model that works on CPU without tokenizer issues
    dry_run_model: str = "microsoft/git-base"
    # Full training target
    model_name:    str = "llava-hf/llava-1.5-7b-hf"

    # LoRA
    lora_r:              int   = 16
    lora_alpha:          int   = 32
    lora_dropout:        float = 0.05
    lora_target_modules: tuple = ("q_proj", "v_proj")

    # Training
    num_epochs:    int   = 3
    batch_size:    int   = 2
    grad_accum:    int   = 4
    learning_rate: float = 2e-4
    warmup_ratio:  float = 0.03
    max_seq_len:   int   = 256

    output_dir:    str = "models/checkpoints"
    logging_steps: int = 10
    save_steps:    int = 200


CFG = FinetuneConfig()


# ── dataset ───────────────────────────────────────────────────────────────────

class VLMDataset(Dataset):
    def __init__(self, json_path: Path, image_dir: Path, processor, max_len: int):
        with open(json_path) as f:
            self.records = json.load(f)
        self.image_dir = image_dir
        self.processor = processor
        self.max_len   = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = self.image_dir / rec["image"]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        convs       = rec.get("conversations", [])
        human_text  = next((c["value"] for c in convs if c["from"] == "human"), "Describe this image.")
        gpt_text    = next((c["value"] for c in convs if c["from"] == "gpt"),   "")
        human_text  = human_text.replace("<image>\n", "").replace("<image>", "").strip()
        full_text   = f"Question: {human_text} Answer: {gpt_text}"

        enc = self.processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = item["input_ids"].clone()
        return item


# ── LoRA ──────────────────────────────────────────────────────────────────────

def apply_lora(model, cfg: FinetuneConfig):
    # Find actual attention projection names in the model
    named = [n for n, _ in model.named_modules()]
    targets = [t for t in cfg.lora_target_modules
               if any(t in n for n in named)]
    if not targets:
        # fallback: target any linear layers we can find
        targets = list(cfg.lora_target_modules)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=targets,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    try:
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    except Exception as e:
        print(f"  [warn] LoRA application failed ({e}). Training full model instead.")
    return model


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  default="data/processed")
    parser.add_argument("--dry_run",   action="store_true",
                        help="2-step smoke test with microsoft/git-base (CPU safe)")
    parser.add_argument("--use_qlora", action="store_true",
                        help="4-bit QLoRA — needs GPU + bitsandbytes")
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    image_dir = data_dir / "images"
    ckpt_dir  = Path(CFG.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── pick model ──
    model_id = CFG.dry_run_model if args.dry_run else CFG.model_name
    print(f"\nModel  : {model_id}")
    print(f"Dry run: {args.dry_run}")
    print(f"Device : {'GPU' if torch.cuda.is_available() else 'CPU'}\n")

    # Verify dataset exists
    if not (data_dir / "train.json").exists():
        print(f"[error] {data_dir}/train.json not found.")
        print("  Run first: python data/prepare_dataset.py --mode synthetic")
        return

    # ── processor ──
    print("Loading processor …")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # ── model ──
    print("Loading model …")
    quant_cfg = None
    if args.use_qlora and not args.dry_run:
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    dtype   = torch.float32  # always fp32 on CPU
    dev_map = "auto" if torch.cuda.is_available() else "cpu"

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=quant_cfg,
        device_map=dev_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    # ── LoRA ──
    if not args.dry_run:
        model = apply_lora(model, CFG)
    else:
        print("[dry-run] Skipping LoRA — base model only for smoke test")

    # ── datasets ──
    print("\nBuilding datasets …")
    train_ds = VLMDataset(data_dir / "train.json", image_dir, processor, CFG.max_seq_len)
    val_ds   = VLMDataset(data_dir / "val.json",   image_dir, processor, CFG.max_seq_len)
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    # ── training args ──
    training_args = TrainingArguments(
        output_dir=CFG.output_dir,
        num_train_epochs=1 if args.dry_run else CFG.num_epochs,
        max_steps=2 if args.dry_run else -1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if args.dry_run else CFG.grad_accum,
        learning_rate=CFG.learning_rate,
        warmup_ratio=CFG.warmup_ratio,
        logging_steps=1 if args.dry_run else CFG.logging_steps,
        save_steps=CFG.save_steps,
        eval_strategy="no",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="none",
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )

    print("\nTraining …")
    trainer.train()

    # ── save ──
    save_path = ckpt_dir / "final"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"\nCheckpoint saved → {save_path}")
    print("Next: python eval/run_eval.py --checkpoint models/checkpoints/final")


if __name__ == "__main__":
    main()
