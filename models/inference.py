"""
models/inference.py
-------------------
Shared inference wrapper used by eval and the API.

Default model: microsoft/git-base (CPU-safe, 700 MB)
After fine-tuning, pass checkpoint="models/checkpoints/final"
"""

import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel


class VLMInference:
    DEFAULT_MODEL = "microsoft/git-base"

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 64,
    ):
        self.checkpoint = checkpoint
        self.base_model_id = self.DEFAULT_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self._load()

    def _load(self):
        adapter_checkpoint = None
        model_id_for_processor = self.base_model_id

        if self.checkpoint:
            ckpt_path = Path(self.checkpoint)
            if ckpt_path.exists() and (ckpt_path / "adapter_config.json").exists():
                adapter_checkpoint = str(ckpt_path)
                model_id_for_processor = self.base_model_id
            else:
                model_id_for_processor = self.checkpoint

        print(f"[VLMInference] Loading base '{model_id_for_processor}' on {self.device} ...")
        self.processor = AutoProcessor.from_pretrained(
            model_id_for_processor, trust_remote_code=True
        )

        base_model = AutoModelForVision2Seq.from_pretrained(
            model_id_for_processor,
            torch_dtype=torch.float32,
            device_map=self.device,
            trust_remote_code=True,
        )

        if adapter_checkpoint:
            print(f"[VLMInference] Attaching LoRA adapter from '{adapter_checkpoint}' ...")
            self.model = PeftModel.from_pretrained(base_model, adapter_checkpoint)
            self.checkpoint = adapter_checkpoint
        else:
            self.model = base_model
            self.checkpoint = model_id_for_processor

        self.model.eval()
        print("[VLMInference] Ready.")

    @torch.inference_mode()
    def generate(self, image: Image.Image, prompt: str) -> str:
        image = image.convert("RGB")
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        input_len = inputs.get("input_ids", torch.tensor([])).shape[-1]
        generated = output_ids[0][input_len:]
        text = self.processor.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()

    def batch_generate(self, items: list[dict]) -> list[str]:
        return [self.generate(it["image"], it["prompt"]) for it in items]
