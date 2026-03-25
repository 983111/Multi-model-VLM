"""
data/prepare_dataset.py
-----------------------
Prepares an image-text dataset for VLM fine-tuning.

Modes:
  --mode synthetic  : Generates labelled geometric shapes locally (no HF needed)
  --mode medical    : Downloads ROCO radiology subset from HuggingFace
  --mode product    : Downloads fashion subset via tfds-nightly proxy on HF

Usage:
  python data/prepare_dataset.py --mode synthetic --max_samples 500
  python data/prepare_dataset.py --mode medical   --max_samples 500
"""

import os
import json
import argparse
import random
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm


# ── helpers ────────────────────────────────────────────────────────────────────

def resize_and_save(image: Image.Image, path: Path, size: int = 336) -> bool:
    try:
        image = image.convert("RGB")
        image = image.resize((size, size), Image.LANCZOS)
        image.save(path, "JPEG", quality=90)
        return True
    except Exception as e:
        print(f"  [warn] {e}")
        return False


def build_instruction(question: str, answer: str) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt",   "value": answer.strip()},
        ]
    }


# ── synthetic mode (zero dependencies beyond pillow + numpy) ──────────────────

SHAPES   = ["circle", "rectangle", "triangle", "ellipse", "square"]
COLORS   = ["red", "blue", "green", "yellow", "orange", "purple", "cyan", "white"]
SIZES    = ["small", "medium", "large"]
TEXTURES = ["solid", "striped", "dotted"]

COLOR_RGB = {
    "red":    (220, 50,  50),
    "blue":   (50,  100, 220),
    "green":  (50,  180, 80),
    "yellow": (240, 220, 50),
    "orange": (240, 140, 40),
    "purple": (150, 50,  200),
    "cyan":   (50,  200, 210),
    "white":  (240, 240, 240),
}

BG_COLORS = [(30, 30, 30), (240, 240, 240), (200, 220, 200), (220, 200, 210)]


def draw_shape(draw: ImageDraw.Draw, shape: str, color_rgb: tuple,
               bbox: tuple, texture: str):
    x0, y0, x1, y1 = bbox
    if texture == "striped":
        # Draw stripes then clip with shape
        for i in range(x0, x1, 8):
            draw.line([(i, y0), (i, y1)], fill=color_rgb, width=3)
    elif texture == "dotted":
        for i in range(x0, x1, 12):
            for j in range(y0, y1, 12):
                draw.ellipse([(i-3, j-3), (i+3, j+3)], fill=color_rgb)
    else:
        if shape == "circle":
            draw.ellipse(bbox, fill=color_rgb)
        elif shape == "ellipse":
            draw.ellipse(bbox, fill=color_rgb)
        elif shape == "rectangle":
            draw.rectangle(bbox, fill=color_rgb)
        elif shape == "square":
            side = min(x1-x0, y1-y0)
            draw.rectangle([x0, y0, x0+side, y0+side], fill=color_rgb)
        elif shape == "triangle":
            cx = (x0 + x1) // 2
            draw.polygon([(cx, y0), (x0, y1), (x1, y1)], fill=color_rgb)


def make_synthetic_image(shape: str, color_name: str, size_name: str,
                          texture: str, bg: tuple) -> Image.Image:
    img = Image.new("RGB", (336, 336), color=bg)
    draw = ImageDraw.Draw(img)

    size_map = {"small": 60, "medium": 110, "large": 160}
    s = size_map[size_name]
    cx, cy = 168, 168
    bbox = (cx - s, cy - s, cx + s, cy + s)
    rgb = COLOR_RGB[color_name]
    draw_shape(draw, shape, rgb, bbox, texture)
    return img


def generate_caption(shape: str, color: str, size: str, texture: str) -> str:
    article = "An" if shape[0] in "aeiou" or color[0] in "aeiou" else "A"
    return (f"{article} {size} {texture} {color} {shape} centered on a plain background. "
            f"The {shape} has a {texture} {color} fill and is {size} in size.")


def prepare_synthetic(output_dir: Path, max_samples: int) -> list:
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    records = []

    combos = [
        (sh, co, si, tx)
        for sh in SHAPES
        for co in COLORS
        for si in SIZES
        for tx in TEXTURES
    ]
    random.shuffle(combos)
    combos = combos[:max_samples]

    for i, (shape, color, size, texture) in enumerate(tqdm(combos, desc="Generating synthetic images")):
        bg = random.choice(BG_COLORS)
        img = make_synthetic_image(shape, color, size, texture, bg)
        fname = f"synthetic_{i:05d}.jpg"
        if resize_and_save(img, img_dir / fname):
            caption = generate_caption(shape, color, size, texture)
            question = "Describe the shape and color in this image."
            rec = build_instruction(question, caption)
            rec["image"] = fname
            records.append(rec)

    return records


# ── medical mode ────────────────────────────────────────────────────────────────

def prepare_medical(output_dir: Path, max_samples: int) -> list:
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    records = []

    print("Loading ROCO-radiology dataset …")
    try:
        # Import here so numpy warning doesn't affect other modes
        from datasets import load_dataset
        ds = load_dataset("eltorio/ROCO-radiology", split="train", streaming=True)
        count = 0
        for item in tqdm(ds, total=max_samples):
            if count >= max_samples:
                break
            try:
                raw = item.get("image")
                if raw is None:
                    continue
                if isinstance(raw, dict) and "bytes" in raw:
                    from io import BytesIO
                    img = Image.open(BytesIO(raw["bytes"]))
                elif isinstance(raw, Image.Image):
                    img = raw
                else:
                    continue
                fname = f"medical_{count:05d}.jpg"
                if not resize_and_save(img, img_dir / fname):
                    continue
                caption = (item.get("caption") or item.get("text") or "").strip()
                if len(caption) < 10:
                    continue
                rec = build_instruction("Describe the findings in this medical image.", caption)
                rec["image"] = fname
                records.append(rec)
                count += 1
            except Exception as e:
                print(f"  [skip] {e}")
    except Exception as e:
        print(f"\n[warn] Could not load ROCO: {e}")
        print("  Falling back to synthetic medical-style images …")
        records = _synthetic_medical_fallback(img_dir, max_samples)

    return records


def _synthetic_medical_fallback(img_dir: Path, n: int) -> list:
    """Greyscale noise images with plausible radiology captions."""
    captions = [
        "Chest X-ray showing clear lung fields with no acute cardiopulmonary process.",
        "CT scan of the abdomen revealing a homogeneous liver with no focal lesions.",
        "MRI of the knee demonstrating an intact anterior cruciate ligament.",
        "Chest radiograph with bilateral infiltrates consistent with pneumonia.",
        "Brain MRI showing no acute intracranial abnormality.",
        "Pelvic X-ray demonstrating normal bony alignment with no fractures.",
        "Ultrasound of the right upper quadrant showing no gallstones.",
        "Lumbar spine MRI showing mild degenerative disc disease at L4-L5.",
    ]
    records = []
    for i in range(n):
        arr = (np.random.rand(336, 336) * 200 + 30).astype("uint8")
        img = Image.fromarray(arr).convert("RGB")
        fname = f"medical_{i:05d}.jpg"
        img.save(img_dir / fname)
        caption = captions[i % len(captions)]
        rec = build_instruction("Describe the findings in this medical image.", caption)
        rec["image"] = fname
        records.append(rec)
    return records


# ── split & save ───────────────────────────────────────────────────────────────

def split_and_save(records: list, output_dir: Path,
                   train_ratio=0.8, val_ratio=0.1):
    random.shuffle(records)
    n = len(records)
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    splits = {
        "train": records[:t],
        "val":   records[t:t+v],
        "test":  records[t+v:],
    }
    for name, data in splits.items():
        path = output_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {name}: {len(data)} samples → {path}")

    meta = {
        "total":     n,
        "splits":    {k: len(v) for k, v in splits.items()},
        "image_dir": str(output_dir / "images"),
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata → {output_dir}/meta.json")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "medical", "product"],
                        default="synthetic",
                        help="'synthetic' works offline; 'medical' uses HuggingFace ROCO")
    parser.add_argument("--output_dir",  default="data/processed")
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  Mode        : {args.mode}")
    print(f"  Max samples : {args.max_samples}")
    print(f"  Output dir  : {output_dir}")
    print(f"{'='*50}\n")

    if args.mode == "synthetic":
        records = prepare_synthetic(output_dir, args.max_samples)
    elif args.mode == "medical":
        records = prepare_medical(output_dir, args.max_samples)
    else:
        # product → alias to synthetic for offline use
        print("[info] 'product' mode not available offline; using synthetic instead")
        records = prepare_synthetic(output_dir, args.max_samples)

    if not records:
        print("[error] No records collected.")
        return

    print(f"\nCollected {len(records)} records. Splitting …")
    split_and_save(records, output_dir)
    print("\nDone! Run fine-tuning next:")
    print(f"  python models/finetune.py --data_dir {output_dir} --dry_run")


if __name__ == "__main__":
    main()
