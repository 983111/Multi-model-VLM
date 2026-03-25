"""
api/main.py — FastAPI VLM service
Run: python api/main.py
Docs: http://localhost:8000/docs
"""

import io
import json
import sys
import base64
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.inference import VLMInference

app = FastAPI(
    title="Multimodal VLM API",
    description="Vision-language model fine-tuned on custom image-text pairs",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_model: Optional[VLMInference] = None
CHECKPOINT_DIR = Path("models/checkpoints/final")
EVAL_RESULTS   = Path("eval/results/results.json")


def get_model() -> VLMInference:
    global _model
    if _model is None:
        ckpt = str(CHECKPOINT_DIR) if CHECKPOINT_DIR.exists() else None
        _model = VLMInference(checkpoint=ckpt)
    return _model


class PredictRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())   # suppress pydantic warning

    image_b64:      str = Field(..., description="Base64-encoded image")
    prompt:         str = Field(default="Describe this image.")
    max_new_tokens: int = Field(default=64, ge=1, le=256)


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction:  str
    latency_s:   float
    model_used:  str
    prompt_used: str


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest]


class BatchPredictResponse(BaseModel):
    results:         list[PredictResponse]
    total_latency_s: float


def b64_to_pil(b64_str: str) -> Image.Image:
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


@app.get("/")
def root():
    ckpt = str(CHECKPOINT_DIR) if CHECKPOINT_DIR.exists() else "base (no fine-tuned checkpoint)"
    return {"status": "ok", "checkpoint": ckpt,
            "docs": "/docs", "endpoints": ["/predict", "/predict/batch", "/predict/file", "/eval"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = get_model()
    image = b64_to_pil(req.image_b64)
    model.max_new_tokens = req.max_new_tokens
    t0   = time.time()
    pred = model.generate(image, req.prompt)
    return PredictResponse(prediction=pred, latency_s=round(time.time()-t0, 3),
                           model_used=model.checkpoint, prompt_used=req.prompt)


@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...),
                       prompt: str = "Describe this image.",
                       max_new_tokens: int = 64):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Must upload an image file")
    model = get_model()
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    model.max_new_tokens = max_new_tokens
    t0   = time.time()
    pred = model.generate(image, prompt)
    return {"filename": file.filename, "prediction": pred,
            "latency_s": round(time.time()-t0, 3), "model_used": model.checkpoint}


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    if len(req.items) > 16:
        raise HTTPException(400, "Max 16 items per batch")
    model  = get_model()
    t_all  = time.time()
    results = []
    for item in req.items:
        image = b64_to_pil(item.image_b64)
        model.max_new_tokens = item.max_new_tokens
        t0   = time.time()
        pred = model.generate(image, item.prompt)
        results.append(PredictResponse(prediction=pred, latency_s=round(time.time()-t0,3),
                                       model_used=model.checkpoint, prompt_used=item.prompt))
    return BatchPredictResponse(results=results, total_latency_s=round(time.time()-t_all,3))


@app.get("/eval")
def get_eval():
    if not EVAL_RESULTS.exists():
        raise HTTPException(404, "No eval results. Run: python eval/run_eval.py --dry_run")
    return json.loads(EVAL_RESULTS.read_text())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
