"""
Labely inference gateway - FastAPI (port 8000)

  model = sam2          -> SAM 2.1 (public checkpoint baked into the image). Prompts: points / box, or prompt_free
  model = sam3          -> SAM3 text / concept prompts (needs HF_TOKEN; gated facebook/sam3 repo)
  model = micro-sam-lm  -> Segment Anything for Microscopy, light microscopy model  } proxied to the
  model = micro-sam-em  -> Segment Anything for Microscopy, electron microscopy model} micro-sam container (:8001)

All models share the same request/response schema and post-processing: bbox | segment (RLE) | polygon.
SAM3 is loaded lazily (or at startup when HF_TOKEN is set) so the service stays up without a token.
"""
import os
import io
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
import uvicorn

from segmentation_common import (
    InferenceRequest, InferenceResponse, Point, OUTPUT_TYPES,
    resolve_model, validate_request, normalize_boxes, points_to_arrays,
    load_image, image_to_data_uri, build_response,
)

logger = logging.getLogger("uvicorn.info")

SAM2_SIZE = os.getenv("SAM2_SIZE", "large")  # tiny | small | base_plus | large
_CFG = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}
CHECKPOINT = os.getenv("SAM2_CHECKPOINT", f"/app/checkpoints/sam2.1_hiera_{SAM2_SIZE}.pt")
MICROSAM_URL = os.getenv("MICROSAM_URL", "http://micro-sam-inference:8001")

_predictor = None
_sam3 = None  # (model, processor)
_sam3_error = None


def _device():
    import torch
    device = os.getenv("DEVICE", "cuda:0")
    if device != "cpu" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    return device


def get_predictor():
    global _predictor
    if _predictor is None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        device = _device()
        logger.info(f"Loading SAM2 ({SAM2_SIZE}) from {CHECKPOINT} on {device}")
        model = build_sam2(_CFG[SAM2_SIZE], CHECKPOINT, device=device)
        _predictor = SAM2ImagePredictor(model)
    return _predictor


def get_sam3():
    """Load SAM3 on first use. Raises HTTPException 503 with the reason if it cannot load."""
    global _sam3, _sam3_error
    if _sam3 is not None:
        return _sam3
    if _sam3_error is not None:
        raise HTTPException(status_code=503, detail=f"SAM3 unavailable: {_sam3_error}")
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        device = _device()
        logger.info(f"Loading SAM3 on {device} (HF_TOKEN set: {bool(os.getenv('HF_TOKEN'))})")
        model = build_sam3_image_model()
        if device != "cpu":
            model = model.to(device)
        model.eval()
        _sam3 = (model, Sam3Processor(model))
        logger.info("SAM3 loaded")
        return _sam3
    except Exception as e:  # gated repo / no token / download failure
        _sam3_error = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
        logger.error(f"SAM3 failed to load: {_sam3_error}")
        raise HTTPException(status_code=503, detail=f"SAM3 unavailable: {_sam3_error}. "
                            "Text prompts need a Hugging Face token with access to facebook/sam3 (HF_TOKEN in .env).")


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_predictor()  # SAM2 always
    if os.getenv("HF_TOKEN"):
        try:
            get_sam3()  # warm up SAM3 if a token is present
        except HTTPException:
            pass
    yield


app = FastAPI(title="Labely Inference Service (SAM2 / SAM3 / micro-sam)", version="4.0.0", lifespan=lifespan)


# ---------- endpoints ----------
def _microsam_health() -> dict:
    try:
        r = requests.get(f"{MICROSAM_URL}/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "unreachable", "error": f"{type(e).__name__}: {e}"}


@app.get("/health")
async def health():
    sam3_status = ("loaded" if _sam3 is not None else
                   ("error: " + _sam3_error) if _sam3_error else
                   ("not loaded yet" if os.getenv("HF_TOKEN") else "disabled (no HF_TOKEN)"))
    return {
        "status": "healthy",
        "models": {
            "sam2": f"loaded (SAM2.1-{SAM2_SIZE})" if _predictor is not None else "not loaded",
            "sam3": sam3_status,
            "micro-sam": _microsam_health(),
        },
        "default_model": "sam2 (sam3 when a text prompt is sent)",
        "prompt_types": {
            "sam2": ["points", "box", "prompt_free"],
            "sam3": ["prompt (text)"],
            "micro-sam-lm": ["points", "box", "prompt_free"],
            "micro-sam-em": ["points", "box", "prompt_free"],
        },
        "supported_output_types": list(OUTPUT_TYPES),
    }


def run_inference(req: InferenceRequest, image: Optional[Image.Image] = None) -> InferenceResponse:
    model = resolve_model(req)
    output_type = validate_request(req, model)

    if model.startswith("micro-sam"):
        return proxy_microsam(req, image)

    if image is None:
        image = load_image(req.image)
    W, H = image.size

    if model == "sam3":
        masks, scores = run_sam3_text(image, req.prompt)
        return build_response(model, masks, scores, (W, H), output_type,
                              req.polygon_tolerance, req.min_polygon_area,
                              min_area=req.min_area, max_objects=req.max_objects)

    # sam2
    if req.prompt_free:
        masks, scores = run_sam2_automatic(image, req.points_per_side, req.min_area)
        return build_response(model, masks, scores, (W, H), output_type,
                              req.polygon_tolerance, req.min_polygon_area,
                              min_area=req.min_area, max_objects=req.max_objects)

    boxes = normalize_boxes(req.box)
    multi_box = boxes is not None and len(boxes) > 1
    if multi_box and req.points:
        raise HTTPException(status_code=400, detail="'points' cannot be combined with several boxes; send one box or only boxes")
    masks, scores = run_sam2_geometric(image, req.points, boxes, req.multimask)
    return build_response(model, masks, scores, (W, H), output_type,
                          req.polygon_tolerance, req.min_polygon_area,
                          keep_order=multi_box, box_indices=list(range(len(masks))) if multi_box else None)


def proxy_microsam(req: InferenceRequest, image: Optional[Image.Image]) -> InferenceResponse:
    """Forward the request unchanged to the micro-sam container and return its response."""
    payload = req.model_dump(by_alias=False)
    if image is not None:  # multipart upload path: image already decoded here
        payload["image"] = image_to_data_uri(image)
    elif not (payload["image"].startswith("data:image") or payload["image"].startswith("http")):
        payload["image"] = image_to_data_uri(load_image(payload["image"]))  # server-side path -> inline
    try:
        r = requests.post(f"{MICROSAM_URL}/predict", json=payload, timeout=600)
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"micro-sam service unreachable at {MICROSAM_URL}: {e}")
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise HTTPException(status_code=r.status_code, detail=f"micro-sam: {detail}")
    return InferenceResponse(**r.json())


def run_sam3_text(image: Image.Image, prompt: str):
    """SAM3 concept segmentation: returns (masks (N,H,W) numpy, scores (N,) numpy)."""
    import torch
    model, processor = get_sam3()
    use_cuda = next(model.parameters()).is_cuda
    # SAM3 is meant to run under bf16 autocast (as in Meta's examples); without it fc layers hit a dtype mismatch
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
        state = processor.set_image(image)
        out = processor.set_text_prompt(state=state, prompt=prompt)
    masks = out.get("masks", [])
    scores = out.get("scores", [])
    if hasattr(masks, "cpu"):  # bf16 tensors cannot go straight to numpy
        masks = masks.detach().float().cpu().numpy()
    if hasattr(scores, "cpu"):
        scores = scores.detach().float().cpu().numpy()
    masks = np.asarray(masks)
    if masks.ndim == 4:  # (N,1,H,W)
        masks = masks[:, 0]
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    W, H = image.size
    if masks.ndim != 3 or len(masks) == 0:
        logger.info(f"SAM3 found nothing for prompt '{prompt}'")
        return np.zeros((0, H, W), dtype=bool), np.zeros((0,), dtype=np.float32)
    if masks.shape[1:] != (H, W):  # SAM3 may return masks at model resolution
        import cv2
        masks = np.stack([cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) for m in masks])
    logger.info(f"SAM3 prompt='{prompt}' -> {len(masks)} object(s), scores={[round(float(s), 3) for s in scores]}")
    return masks, scores


def run_sam2_geometric(image: Image.Image, points, boxes: Optional[np.ndarray], multimask: bool):
    point_coords, point_labels = points_to_arrays(points)
    box_arr = None
    if boxes is not None:
        box_arr = boxes if len(boxes) > 1 else boxes[0]  # (B,4) batched, or (4,) single

    import torch
    predictor = get_predictor()
    with torch.inference_mode():
        predictor.set_image(np.array(image))
        masks, scores, _ = predictor.predict(
            point_coords=point_coords, point_labels=point_labels,
            box=box_arr, multimask_output=multimask,
        )
    masks = np.asarray(masks)
    scores = np.asarray(scores, dtype=np.float32)
    if masks.ndim == 4:  # batched boxes: (B, K, H, W) / (B, K) -> pick best of K per box
        best = scores.argmax(axis=1)
        masks = masks[np.arange(len(masks)), best]
        scores = scores[np.arange(len(scores)), best]
    logger.info(f"SAM2 points={0 if point_coords is None else len(point_coords)} "
                f"boxes={0 if boxes is None else len(boxes)} -> {len(masks)} mask(s), "
                f"scores={[round(float(s), 3) for s in scores]}")
    return masks, scores


def run_sam2_automatic(image: Image.Image, points_per_side: int, min_area: float):
    """Prompt-free: SAM2 automatic mask generation over a point grid. Returns (list of bool masks, scores)."""
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.utils.amg import rle_to_mask
    predictor = get_predictor()
    gen = SAM2AutomaticMaskGenerator(
        model=predictor.model,
        points_per_side=max(4, min(int(points_per_side), 128)),
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        min_mask_region_area=0,  # min_area is applied at full resolution in build_response
        output_mode="uncompressed_rle",  # keep memory low on large images; decode one at a time below
    )
    # SAM resizes its input to 1024 px internally, so running the grid on a huge image only adds cost in
    # mask upsampling / post-processing. Downscale first and upsample the masks afterwards.
    W, H = image.size
    max_side = int(os.getenv("PROMPT_FREE_MAX_SIDE", "2048"))
    scale = min(1.0, max_side / max(W, H))
    work = image.resize((round(W * scale), round(H * scale)), Image.BILINEAR) if scale < 1.0 else image
    with torch.inference_mode():
        anns = gen.generate(np.array(work))
    anns.sort(key=lambda a: -a["predicted_iou"])
    masks = [rle_to_mask(a["segmentation"]) for a in anns]
    if scale < 1.0:
        import cv2
        masks = [cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks]
    scores = np.array([a["predicted_iou"] for a in anns], dtype=np.float32)
    logger.info(f"SAM2 prompt-free (grid {points_per_side}x{points_per_side}, work size {work.size}) -> {len(masks)} object(s)")
    return masks, scores


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """JSON body: image + (model) + prompt | points/box | prompt_free. See /health for what each model accepts."""
    return run_inference(request)


@app.post("/predict/upload", response_model=InferenceResponse)
async def predict_upload(
    image: UploadFile = File(..., description="JPEG/PNG file part"),
    model: Optional[str] = Form(None, description="sam2 | sam3 | micro-sam-lm | micro-sam-em"),
    prompt: Optional[str] = Form(None, description="text prompt (sam3)"),
    points: Optional[str] = Form(None, description='JSON: [{"x":100,"y":200,"label":1}, ...]'),
    box: Optional[str] = Form(None, description='JSON [x1,y1,x2,y2], [[...],[...]] or "x1,y1,x2,y2"'),
    prompt_free: bool = Form(False),
    output_type: str = Form("segment"),
    multimask: bool = Form(False),
    polygon_tolerance: float = Form(2.0),
    min_polygon_area: float = Form(0.0),
    min_area: float = Form(0.0),
    max_objects: Optional[int] = Form(None),
    points_per_side: int = Form(32),
):
    """multipart/form-data variant: send the raw image bytes as a file part, prompts as text fields."""
    try:
        pil = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image file part: {e}")
    try:
        pts = [Point(**p) for p in json.loads(points)] if points else None
        bx = None
        if box:
            bx = json.loads(box) if box.strip().startswith("[") else [float(v) for v in box.split(",")]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse points/box: {e}")
    req = InferenceRequest(image="(uploaded)", model=model, prompt=prompt, points=pts, box=bx, prompt_free=prompt_free,
                           output_type=output_type, multimask=multimask, polygon_tolerance=polygon_tolerance,
                           min_polygon_area=min_polygon_area, min_area=min_area, max_objects=max_objects,
                           points_per_side=points_per_side)
    return run_inference(req, image=pil)


@app.post("/echo")
async def echo(request: Request):
    """Debugging aid: reflects back exactly what the server received (headers + parsed body)."""
    body = await request.body()
    ctype = request.headers.get("content-type", "")
    out = {
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "content_type": ctype,
        "body_bytes": len(body),
    }
    if ctype.startswith("multipart/form-data"):
        try:
            form = await request.form()
            fields = {}
            for k, v in form.multi_items():
                if hasattr(v, "filename"):  # UploadFile
                    data = await v.read()
                    fields[k] = {"type": "file", "filename": v.filename,
                                 "content_type": v.content_type, "size": len(data)}
                else:
                    fields[k] = {"type": "text", "value": v}
            out["multipart_fields"] = fields
        except Exception as e:
            out["multipart_parse_error"] = str(e)
    elif ctype.startswith("application/json"):
        try:
            j = json.loads(body)
            if isinstance(j, dict) and isinstance(j.get("image"), str) and len(j["image"]) > 80:
                j["image"] = j["image"][:80] + f"... ({len(j['image'])} chars)"
            out["json"] = j
        except Exception as e:
            out["json_parse_error"] = str(e)
    else:
        out["body_preview"] = body[:500].decode("utf-8", errors="replace")
    return out


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000,
                reload=os.getenv("DEBUG", "false").lower() == "true")
