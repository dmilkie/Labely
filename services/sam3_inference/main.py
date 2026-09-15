"""
SAM2 inference service - FastAPI
Segments an image from point and/or box prompts (SAM2 has no text prompting).
Checkpoint is downloaded from Meta's public CDN at image build time; no HF token needed.
"""
import os
import io
import base64
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import json
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel, Field
import uvicorn

logger = logging.getLogger("uvicorn.info")

SAM2_SIZE = os.getenv("SAM2_SIZE", "large")  # tiny | small | base_plus | large
_CFG = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}
CHECKPOINT = os.getenv("SAM2_CHECKPOINT", f"/app/checkpoints/sam2.1_hiera_{SAM2_SIZE}.pt")

_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = os.getenv("DEVICE", "cuda:0")
        if device != "cpu" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        logger.info(f"Loading SAM2 ({SAM2_SIZE}) from {CHECKPOINT} on {device}")
        model = build_sam2(_CFG[SAM2_SIZE], CHECKPOINT, device=device)
        _predictor = SAM2ImagePredictor(model)
    return _predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_predictor()
    yield


app = FastAPI(title="SAM2 Inference Service", version="2.0.0", lifespan=lifespan)


# ---------- request / response models ----------
class Point(BaseModel):
    x: float  # pixel x
    y: float  # pixel y
    label: int = 1  # 1 = foreground (include), 0 = background (exclude)


class InferenceRequest(BaseModel):
    image: str  # data:image/...;base64,<...>  or  http(s) URL  or  server-side path
    points: Optional[List[Point]] = None
    box: Optional[List[float]] = Field(default=None, description="[x1, y1, x2, y2] in pixels")
    output_type: Optional[str] = "segment"  # "bbox", "segment" (RLE mask) or "polygon" (contour vertices)
    multimask: bool = False
    polygon_tolerance: float = 2.0  # px; max deviation when simplifying contours (polygon mode). 0 = every boundary pixel
    min_polygon_area: float = 0.0   # px^2; drop contours (islands) smaller than this (polygon mode)  # True -> return SAM2's 3 candidate masks instead of the best one


class MaskResult(BaseModel):
    mask: Optional[List[int]] = None  # RLE (segment mode only)
    polygons: Optional[List[List[List[int]]]] = None  # polygon mode: [[[x,y],[x,y],...], ...] one list per outer contour, largest first
    score: float
    bbox: List[int]  # [x1, y1, x2, y2]


class InferenceResponse(BaseModel):
    masks: List[MaskResult]
    image_size: List[int]  # [width, height]


# ---------- endpoints ----------
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": f"SAM2.1-{SAM2_SIZE}",
        "prompt_types": ["points", "box"],
        "supported_output_types": ["bbox", "segment", "polygon"],
    }


def run_sam2(image: Image.Image, points: Optional[List[Point]], box: Optional[List[float]],
             output_type: str, multimask: bool,
             polygon_tolerance: float = 2.0, min_polygon_area: float = 0.0) -> InferenceResponse:
    if not points and not box:
        raise HTTPException(status_code=400, detail="Provide 'points' and/or 'box'. SAM2 has no text prompts.")
    if box is not None and len(box) != 4:
        raise HTTPException(status_code=400, detail="'box' must be [x1, y1, x2, y2]")
    output_type = (output_type or "segment").lower()
    if output_type not in ("bbox", "segment", "polygon"):
        raise HTTPException(status_code=400, detail=f"Invalid output_type: {output_type}. Must be 'bbox', 'segment' or 'polygon'")

    width, height = image.size
    point_coords = point_labels = box_arr = None
    if points:
        point_coords = np.array([[p.x, p.y] for p in points], dtype=np.float32)
        point_labels = np.array([p.label for p in points], dtype=np.int32)
    if box:
        box_arr = np.array(box, dtype=np.float32)

    import torch
    predictor = get_predictor()
    with torch.inference_mode():
        predictor.set_image(np.array(image))
        masks, scores, _ = predictor.predict(
            point_coords=point_coords, point_labels=point_labels,
            box=box_arr, multimask_output=multimask,
        )
    logger.info(f"points={0 if point_coords is None else len(point_coords)} box={box_arr is not None} "
                f"-> {len(masks)} mask(s), scores={[round(float(s), 3) for s in scores]}")

    results = []
    for i in np.argsort(-scores):  # best first
        mask = masks[i] > 0.5
        rle = mask_to_rle(mask.astype(np.uint8) * 255) if output_type == "segment" else None
        polys = mask_to_polygons(mask, polygon_tolerance, min_polygon_area) if output_type == "polygon" else None
        results.append(MaskResult(mask=rle, polygons=polys, score=float(scores[i]), bbox=get_bbox(mask)))
    return InferenceResponse(masks=results, image_size=[width, height])


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """JSON body: image as base64 data URI / URL / path, plus points and/or box."""
    image = load_image(request.image)
    return run_sam2(image, request.points, request.box, request.output_type, request.multimask,
                    request.polygon_tolerance, request.min_polygon_area)


@app.post("/predict/upload", response_model=InferenceResponse)
async def predict_upload(
    image: UploadFile = File(..., description="JPEG/PNG file part"),
    points: Optional[str] = Form(None, description='JSON: [{"x":100,"y":200,"label":1}, ...]'),
    box: Optional[str] = Form(None, description='JSON [x1,y1,x2,y2] or "x1,y1,x2,y2"'),
    output_type: str = Form("segment"),
    multimask: bool = Form(False),
    polygon_tolerance: float = Form(2.0),
    min_polygon_area: float = Form(0.0),
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
    return run_sam2(pil, pts, bx, output_type, multimask, polygon_tolerance, min_polygon_area)


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


# ---------- helpers ----------
def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("data:image"):
        _, data = image_str.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    if image_str.startswith("http"):
        import requests
        r = requests.get(image_str, timeout=30)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    return Image.open(image_str).convert("RGB")


def mask_to_rle(mask: np.ndarray) -> List[int]:
    """Label Studio style RLE: alternating [start, length, start, length, ...] on the flattened mask (1-based)."""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs.tolist()


def mask_to_polygons(mask: np.ndarray, tolerance: float = 2.0, min_area: float = 0.0) -> List[List[List[int]]]:
    """Trace the outer boundary of each connected blob in the mask and simplify it to a polygon.

    Returns a list of contours, largest area first; each contour is [[x, y], ...] in pixel coordinates
    (closed implicitly: last vertex connects back to the first). Holes inside a blob are ignored.
    """
    import cv2
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        if tolerance > 0:
            c = cv2.approxPolyDP(c, tolerance, closed=True)
        if len(c) < 3:
            continue
        polys.append((area, c.reshape(-1, 2).astype(int).tolist()))
    polys.sort(key=lambda t: -t[0])
    return [pts for _, pts in polys]


def get_bbox(mask: np.ndarray) -> List[int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax), int(rmax)]


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000,
                reload=os.getenv("DEBUG", "false").lower() == "true")
