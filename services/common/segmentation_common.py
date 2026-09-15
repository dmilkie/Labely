"""
Shared request/response schema and mask post-processing for the Labely inference services.

Used by both containers (SAM2/SAM3 on :8000, micro-sam on :8001) so the JSON API is identical.
Copied into each image as /app/segmentation_common.py.
"""
import base64
import io
from typing import List, Optional, Union

import numpy as np
from PIL import Image
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

MODELS = ("sam2", "sam3", "micro-sam-lm", "micro-sam-em")
OUTPUT_TYPES = ("bbox", "segment", "polygon")


# ---------- request / response schema ----------
class Point(BaseModel):
    x: float  # pixel x
    y: float  # pixel y
    label: int = 1  # 1 = foreground (include), 0 = background (exclude)


class InferenceRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    image: str  # data:image/...;base64,<...>  or  http(s) URL  or  server-side path
    model: Optional[str] = Field(
        default=None,
        description="sam2 (default) | sam3 | micro-sam-lm | micro-sam-em. "
                    "If omitted: sam3 when a text 'prompt' is given, otherwise sam2.")
    prompt: Optional[str] = None  # text prompt, e.g. "dogs" -> SAM3 only
    points: Optional[List[Point]] = None  # SAM2 / micro-sam
    box: Optional[Union[List[float], List[List[float]]]] = Field(
        default=None, description="[x1, y1, x2, y2] or [[x1, y1, x2, y2], ...] in pixels. "
                                  "Several boxes -> one mask per box, returned in the same order")
    prompt_free: bool = Field(
        default=False, alias="prompt-free",
        description="Segment every object without any prompt (sam2 automatic mask generation, "
                    "micro-sam automatic instance segmentation). Accepts 'prompt_free' or 'prompt-free'.")
    output_type: Optional[str] = "segment"  # "bbox", "segment" (RLE mask) or "polygon" (contour vertices)
    multimask: bool = False  # SAM2/micro-sam prompted: return the 3 candidate masks instead of the best one
    polygon_tolerance: float = 2.0  # px; max deviation when simplifying contours (polygon mode). 0 = every boundary pixel
    min_area: float = 0.0           # px^2; drop objects smaller than this; in polygon mode also drops contour islands below it
    # prompt-free options
    max_objects: Optional[int] = None  # keep only the N best-scoring instances
    points_per_side: int = 32       # sam2 prompt-free: density of the point grid (more = smaller objects, slower)


class MaskResult(BaseModel):
    mask: Optional[List[int]] = None  # RLE (segment mode only)
    polygons: Optional[List[List[List[int]]]] = None  # polygon mode: [[[x,y],...], ...] one per outer contour, largest first
    box_index: Optional[int] = None  # when several boxes were sent: which input box this mask belongs to
    score: float
    bbox: List[int]  # [x1, y1, x2, y2]
    area: Optional[int] = None  # mask area in px


class InferenceResponse(BaseModel):
    model: str
    masks: List[MaskResult]
    image_size: List[int]  # [width, height]


# ---------- validation helpers ----------
def resolve_model(req: InferenceRequest) -> str:
    m = (req.model or ("sam3" if req.prompt else "sam2")).lower()
    if m not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{m}'. Choose one of {list(MODELS)}")
    return m


def validate_request(req: InferenceRequest, model: str) -> str:
    """Cross-field checks. Returns the normalized output_type."""
    output_type = (req.output_type or "segment").lower()
    if output_type not in OUTPUT_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid output_type: {output_type}. Must be one of {list(OUTPUT_TYPES)}")
    has_geom = bool(req.points) or bool(req.box)
    if model == "sam3":
        if req.prompt_free:
            raise HTTPException(status_code=400, detail="prompt_free is not supported by sam3; use sam2 or micro-sam-*")
        if has_geom:
            raise HTTPException(status_code=400, detail="sam3 takes a text 'prompt' only; use sam2 or micro-sam-* for points/box")
        if not req.prompt:
            raise HTTPException(status_code=400, detail="sam3 needs a text 'prompt'")
    else:
        if req.prompt:
            raise HTTPException(status_code=400, detail=f"Text prompts are only supported by model 'sam3' (got model '{model}')")
        if req.prompt_free and has_geom:
            raise HTTPException(status_code=400, detail="prompt_free cannot be combined with points/box")
        if not req.prompt_free and not has_geom:
            raise HTTPException(status_code=400, detail="Provide 'points' and/or 'box', or set prompt_free: true")
    return output_type


def normalize_boxes(box) -> Optional[np.ndarray]:
    """Accept [x1,y1,x2,y2] or [[...],[...]] -> (B,4) float32 array, or None."""
    if box is None or (isinstance(box, list) and len(box) == 0):
        return None
    try:
        arr = np.asarray(box, dtype=np.float32)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="'box' must be [x1, y1, x2, y2] or a list of such boxes")
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise HTTPException(status_code=400, detail="'box' must be [x1, y1, x2, y2] or a list of such boxes")
    return arr


def points_to_arrays(points: Optional[List[Point]]):
    if not points:
        return None, None
    coords = np.array([[p.x, p.y] for p in points], dtype=np.float32)
    labels = np.array([p.label for p in points], dtype=np.int32)
    return coords, labels


# ---------- image loading ----------
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


def image_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------- mask encoding ----------
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


def build_response(model: str, masks, scores, image_size, output_type: str,
                   polygon_tolerance: float = 2.0, min_area: float = 0.0,
                   keep_order: bool = False, box_indices: Optional[List[int]] = None,
                   max_objects: Optional[int] = None) -> InferenceResponse:
    """Turn (N,H,W) masks + (N,) scores into the API response.

    masks may be a numpy array or a list of 2-D arrays (bool/float). Sorted best-score first unless keep_order.
    """
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    n = len(scores)
    order = list(range(n)) if keep_order else [int(i) for i in np.argsort(-scores)]
    results = []
    for i in order:
        mask = np.asarray(masks[i])
        mask = mask > 0.5 if mask.dtype != bool else mask
        area = int(mask.sum())
        if area == 0 or area < min_area:
            continue
        rle = mask_to_rle(mask.astype(np.uint8) * 255) if output_type == "segment" else None
        polys = mask_to_polygons(mask, polygon_tolerance, min_area) if output_type == "polygon" else None
        results.append(MaskResult(mask=rle, polygons=polys, score=float(scores[i]), bbox=get_bbox(mask), area=area,
                                  box_index=(box_indices[i] if box_indices is not None else None)))
        if max_objects is not None and len(results) >= max_objects:
            break
    return InferenceResponse(model=model, masks=results, image_size=list(image_size))


def label_image_to_masks(label_img: np.ndarray):
    """Split an integer label image (0 = background) into a list of boolean masks."""
    ids = np.unique(label_img)
    ids = ids[ids != 0]
    return [label_img == i for i in ids]
