"""
micro-sam inference service - FastAPI (port 8001)

Segment Anything for Microscopy (https://github.com/computational-cell-analytics/micro-sam).
  model = micro-sam-lm -> vit_<size>_lm            (light microscopy: cells, nuclei, ...)
  model = micro-sam-em -> vit_<size>_em_organelles (electron microscopy: mitochondria, nuclei, ...)
Prompts: points / box (SAM predictor) or prompt_free (automatic instance segmentation decoder, AIS).
Weights download from Zenodo on first use into MICROSAM_CACHEDIR; no token needed.

Normally reached through the gateway on :8000 (model=micro-sam-*), but speaks the same API directly.
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
import uvicorn

from segmentation_common import (
    InferenceRequest, InferenceResponse, OUTPUT_TYPES,
    resolve_model, validate_request, normalize_boxes, points_to_arrays,
    load_image, build_response, label_image_to_masks,
)

logger = logging.getLogger("uvicorn.info")

SIZE = os.getenv("MICROSAM_SIZE", "b")  # t | b | l | h  (vit_t .. vit_h); b is the documented default
MODEL_TYPES = {
    "micro-sam-lm": f"vit_{SIZE}_lm",
    "micro-sam-em": f"vit_{SIZE}_em_organelles",
}
PRELOAD = [m.strip() for m in os.getenv("MICROSAM_PRELOAD", "micro-sam-lm,micro-sam-em").split(",") if m.strip()]

_models = {}   # name -> (predictor, decoder)
_errors = {}   # name -> str


def _device():
    import torch
    device = os.getenv("DEVICE", "cuda:0")
    if device != "cpu" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    if device.startswith("cuda:"):  # micro-sam only accepts 'cpu' | 'cuda' | 'mps'
        import os as _os
        _os.environ.setdefault("CUDA_VISIBLE_DEVICES", device.split(":", 1)[1])
        device = "cuda"
    return device


def get_model(name: str):
    if name in _models:
        return _models[name]
    if name in _errors:
        raise HTTPException(status_code=503, detail=f"{name} unavailable: {_errors[name]}")
    try:
        from micro_sam.instance_segmentation import get_predictor_and_decoder
        mt = MODEL_TYPES[name]
        logger.info(f"Loading micro-sam {mt} on {_device()} (cache: {os.getenv('MICROSAM_CACHEDIR', '~/.cache/micro_sam')})")
        predictor, decoder = get_predictor_and_decoder(model_type=mt, checkpoint_path=None, device=_device())
        _models[name] = (predictor, decoder)
        logger.info(f"micro-sam {mt} loaded")
        return _models[name]
    except Exception as e:
        _errors[name] = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
        logger.error(f"micro-sam {name} failed to load: {_errors[name]}")
        raise HTTPException(status_code=503, detail=f"{name} unavailable: {_errors[name]}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    for name in PRELOAD:
        if name in MODEL_TYPES:
            try:
                get_model(name)
            except HTTPException:
                pass
    yield


app = FastAPI(title="micro-sam Inference Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models": {name: ("loaded" if name in _models else ("error: " + _errors[name]) if name in _errors else "not loaded yet")
                   for name in MODEL_TYPES},
        "model_types": MODEL_TYPES,
        "prompt_types": ["points", "box", "prompt_free"],
        "supported_output_types": list(OUTPUT_TYPES),
    }


def _to_model_input(image: Image.Image) -> np.ndarray:
    """micro-sam accepts 2-D grayscale or HxWx3 RGB. Collapse RGB that is really grayscale."""
    arr = np.array(image)
    if arr.ndim == 3 and arr.shape[2] == 3 and np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]):
        return arr[..., 0]
    return arr


@app.post("/predict", response_model=InferenceResponse)
async def predict(req: InferenceRequest):
    model = resolve_model(req)
    if model not in MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"This service only serves {list(MODEL_TYPES)}; got '{model}'")
    output_type = validate_request(req, model)
    image = load_image(req.image)
    W, H = image.size
    predictor, decoder = get_model(model)

    if req.prompt_free:
        masks, scores = run_ais(predictor, decoder, image, req.min_area)
        return build_response(model, masks, scores, (W, H), output_type,
                              req.polygon_tolerance, req.min_polygon_area,
                              min_area=req.min_area, max_objects=req.max_objects)

    boxes = normalize_boxes(req.box)
    multi_box = boxes is not None and len(boxes) > 1
    if multi_box and req.points:
        raise HTTPException(status_code=400, detail="'points' cannot be combined with several boxes; send one box or only boxes")
    masks, scores = run_prompted(predictor, image, req.points, boxes, req.multimask)
    return build_response(model, masks, scores, (W, H), output_type,
                          req.polygon_tolerance, req.min_polygon_area,
                          keep_order=multi_box, box_indices=list(range(len(masks))) if multi_box else None)


def run_prompted(predictor, image: Image.Image, points, boxes: Optional[np.ndarray], multimask: bool):
    """Point / box prompts through the (segment_anything) SamPredictor wrapped by micro-sam."""
    import torch
    rgb = np.array(image.convert("RGB"))
    point_coords, point_labels = points_to_arrays(points)
    out_masks, out_scores = [], []
    with torch.inference_mode():
        predictor.set_image(rgb)
        if boxes is not None and len(boxes) > 1:
            for b in boxes:  # SAM v1 predictor takes one box at a time
                m, s, _ = predictor.predict(box=b, multimask_output=False)
                out_masks.append(m[0]); out_scores.append(float(s[0]))
        else:
            m, s, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels,
                                        box=boxes[0] if boxes is not None else None,
                                        multimask_output=multimask)
            out_masks.extend(list(m)); out_scores.extend([float(v) for v in s])
    logger.info(f"micro-sam prompted points={0 if point_coords is None else len(point_coords)} "
                f"boxes={0 if boxes is None else len(boxes)} -> {len(out_masks)} mask(s)")
    return out_masks, np.array(out_scores, dtype=np.float32)


def run_ais(predictor, decoder, image: Image.Image, min_area: float):
    """Prompt-free automatic instance segmentation with micro-sam's extra decoder."""
    import torch
    from micro_sam.instance_segmentation import InstanceSegmentationWithDecoder
    ais = InstanceSegmentationWithDecoder(predictor, decoder)
    with torch.inference_mode():
        ais.initialize(_to_model_input(image))
        seg = ais.generate(min_size=int(min_area), output_mode="binary_mask")
    if isinstance(seg, np.ndarray) and seg.ndim == 2:  # label image
        masks = label_image_to_masks(seg)
        scores = np.ones(len(masks), dtype=np.float32)
    else:  # list of SAM-style dicts
        masks = [np.asarray(a["segmentation"]) for a in seg]
        scores = np.array([float(a.get("predicted_iou", 1.0)) for a in seg], dtype=np.float32)
    logger.info(f"micro-sam AIS -> {len(masks)} instance(s)")
    return masks, scores


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001)
