"""
Shared request/response schema and mask post-processing for the Labely inference services.

Used by both containers (SAM2/SAM3 on :8000, micro-sam on :8001) so the JSON API is identical.
Copied into each image as /app/segmentation_common.py.
"""
import base64
import os
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


# ---------- CUDA / driver compatibility ----------
# Minimum NVIDIA driver (Linux / Windows) that supports each CUDA runtime minor version, per NVIDIA's
# CUDA Toolkit release notes. Used only to phrase the advice; the decision comes from the probe below.
_MIN_DRIVER = {
    (12, 0): "525", (12, 1): "530", (12, 2): "535", (12, 3): "545", (12, 4): "550",
    (12, 5): "555", (12, 6): "560", (12, 8): "570", (12, 9): "575", (13, 0): "580",
}
DRIVER_DOWNLOAD_URL = "https://www.nvidia.com/drivers"


def _parse_ver(v):
    try:
        major, minor = str(v).split(".")[:2]
        return int(major), int(minor)
    except (ValueError, AttributeError):
        return None


def cuda_compat_report() -> dict:
    """Compare the CUDA version this image was built for with what the host driver supports, and run a
    tiny GPU kernel. Returns a dict; `ok` False plus a human-readable `warning` when the GPU cannot be used.

    Typical failure: image built for CUDA 12.9, host driver only supports 12.2 -> kernels fail with
    "CUDA error: named symbol not found". The fix is always on the HOST: update the NVIDIA display driver.
    Set LABELY_FAKE_DRIVER_CUDA=12.2 to exercise the warning path on a compatible machine (testing only).
    """
    import torch
    rep = {"build_cuda": torch.version.cuda, "torch": torch.__version__, "driver_cuda": None,
           "driver_version": None, "gpu": None, "ok": True, "warning": None}
    if not torch.cuda.is_available():
        rep["ok"] = False
        rep["warning"] = ("No CUDA device visible to this container. Check that the NVIDIA driver is installed on the "
                          "host, Docker has GPU support (`docker run --gpus all ...` works) and the compose file keeps "
                          "the nvidia device reservation.")
        return rep
    try:
        rep["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    # driver-supported CUDA version, straight from libcuda (injected by the NVIDIA container toolkit)
    try:
        import ctypes
        lib = ctypes.CDLL("libcuda.so.1")
        v = ctypes.c_int()
        lib.cuInit(0)
        if lib.cuDriverGetVersion(ctypes.byref(v)) == 0 and v.value:
            rep["driver_cuda"] = f"{v.value // 1000}.{(v.value % 1000) // 10}"
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        if out:
            rep["driver_version"] = out[0].strip()
    except Exception:
        pass
    fake = os.getenv("LABELY_FAKE_DRIVER_CUDA")  # testing only: pretend the driver is this old and the probe failed
    if fake:
        rep["driver_cuda"] = fake
        rep["faked"] = True

    # functional probe: a real kernel launch + matmul catches "named symbol not found" and friends.
    # This is the deciding test. A driver older than the image's CUDA version often still works thanks to
    # CUDA minor-version compatibility (e.g. a 12.6 build on a 12.2 driver), so the version gap alone is
    # only reported as a note.
    probe_error = None
    if fake:
        probe_error = "simulated (LABELY_FAKE_DRIVER_CUDA)"
    else:
        try:
            x = torch.randn(64, 64, device="cuda")
            (x @ x).sum().item()
            torch.cuda.synchronize()
        except Exception as e:
            probe_error = str(e).splitlines()[0][:200]

    build, drv = _parse_ver(rep["build_cuda"]), _parse_ver(rep["driver_cuda"])
    version_gap = build is not None and drv is not None and drv < build
    need = _MIN_DRIVER.get(build, "the latest") if build else "the latest"
    drv_txt = f"NVIDIA driver {rep['driver_version'] or '?'} (supports CUDA <= {rep['driver_cuda'] or '?'})"
    if probe_error:
        rep["ok"] = False
        rep["warning"] = (
            f"CUDA MISMATCH: this image was built for CUDA {rep['build_cuda']} but the host's {drv_txt} "
            f"cannot run it (GPU probe failed: {probe_error}). FIX ON THE HOST MACHINE: update the NVIDIA "
            f"display driver to version {need} or newer ({DRIVER_DOWNLOAD_URL}), then restart Docker Desktop "
            f"and run `docker compose up -d` again. (Alternative: rebuild this image with a CUDA pin matching "
            f"the driver, see services/*/Dockerfile.)"
        )
    elif version_gap:
        rep["note"] = (
            f"Host {drv_txt} is older than this image's CUDA {rep['build_cuda']}; it works through CUDA "
            f"minor-version compatibility. Updating the NVIDIA driver to {need}+ is recommended."
        )
    return rep


def log_cuda_compat(logger, service: str) -> dict:
    """Run cuda_compat_report() and log it prominently. Returns the report."""
    import os as _os
    rep = cuda_compat_report()
    if rep["ok"]:
        logger.info(f"[{service}] GPU {rep['gpu']}: driver {rep['driver_version']} (CUDA <= {rep['driver_cuda']}), "
                    f"image built for CUDA {rep['build_cuda']} - compatible")
        if rep.get("note"):
            logger.warning(f"[{service}] {rep['note']}")
    else:
        bar = "!" * 100
        logger.error(f"\n{bar}\n[{service}] {rep['warning']}\n{bar}")
    return rep
