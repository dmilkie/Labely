"""
Minimal example: send a JPEG + point/box prompt to the SAM2 inference service.

Endpoint:  POST http://localhost:8000/predict
Body (JSON):
    {
      "image":       "data:image/jpeg;base64,<base64 of the jpeg bytes>",
      "prompt":      "dogs",                                 # text prompt -> SAM3 (needs HF_TOKEN)
      "points":      [{"x": 640, "y": 420, "label": 1}],   # label 1 = include, 0 = exclude  -> SAM2
      "box":         [x1, y1, x2, y2],                       # optional, pixels
      "output_type": "bbox"                                  # or "segment" (adds RLE mask)
    }
    Send either "prompt" (SAM3) or "points"/"box" (SAM2).
Response (JSON):
    {
      "masks": [ {"mask": [...] or null, "score": 0.97, "bbox": [x1, y1, x2, y2]} ],
      "image_size": [width, height]
    }

Usage:
    python sam3_predict.py IMAGE                    # one point at the image centre (model sam2)
    python sam3_predict.py IMAGE --model micro-sam-lm 640,420   # choose the model: sam2 | sam3 | micro-sam-lm | micro-sam-em
    python sam3_predict.py IMAGE --auto             # prompt-free: every object (sam2 or micro-sam-*)
    python sam3_predict.py IMAGE --auto --model micro-sam-lm --min-area 200
    python sam3_predict.py IMAGE 640,420            # one foreground point
    python sam3_predict.py IMAGE 640,420 300,400    # several points
    python sam3_predict.py IMAGE box:100,50,900,700 # a box
    python sam3_predict.py IMAGE box:... box:...     # several boxes -> one mask per box, in order
    python sam3_predict.py IMAGE text:dogs          # a text prompt (SAM3, needs HF_TOKEN on the server)
    add  --segment  to also get the RLE mask, and  --save out.png  to write a mask overlay
    add  --polygon  to get simplified contour vertices [[x,y],...] instead of a mask (lasso / ROI)
    add  --min-area N  (prompt-free) to drop objects smaller than N px^2, --max-objects N to keep the N best
"""
import base64
import json
import sys

import requests

URL = "http://localhost:8000/predict"

argv = sys.argv[1:]


def take_option(name, default=None, cast=str):
    """Pop '--name value' from argv."""
    if name in argv:
        i = argv.index(name)
        val = cast(argv[i + 1])
        del argv[i:i + 2]
        return val
    return default


SAVE_PATH = take_option("--save")
MODEL = take_option("--model")
MIN_AREA = take_option("--min-area", 0.0, float)
MAX_OBJECTS = take_option("--max-objects", None, int)
AUTO = "--auto" in argv
OUTPUT_TYPE = "polygon" if "--polygon" in argv else ("segment" if "--segment" in argv or SAVE_PATH else "bbox")
args = [a for a in argv if not a.startswith("--")]
IMAGE_PATH = args[0] if args else "pexels-chevanon-1108099.jpg"

# 1. Read the JPEG bytes and base64-encode them into a data URI
with open(IMAGE_PATH, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")
payload = {"image": f"data:image/jpeg;base64,{b64}", "output_type": OUTPUT_TYPE}
if MODEL:
    payload["model"] = MODEL
if AUTO:
    payload["prompt_free"] = True
    payload["min_area"] = MIN_AREA
    if MAX_OBJECTS is not None:
        payload["max_objects"] = MAX_OBJECTS

# 2. Add the prompt (points and/or box, pixel coordinates)
points, box, text = [], None, None
for a in args[1:]:
    if a.startswith("text:"):
        text = a[5:]
    elif a.startswith("box:"):
        b = [float(v) for v in a[4:].split(",")]
        if box is None:
            box = b
        elif isinstance(box[0], list):
            box.append(b)
        else:
            box = [box, b]
    else:
        x, y = a.split(",")
        points.append({"x": float(x), "y": float(y), "label": 1})
if AUTO:
    pass  # no prompt needed
elif text:
    payload["prompt"] = text
elif not points and box is None:
    from PIL import Image
    w, h = Image.open(IMAGE_PATH).size
    points = [{"x": w / 2, "y": h / 2, "label": 1}]
if points:
    payload["points"] = points
if box is not None:
    payload["box"] = box

# 3. POST (Content-Type: application/json is set by json=)
resp = requests.post(URL, json=payload, timeout=120)
resp.raise_for_status()
result = resp.json()

# 4. Use the result
w, h = result["image_size"]
print(f"model: {result.get('model')}   image size: {w}x{h}")
print(f"prompt: {'PROMPT-FREE' if AUTO else ''} text={text!r} points={points} box={box}")
print(f"objects: {len(result['masks'])}")
for i, m in enumerate(result["masks"][:25]):
    x1, y1, x2, y2 = m["bbox"]
    line = f"  #{i}: score={m['score']:.3f}  bbox=[x1={x1}, y1={y1}, x2={x2}, y2={y2}]"
    if m.get("box_index") is not None:
        line += f"  (input box {m['box_index']})"
    if m.get("mask") is not None:
        line += f"  rle_len={len(m['mask'])}"
    if m.get("polygons"):
        line += f"  polygons={len(m['polygons'])} vertices={[len(p) for p in m['polygons']]}"
    if m.get("area") is not None:
        line += f"  area={m['area']}"
    print(line)
if len(result["masks"]) > 25:
    print(f"  ... {len(result['masks']) - 25} more")

# Optional: draw the polygon outline and save it
if SAVE_PATH and result["masks"] and result["masks"][0].get("polygons"):
    from PIL import Image, ImageDraw
    img = Image.open(IMAGE_PATH).convert("RGB")
    d = ImageDraw.Draw(img)
    import colorsys
    for k, m in enumerate(result["masks"]):
        col = tuple(int(255 * c) for c in colorsys.hsv_to_rgb((k * 0.61803) % 1.0, 1.0, 1.0))
        for poly in m.get("polygons") or []:
            d.polygon([tuple(p) for p in poly], outline=col, width=max(2, w // 600))
    img.save(SAVE_PATH)
    print(f"saved polygon overlay to {SAVE_PATH}")

# Optional: decode the RLE and save a red overlay so you can eyeball the mask
if SAVE_PATH and result["masks"] and result["masks"][0].get("mask"):
    import numpy as np
    from PIL import Image
    import colorsys
    img = np.array(Image.open(IMAGE_PATH).convert("RGB")).astype(np.float32)
    for k, m in enumerate(result["masks"]):
        rle = m.get("mask") or []
        flat = np.zeros(w * h, dtype=bool)
        for start, length in zip(rle[0::2], rle[1::2]):
            flat[start - 1:start - 1 + length] = True
        mask = flat.reshape(h, w)
        col = np.array([255 * c for c in colorsys.hsv_to_rgb((k * 0.61803) % 1.0, 1.0, 1.0)])
        img[mask] = img[mask] * 0.4 + col * 0.6
    Image.fromarray(img.astype(np.uint8)).save(SAVE_PATH)
    print(f"saved overlay to {SAVE_PATH}")

trimmed = json.loads(json.dumps(result))
trimmed["masks"] = trimmed["masks"][:3] + (["..."] if len(trimmed["masks"]) > 3 else [])
for m in trimmed["masks"]:
    if not isinstance(m, dict):
        continue
    if m.get("mask"):
        m["mask"] = m["mask"][:8] + ["..."]
    if m.get("polygons"):
        m["polygons"] = [p[:6] + ["..."] for p in m["polygons"]]
print("\nraw JSON:\n" + json.dumps(trimmed, indent=2))
