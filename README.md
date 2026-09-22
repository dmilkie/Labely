# Labely - SAM3 Interactive Segmentation for Label Studio

A production-ready architecture for integrating Meta's SAM3 (Segment Anything Model 3) with Label Studio for interactive image segmentation and annotation.

## 🎯 Overview

Labely provides a clean, modular architecture that connects Label Studio with SAM3 inference capabilities. The system is designed with separation of concerns: a lightweight adapter handles Label Studio integration while a dedicated inference service handles all SAM3 model operations.

## 🏗️ Architecture

```
┌─────────────────┐
│ Label Studio    │  Port: 8080
│ (Web UI)        │
└────────┬────────┘
         │
         ↓ HTTP API
┌─────────────────┐
│ LS Adapter      │  Port: 9090
│ (Lightweight)   │  ← Label Studio ML Backend
└────────┬────────┘
         │
         ↓ HTTP API
┌─────────────────┐
│ SAM3 Inference  │  Port: 8000
│ (FastAPI)       │  ← Pure inference service
└─────────────────┘
```

## ✨ Features

- **Modular Design**: Separated inference service and adapter for better scalability
- **Text-Prompt Segmentation**: Use natural language prompts to segment objects
- **Flexible Output Types**: Support for both bounding boxes and segmentation masks
- **Auto-Label Matching**: Intelligent label matching from prompts to Label Studio configuration
- **Production Ready**: Docker-based deployment with proper error handling

## 📁 Project Structure

```
Labely/
├── services/
│   ├── sam3_inference/      # SAM3 inference service
│   │   ├── main.py          # FastAPI application
│   │   ├── requirements.txt # Inference dependencies
│   │   └── Dockerfile
│   └── ls_adapter/          # Label Studio adapter
│       ├── model.py         # Adapter logic
│       ├── app.py           # Application entry point
│       ├── requirements.txt # Adapter dependencies
│       └── Dockerfile
├── docker-compose.yml       # Service orchestration
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Docker Desktop (Windows/macOS) or Docker Engine + Compose (Linux)
- NVIDIA GPU with a recent driver (>= 570 for the unpinned micro-sam image, see its Dockerfile) and the NVIDIA Container Toolkit
- Optional: a Hugging Face token with access to `facebook/sam3` for text prompts (see [Configuration](#-configuration))

### First start (builds images, downloads weights)

```bash
cp .env.example .env                        # then put HF_TOKEN=hf_... in .env if you want SAM3 text prompts
python scripts/fetch_microsam_weights.py    # stages the micro-sam weights (~830 MB) for the image build
docker compose up -d --build sam3-inference micro-sam-inference
```

The first build takes a while: the SAM2/SAM3 image downloads the SAM 2.1 checkpoint (~900 MB) and the
micro-sam image solves a conda environment (~10 min) and bakes in the staged micro-sam weights. On first
container start, SAM3 (if `HF_TOKEN` is set) downloads its weights into the Hugging Face cache volume;
nothing else is downloaded at runtime.

### Cold start (images already built)

Only the inference containers are needed for an external client such as LabVIEW:

```bash
docker compose up -d sam3-inference micro-sam-inference
```

Add Label Studio, its database and the ML adapter too with plain `docker compose up -d`.

Startup times on an RTX 4070: port **8000** answers after ~40 s (SAM2 + SAM3 loading), port **8001** after ~15 s.
Both containers have `restart: unless-stopped`, so after a reboot Docker starts them automatically; you only
need the command above after an explicit `docker compose down` / `stop`.

### Verify

```bash
curl -f http://localhost:8000/ready   # HTTP 200 only when every model is loaded (503 + reasons otherwise)
curl http://localhost:8000/health     # gateway: reports sam2, sam3 and the micro-sam container state
curl http://localhost:8001/health     # micro-sam directly
curl http://localhost:9090/health     # Label Studio ML adapter (if started)
```

`/ready` is what a client should poll at start-up: connection refused means the models are still loading,
503 lists what is missing (including a CUDA/driver mismatch warning with the fix), 200 means go.
`/health` additionally carries a `cuda` block (image CUDA version, host driver, GPU, ok flag).

Then try the sample client:

```bash
python sam3_predict.py samples/human_mitosis.png --model micro-sam-lm --auto --polygon
```

### Stop / restart / rebuild

```bash
docker compose stop sam3-inference micro-sam-inference        # stop
docker compose restart sam3-inference micro-sam-inference     # restart
docker compose up -d --build sam3-inference micro-sam-inference   # rebuild after editing services/
```

Rebuilding the SAM2/SAM3 image after a Python-only change takes seconds (layers are cached); the micro-sam
image only re-solves conda when its Dockerfile changes.

### Known gotchas

- **No `HF_TOKEN`**: everything works except text prompts, which return HTTP 503 with the reason.
- **After an NVIDIA driver update** (Docker Desktop / WSL2): GPU containers fail to start with an `ld.so`
  assertion in the NVIDIA prestart hook, and running ones throw CUDA errors. Restart Docker Desktop
  (`wsl --shutdown` first), then `docker compose restart sam3-inference micro-sam-inference ls-adapter label-studio`
  to re-create the host port forwards, which can otherwise stay dead for auto-restarted containers.
- **Requests to `localhost` hang** (Windows + Docker Desktop): `localhost` may resolve to `::1`, where Docker
  Desktop keeps a listener it never forwards. Use `http://127.0.0.1:8000` in clients (the sample client does).
- **Port 8080 clash**: LabVIEW's `ApplicationWebServer` also listens on 8080 on machines with LabVIEW installed,
  which hides the Label Studio UI. Change the `label-studio` port mapping in `docker-compose.yml` if you need both.
- **Old NVIDIA driver / CUDA mismatch**: each image is built for a CUDA version (gateway 12.6, micro-sam 12.9
  by default). If the host driver is older than that CUDA version needs, kernels fail with
  "CUDA error: named symbol not found". Both services check this at start-up: they log a
  `CUDA MISMATCH` banner, skip loading models, report `"cuda": {"ok": false, "warning": ...}` in `/health`
  and list the warning in `/ready` (HTTP 503), including the minimum driver version. **Fix on the host:**
  update the NVIDIA display driver (https://www.nvidia.com/drivers, >= 570 for CUDA 12.8+), restart Docker
  Desktop, `docker compose up -d`. Alternative: rebuild micro-sam pinned to an older CUDA,
  `docker compose build --build-arg CUDA_PIN='"cuda-version=12.6"' --build-arg CUDA_VERSION=12.6 micro-sam-inference`.

## 📦 Deployment (pre-built images)

For machines that only run the API (e.g. a LabVIEW acquisition PC) use [`compose.deploy.yml`](compose.deploy.yml):
it pulls the two inference images from GHCR, has no build step and no Label Studio. Ports 8000/8001 are
published on all interfaces (binding to 127.0.0.1 only breaks `localhost` clients on Windows, see the comment
in the file); restrict with the host firewall if needed. Clients should address `http://127.0.0.1:8000`.

### Publish images (from a dev machine)

```bash
python scripts/fetch_microsam_weights.py
LABELY_TAG=v1.0.0 docker compose build sam3-inference micro-sam-inference   # tags ghcr.io/dmilkie/labely-{gateway,micro-sam}:v1.0.0
docker login ghcr.io                                                        # once; PAT with write:packages
LABELY_TAG=v1.0.0 docker compose push sam3-inference micro-sam-inference
```

Set `LABELY_TAG` in `.env` instead of the environment if you prefer. Repeat with `LABELY_TAG=latest` to move
the floating tag. Make the two packages public in GitHub so target machines need no `docker login`.
Image sizes: gateway ~8 GB, micro-sam ~15 GB (CUDA + torch + baked weights).

### Target machine

One-time: Docker Desktop (WSL2 backend, start on login), NVIDIA driver >= 570, a deploy folder with
`compose.deploy.yml` and a `.env` containing `LABELY_TAG=v1.0.0` and optionally `HF_TOKEN=hf_...`.

Then, from that folder (this is the sequence a client such as LabVIEW can run via System Exec):

```bash
docker info                                   # 1. engine up? retry / start Docker Desktop if not
docker compose -f compose.deploy.yml pull     # 2. fetch the pinned tag (non-fatal offline: local images are used)
docker compose -f compose.deploy.yml up -d    # 3. idempotent; recreates only containers whose image changed
curl -f http://127.0.0.1:8000/ready           # 4. poll until HTTP 200 (allow ~90 s warm, minutes on a first start with SAM3)
```

On failure, `docker compose -f compose.deploy.yml logs --tail 50` shows why (missing token, stale WSL2 driver, ...).

### Update / roll back

Change `LABELY_TAG` in `.env`, then run steps 2-4 again. SAM3 weights live in the `hf-cache` volume and the
micro-sam weights are inside the image, so updates never re-download model files.

## 🔧 Configuration

### Label Studio Setup

1. Access Label Studio at `http://localhost:8080`
2. Create a new project
3. Navigate to **Settings → Machine Learning**
4. Add ML Backend:
   - **URL**: `http://ls-adapter:9090` (or `http://localhost:9090` if accessing from host)
   - Check **"Use for interactive preannotations"**
5. Configure **Extra params** (optional):
   ```json
   {
     "prompt": "detect all objects",
     "output_type": "bbox",
     "label": "dog"
   }
   ```
6. Save configuration

### Labeling Configuration

For **bounding box detection**:
```xml
<View>
  <Image name="image" value="$image" zoom="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="dog" background="#FFA39E"/>
    <Label value="cat" background="#FFD700"/>
  </RectangleLabels>
</View>
```

For **segmentation masks**:
```xml
<View>
  <Image name="image" value="$image" zoom="true"/>
  <BrushLabels name="label" toName="image">
    <Label value="object" background="#FF0000"/>
  </BrushLabels>
</View>
```

## 📡 API Documentation

### Inference API (port 8000)

One endpoint, four models, selected with the `model` field:

| `model` | backend | prompts | needs |
|---|---|---|---|
| `sam2` (**default**) | SAM 2.1 (Meta) | `points`, `box`, `prompt_free` | nothing (public checkpoint baked into the image) |
| `sam3` | SAM3 (Meta) | `prompt` (text, e.g. `"dogs"`) | `HF_TOKEN` in `.env` with access to the gated `facebook/sam3` repo |
| `micro-sam-lm` | Segment Anything for Microscopy, light-microscopy model | `points`, `box`, `prompt_free` | nothing (weights from Zenodo on first start) |
| `micro-sam-em` | Segment Anything for Microscopy, electron-microscopy model | `points`, `box`, `prompt_free` | nothing |

If `model` is omitted it is `sam2`, except that a request carrying a text `prompt` defaults to `sam3`.
`prompt_free: true` (also accepted as `"prompt-free"`) segments **every object** with no prompt: SAM2 uses
automatic mask generation over a point grid, micro-sam uses its dedicated instance-segmentation decoder.

The micro-sam models run in their own conda-based container (`micro-sam-inference`, port 8001) because
they need conda-only dependencies; the gateway on 8000 forwards `model: micro-sam-*` requests to it, so
clients only ever talk to port 8000. Without a token SAM3 requests return HTTP 503 with the reason; without
the micro-sam container those requests return 503 too, everything else keeps working.

To enable SAM3: request access at https://huggingface.co/facebook/sam3, create a read token,
put `HF_TOKEN=hf_...` in `.env` (see `.env.example`), then `docker compose up -d --force-recreate sam3-inference`.

#### POST `/predict`

**Request** (all fields except `image` optional):
```json
{
  "image": "data:image/jpeg;base64,...",
  "model": "sam2",
  "prompt": "dogs",
  "points": [{"x": 640, "y": 420, "label": 1}],
  "box": [100, 50, 900, 700],
  "prompt_free": false,
  "output_type": "segment",
  "polygon_tolerance": 2.0,
  "min_area": 0,
  "max_objects": null
}
```

**Parameters** (only `image` is required):

| field | type | default | applies to | meaning |
|---|---|---|---|---|
| `image` | string | required | all | Base64 data URI, http(s) URL, or server-side path |
| `model` | string | `"sam2"` (`"sam3"` if `prompt` is set) | all | `sam2` \| `sam3` \| `micro-sam-lm` \| `micro-sam-em` |
| `prompt` | string | `null` | `sam3` | text / concept prompt; returns one mask per detected instance |
| `points` | list | `null` | `sam2`, `micro-sam-*` | `[{"x": px, "y": px, "label": 1}]`; `label` 1 = include, 0 = exclude (default 1) |
| `box` | list | `null` | `sam2`, `micro-sam-*` | `[x1, y1, x2, y2]` in pixels, or a list of boxes. Several boxes give one mask per box in input order with `box_index`; not combinable with `points` |
| `prompt_free` | bool | `false` | `sam2`, `micro-sam-*` | segment every object without a prompt (alias `"prompt-free"`) |
| `output_type` | string | `"segment"` | all | `"bbox"`, `"segment"` (RLE mask) or `"polygon"` (contour vertices) |
| `polygon_tolerance` | float | `2.0` | polygon mode | max simplification error in px; `0` keeps every boundary pixel |
| `min_area` | float | `0` | all | drop objects smaller than this many px²; in polygon mode also drops stray contour islands below it |
| `max_objects` | int | `null` (no limit) | all | keep only the N best-scoring objects |
| `points_per_side` | int | `32` | `sam2` + `prompt_free` | density of the point grid; more finds smaller objects, slower (clamped to 4..128) |
| `multimask` | bool | `false` | prompted `sam2`, `micro-sam-*` | return SAM's 3 candidate masks for a single prompt instead of the best one |

**Response:**
```json
{
  "model": "sam2",
  "masks": [
    {
      "mask": [0, 100, 255, ...],        // RLE [start, length, ...] 1-based on the row-major flattened image (segment mode only)
      "polygons": [[[x, y], [x, y], ...]], // polygon mode only; outer contours, largest first
      "box_index": null,                 // input box this mask belongs to when several boxes were sent
      "score": 0.97,                     // model confidence (predicted IoU); 1.0 for micro-sam prompt-free instances
      "bbox": [50, 50, 150, 150],        // [x1, y1, x2, y2]
      "area": 12345                      // mask area in px
    }
  ],
  "image_size": [640, 480]
}
```

Results are sorted best score first, except multi-box requests which keep input order.

Model size is chosen at build time: `docker compose build --build-arg SAM2_SIZE=base_plus sam3-inference`
(`tiny` | `small` | `base_plus` | `large`, default `large`); micro-sam size via `MICROSAM_SIZE` in `.env` (`t` | `b` | `l` | `h`).

Example client ([sam3_predict.py](sam3_predict.py)):
```bash
python sam3_predict.py image.jpg text:dogs --polygon --save overlay.png      # sam3 text prompt
python sam3_predict.py image.jpg 640,420 --segment                           # sam2 point
python sam3_predict.py image.jpg box:100,50,900,700 box:900,50,1500,700      # sam2, two boxes
python sam3_predict.py image.jpg --auto --min-area 500                       # sam2 prompt-free
python sam3_predict.py samples/human_mitosis.png --model micro-sam-lm --auto --polygon   # micro-sam prompt-free
```

#### GET `/health`

Reports which models are loaded, including the micro-sam container's own status.

`POST /predict/upload` accepts the same fields as `multipart/form-data` (file part `image`, text fields for the rest).
`POST /echo` reflects back the headers and parsed body of any request, useful when debugging a client.

### Label Studio Adapter API

The adapter implements the standard [Label Studio ML Backend API](https://labelstud.io/guide/ml.html):

- `POST /setup` - Initialize model with project configuration
- `POST /predict` - Generate predictions
- `GET /health` - Health check

## 🛠️ Development

### Local Development - Inference Service

```bash
cd services/sam3_inference
pip install -r requirements.txt
python main.py
```

### Local Development - Adapter

```bash
cd services/ls_adapter
pip install -r requirements.txt
python app.py
```

### Environment Variables

#### SAM3 Inference Service

| Variable | Description | Default |
|----------|-------------|---------|
| `DEVICE` | GPU device | `cuda:0` |
| `DEBUG` | Enable debug mode | `false` |
| `HF_TOKEN` | Hugging Face token (optional) | - |
| `SAM3_TEXT_PROMPT` | Default text prompt | `segment all objects` |

#### Label Studio Adapter

| Variable | Description | Default |
|----------|-------------|---------|
| `SAM3_INFERENCE_URL` | Inference service URL | `http://sam3-inference:8000` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `LABEL_STUDIO_URL` | Label Studio URL | `http://label-studio:8080` |
| `LABEL_STUDIO_ACCESS_TOKEN` | Label Studio access token | - |

## 📊 System Requirements

- **SAM3 Inference Service**:
  - NVIDIA GPU with CUDA support
  - ~6GB VRAM recommended
  - CUDA 12.6+ recommended

- **Label Studio Adapter**:
  - Lightweight service
  - ~512MB RAM

- **Label Studio**:
  - ~1GB RAM
  - PostgreSQL (included in docker-compose)

## 🐛 Troubleshooting

### Inference Service Not Starting

**Check GPU availability:**
```bash
nvidia-smi
```

**View logs:**
```bash
docker compose logs sam3-inference
```

**Common issues:**
- GPU not detected: Ensure NVIDIA Container Toolkit is installed
- CUDA out of memory: Reduce batch size or use a smaller model variant

### Adapter Connection Issues

**Test connectivity:**
```bash
docker compose exec ls-adapter ping sam3-inference
```

**View logs:**
```bash
docker compose logs ls-adapter
```

**Common issues:**
- Connection refused: Check if SAM3 inference service is running
- Timeout errors: Verify network configuration in docker-compose.yml

### No Predictions Showing in Label Studio

1. **Check label matching**: Ensure the label name in your prompt matches a label in your Label Studio configuration
2. **Verify extra params**: Check that `output_type` matches your labeling configuration (bbox vs segment)
3. **Check logs**: Review adapter logs for conversion errors

### Bounding Box Coordinates Incorrect

The service automatically detects whether SAM3 returns normalized (0-1) or pixel coordinates. If issues persist:
- Check logs for "Raw box" values
- Verify image dimensions match `image_size` in response

## 🔍 Advanced Usage

### Custom Prompts

Configure prompts in Label Studio's ML Backend settings:

```json
{
  "prompt": "detect all fire and smoke",
  "output_type": "bbox"
}
```

The adapter intelligently matches prompt keywords to available labels in your configuration.

### Multiple Labels

For projects with multiple labels, the adapter will:
1. Use explicitly specified `label` parameter if provided
2. Match label names from the prompt
3. Fall back to the first available label

### Output Type Selection

- **`bbox`**: Returns bounding boxes only (faster, less accurate)
- **`segment`**: Returns full segmentation masks (slower, more accurate)

## 📝 License

MIT License

## 🙏 Acknowledgments

- [Meta SAM3](https://github.com/facebookresearch/sam3) - Segment Anything Model 3
- [Label Studio](https://labelstud.io/) - Open source data labeling platform
# Labely
