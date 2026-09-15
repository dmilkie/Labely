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

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (for SAM3 inference)
- NVIDIA Container Toolkit

### Start All Services

```bash
docker compose up -d
```

### Verify Services

```bash
# SAM3 Inference Service
curl http://localhost:8000/health

# Label Studio Adapter
curl http://localhost:9090/health

# Label Studio Web UI
open http://localhost:8080
```

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

The inference container runs two models behind one endpoint:

| prompt | model | needs |
|---|---|---|
| `prompt` (text, e.g. `"dogs"`) | **SAM3** | `HF_TOKEN` in `.env` with access to the gated `facebook/sam3` repo |
| `points` / `box` | **SAM 2.1** | nothing (public checkpoint baked into the image) |

Without a token the service still starts; text prompts return HTTP 503 with the reason.
To enable SAM3: request access at https://huggingface.co/facebook/sam3, create a read token,
put `HF_TOKEN=hf_...` in `.env` (see `.env.example`), then `docker compose up -d --force-recreate sam3-inference`.
Weights download into the mounted HF cache on first use.

#### POST `/predict`

**Request:**
```json
{
  "image": "data:image/jpeg;base64,...",
  "prompt": "dogs",
  "points": [{"x": 640, "y": 420, "label": 1}],
  "box": [100, 50, 900, 700],
  "output_type": "segment",
  "multimask": false
}
```

**Parameters:**
- `image` (string, required): Base64 data URI, http(s) URL, or server-side path
- `prompt` (string, optional): text / concept prompt, routed to SAM3; returns one mask per detected instance
- `points` (list, optional): pixel coordinates; `label` 1 = include, 0 = exclude (SAM2)
- `box` (list, optional): `[x1, y1, x2, y2]` in pixels, or a list of boxes `[[...], [...]]` (SAM2). Several boxes give one mask per box, returned in input order with `box_index`; points cannot be combined with several boxes
- Send either `prompt` or `points`/`box`, not both
- `output_type` (string, optional): `"bbox"`, `"segment"` (RLE mask) or `"polygon"` (contour vertices) (default: `"segment"`)
- `multimask` (bool, optional): return SAM2's 3 candidate masks instead of the best one
- `polygon_tolerance` (float, optional, polygon mode): max simplification error in px (default 2.0; 0 = keep every boundary pixel)
- `min_polygon_area` (float, optional, polygon mode): drop islands smaller than this many px² (default 0)

**Response:**
```json
{
  "masks": [
    {
      "mask": [0, 100, 255, ...],  // RLE [start, length, ...] (segment mode only)
      "polygons": [[[x, y], [x, y], ...], ...],  // polygon mode only; outer contours, largest first
      "score": 0.97,
      "bbox": [50, 50, 150, 150]   // [x1, y1, x2, y2]
    }
  ],
  "image_size": [640, 480]
}
```

Model size is chosen at build time: `docker compose build --build-arg SAM2_SIZE=base_plus sam3-inference`
(`tiny` | `small` | `base_plus` | `large`, default `large`).

Example client: `python sam3_predict.py image.jpg text:dogs --polygon --save overlay.png` or `python sam3_predict.py image.jpg 640,420 --segment`

#### GET `/health`

```json
{
  "status": "healthy",
  "model": "SAM2.1-large",
  "prompt_types": ["points", "box"],
  "supported_output_types": ["bbox", "segment", "polygon"]
}
```

`POST /predict/upload` accepts the same prompts as `multipart/form-data` (file part `image`, text fields `points`, `box`, `output_type`, ...).
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
