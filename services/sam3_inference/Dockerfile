# SAM2 + SAM3 inference service Dockerfile
# SAM2 works with no token; SAM3 text prompts need HF_TOKEN (see .env.example)
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

WORKDIR /tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install SAM2 from source. SAM2_BUILD_CUDA=0 skips the optional CUDA extension
# (only used for mask post-processing) so no nvcc is needed in the runtime image.
RUN git clone https://github.com/facebookresearch/sam2.git && \
    cd sam2 && \
    SAM2_BUILD_CUDA=0 pip install --no-cache-dir --no-build-isolation . && \
    cd .. && rm -rf sam2

# Install SAM3 (text/concept prompts). Weights are NOT baked in: they are a gated HF repo and are
# downloaded at first use into the mounted HF cache using HF_TOKEN from .env.
RUN pip install --no-cache-dir einops decord pycocotools huggingface-hub && \
    git clone https://github.com/facebookresearch/sam3.git && \
    cd sam3 && \
    pip install --no-cache-dir --no-build-isolation . && \
    cd .. && rm -rf sam3

# Download the SAM 2.1 checkpoint from Meta's public CDN (no auth).
# Override SAM2_SIZE at build time: tiny | small | base_plus | large
ARG SAM2_SIZE=large
ENV SAM2_SIZE=${SAM2_SIZE}
RUN mkdir -p /app/checkpoints && \
    curl -fL -o /app/checkpoints/sam2.1_hiera_${SAM2_SIZE}.pt \
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_${SAM2_SIZE}.pt

WORKDIR /app

COPY services/sam3_inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/common/segmentation_common.py .
COPY services/sam3_inference/ .

ENV PYTHONUNBUFFERED=1
ENV DEVICE=cuda:0

EXPOSE 8000

CMD ["python", "main.py"]
