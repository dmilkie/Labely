# micro-sam (Segment Anything for Microscopy) inference service
# micro_sam depends on conda-only packages (vigra, nifty, elf), so this image is conda-based and
# separate from the SAM2/SAM3 pip image. The gateway on :8000 proxies model=micro-sam-* here.
#
# Before building, stage the weights (they are baked into the image so a cold start never downloads):
#     python scripts/fetch_microsam_weights.py          # -> services/microsam_inference/weights/
FROM mambaorg/micromamba:2.0-ubuntu22.04

USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# GPU build of torch per the micro-sam install docs: micro_sam "libtorch=*=cuda*".
# CONDA_OVERRIDE_CUDA tells the solver a CUDA driver will be present at runtime (there is no GPU during
# docker build). CUDA_PIN fixes the CUDA runtime the torch build targets. Default 12.6: it runs on any
# NVIDIA driver >= 525 (via CUDA minor-version compatibility) and matches the gateway image. Without the
# pin conda picks the newest build (12.9), which needs driver >= 575 and otherwise fails at load time with
# "CUDA error: named symbol not found". Override: --build-arg CUDA_VERSION=12.9 --build-arg CUDA_PIN=""
ARG CUDA_VERSION=12.6
ARG CUDA_PIN="cuda-version=12.6"
RUN CONDA_OVERRIDE_CUDA=${CUDA_VERSION} micromamba install -y -n base -c conda-forge \
        micro_sam \
        ${CUDA_PIN} \
        "libtorch=*=cuda*" \
        fastapi uvicorn python-multipart requests py-opencv pillow \
    && micromamba clean -a -y

ENV PATH=/opt/conda/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cuda:0
ENV MICROSAM_SIZE=b

# Baked weights. micro_sam looks in $MICROSAM_CACHEDIR/models/<name> and verifies the xxh128 hash,
# so nothing is downloaded at runtime as long as the staged files are intact.
ENV MICROSAM_CACHEDIR=/opt/microsam
COPY services/microsam_inference/weights/ /opt/microsam/models/

WORKDIR /app
COPY services/common/segmentation_common.py .
COPY services/microsam_inference/main.py .

EXPOSE 8001
CMD ["python", "main.py"]
