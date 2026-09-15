# micro-sam (Segment Anything for Microscopy) inference service
# micro_sam depends on conda-only packages (vigra, nifty, elf), so this image is conda-based and
# separate from the SAM2/SAM3 pip image. The gateway on :8000 proxies model=micro-sam-* here.
FROM mambaorg/micromamba:2.0-ubuntu22.04

USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# GPU build of torch per the micro-sam install docs: micro_sam "libtorch=*=cuda*".
# CONDA_OVERRIDE_CUDA tells the solver a CUDA driver will be present at runtime (no GPU during docker build);
# cuda-version pins the CUDA runtime the torch build targets. Keep it <= what the host driver supports
# (driver 535 -> CUDA 12.2, which still runs 12.6 builds via minor-version compatibility; 12.9 builds fail with
# "CUDA error: named symbol not found"). Override at build time: --build-arg CUDA_VERSION=12.8
ARG CUDA_VERSION=12.6
RUN CONDA_OVERRIDE_CUDA=${CUDA_VERSION} micromamba install -y -n base -c conda-forge \
        micro_sam \
        "cuda-version=${CUDA_VERSION}" \
        "libtorch=*=cuda*" \
        fastapi uvicorn python-multipart requests py-opencv pillow \
    && micromamba clean -a -y

ENV PATH=/opt/conda/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cuda:0
ENV MICROSAM_CACHEDIR=/models
ENV MICROSAM_SIZE=b

WORKDIR /app
COPY services/common/segmentation_common.py .
COPY services/microsam_inference/main.py .

EXPOSE 8001
CMD ["python", "main.py"]
