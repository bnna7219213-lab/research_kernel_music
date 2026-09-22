# ============================================================
# T2Music Phase A — Base Image
# Target: NVIDIA A100 80GB, CUDA 12.x, PyTorch 2.5+
# Usage: docker build -t registry/t2music:phase-a .
# ============================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------- system deps ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs wget curl ffmpeg libsndfile1 \
        python3.11 python3.11-dev python3-pip \
        ninja-build && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------- Python base deps ----------
RUN python -m pip install --upgrade pip setuptools wheel

# ---------- PyTorch + CUDA 12.4 ----------
RUN pip install --no-cache-dir \
        torch==2.5.1+cu124 \
        torchvision==0.20.1+cu124 \
        torchaudio==2.5.1+cu124 \
        -f https://download.pytorch.org/whl/torch_stable.html

# ---------- audio / ml stack ----------
RUN pip install --no-cache-dir \
        transformers==4.46.3 \
        accelerate==1.1.1 \
        datasets==3.1.0 \
        soundfile==0.12.1 \
        librosa==0.10.2.post1 \
        audiomentations==0.37.0 \
        pedalboard==0.9.14 \
        einops==0.8.0 \
        einx==0.3.0 \
        omegaconf==2.3.0 \
        tensorboard==2.18.0 \
        wandb==0.18.7 \
        mlflow==2.18.0

# ---------- evaluation ----------
RUN pip install --no-cache-dir \
        frechet-audio-distance==0.3.2 \
        laion-clap==1.1.6 \
        torchlibrosa==0.1.0 \
        pesq==0.0.4 \
        pystoi==0.4.1

# ---------- data / storage ----------
RUN pip install --no-cache-dir \
        webdataset==0.2.106 \
        boto3==1.35.63 \
        smart_open==7.0.5

# ---------- utilities ----------
RUN pip install --no-cache-dir \
        tqdm==4.67.1 \
        loguru==0.7.2 \
        orjson==3.10.11 \
        h5py==3.12.1 \
        typer==0.13.0

# ---------- code ----------
WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir -e /workspace || true

ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
CMD ["help"]
