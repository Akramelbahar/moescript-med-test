FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV MUSCRIPTOR_MODEL=large
ENV PIP_DISABLE_PIP_VERSION_CHECK=1



# =========================================================
# SYSTEM PACKAGES
# =========================================================

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# =========================================================
# UPDATE PIP TOOLS
# =========================================================

RUN python -m pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel


# =========================================================
# TORCH + TORCHAUDIO
# =========================================================

RUN python -m pip install --no-cache-dir \
    torch==2.8.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128


# =========================================================
# FIX RUNPOD CRYPTOGRAPHY CONFLICT
#
# Base image contains Debian cryptography 41.0.7.
# RunPod requires >=50.
#
# --ignore-installed prevents pip trying to uninstall
# Debian's package.
# =========================================================

RUN python -m pip install --no-cache-dir \
    --ignore-installed \
    "cryptography>=50.0.0"


# =========================================================
# RUNPOD
# =========================================================

RUN python -m pip install --no-cache-dir \
    runpod \
    "requests>=2.32"


# =========================================================
# MUSCRIPTOR INFERENCE DEPENDENCIES
# =========================================================

RUN python -m pip install --no-cache-dir \
    "numpy>=1.24" \
    "einops>=0.4" \
    "mido>=1.3" \
    "packaging>=20.0" \
    "safetensors>=0.4" \
    "huggingface-hub>=0.13" \
    "soundfile>=0.14.0" \
    "beat-this==1.1.0"


# =========================================================
# MUSCRIPTOR
# =========================================================

RUN python -m pip install --no-cache-dir \
    --no-deps \
    muscriptor==0.3.0


# =========================================================
# VERIFY
# =========================================================

RUN python -c "import cryptography; print('Cryptography:', cryptography.__version__)" && \
    python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)" && \
    python -c "import torchaudio; print('TorchAudio:', torchaudio.__version__)" && \
    python -c "import runpod; print('RunPod: OK')" && \
    python -c "from muscriptor import TranscriptionModel; print('MuScriptor: OK')"


# =========================================================
# HANDLER
# =========================================================

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
