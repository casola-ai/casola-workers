FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip \
        git wget curl ffmpeg libgl1 libglib2.0-0 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch with CUDA 12.4
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install LightX2V from GitHub (pinned for reproducible builds)
RUN pip install --no-cache-dir "git+https://github.com/ModelTC/LightX2V.git@3d129372e017ad8b1966f1ea73d805ba95a18190"

# Worker runtime deps
COPY lightx2v/requirements.txt .
COPY casola_worker/requirements.txt ./worker-common-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r worker-common-requirements.txt

# Generic worker + shared transport
COPY casola_worker/ ./casola_worker/
COPY --chmod=+x lightx2v/worker.py .
COPY --chmod=+x shared/entrypoint.sh ./shared/
COPY --chmod=+x lightx2v/entrypoint.sh .
COPY lightx2v/configs/ ./configs/

ENV LIGHTX2V_HOST=127.0.0.1
ENV LIGHTX2V_PORT=8000

# Ensure host's NVIDIA driver libraries are in the library path
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib"

ENTRYPOINT ["/app/entrypoint.sh"]
CMD []
