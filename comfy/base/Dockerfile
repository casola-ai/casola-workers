FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch with CUDA 12.4
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/comfyui && \
    pip install --no-cache-dir -r /app/comfyui/requirements.txt

# Worker runtime deps
COPY comfy/base/requirements.txt .
COPY casola_worker/requirements.txt ./worker-common-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r worker-common-requirements.txt

# Generic worker + shared transport
COPY casola_worker/ ./casola_worker/
COPY comfy/base/worker.py .
COPY comfy/base/entrypoint.sh .
RUN chmod +x worker.py entrypoint.sh

ENV COMFYUI_HOST=127.0.0.1
ENV COMFYUI_PORT=8188

# Ensure host's NVIDIA driver libraries are in the library path
# The container runtime mounts the host driver at /usr/local/nvidia/lib and /usr/local/nvidia/lib64
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib"

ENTRYPOINT ["/app/entrypoint.sh"]
CMD []
