# RunPod Serverless Dockerfile for Chandra OCR
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/runpod-volume/.cache/torch
ENV HF_HOME=/runpod-volume/.cache/huggingface
ENV MODEL_CHECKPOINT=datalab-to/chandra
ENV TORCH_DEVICE=cuda
ENV MAX_OUTPUT_TOKENS=12384

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Install Python dependencies
RUN pip3 install --no-cache-dir -e . && \
    pip3 install --no-cache-dir runpod requests

# Install flash-attention for better performance (optional but recommended)
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation || echo "Flash attention installation failed, continuing without it"

# Pre-download the model during build (optional - comment out if you want to download on first run)
# RUN python3 -c "from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor; \
#     Qwen3VLForConditionalGeneration.from_pretrained('datalab-to/chandra'); \
#     Qwen3VLProcessor.from_pretrained('datalab-to/chandra')"

# Create cache directories
RUN mkdir -p /runpod-volume/.cache/torch /runpod-volume/.cache/huggingface

# Set the entrypoint
CMD ["python3", "-u", "runpod_handler.py"]
