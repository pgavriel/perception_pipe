# Dockerfile (GPU-enabled)

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Optional: set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-venv
# RUN pip install opencv-python
RUN apt-get update && apt-get install -y python3-tk
RUN pip install watchdog
RUN pip install opencv-python
RUN pip install -U ultralytics
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev 

CMD ["/bin/bash"]
