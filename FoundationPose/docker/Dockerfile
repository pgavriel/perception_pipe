# Dockerfile (GPU-enabled)

# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM foundationpose

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# RUN apt-get update && apt-get install -y \
#     git \
#     python3 \
#     python3-pip \
#     python3-venv \
#     curl \
#     ffmpeg \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Optional: set working directory
# WORKDIR /workspace

# # Clone the SAM2 repository
# RUN git clone https://github.com/NVlabs/FoundationPose.git

# WORKDIR /workspace/FoundationPose

# # Upgrade pip and install Python requirements
# RUN pip3 install --upgrade pip && pip3 install -r ./requirements.txt

RUN pip3 install watchdog opencv-python

# RUN pip3 install pytorch3d 

WORKDIR /workspace

CMD ["/bin/bash"]