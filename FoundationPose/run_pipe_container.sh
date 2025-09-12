#!/bin/bash

# Name the image and container
IMAGE_NAME=foundationpose_custom
CONTAINER_NAME=foundationpose_pipe_node

# Build the Docker image
echo 'Building...'
docker build -t $IMAGE_NAME ./docker

# Path to your local scripts folder (adjust this)
LOCAL_SCRIPT_DIR=$(realpath ./scripts)
LOCAL_CONFIG_DIR=$(realpath ./config)
LOCAL_INPUT_DIR="/home/csrobot/Perception_Pipeline/Input"
LOCAL_DEBUG_DIR="/home/csrobot/Perception_Pipeline/Debug"
LOCAL_OUTPUT_DIR="/home/csrobot/Perception_Pipeline/Output"
LOCAL_MESH_DIR="/home/csrobot/Perception_Pipeline/Meshes"
LOCAL_GIT_DIR="/home/csrobot/Perception_Pipeline/FoundationPose"

xhost +local:docker

# Run the container with GPU access and volume mounting
echo 'Running...'
docker run -it --rm \
  --gpus all \
  --net=host \
  --name $CONTAINER_NAME \
  -v "$LOCAL_SCRIPT_DIR":/workspace/scripts \
  -v "$LOCAL_CONFIG_DIR":/workspace/config \
  -v "$LOCAL_INPUT_DIR":/workspace/input \
  -v "$LOCAL_DEBUG_DIR":/workspace/debug \
  -v "$LOCAL_OUTPUT_DIR":/workspace/output \
  -v "$LOCAL_MESH_DIR":/workspace/input/meshes \
  -v "$LOCAL_GIT_DIR":/workspace/FoundationPose \
  -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -e DISPLAY=$DISPLAY \
  $IMAGE_NAME \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate my && python3 -u /workspace/scripts/fp_pipe_node.py"

# Running the startup python script requires sourcing conda and activating the environment before running