#!/bin/bash

# Name the image and container
IMAGE_NAME=yolo_perception_pipe
CONTAINER_NAME=yolo_perception_pipe

# Build the Docker image
docker build -t $IMAGE_NAME .

# Path to your local scripts folder (adjust this)
LOCAL_SCRIPT_DIR=$(realpath ./scripts)
LOCAL_CONFIG_DIR=$(realpath ./config)
LOCAL_INPUT_DIR="/home/csrobot/Perception_Pipeline/Input"
LOCAL_DEBUG_DIR="/home/csrobot/Perception_Pipeline/Debug"
LOCAL_OUTPUT_DIR="/home/csrobot/Perception_Pipeline/Output"

xhost +local:docker

# Run the container with GPU access and volume mounting
docker run -it --rm \
  --gpus all \
  --net=host \
  --name $CONTAINER_NAME \
  -v "$LOCAL_SCRIPT_DIR":/workspace/scripts \
  -v "$LOCAL_CONFIG_DIR":/workspace/config \
  -v "$LOCAL_INPUT_DIR":/workspace/input \
  -v "$LOCAL_DEBUG_DIR":/workspace/debug \
  -v "$LOCAL_OUTPUT_DIR":/workspace/output \
  -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -e DISPLAY=$DISPLAY \
  $IMAGE_NAME \
  python3 -u /workspace/scripts/yolo_pipe_node.py
