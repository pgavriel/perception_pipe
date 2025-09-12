#!/bin/bash

# Name the image and container
IMAGE_NAME=sam2_gpu
CONTAINER_NAME=sam2_container

# Build the Docker image
docker build -t $IMAGE_NAME .

# Path to your local scripts folder (adjust this)
LOCAL_SCRIPT_DIR=$(realpath ./scripts)
LOCAL_DATA_DIR="/home/csrobot/Desktop/anno-demo"

xhost +local:docker

# Run the container with GPU access and volume mounting
docker run -it --rm \
  --gpus all \
  --net=host \
  --name $CONTAINER_NAME \
  -v "$LOCAL_SCRIPT_DIR":/workspace/scripts \
  -v "$LOCAL_DATA_DIR":/workspace/data \
  -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -e DISPLAY=$DISPLAY \
  $IMAGE_NAME \
  /bin/bash
