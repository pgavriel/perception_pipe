# Perception Pipe
Provides Dockerized versions of YOLO, SAM2, and FoundationPose for robotic perception.  
Each model has a **run_pipe_container.sh** script which starts the corresponding model container, mounting necessary local directories, and running a script at startup which loads the model into memory, loads the config.json specified in the startup script, and starts a filesystem watcher to monitor the input directory specified in the json config. When a file or set of files in the input directory match the 'input_patterns' specified in that models config, the model will run inference on those files, save it's output to the specified output directory, and cache that file name so it does not continually run inference.  
In each [MODEL]/scripts folder, there is the **[MODEL]_pipe_node.py** startup python script which is run inside the container and determines the containers behavior when input files are found. 
The paths for local directories in each **run_pipe_container.sh** script will need to be modified to mount the appropriate folders to the container.  
Each **run_pipe_container.sh** script will first attempt to build the container from it's Dockerfile, then run the container after building. Once the container has been built, this step is effectively skipped as long as the Dockerfile is not modified.  

### Running YOLO  
Setup:  
The weights file for the YOLO model you would like to run should be placed inside the YOLO/config/weights folder and the name of the weights file should be set in the YOLO/config/yolo_test_config.json file.  

  
To Run:  
```
cd perception_pipe/YOLO
./run_pipe_container.sh
```
  
### Running SAM2   
Setup:  
It shouldn't be necessary to clone the [SAM2 github repo](https://github.com/facebookresearch/sam2) in order to build the container, as it is part of the build process.  
  
To Run:  
```
cd perception_pipe/SAM
./run_pipe_container.sh
```
  

### Running FoundationPose   
Setup:  
The [FoundationPose github repo](https://github.com/NVlabs/FoundationPose) will need to be cloned locally, and it's path needs to be included as LOCAL_GIT_DIR in the run_pipe_container.sh script for FoundationPose.  
`git clone https://github.com/NVlabs/FoundationPose.git`  
  
To run:  
```
cd perception_pipe/FoundationPose
./run_pipe_container.sh
```
