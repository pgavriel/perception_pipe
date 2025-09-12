print("\nRunning FoundationPose Pipe Node...\n")

import json
import re
import cv2
import numpy as np
import os
import sys
import time
from os.path import join
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
sys.path.insert(0, '/workspace/FoundationPose') 
from estimater import *
from datareader import *



# ===== CONFIG =====
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_FILE = "/workspace/config/fpose_pipe_config.json"
CACHE_FILE = "/workspace/debug/cache_fpose.json"
INPUT_DIR = "/workspace/input"
MESH_DIR = "/workspace/input/meshes"
OUTPUT_DIR = "/workspace/output"
DEBUG_DIR = "/workspace/debug/foundationpose"

# ===== Load patterns from config =====
print(f"Loading Config: {CONFIG_FILE}")
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)
print("Config loaded.")
patterns = [re.compile(p) for p in config["input_patterns"]]

# Load Model Dictionary - Maps from detection name to mesh file
print(f"Loading Model Dictionary: {config['mesh_library']}")
with open(join(MESH_DIR,config["mesh_library"]), "r") as f:
    mesh_library = json.load(f)
print("Library loaded:")
print(f"{'Object Label'.ljust(25)}{'Mesh File'.ljust(25)}")
for k, v in mesh_library.items():
    print(f"{str(k).ljust(25)}-> {v}")


# Setup FoundationPose Model
set_logging_format()
set_seed(0)
scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
cam_intrinsics_file = join("/workspace/config/camera_intrinsics",config["intrinsics_file"])
cam_intrinsics = np.loadtxt(f'{cam_intrinsics_file}').reshape(3,3)
print(f"\nCamera Intrinsics: {config['intrinsics_file']}\n{cam_intrinsics}\n")
# LOAD YOLO MODEL
# model_path = join("/workspace/config/weights",config["model_weights_file"])

# print(f"Loading Model: {model_path}")
# model = YOLO(model_path) # give path to .pt file
# print(f"Model Loaded.")

# ===== Load processed cache =====
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        processed = set(json.load(f))
else:
    processed = set()

print(f"Loaded {len(processed)} processed names from cache.\n")

# ===== Helper to update cache =====
def add_to_cache(name,write=False):
    processed.add(name)
    if write:
        with open(CACHE_FILE, "w") as f:
            json.dump(sorted(processed), f)

# ===== Matching logic =====
def check_for_matches():
    global model, config, cam_intrinsics
    files = os.listdir(INPUT_DIR)
    matched_names = {}

    # Step 1: Match files to patterns
    for filename in files:
        for i, pattern in enumerate(patterns):
            match = pattern.match(filename)
            if match:
                name = match.group("name")
                matched_names.setdefault(name, {})[i] = filename

    # Step 2: Check for complete sets
    for name, pattern_files in matched_names.items():
        if name in processed:
            continue
        if len(pattern_files) == len(patterns):
            print(f"Found match for '{name}':")
            for i, fname in pattern_files.items():
                print(f"  Pattern {i}: {fname}")

            # ===== PROCESS INPUT FILES
            # Load prediction file to get class name
            annotation_file = join(INPUT_DIR,pattern_files[3])
            with open(annotation_file, "r") as f:
                detections = json.load(f)
            if len(detections) > 0:
                class_name = detections[0]["class_name"]
            else:
                print(f"No detections found for {pattern_files[3]}")
                add_to_cache(name)
                continue
            # Load corresponding object model
            if class_name in mesh_library:
                mesh_file = join(MESH_DIR,mesh_library[class_name])
                mesh = trimesh.load(mesh_file, force='mesh')
                to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
                bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
            else:
                print(f"Class name \"{class_name}\" not found in mesh library")
                add_to_cache(name)
                continue
            # Clear Debug Dir
            debug = config.get("debug_level",1)
            os.system(f'rm -rf {DEBUG_DIR}/* && mkdir -p {DEBUG_DIR}/track_vis {DEBUG_DIR}/ob_in_cam')
            # Initialize Estimator
            est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=DEBUG_DIR, debug=debug, glctx=glctx)
            logging.info("Estimator initialization done.")
            # Load images
            print("Loading images... ")
            color = cv2.imread(join(INPUT_DIR,pattern_files[0]),-1)
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            print(f"[Color] {pattern_files[0]} - {color.shape}")
            depth = cv2.imread(join(INPUT_DIR,pattern_files[1]),-1)/1e3
            depth[(depth<0.001) | (depth>=np.inf)] = 0
            print(f"[Depth] {pattern_files[1]} - {depth.shape}")
            mask = cv2.imread(join(INPUT_DIR,pattern_files[2]),-1)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(bool)
            print(f"[Mask ] {pattern_files[2]} - {mask.shape}")
            print("Done.\n")
            # Get registered pose
            pose = est.register(K=cam_intrinsics, rgb=color, depth=depth, ob_mask=mask, iteration=config["est_refine_iter"])

            if debug>=3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{DEBUG_DIR}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, cam_intrinsics)
                valid = depth>=0.001
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{DEBUG_DIR}/scene_complete.ply', pcd)
            # Save pose txt file to debug
            os.makedirs(f'{DEBUG_DIR}/ob_in_cam', exist_ok=True)
            np.savetxt(f'{DEBUG_DIR}/ob_in_cam/{name}_pose.txt', pose.reshape(4,4))

            # TODO: No way to currently do pose tracking, 

            # ===== SAVE MODEL OUTPUT
            # NOTE: For now, output will be the visualization image, but in a proper
            #   pipeline it would be the object pose itself
            if config["save_output"]:
                # Save visualized pose image
                center_pose = pose@np.linalg.inv(to_origin)
                color = cv2.cvtColor(color,cv2.COLOR_BGR2RGB)
                vis = draw_posed_3d_box(cam_intrinsics, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_intrinsics, thickness=3, transparency=0, is_input_rgb=True)
                cv2.imwrite(join(OUTPUT_DIR,f"{name}_pose.png"),vis)
                # print("[IMAGE NOT SAVED]")

            
            # When done, cache file so it doesn't process twice
            print(f"[DONE] Saving {name} to cache...")
            add_to_cache(name)

            # Show model output
            if config["show_output"]:
                print("Show output not implemented.")
                # cv2.imshow("Pose",vis)

# ===== Watchdog event handler =====
class DirectoryHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            time.sleep(0.5)
            check_for_matches()

    def on_modified(self, event):
        if not event.is_directory:
            time.sleep(0.5)
            check_for_matches()

# ===== Main loop =====
if __name__ == "__main__":
    check_for_matches()  # Run once at startup
    # cv2.namedWindow("SAM Output")
    event_handler = DirectoryHandler()
    observer = Observer()
    observer.schedule(event_handler, INPUT_DIR, recursive=False)
    observer.start()

    print(f"Watching directory: {INPUT_DIR}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    # cv2.destroyAllWindows()
