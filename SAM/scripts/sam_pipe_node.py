import json
import re
import os
import time
import numpy as np
import torch
from os.path import join
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ===== LOAD SAM MODEL =====
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam_functions import *
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "/workspace/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))


# ===== CONFIG =====
CONFIG_FILE = "/workspace/config/sam_pipe_config.json"
CACHE_FILE = "/workspace/debug/cache_sam.json"
INPUT_DIR = "/workspace/input"
OUTPUT_DIR = "/workspace/input"

# ===== Load patterns from config =====
print(f"Loading Config: {CONFIG_FILE}")
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)
print("Config loaded.")

patterns = [re.compile(p) for p in config["input_patterns"]]

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
    global predictor
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

            # Load the image file into the predictor model
            image_file = join(INPUT_DIR,pattern_files[0])
            print(f"Loading Image \"{image_file}\"")
            current_image = cv2.imread(image_file)
            print(f"Shape: {current_image.shape}")
            # display_image = current_image.copy()
            predictor.set_image(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
            print(f"[Loaded] {image_file}")
            
            #TODO: Extract model prompts from prompt file
            # Load detections from YOLO JSON

            prompt_file = pattern_files[1]
            with open(join(INPUT_DIR,prompt_file), "r") as f:
                detections = json.load(f)
                print(f"{name} json loaded.")

            # Assemble all bounding boxes as numpy array
            # SAM2 expects shape (N, 4) with [x1, y1, x2, y2]
            boxes = []
            if len(detections) > 0:
                # boxes = np.array([det["bbox"] for det in detections], dtype=np.float32)
                boxes = np.array([detections[0]["bbox"]], dtype=np.float32)
            else:
                print("No detections in json file, skipping...\n")
                continue
            # Create positive labels for each box
            # Usually SAM expects labels as 1 (positive prompt)
            labels = np.ones(len(boxes), dtype=np.int32)

            # prompt = np.array([[375,300]])
            
            # Run model inference
            try:
                masks, scores, _ = predictor.predict(
                    box=boxes[None,:],
                    multimask_output=False
                )
            except:
                print("Something went wrong\n")
                continue
            # masks, scores, _ = predictor.predict(
            #     box=boxes,
            #     labels=labels,
            #     multimask_output=False
            # )

            current_mask = masks[0]
            print(f"Current: {current_mask.shape} {type(current_mask)}")
            colored_mask = np.zeros_like(current_image)
            colored_mask[current_mask.astype(bool)] = (255, 255, 255)  # White mask
            
            # Save model output
            if config["save_output"]:
                cv2.imwrite(join(OUTPUT_DIR,f"{name}_mask.png"),colored_mask)

            # When done, cache file so it doesn't process twice
            print(f"[DONE] Saving {name} to cache...\n")
            add_to_cache(name)

            # Show model output
            if config["show_output"]:
                cv2.imshow("SAM Output",colored_mask)
                cv2.waitKey(5000)

    # cv2.destroyWindow("SAM Output")


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
