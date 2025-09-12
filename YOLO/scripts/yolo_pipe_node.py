print("\nRunning YOLO Pipe Node...\n")

from ultralytics import YOLO, settings
import json
import re
import cv2
import numpy as np
import os
import time
from os.path import join
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ===== CONFIG =====
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_FILE = "/workspace/config/yolo_test_config.json"
CACHE_FILE = "/workspace/debug/cache_yolo.json"
INPUT_DIR = "/workspace/input"
OUTPUT_DIR = "/workspace/input"
DEBUG_DIR = "/workspace/debug"

# ===== Load patterns from config =====
print(f"Loading Config: {CONFIG_FILE}")
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)
print("Config loaded.")
patterns = [re.compile(p) for p in config["input_patterns"]]

# LOAD YOLO MODEL
model_path = join("/workspace/config/weights",config["model_weights_file"])

print(f"Loading Model: {model_path}")
model = YOLO(model_path) # give path to .pt file
print(f"Model Loaded.")

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
    global model, config
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

            # PROCESS INPUT FILES
            # Load image
            image_file = join(INPUT_DIR,pattern_files[0])
            print(f"Loading Image \"{image_file}\"")
            current_image = cv2.imread(image_file)
            print(f"Shape: {current_image.shape}")
            # Run model prediction
            results = model(current_image, 
                            imgsz=config["input_width"], 
                            visualize=False, 
                            conf=config["min_conf"], 
                            iou=config["iou"], 
                            max_det=config["max_det"])

            for result in results:
                output_data = []
                if config["save_debug"]:# Save annotated debug image
                    save_file = join(DEBUG_DIR,f"{name}_yolo_debug.jpg")
                    result.save(filename=save_file)  
                    print(f"Result Saved: {save_file}")
                for b in result.boxes:
                    xyxy = b.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    conf = float(b.conf[0])
                    cls = int(b.cls[0])
                    output_data.append({
                        "bbox": xyxy,
                        "confidence": conf,
                        "class_id": cls,
                        "class_name": model.names[cls]
                    })

            # Save model output annotations JSON
            if config["save_output"]:
                # Write to JSON file
                with open(join(OUTPUT_DIR,f"{name}_yolo.json"), "w") as f:
                    json.dump(output_data, f, indent=4)
                # print("Save output not implemented") # WRITE JSON FILE
                # cv2.imwrite(join(OUTPUT_DIR,f"{name}_yolo.json"),colored_mask)

            # When done, cache file so it doesn't process twice
            print(f"[DONE] Saving {name} to cache...")
            add_to_cache(name)

            # Show model output
            if config["show_output"]:
                print("Show output not implemented.")

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
