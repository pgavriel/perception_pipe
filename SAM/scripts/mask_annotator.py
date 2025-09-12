import os
import cv2
import glob
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# ---- Config ----
IMAGE_DIR = "/workspace/data/"#gear1/masks/"  # Update path as needed
# MODEL_TYPE = "vit_b"             # or vit_l, depending on what you've downloaded
# CHECKPOINT = "/workspace/sam2/sam_vit_b.pth"  # Update with actual checkpoint
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- SAM Setup ----
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))



# ---- Global State ----
images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))  # or png/tiff/etc.
index = 0
points = []
labels = []
current_mask = None
current_image = None
display_image = None
image_filename = None

# ---- Utility Functions ----
def load_image(i):
    global current_image, display_image, image_filename, points, labels, current_mask
    image_filename = images[i]
    current_image = cv2.imread(image_filename)
    display_image = current_image.copy()
    predictor.set_image(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
    points.clear()
    labels.clear()
    current_mask = None
    print(f"[Loaded] {image_filename}")

def update_mask():
    global current_mask, display_image
    if not points:
        current_mask = None
        display_image = current_image.copy()
        return

    input_points = np.array(points)
    input_labels = np.array(labels)

    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False
    )

    current_mask = masks[0]
    print(f"Current: {current_mask.shape} {type(current_mask)}")
    colored_mask = np.zeros_like(current_image)
    colored_mask[current_mask.astype(bool)] = (0, 255, 0)  # Green mask
    # display_image = cv2.addWeighted(current_image, 0.7, colored_mask, 0.3, 0)

    # Draw label points
    for pt, label in zip(points, labels):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(display_image, tuple(pt), 5, color, -1)


    cv2.imshow("Mask",colored_mask)

def save_mask():
    if current_mask is None:
        print("[!] No mask to save")
        return
    out_path = image_filename.replace(".jpg", "_mask.jpg")  # or .png, etc.
    cv2.imwrite(out_path, (current_mask * 255).astype(np.uint8))
    print(f"[Saved] {out_path}")

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        labels.append(1)
        update_mask()
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.append([x, y])
        labels.append(0)
        update_mask()

# ---- Main Loop ----
def main():
    global index, display_image

    if not images:
        print(f"[Error] No images found in {IMAGE_DIR}")
        return

    load_image(index)
    cv2.namedWindow("SAM Annotator", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("SAM Annotator", mouse_callback)

    while True:
        cv2.imshow("SAM Annotator", display_image)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q") or key == 27:  # ESC
            break
        elif key == ord("d"):  # Next image
            index = (index + 1) % len(images)
            load_image(index)
            update_mask()
        elif key == ord("a"):  # Previous image
            index = (index - 1) % len(images)
            load_image(index)
            update_mask()
        elif key == ord("r"):  # Reset labels
            points.clear()
            labels.clear()
            update_mask()
        elif key == ord("x"):  # Save mask
            save_mask()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
