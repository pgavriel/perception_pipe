import gradio as gr
import argparse
import os
import numpy as np
from PIL import Image, ImageDraw
import random
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
import cv2

# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# FUNCTIONS FROM SAM2 TO MODIFY IMAGE LOADING BEHAVIOR
# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
def _load_img_as_tensor(img_path, image_size):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_width, video_height = img_pil.size  # the original video size
    return img, video_height, video_width

def load_video_frames_from_scan(video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    compute_device=torch.device("cuda")
    ):
    '''Expects images named like {str}_{frame_num}.jpg'''
    if isinstance(video_path, str) and os.path.isdir(video_path):
        image_folder = video_path
    else:
        raise NotImplementedError("Video path is not a string and directory.")   
    file_types = [".jpg", ".jpeg", ".JPG", ".JPEG"]
    frame_names = [
        p
        for p in os.listdir(image_folder)
        if os.path.splitext(p)[-1] in file_types
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[1]))
    
    num_frames = len(frame_names)

    print(f"Found {num_frames} frames.")
    if num_frames == 0:
        raise RuntimeError(f"no {file_types} images found in {image_folder}")
    img_paths = [os.path.join(image_folder, frame_name) for frame_name in frame_names]
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    # if async_loading_frames:
    #     lazy_images = AsyncVideoFrameLoader(
    #         img_paths,
    #         image_size,
    #         offload_video_to_cpu,
    #         img_mean,
    #         img_std,
    #         compute_device,
    #     )
    #     return lazy_images, lazy_images.video_height, lazy_images.video_width

    images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)
    for n, img_path in enumerate(tqdm(img_paths, desc="frame loading (JPEG)")):
        images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, video_height, video_width

@torch.inference_mode()
def init_state(
    model,
    video_path,
    offload_video_to_cpu=False,
    offload_state_to_cpu=False,
    async_loading_frames=False,
    ):
    """Initialize an inference state."""
    compute_device = model.device  # device of the model
    images, video_height, video_width = load_video_frames_from_scan(
        video_path=video_path,
        image_size=model.image_size,
        offload_video_to_cpu=offload_video_to_cpu,
        async_loading_frames=async_loading_frames,
        compute_device=compute_device,
    )
    inference_state = {}
    inference_state["images"] = images
    inference_state["num_frames"] = len(images)
    # whether to offload the video frames to CPU memory
    # turning on this option saves the GPU memory with only a very small overhead
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    # whether to offload the inference state to CPU memory
    # turning on this option saves the GPU memory at the cost of a lower tracking fps
    # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
    # and from 24 to 21 when tracking two objects)
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    # the original video height and width, used for resizing final output scores
    inference_state["video_height"] = video_height
    inference_state["video_width"] = video_width
    inference_state["device"] = compute_device
    if offload_state_to_cpu:
        inference_state["storage_device"] = torch.device("cpu")
    else:
        inference_state["storage_device"] = compute_device
    # inputs on each frame
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    # visual features on a small number of recently visited frames for quick interactions
    inference_state["cached_features"] = {}
    # values that don't change across frames (so we only need to hold one copy of them)
    inference_state["constants"] = {}
    # mapping between client-side object id and model-side object index
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
    inference_state["output_dict_per_obj"] = {}
    # A temporary storage to hold new outputs when user interact with a frame
    # to add clicks or mask (it's merged into "output_dict" before propagation starts)
    inference_state["temp_output_dict_per_obj"] = {}
    # Frames that already holds consolidated outputs from click or mask inputs
    # (we directly use their consolidated outputs during tracking)
    # metadata for each tracking frame (e.g. which direction it's tracked)
    inference_state["frames_tracked_per_obj"] = {}
    # Warm up the visual backbone and cache the image feature on frame 0
    model._get_image_feature(inference_state, frame_idx=0, batch_size=1)
    return inference_state

# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# LOAD SAM2 MODEL & GLOBALS
# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
print("Loading SAM2 model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large", clear_non_cond_mem_around_input=True, device=device)
inference_state = None
video_segments = {} # Stores propagated mask data
original_image = None
print("Model loaded on", device)

frame_names = []
LABEL_LIST = ["bar_12mm","bar_16mm","bar_4mm","bar_8mm",
"conn_bnc","conn_dsub","conn_ethernet","conn_usb","conn_wp",
"gear_large","gear_med","gear_small",
"nut_m12","nut_m16","nut_m4","nut_m8",
"rod_12mm","rod_16mm","rod_4mm","rod_8mm"]

# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# MODEL MANAGEMENT
# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def set_sam_video_folder(frame_dir):
    global inference_state, model, frame_names
    inference_state = init_state(model,video_path=frame_dir)
    model.reset_state(inference_state)
    frame_names = [
        p for p in os.listdir(frame_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[1]))

def reset_sam(current_idx, image_paths):
    global model, inference_state, video_segments
    model.reset_state(inference_state)
    video_segments = {}
    image, status = load_image(current_idx, image_paths)
    print("SAM State Reset")
    return image, status, current_idx

def clear_inference_frames(current_idx, image_paths):
    global inference_state, video_segments
    video_segments = {}
    image, status = load_image(current_idx, image_paths)
    print("Inference Frames Cleared")
    return image, status, current_idx

def propagate_frames(current_idx, image_paths):
    global inference_state, video_segments, model
    start_frame = 0 
    print(f"Propagating Frames from frame {start_frame}...")
    # Print inference_state information
    print("===BEFORE===\n",inspect_inference_state(),"\n=====")


    # for v in inference_state["temp_output_dict_per_obj"].values():
    #         v["cond_frame_outputs"].clear()
    #         v["non_cond_frame_outputs"].clear()
    # for v in inference_state["frames_tracked_per_obj"].values():
    #     v.clear()
        
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(inference_state,start_frame_idx=start_frame):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    print(f"Vid Seg Keys: {video_segments.keys()}")
    print(f"Seg 0: {video_segments.get(0,None)}")

    image, status = load_image(current_idx, image_paths)
    print("===AFTER===\n",inspect_inference_state(),"\n=====")
    print("Propagation Finished.")
    return image


def inspect_inference_state():
    ''' Print out data inside current inference_state'''
    global inference_state
    print(f"Inference State Type: {type(inference_state)}, Len: {len(inference_state.keys())}")
    for k, v in inference_state.items():
        if k not in ['images','temp_output_dict_per_obj','cached_features','output_dict_per_obj','constants']:
        # if False:
            print(f"> [ {k} ]: {v}")
        elif k in ['output_dict_per_obj','temp_output_dict_per_obj']:
            print(f"> [ {k} ]: Keys: {v.keys()}")
        else:
            print(f"> [ {k} ]: Type: {type(v)}")

# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# DRAW FUNCTIONS
# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
def overlay_mask_on_image(image: Image.Image, masks: np.ndarray, labels, alpha=0.75):
    """
    Draw one or more semi-transparent segmentation masks on a PIL image.
    Supports:
      - 2D masks (H, W)
      - 3D masks (N, H, W)
    Each mask gets its own distinct color.
    """
    # Convert input image to RGBA
    base = image.convert("RGBA")

    # Ensure masks is numpy array
    masks = np.array(masks)
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]  # (H, W) -> (1, H, W)
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(1)  # (N,1,H,W) -> (N,H,W)

    num_masks, H, W = masks.shape

    # Use a tab20 colormap for up to 20 distinct colors (loop if >20)
    cmap = plt.get_cmap("tab20")
    for i, label in zip(range(masks.shape[0]),labels):
        color_id = LABEL_LIST.index(label)
        mask = masks[i].astype(bool)
        if not mask.any():
            continue

        color = (np.array(cmap(color_id % 20)[:3]) * 255).astype(np.uint8)
        overlay = Image.new("RGBA", base.size, tuple(color) + (int(alpha * 255),))
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))#, mode="L")

        # Use alpha_composite correctly by premasking the overlay
        overlay.putalpha(mask_img.point(lambda p: p * alpha))
        base = Image.alpha_composite(base, overlay)

    return base.convert("RGB")


def overlay_masks(image, masks, alpha=0.75):
    '''
    Called when loading an image, gets masks from video_segments (only after propagating frames)
    '''
    # Use a tab20 colormap for up to 20 distinct colors (loop if >20)
    cmap = plt.get_cmap("tab20")

    base = image.convert("RGBA")
    for obj_name, m in masks.items():
        num_masks, H, W = m.shape
        obj_id = LABEL_LIST.index(obj_name)
        print(f"Object Name: {obj_name} - ID: {obj_id}")

        for i in range(m.shape[0]):
            mask = m[i].astype(bool)
            if not mask.any(): # Skip empty masks
                continue

        color = (np.array(cmap(obj_id % 20)[:3]) * 255).astype(np.uint8)
        overlay = Image.new("RGBA", base.size, tuple(color) + (int(alpha * 255),))
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))#, mode="L")

        # Use alpha_composite correctly by premasking the overlay
        overlay.putalpha(mask_img.point(lambda p: p * alpha))
        base = Image.alpha_composite(base, overlay)

    return base.convert("RGB")

########### CURRENTLY UNUSED - Implement to optionally draw prompts and bounding boxes with checkboxes
# def draw_points_on_image(image: Image.Image, coords, labels, marker_size=6):
#     """
#     Draw positive (green) and negative (red) points on a PIL image.
#     """
#     draw = ImageDraw.Draw(image)
#     for (x, y), label in zip(coords, labels):
#         color = "green" if label == 1 else "red"
#         r = marker_size
#         draw.ellipse((x - r, y - r, x + r, y + r), outline="white", fill=color, width=2)
#     return image

# def draw_box_on_image(image: Image.Image, box, color="green", width=3):
#     """
#     Draw a bounding box on a PIL image.
#     """
#     draw = ImageDraw.Draw(image)
#     x0, y0, x1, y1 = box
#     draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
#     return image


# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# Image sequence management
# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
def list_images(folder):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(valid_exts)])
    return [os.path.join(folder, f) for f in files]

def load_image(idx, image_paths):
    global original_image
    if not image_paths:
        return None, "No images found."
    idx = max(0, min(idx, len(image_paths)-1))
    image = Image.open(image_paths[idx]).convert("RGB")
    original_image = image.copy()
    masks = video_segments.get(idx,{})
    print(f"Loading image IDX:{idx}, Masks:{len(masks.keys())}")
    #TODO: IF LEN MASKS == 0: CHECK INFERENCE STATE FOR MASKS TO DRAW ON CURRENT FRAME
    #masks = fake_sam2_segmentation(image)
    overlay = overlay_masks(image, masks)
    status = f"Image {idx+1}/{len(image_paths)}: {os.path.basename(image_paths[idx])}"
    return overlay, status

def update_folder(folder):
    if not os.path.isdir(folder):
        return None, "Invalid folder path.", [], 0
    image_paths = list_images(folder)
    if not image_paths:
        return None, "No images found in folder.", [], 0
    image, status = load_image(0, image_paths)
    return image, status, image_paths, 0

def next_image(current_idx, image_paths):
    if not image_paths:
        return None, "No images loaded.", current_idx
    new_idx = (current_idx + 1) % len(image_paths)  # min(current_idx + 1, len(image_paths) - 1)
    image, status = load_image(new_idx, image_paths)
    return image, status, new_idx

def prev_image(current_idx, image_paths):
    if not image_paths:
        return None, "No images loaded.", current_idx
    new_idx = (current_idx - 1) % len(image_paths)  # max(current_idx - 1, 0)
    image, status = load_image(new_idx, image_paths)
    return image, status, new_idx


def handle_click(evt: gr.SelectData, image_paths, current_idx, label, click_log,negative_prompt):
    global model, inference_state, original_image
    if not image_paths:
        return "No image loaded.", click_log
    x, y = evt.index
    filename = os.path.basename(image_paths[current_idx])
    entry = {"file": filename, "x": x, "y": y, "label": label}
    click_log.append(entry)
    status = f"Added prompt for {label} at ({x}, {y}) on {filename}"

    # Check inference state to see if object already exists
    if label in inference_state['obj_ids']:
        label_idx = inference_state['obj_id_to_idx'].get(label)
        print(f"Preexisting object - Internal ID: {label_idx}")
    else:
        print(f"Adding new object - No previous data to gather")

    points = np.array([[x, y]], dtype=np.float32)
    if not negative_prompt:
        plabels = np.array([1], np.int32) 
    else:
        plabels = np.array([0], np.int32) 
    print(f"Adding prompt: FRAME_ID={current_idx}, OBJ_ID={label}, PTS={points}, LBL={plabels}")

    _, out_obj_ids, out_mask_logits = model.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=current_idx,
        obj_id=label,
        points=points,
        labels=plabels,
        clear_old_points=False
    )
    print(f"Out OBJ IDs: {out_obj_ids}")
    print(f"Out logits shape: {out_mask_logits.shape}")
    # show_points(points, plabels, image)

    inspect_inference_state()

    draw_mask = (out_mask_logits > 0.0).cpu().numpy()
    # If you ever get multiple masks (e.g. multiple objects at once), you can pick one explicitly:
    # draw_mask = (out_mask_logits[0, 0] > 0).cpu().numpy()
    image = original_image.copy()
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    out_image = overlay_mask_on_image(image,draw_mask,out_obj_ids)

    return status, click_log, out_image

# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# Annotation Output
# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
def mask_to_bbox(mask: np.ndarray):
    """
    Given a 2D boolean mask, return (x_min, y_min, x_max, y_max) in pixel coords.
    Returns None if mask is empty.
    """
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError("mask_to_bbox expects a 2D boolean array")

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None  # no mask content

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return (x_min, y_min, x_max, y_max)

def compute_iou(box1, box2):
    """Compute IoU between two YOLO-format boxes (xc, yc, w, h)."""
    # Convert to corner coordinates
    def to_xy(box):
        xc, yc, w, h = box
        return [
            xc - w / 2,
            yc - h / 2,
            xc + w / 2,
            yc + h / 2,
        ]

    x1_min, y1_min, x1_max, y1_max = to_xy(box1)
    x2_min, y2_min, x2_max, y2_max = to_xy(box2)

    # Intersection box
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - inter_area

    return inter_area / union if union > 0 else 0.0

def remove_duplicate_annotations(annotations, overlap_thresh=0.8):
    """
    Removes duplicates based on label and IoU overlap threshold.
    annotations: list of (label, xc, yc, w, h)
    """
    filtered = []

    for ann in annotations:
        keep = True
        for existing in filtered:
            if ann[0] == existing[0]:  # same label
                iou = compute_iou(ann[1:], existing[1:])
                if iou > overlap_thresh:
                    keep = False
                    break
        if keep:
            filtered.append(ann)

    return filtered

def save_yolo_annotation(
    image_path: str,
    label: str,
    bbox: tuple,
    output_dir: str,
    image_size: tuple,
    overlap_thresh: float = 0.8
):
    """
    Save or append YOLO-format bounding box annotations for a given image.

    Args:
        image_path: path to the image file (used to derive annotation filename)
        label: class name (string)
        bbox: (x1, y1, x2, y2) pixel coordinates
        output_dir: directory to write the annotation file
        image_size: (width, height) of the image for normalization
        overlap_thresh: IoU threshold above which to remove duplicate entries
    """

    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    anno_path = os.path.join(output_dir, f"{img_name}.txt")

    w_img, h_img = image_size
    x1, y1, x2, y2 = bbox

    # Convert to YOLO normalized format
    x_center = ((x1 + x2) / 2) / w_img
    y_center = ((y1 + y2) / 2) / h_img
    width = abs(x2 - x1) / w_img
    height = abs(y2 - y1) / h_img

    new_anno = (label, x_center, y_center, width, height)

    # Load existing annotations if present
    existing = []
    if os.path.exists(anno_path):
        with open(anno_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    existing.append((parts[0], *map(float, parts[1:])))

    # Append and deduplicate
    merged = remove_duplicate_annotations(existing + [new_anno], overlap_thresh)

    # Write updated file
    with open(anno_path, "w") as f:
        for lbl, xc, yc, w, h in merged:
            f.write(f"{lbl} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    print(f"Annotation saved: {anno_path}")
    return anno_path

def export_yolo_annotations(output_dir=None, image_dir=None, overlap_thresh=0.8):
    """
    Iterate through all video segments and save YOLO-style bounding boxes.

    Args:
        video_segments: dict[int, dict[label, np.ndarray]]
        frame_names: list[str]
        output_dir: folder to store YOLO .txt files
        image_dir: folder containing the original image files
        overlap_thresh: IoU threshold for deduplication
    """
    global video_segments, frame_names
    image_dir = "/workspace/data"
    output_dir = "/workspace/data/labels" #TODO: HARDCODED FOR TESTING, fix later
    from PIL import Image

    for frame_idx, obj_masks in video_segments.items():
        image_name = frame_names[frame_idx]
        image_path = os.path.join(image_dir, image_name)
        print(f"Image path: {image_path}")

        # Load image to get size
        with Image.open(image_path) as im:
            width, height = im.size

        for label, mask in obj_masks.items():
            bbox = mask_to_bbox(mask)
            if bbox is None:
                continue  # skip empty masks

            save_yolo_annotation(
                image_path=image_path,
                label=label,
                bbox=bbox,
                output_dir=output_dir,
                image_size=(width, height),
                overlap_thresh=overlap_thresh
            )

        print(f"✅ Exported annotations for {image_name}")

# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# GUI Definition
# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
def build_ui(default_folder):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown("## SAM2 Annotation Prototype — Frame Navigation\nSelect a folder of images to browse and visualize fake segmentations.")
            with gr.Column():
                folder_box = gr.Textbox(value=default_folder, label="Image Folder Path (Local to docker container)")
                select_btn = gr.Button("Load Folder")

        output_image = gr.Image(label="Segmentation Overlay", show_label=False)
        with gr.Row():
            status_text = gr.Textbox(label="Status", interactive=False)
            label_dropdown = gr.Dropdown(
                choices=LABEL_LIST, 
                label="Select Label",
                value=LABEL_LIST[0]
            )
        

        with gr.Row():
            prev_btn = gr.Button("⬅️ Previous")
            propagate_btn = gr.Button("Propagate")
            next_btn = gr.Button("Next (Confirm) ➡️")

        
        with gr.Row():
            with gr.Column():
                prompt_type_box = gr.Checkbox(label="Negative Prompt", value=False)
                bb_box = gr.Checkbox(label="Draw Bounding Boxes", value=True)
            reset_sam_btn = gr.Button("Reset SAM")
            clear_frames_btn = gr.Button("Clear Inference Masks")
            print_inf_btn = gr.Button("Print Inference State")

        export_ann_btn = gr.Button("Export YOLO Annotations")

        # hidden state elements
        image_paths_state = gr.State([])
        current_idx_state = gr.State(0)
        click_log_state = gr.State([])


        # === BUTTON CALLBACKS ====
        select_btn.click(update_folder, 
                        inputs=folder_box, 
                        outputs=[output_image, status_text, image_paths_state, current_idx_state])
        next_btn.click(next_image, 
                        inputs=[current_idx_state, image_paths_state],
                        outputs=[output_image, status_text, current_idx_state])
        propagate_btn.click(propagate_frames, 
                        inputs=[current_idx_state, image_paths_state],
                        outputs=[output_image])
        prev_btn.click(prev_image, 
                        inputs=[current_idx_state, image_paths_state],
                        outputs=[output_image, status_text, current_idx_state])
        reset_sam_btn.click(reset_sam,
                        inputs=[current_idx_state, image_paths_state],
                        outputs=[output_image, status_text, current_idx_state])
        clear_frames_btn.click(clear_inference_frames,
                        inputs=[current_idx_state, image_paths_state],
                        outputs=[output_image, status_text, current_idx_state])
        print_inf_btn.click(inspect_inference_state,
                        inputs=[],
                        outputs=[])
        
        export_ann_btn.click(export_yolo_annotations,
                        inputs=[],
                        outputs=[])

        # output_image.select(
        #     fn=handle_click,
        #     js=js,
        #     outputs=[status_text]
        # )
        output_image.select(
            fn=handle_click,
            inputs=[image_paths_state, current_idx_state, label_dropdown, click_log_state,prompt_type_box],
            outputs=[status_text, click_log_state,output_image]
        )

        # auto-load folder if provided at launch
        demo.load(update_folder, inputs=folder_box,
            outputs=[output_image, status_text, image_paths_state, current_idx_state],
            trigger_mode="once")

    return demo

# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# Entry Point / MAIN
# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/workspace/data", help="Folder containing image sequence")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    # sam2 = load_sam2_model(args.data)
    set_sam_video_folder(args.data)


    # frame_idx = 0
    # plt.figure(figsize=(9, 6))
    # plt.title(f"frame {frame_idx}")
    # plt.imshow(Image.open(os.path.join(args.data, frame_names[frame_idx])))
    # plt.savefig("/workspace/data/test.png")

    demo = build_ui(args.data)
    # demo.update_folder(args.data)
    demo.launch(server_name="0.0.0.0", server_port=args.port)
