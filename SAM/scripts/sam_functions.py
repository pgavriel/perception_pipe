import cv2
import numpy as np

# ---- Utility Functions ----
def load_image(predictor, file_name):
    global current_image, display_image, image_filename, points, labels, current_mask
    # image_filename = images[i]
    print(f"Loading Image \"{file_name}\"")
    current_image = cv2.imread(file_name)
    print(f"Shape: {current_image.shape}")
    # display_image = current_image.copy()
    predictor.set_image(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
    # points.clear()
    # labels.clear()
    current_mask = None
    return predictor
# def update_mask():
#     global current_mask, display_image
#     if not points:
#         current_mask = None
#         display_image = current_image.copy()
#         return

#     input_points = np.array(points)
#     input_labels = np.array(labels)

#     masks, scores, _ = predictor.predict(
#         point_coords=input_points,
#         point_labels=input_labels,
#         multimask_output=False
#     )

#     current_mask = masks[0]
#     print(f"Current: {current_mask.shape} {type(current_mask)}")
#     colored_mask = np.zeros_like(current_image)
#     colored_mask[current_mask.astype(bool)] = (0, 255, 0)  # Green mask
#     # display_image = cv2.addWeighted(current_image, 0.7, colored_mask, 0.3, 0)

#     # Draw label points
#     for pt, label in zip(points, labels):
#         color = (0, 255, 0) if label == 1 else (0, 0, 255)
#         cv2.circle(display_image, tuple(pt), 5, color, -1)


#     cv2.imshow("Mask",colored_mask)

# def save_mask():
#     if current_mask is None:
#         print("[!] No mask to save")
#         return
#     out_path = image_filename.replace(".jpg", "_mask.jpg")  # or .png, etc.
#     cv2.imwrite(out_path, (current_mask * 255).astype(np.uint8))
#     print(f"[Saved] {out_path}")