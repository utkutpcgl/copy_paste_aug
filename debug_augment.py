import os
import random
import yaml
import cv2
import albumentations as A
from cpp_utils.simple_copy_paste import apply_copy_paste_augmentations
import numpy as np
import shutil

# Load configuration from the central config.yaml.
CFG_PATH = os.path.join(os.path.dirname(__file__), 'cpp_utils/config.yaml')
assert os.path.exists(CFG_PATH), f"Config file not found at {CFG_PATH}"
with open(CFG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Extract debugging parameters.
debug_config = config.get("debug", {})
num_augmentations = debug_config.get("num_augmentations", 100)
debug_folder = debug_config.get("debug_folder", "debug")
dataset_yaml = debug_config.get("dataset_yaml", None)


# Folder to save augmented images.
if os.path.exists(debug_folder):
    shutil.rmtree(debug_folder)
os.makedirs(debug_folder, exist_ok=True)

YAML_FILE = dataset_yaml
assert YAML_FILE is not None, "dataset_yaml must be specified in the config file."

# Load image data from the YAML file.
with open(YAML_FILE, "r") as f:
    data = yaml.safe_load(f)

if isinstance(data, dict):
    if "images" in data:
        images_info = data["images"]
    elif "train" in data:
        train_dirs = data["train"] if isinstance(data["train"], list) else [data["train"]]
        base_path = data.get("path", "")
        image_extensions = ('.jpg', '.jpeg', '.png')
        image_files = []
        for folder in train_dirs:
            full_folder_path = os.path.join(base_path, folder) if base_path and not os.path.isabs(folder) else folder
            if not os.path.isdir(full_folder_path):
                print(f"Warning: Directory {full_folder_path} does not exist or is not a directory.")
                continue
            for f in os.listdir(full_folder_path):
                if f.lower().endswith(image_extensions):
                    image_files.append(os.path.join(full_folder_path, f))
        images_info = [{"file": img, "bboxes": []} for img in image_files]
    else:
        raise ValueError("Unknown YAML structure: expecting keys 'images' or 'train'.")
else:
    images_info = data

random.shuffle(images_info)

def draw_yolo_bbox(image, bbox, class_label):
    """Draw a single YOLO format bbox on the image with class label"""
    h, w = image.shape[:2]
    x_center, y_center, width, height = bbox
    x1 = int((x_center - width/2) * w)
    y1 = int((y_center - height/2) * h)
    x2 = int((x_center + width/2) * w)
    y2 = int((y_center + height/2) * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    color = (0, 255, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, f"class_{class_label}", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

for i, item in enumerate(images_info[:num_augmentations]):
    image_path = item.get("file")
    bboxes = item.get("bboxes", None)
    if not bboxes:
        label_path = image_path.replace("images/", "labels/")
        label_path = os.path.splitext(label_path)[0] + ".txt"
        if os.path.exists(label_path):
            bboxes = []
            with open(label_path, "r") as f:
                for line in f:
                    tokens = line.strip().split()
                    if len(tokens) == 5:
                        bbox = [float(token) for token in tokens]
                        bboxes.append(bbox)
        else:
            bboxes = []
    if not image_path:
        print(f"Skipping entry {i}: no image path provided.")
        continue
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        continue
    processed_bboxes = []
    class_labels = []
    for bbox in bboxes:
        assert len(bbox) == 5, "Each bounding box must contain 5 elements: class and 4 coordinates."
        class_labels.append(float(bbox[0]))
        processed_bboxes.append(bbox[1:])
    processed_bboxes = np.array(processed_bboxes, dtype=np.float32)
    class_labels = np.array(class_labels, dtype=np.float32)
    final_aug_image, final_aug_bboxes, final_aug_class_labels = apply_copy_paste_augmentations(
        image=image, 
        processed_bboxes=processed_bboxes, 
        class_labels=class_labels
    )
    final_aug_bboxes = np.array(final_aug_bboxes, dtype=np.float32)
    final_aug_class_labels = np.array(final_aug_class_labels, dtype=np.float32)
    vis_image = final_aug_image.copy()
    for bbox, class_label in zip(final_aug_bboxes, final_aug_class_labels):
        vis_image = draw_yolo_bbox(vis_image, bbox, class_label)
    output_filename = f"augmented_{i}.jpg"
    output_path = os.path.join(debug_folder, output_filename)
    cv2.imwrite(output_path, vis_image)
    # print(f"Saved {output_path} with updated bounding boxes: {final_aug_bboxes} and updated class labels: {final_aug_class_labels}") 