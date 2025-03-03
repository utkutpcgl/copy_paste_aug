import os
import random
import yaml
import cv2
import albumentations as A
from cpp_utils.simple_copy_paste import apply_copy_paste_augmentations
import numpy as np
import shutil
# Folder to save augmented images.
DEBUG_FOLDER = "debug"
if os.path.exists(DEBUG_FOLDER):
    shutil.rmtree(DEBUG_FOLDER)
os.makedirs(DEBUG_FOLDER, exist_ok=True)

YAML_FILE = "/home/utkutopcuoglu/Projects/Datasets/person_datasets/person_data.yaml"

# Number of augmentations per image.
NUM_AUGMENTATIONS = 5

# Load image data from the YAML file.
with open(YAML_FILE, "r") as f:
    data = yaml.safe_load(f)

if isinstance(data, dict):
    if "images" in data:
        # Existing case: the YAML file directly defines an images list.
        images_info = data["images"]
    elif "train" in data:
        # YOLO dataset config case with multiple train folders.
        # Ensure that train_dirs is a list even if a single folder is provided.
        train_dirs = data["train"] if isinstance(data["train"], list) else [data["train"]]
        # Optionally, use the "path" key to resolve relative folder paths.
        base_path = data.get("path", "")
        image_extensions = ('.jpg', '.jpeg', '.png')
        image_files = []
        for folder in train_dirs:
            # If base_path exists and folder is not an absolute path, join them.
            full_folder_path = os.path.join(base_path, folder) if base_path and not os.path.isabs(folder) else folder
            if not os.path.isdir(full_folder_path):
                print(f"Warning: Directory {full_folder_path} does not exist or is not a directory.")
                continue
            # List image files in the current directory.
            for f in os.listdir(full_folder_path):
                if f.lower().endswith(image_extensions):
                    image_files.append(os.path.join(full_folder_path, f))
        # Build images_info so that each entry contains a file path and empty bounding boxes.
        images_info = [{"file": img, "bboxes": []} for img in image_files]
    else:
        raise ValueError("Unknown YAML structure: expecting keys 'images' or 'train'.")
else:
    images_info = data

# Shuffle the images to process them in a random order.
random.shuffle(images_info)

# Optionally, select only a subset if you want to process only a few random images.
# For example, to process 5 images (if available), uncomment the following line:
# images_info = images_info[:5]

# Define the augmentation transform with YOLO bbox format.

def draw_yolo_bbox(image, bbox, class_label):
    """Draw a single YOLO format bbox on the image with class label"""
    h, w = image.shape[:2]
    # YOLO format: x_center, y_center, width, height (normalized)
    x_center, y_center, width, height = bbox
    
    # Convert to pixel coordinates
    x1 = int((x_center - width/2) * w)
    y1 = int((y_center - height/2) * h)
    x2 = int((x_center + width/2) * w)
    y2 = int((y_center + height/2) * h)
    
    # Ensure coordinates are within image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # Draw rectangle and label
    color = (0, 255, 0)  # Green color for bboxes
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, f"class_{class_label}", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


# Process each image.
for i, item in enumerate(images_info[:NUM_AUGMENTATIONS]):
    # Expecting each entry to have a 'file' key.
    image_path = item.get("file")
    
    # Try to get bounding boxes from the YAML entry.
    bboxes = item.get("bboxes", None)
    
    # If no bounding boxes were provided, try to read from the associated txt file.
    if not bboxes:
        # Replace 'images/' with 'labels/' and change the file extension to .txt.
        label_path = image_path.replace("images/", "labels/")
        label_path = os.path.splitext(label_path)[0] + ".txt"
        if os.path.exists(label_path):
            bboxes = []
            with open(label_path, "r") as f:
                for line in f:
                    tokens = line.strip().split()
                    if len(tokens) == 5:
                        # YOLO format: class, x_center, y_center, width, height.
                        bbox = [float(token) for token in tokens]
                        bboxes.append(bbox)
        else:
            bboxes = []

    if not image_path:
        print(f"Skipping entry {i}: no image path provided.")
        continue

    # Load the image.
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        continue

    # Split each bounding box into coordinates and corresponding class label.
    processed_bboxes = []
    class_labels = []
    for bbox in bboxes:
        assert len(bbox) == 5, "Each bounding box must contain 5 elements: class and 4 coordinates."
        class_labels.append(float(bbox[0]))  # Convert class label to float
        processed_bboxes.append(bbox[1:])
    
    # Convert lists to numpy float32 arrays.
    processed_bboxes = np.array(processed_bboxes, dtype=np.float32)
    class_labels = np.array(class_labels, dtype=np.float32)
    
    # Generate several augmented versions for this image.
    # Pass the transformed bounding boxes and the separate class_labels.
    final_aug_image, final_aug_bboxes, final_aug_class_labels = apply_copy_paste_augmentations(
        image=image, 
        processed_bboxes=processed_bboxes, 
        class_labels=class_labels,
        resized_shape=(640, 640),
        ori_shape=(640, 640)
    )
    
    # Ensure returned values are numpy float32 arrays.
    final_aug_bboxes = np.array(final_aug_bboxes, dtype=np.float32)
    final_aug_class_labels = np.array(final_aug_class_labels, dtype=np.float32)
    
    # Draw bounding boxes on the augmented image
    vis_image = final_aug_image.copy()
    for bbox, class_label in zip(final_aug_bboxes, final_aug_class_labels):
        vis_image = draw_yolo_bbox(vis_image, bbox, class_label)
    
    # Save the augmented image with bboxes to the debug folder
    output_filename = f"augmented_{i}.jpg"
    output_path = os.path.join(DEBUG_FOLDER, output_filename)
    cv2.imwrite(output_path, vis_image)
    
    print(f"Saved {output_path} with updated bounding boxes: {final_aug_bboxes} and updated class labels: {final_aug_class_labels}") 