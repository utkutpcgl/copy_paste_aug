#!/usr/bin/env python
"""
fix_bboxes.py

This script fixes all YOLO-format bounding boxes in your dataset. It does the following:
  1. Converts each YOLO bbox [x_center, y_center, width, height] to [x1, y1, x2, y2].
  2. Clips coordinates to lie within [0, 1].
  3. Converts the clipped coordinates back to YOLO format.
  
If the YAML configuration contains an "images" key (with bounding boxes stored inside), 
the YAML file will be updated (with a backup of the original). Otherwise, for keys such as 
"train", "val", or "test", each entry is assumed to be an images folder. The corresponding
labels folder is derived by replacing "images" with "labels" in the folder path and all .txt 
files in that labels folder are processed.
"""

import os
import yaml
import argparse
import numpy as np

def fix_bbox(yolo_bbox):
    """
    Fix a single YOLO bounding box.
    
    Given a YOLO-format bounding box as [x_center, y_center, width, height],
    this function converts it to [x1, y1, x2, y2], clips the coordinates to be within [0, 1],
    and then converts it back to YOLO format.
    
    Returns:
        A list [new_x_center, new_y_center, new_w, new_h] with the clipped coordinates,
        or None if the box becomes degenerate.
    """
    assert len(yolo_bbox) == 4, "Bounding box must be length 4."
    x_center, y_center, w, h = yolo_bbox
    # Convert YOLO to x1, y1, x2, y2.
    x1 = x_center - w / 2.0
    y1 = y_center - h / 2.0
    x2 = x_center + w / 2.0
    y2 = y_center + h / 2.0

    # Clip the coordinates to lie within [0, 1].
    x1_clipped = max(0.0 + 1e-4, x1)
    y1_clipped = max(0.0 + 1e-4, y1)
    x2_clipped = min(1.0 - 1e-4, x2)
    y2_clipped = min(1.0 - 1e-4, y2)

    assert x2_clipped > x1_clipped, f"x2_clipped ({x2_clipped}) must be greater than x1_clipped ({x1_clipped})"
    assert y2_clipped > y1_clipped, f"y2_clipped ({y2_clipped}) must be greater than y1_clipped ({y1_clipped})"

    # Compute the new width and height.
    new_w = x2_clipped - x1_clipped
    new_h = y2_clipped - y1_clipped
    if new_w <= 0 or new_h <= 0:
        print(f"Warning: The bounding box {yolo_bbox} became degenerate after clipping.")
        return None

    # Convert back to YOLO format.
    new_x_center = (x1_clipped + x2_clipped) / 2.0
    new_y_center = (y1_clipped + y2_clipped) / 2.0
    return [new_x_center, new_y_center, new_w, new_h]

def process_yaml_with_images(yaml_path, data):
    """
    Processes a YAML file that contains an 'images' list with bounding boxes.
    Each image entry is expected to hold a "file" key and a "bboxes" key where each bbox
    is in the format [class, x_center, y_center, width, height]. The function fixes each bbox,
    removes any degenerate ones, and writes back the updated YAML file.
    """
    images_info = data["images"]
    modified = False

    for image_entry in images_info:
        if "bboxes" in image_entry and image_entry["bboxes"]:
            new_bboxes = []
            for bbox in image_entry["bboxes"]:
                assert len(bbox) == 5, f"Expected bbox with 5 values, got: {bbox}"
                class_id = bbox[0]
                fixed_bbox = fix_bbox(bbox[1:])
                if fixed_bbox is not None:
                    new_bboxes.append([class_id] + fixed_bbox)
                else:
                    print(f"Removed degenerate bounding box in file {image_entry.get('file')}: {bbox}")
                    modified = True
            if new_bboxes != image_entry["bboxes"]:
                image_entry["bboxes"] = new_bboxes
                modified = True

    if modified:
        backup_file = yaml_path + ".bak"
        os.rename(yaml_path, backup_file)
        with open(yaml_path, "w") as f:
            yaml.dump(data, f)
        print(f"Updated YAML file saved. Original backed up as {backup_file}")
    else:
        print("No modifications needed for bounding boxes in the YAML file.")

def fix_label_file(label_file):
    """
    Fixes the bounding boxes in a YOLO-format label file.
    
    Each line in the file should be formatted as:
        class x_center y_center width height
    The function updates the file in place (after creating a backup) if any bounding box is fixed.
    """
    fixed_lines = []
    updated = False
    with open(label_file, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) != 5:
                print(f"Skipping malformed line in {label_file}: {line.strip()}")
                fixed_lines.append(line.strip())
                continue
            try:
                class_id = int(tokens[0])
                bbox = [float(x) for x in tokens[1:]]
            except Exception as e:
                print(f"Skipping line with invalid numbers in {label_file}: {line.strip()} (Error: {e})")
                fixed_lines.append(line.strip())
                continue
            fixed_bbox = fix_bbox(bbox)
            if fixed_bbox is not None:
                if not np.allclose(bbox, fixed_bbox, atol=1e-6):
                    updated = True
                fixed_line = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in fixed_bbox)
                fixed_lines.append(fixed_line)
            else:
                updated = True
                print(f"Removing degenerate bounding box from {label_file}: {line.strip()}")
    if updated:
        backup_path = label_file + ".bak"
        os.rename(label_file, backup_path)
        with open(label_file, "w") as f:
            for l in fixed_lines:
                f.write(l + "\n")
        print(f"Updated {label_file} (backup created at {backup_path})")

def process_labels_in_dirs(key, yaml_path, data):
    """
    Processes label files for a given key (e.g., 'train', 'val', 'test') in the dataset YAML.
    
    Each entry in data[key] is assumed to be an images folder path. The corresponding labels folder
    is derived by replacing 'images' with 'labels' in the folder path. All .txt files in the labels folder
    are processed and updated using fix_label_file.
    """
    folders = data[key] if isinstance(data[key], list) else [data[key]]
    base_path = data.get("path", "")
    for folder in folders:
        full_folder_path = os.path.join(base_path, folder) if base_path and not os.path.isabs(folder) else folder
        
        # Derive the labels folder from the images folder path.
        if "images" in full_folder_path:
            label_folder = full_folder_path.replace("images", "labels", 1)
        else:
            print(f"Warning: Expected 'images' in folder path '{full_folder_path}' to derive the corresponding labels folder. Skipping.")
            continue

        if not os.path.isdir(label_folder):
            print(f"Warning: The derived label folder '{label_folder}' does not exist.")
            continue

        print(f"Processing label folder: {label_folder}")
        for filename in os.listdir(label_folder):
            if filename.lower().endswith(".txt"):
                label_file = os.path.join(label_folder, filename)
                fix_label_file(label_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix bounding boxes in a YOLO dataset.")
    parser.add_argument("--yaml", type=str, default="/home/utkutopcuoglu/Projects/Datasets/person_datasets/person_data.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    yaml_file = args.yaml
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    if isinstance(data, dict):
        if "images" in data:
            process_yaml_with_images(yaml_file, data)
        else:
            # Process keys like 'train', 'val', or 'test'. These entries should be images folders,
            # from which the corresponding labels folders are derived.
            keys_to_process = [key for key in ["train", "val", "test"] if key in data]
            if not keys_to_process:
                raise ValueError("YAML does not contain 'images', 'train', 'val', or 'test' keys. Cannot process dataset.")
            for key in keys_to_process:
                process_labels_in_dirs(key, yaml_file, data)
    else:
        raise ValueError("Expected YAML content to be a dictionary.") 