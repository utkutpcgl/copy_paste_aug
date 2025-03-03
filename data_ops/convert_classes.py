#!/usr/bin/env python
"""
convert_classes.py

This script converts all class indices in a YOLO dataset's bounding boxes from 0 to 3.
It works in two modes:

1. If the YAML file contains an "images" key, each image entry is expected to have a "bboxes" list,
   where every bounding box is a list in the format:
       [class, x_center, y_center, width, height]
   The script asserts that the class is 0 and updates it to 3. The YAML file is backed up before
   writing any changes.

2. Otherwise, keys such as "train", "val", or "test" are processed. Each entry under these keys is
   assumed to be an images folder path. The corresponding labels folder is derived by replacing "images"
   with "labels" in the folder path. Each .txt label file is then processed: each line (expected to contain
   5 tokens as "class x_center y_center width height") is checked to assert the class is 0 and updated to 3.
   A backup of any modified label file is made.
"""

import os
import yaml
import argparse

def convert_bbox_class(bbox):
    """
    Convert the bounding box class index from 0 to 3.

    Args:
        bbox (list): A YOLO-format bbox [class, x_center, y_center, width, height].

    Returns:
        list: Updated bbox with the class index set to 3.

    Raises:
        AssertionError: If the original class index is not 0.
    """
    assert len(bbox) == 5, f"Expected bbox with 5 values, got {bbox}"
    current_class = bbox[0]
    assert current_class == 0, f"Expected class index 0, got {current_class}"
    return [3] + bbox[1:]

def process_yaml_with_images(yaml_path, data):
    """
    Process a YAML file that contains an "images" list with bounding boxes.
    Each bounding box is expected in the format [class, x_center, y_center, width, height].
    Assert that every class is 0, update it to 3, and write back the YAML file (backing up the original).

    Args:
        yaml_path (str): Path to the YAML file.
        data (dict): The YAML data.
    """
    images_info = data.get("images", [])
    modified = False

    for image_entry in images_info:
        if "bboxes" in image_entry and image_entry["bboxes"]:
            new_bboxes = []
            for bbox in image_entry["bboxes"]:
                new_bbox = convert_bbox_class(bbox)
                new_bboxes.append(new_bbox)
            if new_bboxes != image_entry["bboxes"]:
                image_entry["bboxes"] = new_bboxes
                modified = True

    if modified:
        backup_file = yaml_path + ".bak"
        os.rename(yaml_path, backup_file)
        with open(yaml_path, "w") as f:
            yaml.dump(data, f)
        print(f"Updated YAML file; backup saved as {backup_file}")
    else:
        print("No changes needed in YAML file.")

def convert_label_file(label_file):
    """
    Convert the class indices in a YOLO-format label file.
    Each line should be:
        class x_center y_center width height
    Assert that the class is 0 and update it to 3. Updates the file in place after creating a backup.

    Args:
        label_file (str): Path to the label file.
    """
    updated_lines = []
    modified = False

    with open(label_file, "r") as f:
        for line in f:
            tokens = line.strip().split()
            # Skip empty lines
            if not tokens:
                continue
            assert len(tokens) == 5, f"Malformed line in {label_file}: {line.strip()}"
            try:
                class_id = int(tokens[0])
            except Exception as e:
                raise ValueError(f"Invalid class token in {label_file}: {line.strip()} (Error: {e})")
            assert class_id == 0, f"Expected class index 0, got {class_id} in {label_file}"
            updated_line = f"3 " + " ".join(tokens[1:])
            updated_lines.append(updated_line)
            modified = True

    if modified:
        backup_path = label_file + ".bak"
        os.rename(label_file, backup_path)
        with open(label_file, "w") as f:
            for l in updated_lines:
                f.write(l + "\n")
        print(f"Updated {label_file}; backup saved as {backup_path}")

def process_labels_in_dirs(key, yaml_path, data):
    """
    Process label files for a given key (e.g. "train", "val", or "test") in the dataset YAML.
    Each entry is assumed to be an images folder path. The corresponding labels folder is obtained
    by replacing "images" with "labels". All .txt files in the labels folder are processed.

    Args:
        key (str): Key in the YAML (e.g., "train").
        yaml_path (str): Path to the YAML file.
        data (dict): The YAML data.
    """
    entries = data[key] if isinstance(data[key], list) else [data[key]]
    base_path = data.get("path", "")
    for folder in entries:
        full_folder_path = os.path.join(base_path, folder) if base_path and not os.path.isabs(folder) else folder

        if "images" in full_folder_path:
            label_folder = full_folder_path.replace("images", "labels", 1)
        else:
            print(f"Warning: Expected 'images' in folder path '{full_folder_path}' to derive labels folder; skipping.")
            continue

        if not os.path.isdir(label_folder):
            print(f"Warning: Labels folder '{label_folder}' does not exist; skipping.")
            continue

        print(f"Processing label folder: {label_folder}")
        for filename in os.listdir(label_folder):
            if filename.lower().endswith(".txt"):
                label_file = os.path.join(label_folder, filename)
                convert_label_file(label_file)

def main():
    parser = argparse.ArgumentParser(description="Convert all class indices from 0 to 3 in a YOLO dataset YAML file.")
    parser.add_argument("--yaml", type=str, default="/home/utkutopcuoglu/Projects/Datasets/person_datasets/person_data.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    yaml_file = args.yaml
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    if isinstance(data, dict):
        # If the YAML has "images", process bboxes inside it.
        if "images" in data:
            process_yaml_with_images(yaml_file, data)
        else:
            # Otherwise, process label files corresponding to "train", "val", or "test".
            keys_to_process = [key for key in ["train", "val", "test"] if key in data]
            if not keys_to_process:
                raise ValueError("YAML does not contain 'images', 'train', 'val', or 'test' keys. Cannot process dataset.")
            for key in keys_to_process:
                process_labels_in_dirs(key, yaml_file, data)
    else:
        raise ValueError("Expected YAML content to be a dictionary.")

if __name__ == "__main__":
    main() 