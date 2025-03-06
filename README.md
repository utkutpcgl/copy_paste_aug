# Copy-Paste Augmentation for YOLOv8

This repository provides a simple copy-paste augmentation implementation for YOLOv8 using Albumentations.

## Installation

1. Clone the Ultralytics repository:
```bash
git clone https://github.com/ultralytics/ultralytics.git
```

2. Install the package in editable mode for development:
```bash
cd ultralytics
pip install -e .
```

3. Install additional dependencies:
```bash
pip3 install albumentations
pip3 install scikit-image
```

## Configuration

### Import Path Modification
Update the import path in `simple_copy_paste.py`:
```python
from ultralytics.data.cpp_utils.simple_copy_paste import apply_copy_paste_augmentations
```

``  
simple_cpp/cpp_utils/config.yaml
``

This YAML file contains multiple sections:

- **augmentation**:  
  Contains visualization settings and the parameters for different copy-paste augmentation pipelines (such as *hands_inhouse*, *hands_public*, and *tags*).

- **crop_augmentations**:  
  Contains parameters for pixel-level transforms (brightness, contrast, blur, etc.), spatial transforms (flip, rotation, affine, perspective), as well as random resize and crop configurations.

- **debug**:  
  Contains settings used for debugging (e.g., the number of augmented samples, folder names, and dataset YAML file path).

Ensure you set the correct directory paths and values in this file. All modules reference it via relative paths so that they behave consistently.

---

## Modules

- **Augmentation Pipeline**  
  The copy-paste augmentation pipelines are defined in ``simple_copy_paste.py`` and instantiated based on configuration parameters. These pipelines:
  - Paste object crops ensuring minimal overlap with existing objects.
  - Blend pasted objects using Gaussian smoothing if enabled.
  - Scale and adjust objects based on the provided configuration.

- **Crop Augmentations**  
  In ``crop_augmentations.py``, custom transforms (like ``RotateWithExpansion``, ``RandomResize``, and ``RandomCropCustom``) are created using parameters from the configuration file.

- **Debug Augmentations**  
  Run the script ``debug_augment.py`` to visualize augmentations on test images. Debug images (with drawn bounding boxes) are saved to the folder specified in the configuration.

---

## Running the Augmentations

```

## Integration

1. Copy the `cpp_utils` folder to:
```
ultralytics/ultralytics/data/cpp_utils
```

Then modify ultralytics/ultralytics/data/base.py function get_image_and_label as follows (OR just copy paste copy-paste-aug/simple_cpp/cpp_utils/base.py to ultralytics/ultralytics/data/base.py):

1. Import the apply_copy_paste_augmentations and plot_yolo_predictions functions from the simple_copy_paste.py file.

```python
from ultralytics.data.cpp_utils.simple_copy_paste import apply_copy_paste_augmentations, plot_yolo_predictions, VISUALIZATION_PATH
```

2. Modify the get_image_and_label function as follows:
```python
    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        label = self.update_labels_info(label)
        # NOTE UTKU: here we should return the new label with the COPY PASTE applied.
        if self.augment:
            # NOTE UTKU this is the overall idea:
            # label["cls"] gives the class id. (float32 numpy array)
            # label["instances"]["bboxes"] gives the bounding boxes. (float32 numpy array)
            # label["instances"]["bbox_areas"] gives the bounding bbox areas. (float32 numpy array)
            # label["img"] gives the image. (uint8 numpy array)
            prev_ndim = label["img"].ndim
            label["img"], new_bboxes , label["cls"] = apply_copy_paste_augmentations(
                label["img"],
                label["instances"].bboxes,
                label["cls"],
                resized_shape=label["resized_shape"],
                ori_shape=label["ori_shape"],

            )
            assert label["img"].ndim == prev_ndim, "Image ndim changed after copy paste augmentation."
            # NOTE calling update() automatically updates bbox_areas.
            label["instances"].update(bboxes=new_bboxes)
            
            # NOTE UTKU, ADD visualization path to ultralytics/ultralytics/cfg/default.yaml if you want to visualize the data.
            if VISUALIZATION_PATH:
                plot_yolo_predictions(
                    label["img"],
                    label["instances"].bboxes,
                    label["cls"],
                    save_path=VISUALIZATION_PATH + "/" + str(index) + ".png"
                )

        return label
```

## Visualization
To enable visualization, uncomment and set the following line in the yaml file ``simple_cpp/cpp_utils/config.yaml``:
```yaml
visualization_path: "visualizations"  # Path to save augmented images. Uncomment this if you want to save augmented images.
```
*NOTE*: Only batch size number of images will be saved circularly.