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

### Copy-Paste Settings
Configure your augmentation settings in `simple_copy_paste.py`:

```python
copy_paste_hand_inhouse = A.Compose(
    [
        SelectiveCopyPaste(
            folder="/home/utkutopcuoglu/Projects/ebis/copy-paste-aug/hands_inhouse",
            max_paste_objects=2,
            blend=True,
            sigma=2, # The size of the gaussian kernel for blending. The larger the more smooth the blending, the more transparent the pasted object.
            max_attempts=20,
            p=0.3,
            class_id = 3
        )
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)

copy_paste_hand_public = A.Compose(
    [
        SelectiveCopyPaste(
            folder="/home/utkutopcuoglu/Projects/ebis/copy-paste-aug/hands_public",
            max_paste_objects=5,
            blend=True,
            sigma=2, # The size of the gaussian kernel for blending. The larger the more smooth the blending, the more transparent the pasted object.
            max_attempts=20,
            p=0.6,
            class_id = 3
        )
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)

copy_paste_tag = A.Compose(
    [
        SelectiveCopyPaste(
            folder="/home/utkutopcuoglu/Projects/ebis/copy-paste-aug/tags",
            max_paste_objects=7,
            blend=True,
            sigma=2, # The size of the gaussian kernel for blending. The larger the more smooth the blending, the more transparent the pasted object.
            max_attempts=20,
            p=0.9,
            class_id = 1
        )
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)

```

## Integration

1. Copy the `cpp_utils` folder to:
```
ultralytics/ultralytics/data/cpp_utils
```

Then modify ultralytics/ultralytics/data/base.py function get_image_and_label as follows (OR just copy paste copy-paste-aug/simple_cpp/cpp_utils/base.py to ultralytics/ultralytics/data/base.py):

1. Import the apply_copy_paste_augmentations and plot_yolo_predictions functions from the simple_copy_paste.py file.
```python
from ultralytics.data.cpp_utils.simple_copy_paste import apply_copy_paste_augmentations, plot_yolo_predictions
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
            )
            assert label["img"].ndim == prev_ndim, "Image ndim changed after copy paste augmentation."
            # NOTE calling update() automatically updates bbox_areas.
            label["instances"].update(bboxes=new_bboxes)
            
            # TODO UTKU, ass visualization path to ultralytics/ultralytics/cfg/default.yaml if you want to visualize the data.
            if DEFAULT_CFG.get("visualization_path"):
                plot_yolo_predictions(
                    label["img"],
                    label["instances"].bboxes,
                    label["cls"],
                    save_path=DEFAULT_CFG.visualization_path + "/" + str(index) + ".png"
                )

        return label
```

## Visualization
To enable visualization, add the following to `ultralytics/cfg/default.yaml`:
```yaml
visualization_path: "/path/to/save/visualizations"
```




# copy_paste_aug
