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
from ultralytics.data.cpp_utils.crop_augmentations import augment_crop
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

2. Then modify configs as you wish.

3. Update the import path in `simple_copy_paste.py` if not correct:
```python
from ultralytics.data.cpp_utils.crop_augmentations import augment_crop
```

4. Modify the ultralytics/ultralytics/data/augment.py/v8_transforms function as follows:

```python

    def v8_transforms(dataset, imgsz, hyp, stretch=False):
    """
    Applies a series of image transformations for training.

    This function creates a composition of image augmentation techniques to prepare images for YOLO training.
    It includes operations such as mosaic, copy-paste, random perspective, mixup, and various color adjustments.

    Args:
        dataset (Dataset): The dataset object containing image data and annotations.
        imgsz (int): The target image size for resizing.
        hyp (Namespace): A dictionary of hyperparameters controlling various aspects of the transformations.
        stretch (bool): If True, applies stretching to the image. If False, uses LetterBox resizing.

    Returns:
        (Compose): A composition of image transformations to be applied to the dataset.

    Examples:
        >>> from ultralytics.data.dataset import YOLODataset
        >>> from ultralytics.utils import IterableSimpleNamespace
        >>> dataset = YOLODataset(img_path="path/to/images", imgsz=640)
        >>> hyp = IterableSimpleNamespace(mosaic=1.0, copy_paste=0.5, degrees=10.0, translate=0.2, scale=0.9)
        >>> transforms = v8_transforms(dataset, imgsz=640, hyp=hyp)
        >>> augmented_data = transforms(dataset[0])
    """
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    affine = RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
    )

    pre_transform = Compose([mosaic, affine])
    if hyp.copy_paste_mode == "flip":
        pre_transform.insert(1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    else:
        pre_transform.append(
            CopyPaste(
                dataset,
                pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),
                p=hyp.copy_paste,
                mode=hyp.copy_paste_mode,
            )
        )
    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")

    # Build the complete pipeline and add the copy paste wrapper at the end.
    from ultralytics.data.cpp_utils.simple_copy_paste import CopyPasteUtku   
    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            Albumentations(p=1.0),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
            # Apply copy paste AFTER all other transforms
            CopyPasteUtku() # NOTE Added by UTKU
        ]
    )  # transforms
```

## Visualization
To enable visualization, uncomment and set the following line in the yaml file ``simple_cpp/cpp_utils/config.yaml``:
```yaml
visualization_path: "visualizations"  # Path to save augmented images. Uncomment this if you want to save augmented images.
```
*NOTE*: Only batch size number of images will be saved circularly.


### Visualizing training samples right before the model sees them
Modify the trainer.py code as follows, in the BaseTrainer class, _do_train method:
```python
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                # TODO visualize the batch here with the yolo annotation bounding boxes (with classes) to validate copy paste augmentation is working as desired.
                self.plot_training_samples(batch, ni)
```
