# Albumentation Box Preprocessing

## Converts Bounding Boxes to Albumentations Format

Albumentations expects bounding boxes in the format `(x_min, y_min, x_max, y_max)`.

```python
@handle_empty_array("bboxes")
def convert_bboxes_to_albumentations(
    bboxes: np.ndarray,
    source_format: Literal["coco", "pascal_voc", "yolo"],
    shape: ShapeType,
    check_validity: bool = False,
) -> np.ndarray:
    """Convert bounding boxes from a specified format to the format used by albumentations:
    normalized coordinates of top-left and bottom-right corners of the bounding box in the form of
    `(x_min, y_min, x_max, y_max)` e.g. `(0.15, 0.27, 0.67, 0.5)`.

    Args:
        bboxes: A numpy array of bounding boxes with shape (num_bboxes, 4+).
        source_format: Format of the input bounding boxes. Should be 'coco', 'pascal_voc', or 'yolo'.
        shape: Image shape (height, width).
        check_validity: Check if all boxes are valid boxes.

    Returns:
        np.ndarray: An array of bounding boxes in albumentations format with shape (num_bboxes, 4+).

    Raises:
        ValueError: If `source_format` is not 'coco', 'pascal_voc', or 'yolo'.
        ValueError: If in YOLO format, any coordinates are not in the range (0, 1].
    """
    if source_format not in {"coco", "pascal_voc", "yolo"}:
        raise ValueError(
            f"Unknown source_format {source_format}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'",
        )

    bboxes = bboxes.copy().astype(np.float32)
    converted_bboxes = np.zeros_like(bboxes)
    converted_bboxes[:, 4:] = bboxes[:, 4:]  # Preserve additional columns

    if source_format == "coco":
        converted_bboxes[:, 0] = bboxes[:, 0]  # x_min
        converted_bboxes[:, 1] = bboxes[:, 1]  # y_min
        converted_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # x_max
        converted_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # y_max
    elif source_format == "yolo":
        if check_validity and np.any((bboxes[:, :4] <= 0) | (bboxes[:, :4] > 1)):
            raise ValueError(f"In YOLO format all coordinates must be float and in range (0, 1], got {bboxes}")

        w_half, h_half = bboxes[:, 2] / 2, bboxes[:, 3] / 2
        converted_bboxes[:, 0] = bboxes[:, 0] - w_half  # x_min
        converted_bboxes[:, 1] = bboxes[:, 1] - h_half  # y_min
        converted_bboxes[:, 2] = bboxes[:, 0] + w_half  # x_max
        converted_bboxes[:, 3] = bboxes[:, 1] + h_half  # y_max
    else:  # pascal_voc
        converted_bboxes[:, :4] = bboxes[:, :4]

    if source_format != "yolo":
        converted_bboxes[:, :4] = normalize_bboxes(converted_bboxes[:, :4], shape)

    if check_validity:
        check_bboxes(converted_bboxes)

    return converted_bboxes
```

The output will be in the format `(x_min, y_min, x_max, y_max)` while being passed to SelectiveCopyPaste(A.DualTransform) augmentation.