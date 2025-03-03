import random
import cv2
import albumentations as A
import numpy as np
import os
import time
import shutil

DEBUG_FOLDER_CROPS = "debug_augmented_objects"
if os.path.exists(DEBUG_FOLDER_CROPS):
    shutil.rmtree(DEBUG_FOLDER_CROPS)
os.makedirs(DEBUG_FOLDER_CROPS, exist_ok=True)

# Custom rotation that expands the image canvas so that the entire rotated image fits.
class RotateWithExpansion(A.DualTransform):
    def __init__(self, limit=(-90, 90), border_mode=cv2.BORDER_CONSTANT, 
                 value=(0, 0, 0, 0), mask_value=(0, 0, 0, 0), p=1):
        super(RotateWithExpansion, self).__init__(p)
        self.limit = limit
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def get_params(self):
        # Choose a random angle from the provided range.
        angle = random.uniform(self.limit[0], self.limit[1])
        return {"angle": angle}

    def apply(self, img, angle=0, **kwargs):
        # Assuming img.shape is (h, w, channels)
        (h, w) = img.shape[:2]
        # Define the center and compute the rotation matrix.
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Compute the sine and cosine of the rotation angle.
        abs_cos = abs(rot_mat[0, 0])
        abs_sin = abs(rot_mat[0, 1])

        # Compute new bounding dimensions to ensure the whole image fits.
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)

        # Adjust the rotation matrix to take into account the translation.
        rot_mat[0, 2] += (new_w / 2) - center[0]
        rot_mat[1, 2] += (new_h / 2) - center[1]

        # Perform the warp affine transformation with the expanded canvas.
        rotated = cv2.warpAffine(
            img,
            rot_mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=self.border_mode,
            borderValue=self.value,
        )
        return rotated

    def get_transform_init_args_names(self):
        return ("limit", "border_mode", "value", "mask_value", "p")

# 1. Pixel-Level Transforms (applied only on the RGB image)
pixel_transforms = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),
    A.GaussianBlur(p=0.2),
    A.GaussNoise(std_range=(0.03,0.15), p=0.2),
    A.ImageCompression(quality_range=(75, 100), p=0.2),
    A.CLAHE(clip_limit=2, tile_grid_size=(8,8), p=0.2)
])

# 2. Spatial-Level Transforms (applied to image and alpha mask synchronously)
spatial_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    RotateWithExpansion(limit=(-90, 90), p=1),
    A.Affine(
        translate_percent=(-0.0625, 0.0625),
        scale=(0.9, 1.1),
        rotate=(-15, 15),
        border_mode=cv2.BORDER_CONSTANT,
        p=0.5
    ),
    A.Perspective(scale=(0.05, 0.1), p=0.2)
])


# 3. Custom Random Resize transform that preserves aspect ratio.
class RandomResize(A.DualTransform):
    def __init__(self, scale_range=(0.2, 2), p=0.5):
        super(RandomResize, self).__init__(p)
        self.min_scale = scale_range[0]
        self.max_scale = scale_range[1]

    def get_params(self):
        factor = random.uniform(self.min_scale, self.max_scale)
        return {'factor': factor}

    def apply(self, img, factor=1.0, **kwargs):
        new_w = int(img.shape[1] * factor)
        new_h = int(img.shape[0] * factor)
        new_w = max(new_w, 1)
        new_h = max(new_h, 1)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def get_transform_init_args_names(self):
        return ("min_scale", "max_scale", "p")

# 4. Custom Random Crop transform; crops randomly down to a percentage of the original area.
class RandomCropCustom(A.DualTransform):
    def __init__(self, min_crop=0.75, p=0.5):
        super(RandomCropCustom, self).__init__(p)
        self.min_crop = min_crop
    
    def get_params(self):
        return {"temp": 1}

    def apply(self, img, temp=1, **kwargs):
        h, w = img.shape[:2]
        scale = random.uniform(self.min_crop, 1.0)
        new_h, new_w = int(h * scale), int(w * scale)
        new_h = max(new_h, 1)
        new_w = max(new_w, 1)
        top = random.randint(0, h - new_h) if (h - new_h) > 0 else 0
        left = random.randint(0, w - new_w) if (w - new_w) > 0 else 0
        return img[top:top + new_h, left:left + new_w]

    def get_transform_init_args_names(self):
        return ("min_crop", "p")

# Instantiate our custom transforms.
resize_transform = RandomResize(scale_range=(0.3, 1.2), p=1)
random_crop = RandomCropCustom(min_crop=0.75, p=0.6)

def augment_crop(obj_img):
    """
    Applies all the augmentations defined in the augmentation menu to the cropped segmentation.

    Args:
        obj_img (np.ndarray): Cropped object image with 4 channels (BGR + alpha).

    Returns:
        np.ndarray: Augmented object image (4 channels: BGR + alpha).
    """
    assert obj_img.shape[-1] == 4, "Object image must have 4 channels (BGR + alpha)."
    
    # Separate the BGR image and the alpha mask.
    bgr_img = obj_img[:, :, :3]
    alpha_mask = obj_img[:, :, 3]
    
    # 1. Apply pixel-level augmentations to the BGR image.
    pixel_bgr = pixel_transforms(image=bgr_img)['image']

    # 2. Combine the pixel-augmented BGR with the unchanged alpha mask.
    pixel_image = np.dstack([pixel_bgr, alpha_mask])
    
    # 3. Apply spatial-level transforms to the entire image.
    spatial_pixel_image = spatial_transforms(image=pixel_image)['image']
    
    # 4. Apply random resizing.
    resized_spatial_pixel_image = resize_transform(image=spatial_pixel_image)['image']
    
    # 5. Apply random cropping.
    cropped_resized_spatial_pixel_image = random_crop(image=resized_spatial_pixel_image)['image']

    # Crop the image so that only the actual object is strictly cropped,
    # based on the alpha mask. Transparent pixels should not be filling the box edges.
    alpha_final = cropped_resized_spatial_pixel_image[:, :, 3]
    ys, xs = np.where(alpha_final > 0)
    
    # Assert we found at least one non-transparent pixel.
    assert ys.size > 0 and xs.size > 0, "The augmented image does not contain any non-transparent pixels based on the alpha mask."
    
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    cropped_resized_spatial_pixel_image = cropped_resized_spatial_pixel_image[y_min:y_max+1, x_min:x_max+1]

    # Save the augmented object to a folder for debugging.
    cv2.imwrite(f"{DEBUG_FOLDER_CROPS}/augmented_object_{time.time()}.png", cropped_resized_spatial_pixel_image)
    return cropped_resized_spatial_pixel_image 