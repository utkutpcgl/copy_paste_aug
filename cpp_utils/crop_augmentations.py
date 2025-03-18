import random
import cv2
import albumentations as A
import numpy as np
import os
import time
import shutil
import yaml

# Remove previously saved debug crops and create the folder.
DEBUG_FOLDER_CROPS = "debug_augmented_objects"
if os.path.exists(DEBUG_FOLDER_CROPS):
    shutil.rmtree(DEBUG_FOLDER_CROPS)
os.makedirs(DEBUG_FOLDER_CROPS, exist_ok=True)


# -----------------------------------------------------------------------------
# Custom transforms as defined previously.
# -----------------------------------------------------------------------------
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
        (h, w) = img.shape[:2]
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

# Custom Random Resize transform that preserves aspect ratio.
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

# Custom Random Crop transform; crops randomly down to a percentage of the original area.
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

class RandomGamma(A.ImageOnlyTransform):
    def __init__(self, gamma_limit=(80, 120), p=0.3, always_apply=False):
        super(RandomGamma, self).__init__(always_apply, p)
        self.gamma_limit = gamma_limit

    def get_params(self):
        gamma_value = random.uniform(self.gamma_limit[0], self.gamma_limit[1])
        gamma = gamma_value / 100.0  # e.g., 80 becomes 0.8 and 120 becomes 1.2
        return {"gamma": gamma}

    def apply(self, img, gamma=1.0, **params):
        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(img, table)
    
    def get_transform_init_args_names(self):
        return ("gamma_limit", "p")


# -----------------------------------------------------------------------------
# Load configuration parameters from the central config.yaml.
# -----------------------------------------------------------------------------
# The config file is located at: copy-paste-aug/simple_cpp/cpp_utils/config.yaml
CFG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
assert os.path.exists(CFG_PATH), f"Configuration file not found at {CFG_PATH}"
with open(CFG_PATH, 'r') as f:
    config = yaml.safe_load(f)

crop_config = config.get("crop_augmentations", {})
# Get pixel-level transforms config.
pixel_config = crop_config.get("pixel_transforms", {})
rb_config = pixel_config.get("random_brightness_contrast", {})
hs_config = pixel_config.get("hue_saturation_value", {})
rgb_config = pixel_config.get("rgb_shift", {})
rg_config = pixel_config.get("random_gamma", {})
gb_config = pixel_config.get("gaussian_blur", {})
gn_config = pixel_config.get("gauss_noise", {})
ic_config = pixel_config.get("image_compression", {})
clahe_config = pixel_config.get("clahe", {})

# Build the pixel-level transforms using config parameters.
pixel_transforms = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=rb_config.get("brightness_limit", 0.2),
        contrast_limit=rb_config.get("contrast_limit", 0.2),
        p=rb_config.get("p", 0.3)
    ),
    A.HueSaturationValue(
        hue_shift_limit=hs_config.get("hue_shift_limit", 20),
        sat_shift_limit=hs_config.get("sat_shift_limit", 30),
        val_shift_limit=hs_config.get("val_shift_limit", 20),
        p=hs_config.get("p", 0.3)
    ),
    A.RGBShift(
        r_shift_limit=rgb_config.get("r_shift_limit", 10),
        g_shift_limit=rgb_config.get("g_shift_limit", 10),
        b_shift_limit=rgb_config.get("b_shift_limit", 10),
        p=rgb_config.get("p", 0.1)
    ),
    RandomGamma(
        gamma_limit=tuple(rg_config.get("gamma_limit", [80, 120])),
        p=rg_config.get("p", 0.3)
    ),
    A.GaussianBlur(p=gb_config.get("p", 0.1)),
    A.GaussNoise(
        std_range=tuple(gn_config.get("std_range", [0.03, 0.1])),
        p=gn_config.get("p", 0.1)
    ),
    A.ImageCompression(
        quality_range=tuple(ic_config.get("quality_range", [80, 100])),
        p=ic_config.get("p", 0.1)
    ),
    A.CLAHE(
        clip_limit=clahe_config.get("clip_limit", 2),
        tile_grid_size=tuple(clahe_config.get("tile_grid_size", [8, 8])),
        p=clahe_config.get("p", 0.1)
    )
])

# Get spatial-level transforms config.
spatial_config = crop_config.get("spatial_transforms", {})
hf_config = spatial_config.get("horizontal_flip", {})
vf_config = spatial_config.get("vertical_flip", {})
rwe_config = spatial_config.get("rotate_with_expansion", {})
affine_config = spatial_config.get("affine", {})
persp_config = spatial_config.get("perspective", {})

# Prepare parameters for RotateWithExpansion.
rwe_limit = tuple(rwe_config.get("limit", [-90, 90]))
rwe_border_mode_str = rwe_config.get("border_mode", "BORDER_CONSTANT")
rwe_border_mode = getattr(cv2, rwe_border_mode_str)
rwe_value = tuple(rwe_config.get("value", [0, 0, 0, 0]))
rwe_mask_value = tuple(rwe_config.get("mask_value", [0, 0, 0, 0]))
rwe_p = rwe_config.get("p", 1)

# Prepare parameters for Affine transform.
affine_translate = tuple(affine_config.get("translate_percent", [-0.0625, 0.0625]))
affine_scale = tuple(affine_config.get("scale", [0.9, 1.1]))
affine_rotate = tuple(affine_config.get("rotate", [-15, 15]))
affine_border_mode_str = affine_config.get("border_mode", "BORDER_CONSTANT")
affine_border_mode = getattr(cv2, affine_border_mode_str)
affine_p = affine_config.get("p", 0.5)

# Prepare parameters for Perspective transform.
persp_scale = tuple(persp_config.get("scale", [0.05, 0.1]))
persp_p = persp_config.get("p", 0.2)

# Get random resize/crop config.
# random_resize_config = crop_config.get("random_resize", {})
random_crop_config = crop_config.get("random_crop", {})


# -----------------------------------------------------------------------------
# Instantiate our custom transforms.
# -----------------------------------------------------------------------------
spatial_transforms = A.Compose([
    A.HorizontalFlip(p=hf_config.get("p", 0.5)),
    A.VerticalFlip(p=vf_config.get("p", 0.5)),
    # Affine and Perspective are defined using parameters from the configuration.
    # Note: The custom RotateWithExpansion transform is used instead of Albumentations built-in rotation.
    # See the custom class below.
    # We pass the loaded config values.
    # The p value for RotateWithExpansion is set in rwe_p.
    # Similarly for Affine and Perspective.
    # The cv2 border-mode strings are converted with getattr.
    # RotateWithExpansion is defined later in this file.
    # It rotates the image with expansion so that the entire image fits.
    # -----------------------------------------------------------------------------
    # Use custom RotateWithExpansion (defined below).
    RotateWithExpansion(limit=rwe_limit, border_mode=rwe_border_mode, value=rwe_value, mask_value=rwe_mask_value, p=rwe_p),
    A.Affine(
        translate_percent=affine_translate,
        scale=affine_scale,
        rotate=affine_rotate,
        border_mode=affine_border_mode,
        p=affine_p
    ),
    A.Perspective(
        scale=persp_scale,
        p=persp_p
    )
])

# resize_transform = RandomResize(
#     scale_range=tuple(random_resize_config.get("scale_range", [0.3, 1.2])),
#     p=random_resize_config.get("p", 1)
# )


random_crop = RandomCropCustom(
    min_crop=random_crop_config.get("min_crop", 0.75),
    p=random_crop_config.get("p", 0.6)
)


# -----------------------------------------------------------------------------
# The augment_crop function remains mostly the same, but now uses the updated transforms.
# -----------------------------------------------------------------------------
def augment_crop(obj_img, debug_crops=False, min_long_edge_ratio=0.2, max_long_edge_ratio=0.66, target_long_edge=640):
    """
    Applies all the augmentations defined in the augmentation menu (loaded from config)
    to the cropped segmentation.

    Args:
        obj_img (np.ndarray): Cropped object image with 4 channels (BGR + alpha).
        debug_crops (bool): Save augmented object crops for debugging if True.
        min_long_edge_ratio (float): Minimum ratio (to target_long_edge) for the object's long side.
        max_long_edge_ratio (float): Maximum ratio (to target_long_edge) for the object's long side.
        target_long_edge (int): The reference long edge (e.g. max(target image dimension)).

    Returns:
        np.ndarray: Augmented object image (4 channels: BGR + alpha) or None if augmentation fails.
    """
    if obj_img.shape[-1] != 4:
        print("Object image must have 4 channels (BGR + alpha). Returning None. Object shape: ", obj_img.shape)
        return None

    # --- Custom resizing step based on long-edge ratio ---
    # Determine the current long edge of the cropped object.
    current_long_edge = max(obj_img.shape[:2])
    # Choose a desired long edge randomly between the provided ratios of target_long_edge.
    desired_long_edge = random.uniform(target_long_edge * min_long_edge_ratio, target_long_edge * max_long_edge_ratio)
    scale_factor = desired_long_edge / current_long_edge
    new_w = max(1, int(obj_img.shape[1] * scale_factor))
    new_h = max(1, int(obj_img.shape[0] * scale_factor))
    obj_img = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # --- Continue with existing pixel- and spatial-level augmentations ---
    bgr_img = obj_img[:, :, :3]
    alpha_mask = obj_img[:, :, 3]

    pixel_bgr = pixel_transforms(image=bgr_img)['image']
    pixel_image = np.dstack([pixel_bgr, alpha_mask])
    spatial_pixel_image = spatial_transforms(image=pixel_image)['image']

    # Crop the image so that only the actual object is strictly cropped,
    # based on the alpha mask. Transparent pixels should not fill the box edges.
    cropped_resized_spatial_pixel_image = random_crop(image=spatial_pixel_image)['image']

    # Crop the image so that only the actual object is retained (using the alpha mask).
    alpha_final = cropped_resized_spatial_pixel_image[:, :, 3]
    ys, xs = np.where(alpha_final > 0)
    
    # Assert we found at least one non-transparent pixel.
    if not (ys.size > 0 and xs.size > 0):
        print("The augmented image does not contain any non-transparent pixels based on the alpha mask. Returning None.")
        return None # else return NONE, dont copy paste this object
    
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    cropped_resized_spatial_pixel_image = cropped_resized_spatial_pixel_image[y_min:y_max+1, x_min:x_max+1]


    # Save the augmented object for debugging if required.
    if debug_crops:
        cv2.imwrite(f"{DEBUG_FOLDER_CROPS}/augmented_object_{time.time()}.png", cropped_resized_spatial_pixel_image)
    return cropped_resized_spatial_pixel_image 