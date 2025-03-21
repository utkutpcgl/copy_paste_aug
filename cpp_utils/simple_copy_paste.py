import os
import random
import cv2
import numpy as np
from skimage.filters import gaussian
import albumentations as A
from ultralytics.data.cpp_utils.crop_augmentations import augment_crop
import yaml
# visualization_path: /home/utkutopcuoglu/Projects/ebis/copy-paste-aug/visualization # (str) path to save the visualization images

# Load configuration for augmentation from the central config.yaml.
CFG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
assert os.path.exists(CFG_PATH), f"Configuration file not found at {CFG_PATH}"
with open(CFG_PATH, 'r') as f:
    global_config = yaml.safe_load(f)


# Extract augmentation configuration.
augmentation_config = global_config.get("augmentation", {})
VISUALIZATION_PATH = augmentation_config.get("visualization_path", None)
if VISUALIZATION_PATH:
    os.makedirs(VISUALIZATION_PATH, exist_ok=True)
DEBUG_CROPS_CONFIG = augmentation_config.get("debug_crops", False)
copy_paste_config = augmentation_config.get("copy_paste", {})

# Get individual pipeline configs.
hands_inhouse_config = copy_paste_config.get("hands_inhouse", {})
hands_public_config = copy_paste_config.get("hands_public", {})
tags_config = copy_paste_config.get("tags", {})

# visualization_path is not directly used in this file but can be used elsewhere if needed.
# visualization_path = augmentation_config.get("visualization_path", None)

class SelectiveCopyPaste(A.DualTransform):
    augmented_count = 0


    def __init__(
        self,
        folder,
        max_paste_objects=5,
        blend=True,
        sigma=1,
        max_attempts=20,
        p=1,
        always_apply=False,
        class_id=None,
        max_occlude_ratio=0.5,
    ):
        """
        Args:
            folder (str): Path to folder containing object crop images.
            max_paste_objects (int): Maximum number of objects to paste.
            blend (bool): Whether to blend pasted objects using Gaussian smoothing.
            sigma (float): Sigma value for the Gaussian filter.
            max_attempts (int): Maximum number of attempts to find a non-overlapping paste location.
            p (float): Probability of applying the transform.
            always_apply (bool): If True, the transform is always applied.
        """
        super(SelectiveCopyPaste, self).__init__(p, always_apply)
        self.initial_p = p  # store the initial probability for linear decay
        self.folder = folder
        self.max_paste_objects = max_paste_objects
        self.blend = blend
        self.sigma = sigma
        self.max_attempts = max_attempts
        self.max_occlude_ratio = max_occlude_ratio
        self.object_class = class_id

        # Preload list of valid image files.
        self.object_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        # Set up new long-edge ratio parameters based on the object class.
        # For hands (class_id == 3) we use tag ratios; for tags (class_id == 0) we use hand ratios.
        if self.object_class == 3:
            self.min_long_edge_ratio = hands_inhouse_config.get("hand_min_long_edge_ratio", 0.1)
            self.max_long_edge_ratio = hands_inhouse_config.get("hand_max_long_edge_ratio", 0.6)
        elif self.object_class == 0:
            self.min_long_edge_ratio = tags_config.get("tag_min_long_edge_ratio", 0.2)
            self.max_long_edge_ratio = tags_config.get("tag_max_long_edge_ratio", 0.66)
        else:
            # Use a default ratio if no specific rule is defined.
            self.min_long_edge_ratio = 0.2
            self.max_long_edge_ratio = 0.66

    @staticmethod
    def check_overlap(bbox1, bbox2):
        """
        Check if two bounding boxes (in [x1, y1, x2, y2] format) overlap.

        Args:
            bbox1 (np.ndarray): [x1, y1, x2, y2]
            bbox2 (np.ndarray): [x1, y1, x2, y2]
            
        Returns:
            bool: True if there is overlap, False otherwise.
        """
        bbox1 = np.array(bbox1, dtype=np.float32)
        bbox2 = np.array(bbox2, dtype=np.float32)
        if bbox1.shape[0] != 4 or bbox2.shape[0] != 4:
            raise ValueError("Both bboxes must be in [x1, y1, x2, y2] format.")
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        assert x1_1 <= x2_1 and y1_1 <= y2_1 and x1_2 <= x2_2 and y1_2 <= y2_2, f"bbox1: {bbox1}, bbox2: {bbox2}"
        if x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1:
            return False
        return True

    def get_occlude_ratio(self, candidate_bbox, bbox):
        """
        Compute the occlusion ratio of bbox (box2) by candidate_bbox.
        The occlusion ratio is defined as the fraction of bbox's area that is covered by candidate_bbox.
        
        For example:
            - If candidate_bbox covers 50% of bbox, this returns 0.5.
            - If candidate_bbox covers 25% of bbox, this returns 0.25.
        
        Args:
            candidate_bbox (np.ndarray): The candidate bounding box in [x1, y1, x2, y2] format.
            bbox (np.ndarray): The reference bounding box (box2) in [x1, y1, x2, y2] format.
            
        Returns:
            float: The occlusion ratio.
        """
        candidate_bbox = np.array(candidate_bbox, dtype=np.float32)
        bbox = np.array(bbox, dtype=np.float32)
        
        # Compute intersection coordinates.
        inter_x1 = max(candidate_bbox[0], bbox[0])
        inter_y1 = max(candidate_bbox[1], bbox[1])
        inter_x2 = min(candidate_bbox[2], bbox[2])
        inter_y2 = min(candidate_bbox[3], bbox[3])
        
        # Compute intersection area.
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height
        
        # Compute area of bbox.
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if bbox_area <= 0:
            return 0.0
        
        occlude_ratio = inter_area / bbox_area
        return occlude_ratio

    def is_valid_position(self, candidate_bbox, bboxes, class_labels):
        """
        Check if candidate_bbox (in [x1, y1, x2, y2] format) overlaps any bbox
        in bboxes whose corresponding label equals self.object_class.
        
        Args:
            candidate_bbox (np.ndarray): The candidate bounding box [x1, y1, x2, y2].
            bboxes (np.ndarray): Array of existing bounding boxes (each as [x1, y1, x2, y2]).
            class_labels (np.ndarray): Array of class labels corresponding to each bbox.
        
        Returns:
            bool: True if no overlap with any bbox of class self.object_class; False otherwise.
        """
        for bbox, label in zip(bboxes, class_labels):
            if label == self.object_class:
                if self.check_overlap(candidate_bbox, bbox):
                    return False
            else:
                # Check if the iou of the pasted object with other class bboxes is too high.
                if self.get_occlude_ratio(candidate_bbox, bbox) > self.max_occlude_ratio:
                    return False
        return True

    def apply(self, image, bboxes, class_labels):
        """
        Paste object crops onto the image while avoiding overlap with existing
        or previously pasted objects of class self.object_class.

        Args:
            image (np.ndarray): The target (grayscale) image.
            bboxes (np.ndarray): Array of bounding boxes (each as [x1, y1, x2, y2]) in pixel coordinates.
            class_labels (np.ndarray): Array of class labels corresponding to each bbox.
            
        Returns:
            tuple: (image (np.ndarray), updated_bboxes (np.ndarray), updated_class_labels (np.ndarray))
        """
        assert bboxes.shape[1] == 4, "Bboxes must be in [x1, y1, x2, y2] format."
        h_img, w_img = image.shape[:2]
        pasted_bboxes = []
        pasted_labels = []

        # Randomly choose the number of objects to paste.
        n_objects = random.randint(1, self.max_paste_objects)
        for _ in range(n_objects):
            if not self.object_files:
                break  # No available objects.

            # Randomly select an object crop.
            obj_file = random.choice(self.object_files)
            obj_img = cv2.imread(obj_file, cv2.IMREAD_UNCHANGED)
            assert obj_img is not None, f"Failed to read {obj_file}"
            assert obj_img.shape[-1] == 4, "Object image must have 4 channels (BGR + alpha)."

            # Apply the augmentation pipeline to the cropped object.
            # Pass the new long-edge ratio parameters and derive target_long_edge from the target image.
            obj_img = augment_crop(
                obj_img,
                debug_crops=DEBUG_CROPS_CONFIG,
                min_long_edge_ratio=self.min_long_edge_ratio,
                max_long_edge_ratio=self.max_long_edge_ratio,
                target_long_edge=max(h_img, w_img)
            )

            if obj_img is None: # NOTE if the augmented crop is None, skip this object. Can happen if no alpha channel is found in the object (after cropping or before.)
                continue

            # NOTE Resize objects that are too large to ensure they fit within the target image 
            # while maintaining aspect ratio. The pasted object will be at most half of the target's smallest dimension.
            obj_h, obj_w = obj_img.shape[:2]
            min_edge_size = min(h_img, w_img) // 2  # Maximum allowed size for pasted objects.
            if obj_h > min_edge_size or obj_w > min_edge_size:
                scale = min_edge_size / max(obj_h, obj_w)
                new_h, new_w = int(obj_h * scale), int(obj_w * scale)
                obj_img = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                obj_h, obj_w = obj_img.shape[:2]

            # Attempt to find a valid paste location.
            valid_position_found = False
            candidate_bbox = None
            for _ in range(self.max_attempts):
                max_y = h_img - obj_h
                max_x = w_img - obj_w
                if max_y < 0 or max_x < 0:
                    raise Exception("Object is larger than target image.")
                y1 = random.randint(0, max_y)
                x1 = random.randint(0, max_x)
                candidate_bbox = np.array([x1, y1, x1 + obj_w, y1 + obj_h], dtype=np.float32)
                
                # Combine existing boxes and those already pasted.
                if bboxes.shape[0] > 0 and len(pasted_bboxes) > 0:
                    combined_bboxes = np.vstack((bboxes, np.array(pasted_bboxes, dtype=np.float32)))
                    combined_labels = np.concatenate((class_labels, np.array(pasted_labels, dtype=np.int32)))
                elif bboxes.shape[0] > 0:
                    combined_bboxes = bboxes
                    combined_labels = class_labels
                elif len(pasted_bboxes) > 0:
                    combined_bboxes = np.array(pasted_bboxes, dtype=np.float32)
                    combined_labels = np.array(pasted_labels, dtype=np.int32)
                else:
                    combined_bboxes = np.empty((0, 4), dtype=np.float32)
                    combined_labels = np.empty((0,), dtype=np.int32)
                # Convert candidate_bbox from pixel coordinates to normalized [0,1] coordinates.
                candidate_bbox_norm = candidate_bbox / np.array([w_img, h_img, w_img, h_img], dtype=np.float32)
                if self.is_valid_position(candidate_bbox_norm, combined_bboxes, combined_labels):
                    valid_position_found = True
                    break

            if not valid_position_found:
                continue  # Skip if no valid location found.

            # Convert the pasted object to grayscale while ensuring we use
            # any available alpha channel as a blending mask.
            if len(obj_img.shape) == 2:
                # Object image is already grayscale.
                gray_obj_img = obj_img
                alpha = np.ones((obj_h, obj_w), dtype=np.float32)
            else:
                assert obj_img.shape[2] == 4
                # Object image has an alpha channel.
                gray_obj_img = cv2.cvtColor(obj_img, cv2.COLOR_BGRA2GRAY)
                alpha = obj_img[:, :, 3].astype(np.float32) / 255.0

            if self.blend:
                # Smooth the alpha mask.
                alpha = gaussian(alpha, sigma=self.sigma, preserve_range=True)
                max_val = np.max(alpha)
                if max_val > 0:
                    alpha = alpha / max_val

            # Blend the object crop (grayscale) onto the grayscale image.
            x1, y1, x2, y2 = candidate_bbox.astype(int)
            roi = image[y1:y2, x1:x2].astype(np.float32)
            gray_obj_img = gray_obj_img.astype(np.float32)
            blended = gray_obj_img * alpha + roi * (1 - alpha)
            image[y1:y2, x1:x2] = blended.astype(image.dtype)


            assert np.all((candidate_bbox_norm >= 0) & (candidate_bbox_norm <= 1)), f"All bbox coordinates must be in range [0,1], got {candidate_bbox_norm}"
            pasted_bboxes.append(candidate_bbox_norm)
            pasted_labels.append(self.object_class)

        # Merge the original and pasted boxes.
        if len(pasted_bboxes) > 0:
            pasted_bboxes_arr = np.array(pasted_bboxes, dtype=np.float32)
            pasted_labels_arr = np.array(pasted_labels, dtype=np.int32)
            if bboxes.shape[0] > 0:
                updated_bboxes = np.vstack((bboxes, pasted_bboxes_arr))
                updated_class_labels = np.concatenate((class_labels, pasted_labels_arr))
            else:
                updated_bboxes = pasted_bboxes_arr
                updated_class_labels = pasted_labels_arr
        else:
            updated_bboxes = bboxes
            updated_class_labels = class_labels
        
        return image, updated_bboxes, updated_class_labels

    def __call__(self, image, bboxes=None, **kwargs):
        """
        Convert input image to grayscale and apply the selective copy-paste transform.
        The input bounding boxes are expected to be in [x1, y1, x2, y2] format, optionally with a class_id as a 5th element.
        Returns a dict with the transformed (grayscale) image and updated bounding boxes and class labels.

        NOTE we normally add a dummy box at the beginning of the bboxes and class_labels. But it is automatically removed in the preprocess function of albumentations.
        NOTE only works for grayscale images currently.
        
        Args:
            image (np.ndarray): The target image.
            bboxes (np.ndarray): Array of bounding boxes. Each bbox is expected in [x1, y1, x2, y2] (is converted to [x1, y1, x2, y2, class_id] internally) format.
            class_labels (list): Optional separate list of class labels.
        
        Returns:
            dict: Dictionary with keys 'image', 'bboxes', and 'class_labels'.
                  The returned bboxes are a numpy array in [x1, y1, x2, y2, class_id] format.
                  The returned box will be in yolo x,y,w,h format, as albumentation internally applies
                    self.postprocess(data) automatically converts the boxes to yolo format.
        """
        SelectiveCopyPaste.augmented_count += 1

        # Update the transformation probability (p) using linear decay if enabled.
        if copy_paste_config.get("linear_decay", False):
            max_epochs = copy_paste_config.get("max_epochs")
            total_samples = copy_paste_config.get("total_samples")
            assert max_epochs is not None, "max_epochs must be set in config for linear_decay"
            assert total_samples is not None, "total_samples must be set in config for linear_decay"
            current_iter = SelectiveCopyPaste.augmented_count
            max_iter = max_epochs * total_samples
            if current_iter >= max_iter:
                self.p = 0.0
            else:
                self.p = self.initial_p * (1 - current_iter / max_iter)
            self.p = max(self.p , 0.01) # NOTE never close the aug fully.
            if current_iter % 5000 == 0:
                print(f"Current iteration: {current_iter}, Current probability: {self.p}")

        # Convert the input image to grayscale if it is not already.

        assert image.shape[-1] == 3 or image.ndim == 2, "Input image must have 2 or 3 channels (BGR or grayscale)."
        class_labels_input = kwargs.get("class_labels", np.empty((0, 1), dtype=np.float32))
        assert bboxes.dtype == np.float32, "Bboxes must be a numpy array."
        assert class_labels_input.dtype == np.float32, "class_labels_input must be a numpy array."
        # NOTE remove the dummy box from the bboxes and class_labels here if necessary. It is the first box in bboxes and class_labels.
        is_dummy = bboxes.shape[0] > 0 and np.allclose(bboxes[0][0:4], np.array([[0.5, 0.5, 0.001, 0.001]]), atol=1e-2)
        assert not is_dummy, f"Dummy box found in final_aug_bboxes: {bboxes[0]}"
        
        # Allow a small tolerance for floating point errors.
        tol = 1e-6
        # Dont process if the probability is less than p.
        if max(random.random(), 0.0) > self.p:
            return {"image": image, "bboxes": np.concatenate((np.clip(bboxes[:, :4], 0+tol, 1-tol), bboxes[:, 4:5]), axis=1), "class_labels": class_labels_input}
        
        convert_back_to_bgr = False
        if image.shape[-1] == 3:
            convert_back_to_bgr = True
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if bboxes.shape[0] > 0:
            # Instead of using a strict assertion, check with a tolerance then clip.
            assert np.all((bboxes[:, :4] >= -tol) & (bboxes[:, :4] <= 1 + tol)), \
                f"All bbox coordinates must be in range [0,1] within tolerance {tol}, got {bboxes}"
            bboxes[:, :4] = np.clip(bboxes[:, :4], 0, 1)
        # Process bounding boxes and class labels using numpy arrays.
        if bboxes is not None and len(bboxes) > 0:
            assert len(bboxes[0]) == 5, "Bboxes must be in [x1, y1, x2, y2, class_id] format."
            bboxes_array = np.array(bboxes, dtype=np.float32)
            bboxes_coords = bboxes_array[:, :4]
            bboxes_labels = bboxes_array[:, 4]
        else:
            bboxes_labels = class_labels_input
            bboxes_coords = np.empty((0, 4), dtype=np.float32)

        image_aug, updated_bboxes, updated_class_labels = self.apply(
            image,
            bboxes_coords,
            class_labels=bboxes_labels,
        )
        if updated_bboxes.shape[0] > 0:
            merged = np.hstack((updated_bboxes, updated_class_labels.reshape(-1, 1)))
        else:
            merged = np.empty((0, 5), dtype=np.float32)
        
        if merged.shape[0] > 0:
            # Instead of using a strict assertion, check with a tolerance then clip.
            assert np.all((merged[:, :4] >= -tol) & (merged[:, :4] <= 1 + tol)), \
                f"All bbox coordinates must be in range [0,1] within tolerance {tol}, got {bboxes}"
            merged[:, :4] = np.clip(merged[:, :4], 0+tol, 1-tol) # Adjust safety margin to avoid floating point errors.
        if convert_back_to_bgr:
            image_aug = cv2.cvtColor(image_aug, cv2.COLOR_GRAY2BGR)
        return {"image": image_aug, "bboxes": merged, "class_labels": updated_class_labels}

    def get_transform_init_args_names(self):
        return ("folder", "max_paste_objects", "blend", "sigma", "max_attempts", "p")


copy_paste_hand_inhouse = A.Compose(
    [
        SelectiveCopyPaste(
            folder=hands_inhouse_config.get("folder"),
            max_paste_objects=hands_inhouse_config.get("max_paste_objects", 2),
            blend=hands_inhouse_config.get("blend", True),
            sigma=hands_inhouse_config.get("sigma", 2),
            max_attempts=hands_inhouse_config.get("max_attempts", 20),
            p=hands_inhouse_config.get("p", 0.3),
            class_id=hands_inhouse_config.get("class_id", 3),
            max_occlude_ratio=hands_inhouse_config.get("max_occlude_ratio", 0.5)
        )
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)

copy_paste_hand_public = A.Compose(
    [
        SelectiveCopyPaste(
            folder=hands_public_config.get("folder"),
            max_paste_objects=hands_public_config.get("max_paste_objects", 5),
            blend=hands_public_config.get("blend", True),
            sigma=hands_public_config.get("sigma", 2),
            max_attempts=hands_public_config.get("max_attempts", 20),
            p=hands_public_config.get("p", 0.6),
            class_id=hands_public_config.get("class_id", 3),
            max_occlude_ratio=hands_public_config.get("max_occlude_ratio", 0.5)
        )
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)

copy_paste_tag = A.Compose(
    [
        SelectiveCopyPaste(
            folder=tags_config.get("folder"),
            max_paste_objects=tags_config.get("max_paste_objects", 7),
            blend=tags_config.get("blend", True),
            sigma=tags_config.get("sigma", 2),
            max_attempts=tags_config.get("max_attempts", 20),
            p=tags_config.get("p", 0.9),
            class_id=tags_config.get("class_id", 0),
            max_occlude_ratio=tags_config.get("max_occlude_ratio", 0.5)
        )
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)



def apply_copy_paste_augmentations(image, processed_bboxes, class_labels):
    """
    Applies hand and tag copy-paste augmentations to the given image in a randomly determined order.
    
    The function converts the provided bounding boxes to a NumPy array (if not already),
    adds a dummy box and dummy class label as required by the augmentation pipelines,
    randomly shuffles the order of the augmentation pipelines, applies them sequentially,
    and finally removes the dummy box.
    
    Parameters:
        image (np.ndarray): The image to augment.
        processed_bboxes (list or np.ndarray): List of bounding boxes in YOLO format 
            (each as [x_center, y_center, width, height]).
        class_labels (list): List of class labels corresponding to each bounding box.
        
    Returns:
        final_aug_image (np.ndarray): The augmented image.
        final_aug_bboxes (np.ndarray): The augmented bounding boxes (dummy removed).
        final_aug_class_labels (list): The updated class labels with dummy removed.
        
    Raises:
        AssertionError: If the dummy box persists in the final augmented bounding boxes.
    """
    # Convert to numpy array for Albumentations compatibility.
    assert processed_bboxes.dtype == np.float32, "processed_bboxes must be a numpy array."
    assert class_labels.dtype == np.float32, "processed_bboxes must be a numpy array."

    # Add dummy box at center of image with small dimensions NOTE this is crucial.
    dummy_box = np.array([[0.5, 0.5, 0.0001, 0.0001]], dtype=np.float32)  # x_center, y_center, width, height
    processed_bboxes = np.vstack([dummy_box, processed_bboxes]) if len(processed_bboxes) > 0 else dummy_box
    dummy_class_label = -1
    class_labels = np.insert(class_labels, 0, dummy_class_label)  # Add dummy class label at the beginning.

    # Define augmentation pipelines.
    aug_pipelines = [
        copy_paste_hand_inhouse,
        copy_paste_hand_public,
        copy_paste_tag
    ]
    # Randomly shuffle the order of augmentations.
    random.shuffle(aug_pipelines)

    result = {
        "image": image,
        "bboxes": processed_bboxes,
        "class_labels": class_labels
    }
    # Apply each augmentation sequentially in randomized order.
    for aug_fn in aug_pipelines:
        result = aug_fn(
            image=result["image"],
            bboxes=result["bboxes"],
            class_labels=result["class_labels"],
        )

    final_aug_image = result["image"]
    final_aug_bboxes = result["bboxes"]
    final_aug_class_labels = result.get("class_labels", None)
    
    # Remove the dummy box which is always positioned first.
    final_aug_bboxes = final_aug_bboxes[1:]
    final_aug_class_labels = final_aug_class_labels[1:]
    
    # Assert that the dummy box is no longer present.
    if final_aug_bboxes.shape[0] > 0:
        is_dummy = np.allclose(final_aug_bboxes[0], dummy_box[0], atol=1e-2)
        if is_dummy:
            final_aug_bboxes = final_aug_bboxes[1:]
            final_aug_class_labels = final_aug_class_labels[1:]
    return final_aug_image, final_aug_bboxes, final_aug_class_labels


def plot_yolo_predictions(image, bboxes, class_ids, save_path=None, colors=None, thickness=2):
    """
    Plot YOLO predictions on an image.

    Args:
        image (np.ndarray): Input image in BGR format
        bboxes (np.ndarray): Array of bounding boxes in YOLO format [x_center, y_center, width, height]
        class_ids (np.ndarray): Array of class IDs for each bounding box
        save_path (str, optional): Path to save the output image. If None, won't save
        colors (list, optional): List of colors for different classes
        thickness (int, optional): Thickness of bounding box lines

    Returns:
        np.ndarray: Image with drawn predictions
    """
    if save_path and not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        
    # Default colors if not provided
    if colors is None:
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red
    
    # Make a copy of the image to draw on
    output_image = image.copy()
    h, w = image.shape[:2]
    
    # Draw each bounding box
    for i, bbox in enumerate(bboxes):
        x_center, y_center, box_w, box_h = bbox
        class_id = int(class_ids[i])
        
        # Convert from normalized YOLO format to pixel coordinates
        x_center *= w
        y_center *= h
        box_w *= w
        box_h *= h
        
        # Calculate top-left and bottom-right coordinates
        x1 = int(x_center - box_w/2)
        y1 = int(y_center - box_h/2)
        x2 = int(x_center + box_w/2)
        y2 = int(y_center + box_h/2)
        
        # Get color for this class (cycle through colors if more classes than colors)
        color = colors[class_id % len(colors)]
        
        # Draw rectangle
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)
        
        # Add class label
        label = f"Class {class_id}"
        cv2.putText(output_image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    # Save if path is provided
    if save_path:
        cv2.imwrite(save_path, output_image)
    
    return output_image

def clip_bboxes_safely(bboxes_normalized, tol=1e-6):
    """
    Inputs normalized bboxes in [0,1] YOLO format.
    Clip bounding boxes so that they are strictly within [0,1] in x1, y1, x2, y2 coordinate system with a safety margin `tol`.
    Returns the clipped bboxes in normalized [0,1] YOLO format.
    """
    # Convert YOLO format (x_center, y_center, w, h) to corner format (x1, y1, x2, y2)
    x_center = bboxes_normalized[:, 0]
    y_center = bboxes_normalized[:, 1]
    w_box = bboxes_normalized[:, 2]
    h_box = bboxes_normalized[:, 3]
    x1 = x_center - w_box / 2.0
    y1 = y_center - h_box / 2.0
    x2 = x_center + w_box / 2.0
    y2 = y_center + h_box / 2.0
    # Clip boxes so that they are strictly within [0,1] with a safety margin `tol`, update bboxes_normalized if you have to change x1, y1, x2, y2.
    assert np.all(x1 >= -tol) and np.all(y1 >= -tol) and np.all(x2 <= 1.0 + tol) and np.all(y2 <= 1.0 + tol), \
        f"Bounding boxes (in corner format) must be strictly within [0,1]: got x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}"

    x1 = np.clip(x1, 0 + tol, 1.0 - tol)
    y1 = np.clip(y1, 0 + tol, 1.0 - tol)
    x2 = np.clip(x2, 0 + tol, 1.0 - tol)
    y2 = np.clip(y2, 0 + tol, 1.0 - tol)
    
    # Update bboxes_normalized with the clipped values
    bboxes_normalized[:, 0] = (x1 + x2) / 2.0  # x_center
    bboxes_normalized[:, 1] = (y1 + y2) / 2.0  # y_center
    bboxes_normalized[:, 2] = x2 - x1          # width
    bboxes_normalized[:, 3] = y2 - y1          # height
    return bboxes_normalized


class CopyPasteUtku:
    def __init__(self):
        pass

    def __call__(self, labels):
        """
        Expecting data to be a dictionary with keys:
            "img": np.ndarray,
            "data["instances"].bboxes": np.ndarray (in YOLO format),
            "class_labels": np.ndarray

            # NOTE UTKU this is the overall idea:
            # label["cls"] gives the class id. (float32 numpy array)
            # label["instances"]["bboxes"] gives the bounding boxes. (float32 numpy array)
            # label["instances"]["bbox_areas"] gives the bounding bbox areas. (float32 numpy array)
            # label["img"] gives the image. (uint8 numpy array)

        """
        image = labels["img"]
        bboxes = labels["instances"].bboxes
        class_labels = labels["cls"]

        # Ensure that bboxes and class_labels are in the expected dtype.
        assert isinstance(bboxes, np.ndarray) and bboxes.dtype == np.float32, "Bboxes must be a np.ndarray with dtype float32"
        assert isinstance(class_labels, np.ndarray) and class_labels.dtype == np.float32, "Class labels must be a np.ndarray with dtype float32"
        

        # Normalize bboxes coordinates from pixel values (x_center, y_center, width, height)
        # to normalized [0,1] YOLO format using the image dimensions.
        h_img, w_img = image.shape[:2]
        bboxes_normalized = bboxes.copy()
        if bboxes.shape[0] > 0:
            bboxes_normalized[:, [0, 2]] /= w_img
            bboxes_normalized[:, [1, 3]] /= h_img
            # Clip boxes within [0, 1] range to account for small floating point errors.
            # assert the error margin is small.
            assert np.all((bboxes_normalized[:, :4] >= 0) & (bboxes_normalized[:, :4] <= 1)), \
                f"All bbox coordinates must be in range [0,1]"
            bboxes_normalized = clip_bboxes_safely(bboxes_normalized)

        image, new_bboxes, class_labels = apply_copy_paste_augmentations(
            image,
            bboxes_normalized,
            class_labels,
        )

        new_bboxes_denorm = new_bboxes.copy()
        # Denormalize new_bboxes coordinates from normalized [0,1] YOLO format back to pixel coordinates.
        if new_bboxes.shape[0] > 0:
            new_bboxes_denorm[:, [0, 2]] *= w_img
            new_bboxes_denorm[:, [1, 3]] *= h_img

        labels["img"] = image
        labels["cls"] = class_labels
        labels["instances"].update(bboxes=new_bboxes_denorm)  # update() will automatically update bbox_areas.

        # NOTE UTKU: if you want to visualize the result, set VISUALIZATION_PATH in the configuration.
        if VISUALIZATION_PATH:
            import time
            plot_yolo_predictions(
                labels["img"],
                labels["instances"].bboxes,
                labels["cls"],
                save_path=VISUALIZATION_PATH + "/" + str(time.time()) + ".png"
            )
        return labels