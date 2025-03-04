import os
import random
import cv2
import numpy as np
from skimage.filters import gaussian
import albumentations as A
from cpp_utils.crop_augmentations import augment_crop
# visualization_path: /home/utkutopcuoglu/Projects/ebis/copy-paste-aug/visualization # (str) path to save the visualization images

class SelectiveCopyPaste(A.DualTransform):
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
        obj_size_scale=1
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
        self.folder = folder
        self.max_paste_objects = max_paste_objects
        self.blend = blend
        self.sigma = sigma
        self.max_attempts = max_attempts
        self.obj_size_scale = obj_size_scale
        # Preload list of valid image files.
        self.object_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.object_class = class_id

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
        return True

    def apply(self, image, bboxes, class_labels, resized_shape=None, ori_shape=None):
        """
        Paste object crops onto the image while avoiding overlap with existing
        or previously pasted objects of class self.object_class.

        Args:
            image (np.ndarray): The target (grayscale) image.
            bboxes (np.ndarray): Array of bounding boxes (each as [x1, y1, x2, y2]) in pixel coordinates.
            class_labels (np.ndarray): Array of class labels corresponding to each bbox.
            resized_shape (tuple, optional): The shape (height, width) of the resized template image.
            ori_shape (tuple, optional): The original shape (height, width) of the template image.
            
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
            assert obj_img is not None
            assert obj_img.shape[-1] == 4, "Object image must have 4 channels (BGR + alpha)."

            # Immediately after reading, resize the object if the template image was resized.
            if resized_shape is not None and ori_shape is not None:
                assert isinstance(resized_shape, (tuple, list)) and len(resized_shape) == 2, \
                    "resized_shape must be a tuple of (height, width)"
                assert isinstance(ori_shape, (tuple, list)) and len(ori_shape) == 2, \
                    "ori_shape must be a tuple of (height, width)"
                ori_h, ori_w = ori_shape
                resized_h, resized_w = resized_shape
                scale_w = resized_w / ori_w
                scale_h = resized_h / ori_h
                new_obj_w = int(obj_img.shape[1] * scale_w * self.obj_size_scale)
                new_obj_h = int(obj_img.shape[0] * scale_h * self.obj_size_scale)
                obj_img = cv2.resize(obj_img, (new_obj_w, new_obj_h), interpolation=cv2.INTER_LINEAR)

            # Apply the augmentation pipeline to the cropped object.
            obj_img = augment_crop(obj_img)

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
        Convert input image to grayscale and then apply the selective copy-paste transform.
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
        # Convert the input image to grayscale if it is not already.

        assert image.shape[-1] == 3 or image.ndim == 2, "Input image must have 2 or 3 channels (BGR or grayscale)."
        class_labels_input = kwargs.get("class_labels", np.empty((0, 1), dtype=np.float32))
        assert bboxes.dtype == np.float32, "Bboxes must be a numpy array."
        assert class_labels_input.dtype == np.float32, "class_labels_input must be a numpy array."
        # NOTE remove the dummy box from the bboxes and class_labels here if necessary. It is the first box in bboxes and class_labels.
        is_dummy = bboxes.shape[0] > 0 and np.allclose(bboxes[0][0:4], np.array([[0.5, 0.5, 0.001, 0.001]]), atol=1e-2)
        assert not is_dummy, f"Dummy box found in final_aug_bboxes: {bboxes[0]}"
        
        # Dont process if the probability is less than p.
        if max(random.random(), 0.0) > self.p:
            return {"image": image, "bboxes": bboxes, "class_labels": class_labels_input}
        
        convert_back_to_bgr = False
        if image.shape[-1] == 3:
            convert_back_to_bgr = True
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Allow a small tolerance for floating point errors.
        tol = 1e-6
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
            resized_shape=kwargs.get("resized_shape"),
            ori_shape=kwargs.get("ori_shape")
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
            folder="/home/utkutopcuoglu/Projects/ebis/copy-paste-aug/hands_inhouse",
            max_paste_objects=2,
            blend=True,
            sigma=2, # The size of the gaussian kernel for blending. The larger the more smooth the blending, the more transparent the pasted object.
            max_attempts=20,
            p=0.3,
            class_id = 3,
            obj_size_scale=1.7
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
            class_id = 3,
            obj_size_scale=1.7
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
            class_id = 1,
            obj_size_scale=1
        )
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)



def apply_copy_paste_augmentations(image, processed_bboxes, class_labels, resized_shape=None, ori_shape=None):
    """
    Applies hand and tag copy-paste augmentations to the given image.
    
    The function converts the provided bounding boxes to a NumPy array (if not already),
    adds a dummy box and dummy class label as required by the augmentation pipelines,
    applies the hand and tag transformations sequentially, and finally removes the dummy box.
    
    Parameters:
        image (np.ndarray): The image to augment.
        processed_bboxes (list or np.ndarray): List of bounding boxes in YOLO format 
            (each as [x_center, y_center, width, height]).
        class_labels (list): List of class labels corresponding to each bounding box.
        copy_paste_hand (albumentations.Compose): The copy-paste augmentation for hands.
        copy_paste_tag (albumentations.Compose): The copy-paste augmentation for tags.
        
    Returns:
        final_aug_image (np.ndarray): The augmented image.
        final_aug_bboxes (np.ndarray): The augmented bounding boxes (dummy removed).
        final_aug_class_labels (list): The updated class labels with dummy removed.
        
    Raises:
        AssertionError: If the dummy box persists in the final augmented bounding boxes.
    """

    # Convert to numpy array for Albumentations compatibility
    assert processed_bboxes.dtype == np.float32, "processed_bboxes must be a numpy array."
    assert class_labels.dtype == np.float32, "processed_bboxes must be a numpy array."

    # Add dummy box at center of image with small dimensions NOTE this is crucial.
    dummy_box = np.array([[0.5, 0.5, 0.0001, 0.0001]], dtype=np.float32)  # x_center, y_center, width, height
    processed_bboxes = np.vstack([dummy_box, processed_bboxes]) if len(processed_bboxes) > 0 else dummy_box
    class_labels = np.insert(class_labels, 0, 0)  # Add dummy class label at the beginning. Use np.insert.

    
    # 1 Apply hand augmentation inhouse.
    hand_augmented_inhouse = copy_paste_hand_inhouse(
        image=image, 
        bboxes=processed_bboxes, 
        class_labels=class_labels,
        resized_shape=resized_shape,
        ori_shape=ori_shape
    )
    hand_aug_image_inhouse = hand_augmented_inhouse["image"]
    hand_aug_bboxes_inhouse = hand_augmented_inhouse["bboxes"]
    hand_aug_class_labels_inhouse = hand_augmented_inhouse.get("class_labels", None)

    # 2 Apply hand augmentation public.
    hand_augmented_public = copy_paste_hand_public(
        image=hand_aug_image_inhouse, 
        bboxes=hand_aug_bboxes_inhouse, 
        class_labels=hand_aug_class_labels_inhouse,
        resized_shape=resized_shape,
        ori_shape=ori_shape
    )

    hand_aug_image_public = hand_augmented_public["image"]
    hand_aug_bboxes_public = hand_augmented_public["bboxes"]
    hand_aug_class_labels_public = hand_augmented_public.get("class_labels", None)
    
    # 3 Apply tag augmentation on the result from hand augmentation.
    tag_augmented = copy_paste_tag(
        image=hand_aug_image_public, 
        bboxes=hand_aug_bboxes_public, 
        class_labels=hand_aug_class_labels_public,
        resized_shape=resized_shape,
        ori_shape=ori_shape
    )
    
    final_aug_image = tag_augmented["image"]
    final_aug_bboxes = tag_augmented["bboxes"]
    # Remove the dummy box which is always positioned first.
    final_aug_bboxes = final_aug_bboxes[1:]
    
    final_aug_class_labels = tag_augmented.get("class_labels", None)
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