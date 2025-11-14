import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode, v2


class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def fix_size(img, depth):
    """
    Converts image sizes to the next number divisible by 2^depth. This is to avoid
    the problems occured when scaling the image down and then up.

    Parameters:
        img: image read
        depth: depth of the U-Net model

    Returns:
        image with the size fixed.
    """
    den = 2**depth
    shape = np.array(img.size()[1:])
    new_shape = den * np.ceil(shape / den).astype(np.int32)
    p3d = (0, new_shape[1] - shape[1], 0, new_shape[0] - shape[0], 0, 0)
    if new_shape[0] != shape[0] or new_shape[1] != shape[1]:
        return F.pad(img, p3d, "constant", 0)
    return img


def next_multiple(y):
    lcm = math.lcm(32, 7)  # Least common multiple of 32 and 7
    return ((y // lcm) + 1) * lcm


def fix_size_swinunet(img):
    """
    Converts image sizes to the next number divisible by 7.

    Parameters:
        img: image read
        depth: depth of the U-Net model

    Returns:
        image with the size fixed.
    """
    den = math.lcm(32, 7)  # Least common multiple of 32 and 7
    shape = np.array(img.size()[1:])
    new_shape = den * np.ceil(shape / den).astype(np.int32)
    biggest_side = max(new_shape)
    new_shape = np.array([biggest_side, biggest_side])
    p3d = (0, new_shape[1] - shape[1], 0, new_shape[0] - shape[0], 0, 0)
    if new_shape[0] != shape[0] or new_shape[1] != shape[1]:
        return F.pad(img, p3d, "constant", 0)
    return img


def fix_size_patched_cropped(img):
    """
    Converts image sizes to the next number divisible by patch_size. This is to avoid
    the problems occured when spliting the image.

    Parameters:
        img: image read
        patch_size: patch_size

    Returns:
        image with the size fixed.
    """

    cropped_img = img[:, :512, :512]

    shape = np.array(cropped_img.size()[1:])

    p3d = (0, 512 - shape[1], 0, 512 - shape[0], 0, 0)

    return F.pad(cropped_img, p3d, "constant", 0)


def fix_size_patched(img, patch_size):
    """
    Converts image sizes to the next number divisible by patch_size. This is to avoid
    the problems occured when spliting the image.

    Parameters:
        img: image read
        patch_size: patch_size

    Returns:
        image with the size fixed.
    """

    shape = np.array(img.size()[1:])
    new_shape = np.array(
        [next_divisible(shape[0], patch_size), next_divisible(shape[1], patch_size)]
    )
    p3d = (0, new_shape[1] - shape[1], 0, new_shape[0] - shape[0], 0, 0)

    return F.pad(img, p3d, "constant", 0)


def next_divisible(x, y):
    # If x is already divisible by y, return x
    if x % y == 0:
        return x
    # Otherwise, find the next number greater than x that is divisible by y
    return x + (y - x % y)


def to_ohe(x, N_CLASSES):
    y = torch.squeeze(torch.nn.functional.one_hot(x.long(), num_classes=N_CLASSES), 0)
    return y


def generate_random_palette(num_classes=150, labels2print=None, palette_path=""):
    # Generate random RGB values for each class
    palette = torch.randint(0, 256, (num_classes, 3), dtype=torch.int)

    palette_w = 512
    palette_h = 12000
    class_height = palette_h // num_classes

    legend = np.zeros((palette_h, palette_w, 3), dtype=np.uint8)

    if labels2print is not None:
        for i in range(num_classes):
            start_y = i * class_height
            end_y = (i + 1) * class_height
            legend[start_y:end_y, :, :] = palette[i].cpu().numpy().astype(np.uint8)
            # Add the class label as text on the legend
            label_text = labels2print[i]
            cv2.putText(
                legend,
                # label_text, (start_x + 5, palette_h - 10),
                label_text,
                (10, start_y + 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        cv2.imwrite(f"{palette_path}", legend)

    return palette


def resize_max_edge(max_edge_size, is_label=False):
    def _resize_max_edge(image):
        c, h, w = image.size()
        if w > h:
            new_width = max_edge_size
            new_height = int(max_edge_size * h / w)
        else:
            new_height = max_edge_size
            new_width = int(max_edge_size * w / h)

        return v2.Resize(
            (new_height, new_width), interpolation=InterpolationMode.NEAREST_EXACT
        )(image)

    return _resize_max_edge


def pad_if_needed(min_height, min_width):
    def _pad_if_needed(image):
        c, h, w = image.size()

        assert h == min_height or w == min_width

        diff_w = 0
        diff_h = 0

        if h < min_height:
            diff_h = min_height - h
        if w < min_width:
            diff_w = min_width - w
        # left, top, right and bottom
        return v2.Pad((0, 0, diff_w, diff_h))(image)

    return _pad_if_needed


def convert_gt_to_label_map(gt_3ch: torch.Tensor) -> torch.Tensor:
    """
    Converts a 3-channel binary ground-truth tensor to a 2D label map.

    Args:
        gt_3ch (torch.Tensor): Tensor of shape (H, W, 3) or (3, H, W), with binary values (0 or 1).

    Returns:
        torch.Tensor: Tensor of shape (H, W) with values in {0, 1, 2, 3}.
    """
    if gt_3ch.dim() != 3:
        raise ValueError("Expected 3D tensor with shape (H, W, 3) or (3, H, W)")

    if gt_3ch.shape[0] == 3:
        # Convert from (3, H, W) to (H, W, 3)
        gt_3ch = gt_3ch.permute(1, 2, 0)

    label_map = torch.zeros(gt_3ch.shape[:2], dtype=torch.uint8, device=gt_3ch.device)

    # Class 1 → channel 0
    label_map[gt_3ch[:, :, 0] == 1] = 1

    # Class 2 → channel 1
    label_map[gt_3ch[:, :, 1] == 1] = 2

    # Class 3 → channel 2
    label_map[gt_3ch[:, :, 2] == 1] = 3

    return label_map
