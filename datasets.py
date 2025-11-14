import os

import h5py
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image

from utils import fix_size_swinunet, next_divisible


def pad(img, patch_size):
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


class ADE20KSwinUNet(Dataset):
    def __init__(
        self,
        label_dir,
        img_dir,
        transform=None,
        target_transform=None,
        seed=13,
        train_size=1,
    ):
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        self.image_names = np.array(sorted(os.listdir(img_dir)))
        self.label_names = np.array(sorted(os.listdir(label_dir)))

        np.random.seed(seed)

        if train_size < 1:
            chosen_indices = np.random.choice(
                len(self.image_names),
                size=int(len(self.image_names) * train_size),
                replace=False,
            )

            self.image_names = self.image_names[chosen_indices]
            self.label_names = self.label_names[chosen_indices]

        assert len(self.image_names) == len(self.label_names), "missing some labels"

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_names[idx])
        label_path = os.path.join(self.label_dir, self.label_names[idx])

        image = read_image(img_path, mode=ImageReadMode.RGB)
        original_label = read_image(label_path)
        self.original_img_size = image.size()

        image = self.transform(image.type(torch.float32))  # type: ignore

        self.padded_img_size = image.size()

        label = self.target_transform(original_label.type(torch.float32))  # type: ignore
        mask = (label != 0).float()
        label = label - 1
        label[label < 0] = 0

        return {"image": image, "label": label, "path": img_path, "mask": mask}


class SwinUNetDataset(Dataset):
    def __init__(
        self,
        label_dir,
        img_dir,
        transform=None,
        target_transform=None,
        seed=13,
        train_size=1,
    ):
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        self.image_names = np.array(sorted(os.listdir(img_dir)))
        self.label_names = np.array(sorted(os.listdir(label_dir)))

        np.random.seed(seed)

        if train_size < 1:
            chosen_indices = np.random.choice(
                len(self.image_names),
                size=int(len(self.image_names) * train_size),
                replace=False,
            )

            self.image_names = self.image_names[chosen_indices]
            self.label_names = self.label_names[chosen_indices]

        assert len(self.image_names) == len(self.label_names), "missing some labels"

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_names[idx])
        label_path = os.path.join(self.label_dir, self.label_names[idx])

        image = read_image(img_path, mode=ImageReadMode.RGB)
        label = read_image(label_path)

        if label.shape[0] >= 1:
            label = label[0].unsqueeze(dim=0)

        self.original_img_size = image.size()

        image = self.transform(image.type(torch.float32))  # type: ignore

        self.padded_img_size = image.size()

        label = self.target_transform(label.type(torch.float32))  # type: ignore

        return {
            "image": image,
            "label": label,
            "path": img_path,
        }


class SwinUNetNiiDataset(Dataset):
    def __init__(
        self,
        label_dir,
        img_dir,
        transform=None,
        target_transform=None,
        seed=13,
        train_size=1,
    ):
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        self.image_names = np.array(sorted(os.listdir(img_dir)))
        self.label_names = np.array(sorted(os.listdir(label_dir)))

        np.random.seed(seed)

        if train_size < 1:
            chosen_indices = np.random.choice(
                len(self.image_names),
                size=int(len(self.image_names) * train_size),
                replace=False,
            )

            self.image_names = self.image_names[chosen_indices]
            self.label_names = self.label_names[chosen_indices]

        self.index_map = []
        for i, img_name in enumerate(self.image_names):
            img_path = os.path.join(self.img_dir, img_name)
            img_nii = nib.load(img_path)
            n_slices = img_nii.shape[2]
            del img_nii
            for s in range(n_slices):
                self.index_map.append((i, s))

        assert len(self.image_names) == len(self.label_names), "missing some labels"

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, slice_idx = self.index_map[idx]
        img_path = os.path.join(self.img_dir, self.image_names[file_idx])
        label_path = os.path.join(self.label_dir, self.label_names[file_idx])

        # Load the .nii files
        img_nii = nib.load(img_path)
        label_nii = nib.load(label_path)

        image = np.array(img_nii.get_fdata())[:, :, slice_idx]  # shape: (H, W, N)
        label = np.array(label_nii.get_fdata())[:, :, slice_idx]  # shape: (H, W, N)

        image = torch.from_numpy(image).unsqueeze(dim=0)
        label = torch.from_numpy(label).unsqueeze(dim=0)

        self.original_img_size = image.size()

        image = self.transform(image.type(torch.float32))  # type: ignore

        self.padded_img_size = image.size()

        label = self.target_transform(label.type(torch.float32))  # type: ignore

        return {
            "image": image,
            "label": label,
            "path": img_path,
            "slice_idx": slice_idx,
        }


class SwinUNetVolumetricH5Dataset(Dataset):
    def __init__(
        self,
        example_dir,
        transform=None,
        target_transform=None,
        seed=13,
        train_size=1,
        channel_first=True,
        label_key="label",
    ):
        self.example_dir = example_dir
        self.transform = transform
        self.target_transform = target_transform
        self.channel_first = channel_first
        self.label_key = label_key
        self.example_names = np.array(sorted(os.listdir(example_dir)))

        np.random.seed(seed)

        if train_size < 1:
            chosen_indices = np.random.choice(
                len(self.example_names),
                size=int(len(self.example_names) * train_size),
                replace=False,
            )

            self.example_names = self.example_names[chosen_indices]

        self.index_map = []
        for i, example_name in enumerate(self.example_names):
            example_path = os.path.join(self.example_dir, example_name)
            img_h5 = h5py.File(example_path, "r")
            if self.channel_first:
                n_slices = img_h5["image"].shape[0]
            else:
                n_slices = img_h5["image"].shape[2]
            del img_h5
            for s in range(n_slices):
                self.index_map.append((i, s))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, slice_idx = self.index_map[idx]
        example_path = os.path.join(self.example_dir, self.example_names[file_idx])

        # Load the .nii files
        example = h5py.File(example_path, "r")
        if self.channel_first:
            image = np.array(example["image"])[slice_idx, :, :]
            label = np.array(example[self.label_key])[slice_idx, :, :]
        else:
            image = np.array(example["image"])[:, :, slice_idx]
            label = np.array(example[self.label_key])[:, :, slice_idx]

        image = torch.from_numpy(image).unsqueeze(dim=0)
        label = torch.from_numpy(label).unsqueeze(dim=0)

        self.original_img_size = image.size()

        image = self.transform(image.type(torch.float32))  # type: ignore

        self.padded_img_size = image.size()

        label = self.target_transform(label.type(torch.float32))  # type: ignore

        return {
            "image": image,
            "label": label,
            "path": example_path,
            "slice_idx": slice_idx,  # Use a placeholder for non-volumetric data
        }


class SwinUNetH5Dataset(Dataset):
    def __init__(
        self,
        example_dir,
        transform=None,
        target_transform=None,
        seed=13,
        train_size=1,
        channel_first=True,
        label_key="label",
    ):
        self.example_dir = example_dir
        self.transform = transform
        self.target_transform = target_transform
        self.channel_first = channel_first
        self.label_key = label_key
        self.example_names = np.array(sorted(os.listdir(example_dir)))

        np.random.seed(seed)

        if train_size < 1:
            chosen_indices = np.random.choice(
                len(self.example_names),
                size=int(len(self.example_names) * train_size),
                replace=False,
            )

            self.example_names = self.example_names[chosen_indices]

    def __len__(self):
        return len(self.example_names)

    def __getitem__(self, idx):
        example_path = os.path.join(self.example_dir, self.example_names[idx])

        # Load the .nii files
        example = h5py.File(example_path, "r")

        image = np.array(example["image"])[:, :, :3]
        label = np.array(example[self.label_key])

        image = torch.from_numpy(image).permute(2, 0, 1)
        label = torch.from_numpy(label).permute(2, 0, 1)

        self.original_img_size = image.size()

        image = self.transform(image.type(torch.float32))  # type: ignore

        self.padded_img_size = image.size()

        label = self.target_transform(label.type(torch.float32))  # type: ignore

        return {"image": image, "label": label, "path": example_path}
