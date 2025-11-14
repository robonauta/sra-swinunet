# Swin-UNet with Spatial Reduction Attention (SRA)

This project is an implementation of the Swin-UNet architecture, modified to incorporate Spatial Reduction Attention (SRA) for enhanced computational efficiency and performance in semantic segmentation tasks.

The framework is designed for benchmarking and experimentation, with support for multiple public datasets including Manga109, ACDC, BraTS, DRIVE, Atrial, and ADE20K.

This repository is based on the original Swin-Unet implementation by HuCaoFighting: https://github.com/HuCaoFighting/Swin-Unet.

## 1. Setup and Installation

### Dependencies
This project requires Python 3 and several libraries. You can install the main dependencies using pip:


```bash
pip install -r requirements.txt
```

### Pretrained Model

For a "pretrained" run, the model expects the original Swin-Transformer (Swin-T) weights.

**Download**: You can download the required pretrained model file (`swin_tiny_patch4_window7_224.pth`) from here: https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing

**Location**: Create a folder named `pretrained` in the root of your project and place the downloaded `.pth` file inside it.

The project scripts are configured to look for the weights at this path: `pretrained/swin_tiny_patch4_window7_224.pth`.

## 2. Data Setup
The project expects a specific directory structure for the datasets. All datasets should be placed within a root `datasets/` folder.

```
sra-swinunet/
├── datasets/
│   ├── acdc/
│   │   ├── training/
│   │   ├── validation/
│   │   └── testing/
│   ├── atrial/
│   │   ├── images/
│   │   │   ├── training/
│   │   │   ├── validation/
│   │   │   └── testing/
│   │   └── annotations/
│   │       ├── training/
│   │       ├── validation/
│   │       └── testing/
│   ├── brats/
│   │   ├── training/
│   │   ├── validation/
│   │   └── testing/
│   ├── drive/
│   │   ├── images/
│   │   └── annotations/
│   └── ade20k/
│       ├── images/
│       └── annotations/
├── pretrained/
│   └── swin_tiny_patch4_window7_224.pth
└── driver_sra.py
```

### Dataset Formats

- **ACDC**: Uses `SwinUNetVolumetricH5Dataset`. Expects `.h5` files, with each file containing a 3D volume. Each slice is loaded as a separate 2D image.
- **ADE20K**: Uses `ADE20KSwinUNet`. Expects standard 2D image files. 
- **Atrial**: Uses `SwinUNetNiiDataset`. Expects `.nii` or `.nii.gz` files containing 3D volumes.
- **BraTS**: Uses `SwinUNetH5Dataset`. Expects `.h5` files, with each file containing a 3-channel image and a 3-channel mask.
- **DRIVE**: Uses `SwinUNetDataset`. Expects standard 2D image files (e.g., `.png`, `.jpg`) in the `images` and `annotations` folders.
- **Manga109**: Uses `SwinUNetDataset`. Expects standard 2D image files.


## 3. Running Experiments
The main entry point for launching any training is the `driver_sra.py` script. This script orchestrates which dataset to run, which model configuration to use, and which GPU to use.

**Example Usage**
```Bash
# Run the 'sra' experiment on the 'brats' dataset using devices cuda:0 and cuda:1
python driver_sra.py --model brats --exp sra --devices "[0, 1]"

# Run the '75%' model experiment on the 'atrial' dataset using only cuda:0
python driver_sra.py --model atrial --exp 75 --devices "[0]"
```


**Driver Arguments**

The `driver_sra.py` script accepts the following arguments:

- `--model` or `-m`: (Required) The name of the dataset to run. This name must match the suffix of the corresponding training script.
    - Examples: brats, acdc, drive, atrial, ade20k, manga109.

- `--exp` or `-e`: The type of experiment to run. This controls which model configurations are tested.
    - `sra`: Compares the regular SwinUNet with the SRA-modified version.
    - `sra_75`: Compares the regular SwinUNet with a 75% size SRA-modified version.
    - `75`: Compares the regular SwinUNet with a 75% size regular version.

- `--devices` or `-d`: A list of GPU device IDs to use, formatted as a Python list string. The script will run experiments in parallel across these devices.
    - Example: `"[0]"`, `"[0, 1]"`, `"[2, 3]"`.

**Experiment Configuration**

The `driver_sra.py` script automatically passes the correct model configurations (dimensions, depths, SRA ratios) to the corresponding `train_test_... script`.

Hyperparameters like Learning Rate (LR), Batch Size, and Image Size are configured within each specific `train_test_*.py` file. For example, to change the batch size for BraTS, you would edit `train_test_swinunet_sra_brats.py`.

## 4. Project Structure Overview

- `driver_sra.py`: Main entry point for launching experiments.

- `train_test_swinunet_sra_[dataset].py`: Individual experiment scripts for each dataset (e.g., `..._brats.py`, `..._acdc.py`). These files define data loading, hyperparameters, and model configuration.

- `train.py`: Contains the Trainer class, which handles the complete training and validation loop, logging, and model checkpointing.

- `test.py`: Contains the Tester class, which runs the final evaluation on the test set and saves the results CSV.

- `metrics.py`: Contains the Metrics class responsible for calculating and aggregating all performance metrics (Loss, Accuracy, Precision, Recall, Dice, and F1-score, in both macro and micro variants).

- `datasets.py`: Defines all the PyTorch Dataset classes (e.g., `SwinUNetH5Dataset`, `SwinUNetNiiDataset`) for loading and preprocessing the various data formats.

- `utils.py`: Contains helper functions, including the `EarlyStopper` class, image resizing (`resize_max_edge`), and padding (`pad_if_needed`).

- `SwinUnet/`:  This directory should contain the model definitions, including the base SwinUNet and the modified SwinUnetSRA.