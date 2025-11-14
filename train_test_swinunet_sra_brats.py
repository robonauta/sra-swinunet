import argparse
import json
import os
import uuid

import ml_collections
import torch
from monai.losses import GeneralizedDiceLoss
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from datasets import SwinUNetH5Dataset
from SwinUnet.config import get_config
from SwinUnet.networks.vision_transformer import SwinUnetSRA as ViT_seg
from test import Tester
from train import Trainer
from utils import EarlyStopper, convert_gt_to_label_map, pad_if_needed, resize_max_edge


def main():
    parser = argparse.ArgumentParser(description="Train/Test driver")
    parser.add_argument(
        "--message",
        "-m",
        type=str,
        help="A description of this experiment.",
        default="",
    )
    parser.add_argument("--train_size", "-ts", type=str, help="Train size", default=1.0)
    parser.add_argument("--mode", "-md", type=str, help="Mode", default="regular")
    parser.add_argument("--device", "-d", type=str, help="Device", default="cuda:0")
    parser.add_argument("--runs_set", "-rs", type=str, help="Device")
    parser.add_argument("--dims", "-dims", type=str, help="Model dimension", default=96)
    parser.add_argument(
        "--depths",
        "-depths",
        type=str,
        help="Model encoder depths",
        default="[2, 2, 2, 2]",
    )
    parser.add_argument(
        "--model_size", "-ms", type=str, help="SRA or not", default="regular"
    )
    parser.add_argument(
        "--sra-ratios", "-sr", type=str, help="SRA ratios", default="[1, 1, 1, 1]"
    )
    parser.add_argument("--exp", "-e", type=str, help="Experiment type", default="sra")
    args = parser.parse_args()

    # Variables & Setup
    RUNS_SET = args.runs_set
    EPOCHS = 300
    BATCH_SIZE = 16
    SEED = 13
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 1e-6
    MODEL_NAME = uuid.uuid4().hex
    MODEL_NAME = f"swinunet_{args.exp}_brats"
    TRAIN_SIZE = float(args.train_size)
    RUN_NAME = f"{TRAIN_SIZE:.2f}_{args.mode}"
    ROOT_PATH = f"{MODEL_NAME}/{RUNS_SET}/{args.model_size}/{RUN_NAME}"
    LR = 1e-5
    IMG_SIZE = 448
    N_CLASSES = 4
    DATASET_PERC = 1

    os.makedirs(f"{ROOT_PATH}", exist_ok=True)

    if args.message is not None:
        with open(f"{ROOT_PATH}/description.txt", "w") as f:
            f.write(args.message)

    params = {}

    print(RUN_NAME)

    torch.manual_seed(SEED)

    device = torch.device(args.device)

    torch.cuda.empty_cache()

    # Data

    transforms = v2.Compose(
        [
            resize_max_edge(IMG_SIZE),
            pad_if_needed(IMG_SIZE, IMG_SIZE),
        ]
    )

    target_transforms = v2.Compose(
        [
            resize_max_edge(IMG_SIZE, is_label=True),
            pad_if_needed(IMG_SIZE, IMG_SIZE),
            lambda x: convert_gt_to_label_map(x),
            lambda x: x.unsqueeze(dim=0),
            lambda x: x.long(),
        ]
    )

    train_ds = None
    train_dataloader = None

    if TRAIN_SIZE > 0:
        train_ds = SwinUNetH5Dataset(
            "datasets/brats/training/",
            transforms,
            target_transforms,
            seed=13,
            train_size=DATASET_PERC,
            channel_first=False,
            label_key="mask",
        )

        train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        params["train_size"] = len(train_ds)

    val_ds = SwinUNetH5Dataset(
        "datasets/brats/validation/",
        transforms,
        target_transforms,
        seed=13,
        train_size=DATASET_PERC,
        channel_first=False,
        label_key="mask",
    )
    test_ds = SwinUNetH5Dataset(
        "datasets/brats/testing/",
        transforms,
        target_transforms,
        seed=13,
        train_size=DATASET_PERC,
        channel_first=False,
        label_key="mask",
    )

    params["img_size"] = IMG_SIZE

    val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

    params["val_size"] = len(val_ds)
    params["test_size"] = len(test_ds)

    del train_ds
    del val_ds
    del test_ds

    # Model
    config = ml_collections.ConfigDict()
    config.num_classes = N_CLASSES
    config.n_class = N_CLASSES
    config.base_lr = LR
    config.img_size = IMG_SIZE
    config.seed = SEED
    config.cfg = "SwinUnet/configs/swin_tiny_patch4_window7_224_lite.yaml"
    config.opts = None
    config.batch_size = BATCH_SIZE
    config.zip = None
    config.cache_mode = None
    config.resume = None
    config.accumulation_steps = None
    config.use_checkpoint = None
    config.amp_opt_level = None
    config.tag = None
    config.eval = None
    config.throughput = None
    config.root_path = "pretrained/swin_tiny_patch4_window7_224.pth"
    config.use_checkpoint = True
    config.depths = eval(args.depths)
    config.embed_dim = int(args.dims)
    config.num_heads = [3, 6, 12, 24]
    config.sr_ratios = eval(args.sra_ratios)
    config = get_config(config)

    model = ViT_seg(config, img_size=IMG_SIZE, num_classes=N_CLASSES)

    if args.mode == "pretrained":
        model.load_from(config)

    model.to(device)

    # loss_fn = GeneralizedDiceLoss(
    #     include_background=False,
    #     to_onehot_y=True,
    #     sigmoid=False,
    #     softmax=True,
    # )
    loss_fn = GeneralizedDiceLoss(
        include_background=False, to_onehot_y=False, sigmoid=False, softmax=False
    )
    # loss_fn = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train & Val

    params["model"] = str(model)
    params["model_name"] = MODEL_NAME
    params["run_name"] = RUN_NAME
    params["model_size"] = sum(p.numel() for p in model.parameters())
    params["batch_size"] = BATCH_SIZE
    params["epochs"] = EPOCHS
    params["seed"] = SEED
    params["early_stopping_patience"] = EARLY_STOPPING_PATIENCE
    params["early_min_delta"] = EARLY_STOPPING_MIN_DELTA
    params["learning_rate"] = LR
    params["batch_size"] = BATCH_SIZE
    params["vit_config"] = str(config)
    params["loss_fn"] = str(loss_fn)
    params["optimizer"] = str(optimizer)
    params["model_encoder_depths"] = args.depths
    params["model_dims"] = args.dims
    params["device"] = str(args.device)
    params["norm_mean"] = 0
    params["norm_std"] = 1

    # Convert and write JSON object to file
    with open(f"{ROOT_PATH}/params.json", "w") as outfile:
        json.dump(params, outfile)

    es = EarlyStopper(
        patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA
    )

    if TRAIN_SIZE > 0:
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=None,
            train_dl=train_dataloader,
            val_dl=val_dataloader,
            device=device,
            model_name=MODEL_NAME,
            run_name=RUN_NAME,
            callbacks={"early_stopper": es},
            csv_path=f"{ROOT_PATH}/training_metrics.csv",
            pyplot_path=f"{ROOT_PATH}/learning_curve.png",
            checkpoint_path=f"{ROOT_PATH}/checkpoint",
            n_classes=N_CLASSES,
        )

        trainer.train(epochs=EPOCHS)

        best_model = trainer.get_best_model()
    else:
        best_model = model

    # Testing

    tester = Tester(
        best_model,
        loss_fn=loss_fn,
        test_dl=test_dataloader,
        device=device,
        csv_path=f"{ROOT_PATH}/testing_metrics.csv",
        original_img_size=IMG_SIZE,
        n_classes=N_CLASSES,
    )

    tester.test(preds_path=None)


if __name__ == "__main__":
    main()
