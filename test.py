import os

import cv2
import pandas as pd
import torch
from metrics import Metrics
from torch.utils.data import DataLoader


class Tester:
    def __init__(
        self,
        model,
        loss_fn,
        test_dl: DataLoader,
        device,
        original_img_size,
        palette=None,
        csv_path: str = "",
        model_name="unet",
        n_classes: int = 1,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.best_vloss = 1_000_000
        self.best_model = model
        self.metrics = None
        self.test_dl = test_dl
        self.device = device
        self.csv_path = csv_path
        self.original_img_size = original_img_size
        self.model_name = model_name
        self.palette = palette
        self.n_classes = n_classes

    def test(self, preds_path=None):
        tmetrics = Metrics(
            loss_fn=self.loss_fn,
            set_size=len(self.test_dl),
            device=self.device,
            n_classes=self.n_classes,
        )

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for batch in self.test_dl:
                tinputs, tlabels, tmasks, tpathes = None, None, None, None
                if "image" in batch:
                    tinputs = batch["image"]
                if "label" in batch:
                    tlabels = batch["label"]
                if "mask" in batch:
                    tmasks = batch["masks"]
                if "path" in batch:
                    tpathes = batch["path"]

                tinputs, tlabels = (
                    tinputs.to(self.device),
                    tlabels.to(self.device),
                )

                if tmasks is not None:
                    tmasks = tmasks.to(self.device)

                toutputs = self.model(tinputs)

                if preds_path is not None:
                    for img, path in zip(toutputs, tpathes):
                        p = path.split("/")[-1].split(".")[0]

                        img = img[
                            :, : self.original_img_size[1], : self.original_img_size[2]
                        ]

                        img = (img >= 0.5).to(torch.float32)
                        img = img.permute(1, 2, 0).cpu().numpy() * 255

                        if self.palette is not None:
                            img = img
                            img = self.palette[img]

                        os.makedirs(preds_path, exist_ok=True)

                        cv2.imwrite(f"{preds_path}/{p}.png", img)

                if tmasks is not None:
                    tmetrics.update(
                        torch.squeeze(toutputs, 1),
                        torch.squeeze(tlabels, 1),
                        torch.squeeze(tmasks, 1),
                    )
                else:
                    tmetrics.update(
                        torch.squeeze(toutputs, 1),
                        torch.squeeze(tlabels, 1),
                    )

                # del tinputs, tlabels, toutputs
                # torch.cuda.empty_cache()

        pd.DataFrame(
            {
                **(
                    {
                        "loss": [tmetrics.loss().item()],
                        "accuracy": [tmetrics.accuracy(class_id=0).item()],
                        "precision": [tmetrics.precision(class_id=0).item()],
                        "recall": [tmetrics.recall(class_id=0).item()],
                        "dice": [tmetrics.dice(class_id=0).item()],
                        "f1": [tmetrics.f1(class_id=0).item()],
                    }
                    if self.n_classes == 1
                    else {
                        "label": ["total"] + [i for i in range(self.n_classes)],
                        "loss": [tmetrics.loss().item()] + [None] * self.n_classes,
                        "macro_accuracy": [tmetrics.macro_accuracy().item()]
                        + [None] * self.n_classes,
                        "macro_precision": [tmetrics.macro_precision().item()]
                        + [None] * self.n_classes,
                        "macro_recall": [tmetrics.macro_recall().item()]
                        + [None] * self.n_classes,
                        "macro_dice": [tmetrics.macro_dice().item()]
                        + [None] * self.n_classes,
                        "macro_f1": [tmetrics.macro_f1().item()]
                        + [None] * self.n_classes,
                        "micro_accuracy": [tmetrics.micro_accuracy().item()]
                        + [None] * self.n_classes,
                        "micro_precision": [tmetrics.micro_precision().item()]
                        + [None] * self.n_classes,
                        "micro_recall": [tmetrics.micro_recall().item()]
                        + [None] * self.n_classes,
                        "micro_dice": [tmetrics.micro_dice().item()]
                        + [None] * self.n_classes,
                        "micro_f1": [
                            tmetrics.micro_f1().item(),
                        ]
                        + [None] * self.n_classes,
                        "macro_accuracy_wo_bg": [
                            tmetrics.macro_accuracy(ignore_index=0).item()
                        ]
                        + [None] * self.n_classes,
                        "macro_precision_wo_bg": [
                            tmetrics.macro_precision(ignore_index=0).item()
                        ]
                        + [None] * self.n_classes,
                        "macro_recall_wo_bg": [
                            tmetrics.macro_recall(ignore_index=0).item()
                        ]
                        + [None] * self.n_classes,
                        "macro_dice_wo_bg": [tmetrics.macro_dice(ignore_index=0).item()]
                        + [None] * self.n_classes,
                        "macro_f1_wo_bg": [tmetrics.macro_f1(ignore_index=0).item()]
                        + [None] * self.n_classes,
                        "micro_accuracy_wo_bg": [
                            tmetrics.micro_accuracy(ignore_index=0).item()
                        ]
                        + [None] * self.n_classes,
                        "micro_precision_wo_bg": [
                            tmetrics.micro_precision(ignore_index=0).item()
                        ]
                        + [None] * self.n_classes,
                        "micro_recall_wo_bg": [
                            tmetrics.micro_recall(ignore_index=0).item()
                        ]
                        + [None] * self.n_classes,
                        "micro_dice_wo_bg": [tmetrics.micro_dice(ignore_index=0).item()]
                        + [None] * self.n_classes,
                        "micro_f1_wo_bg": [
                            tmetrics.micro_f1(ignore_index=0).item(),
                        ]
                        + [None] * self.n_classes,
                        "accuracy": [None]
                        + [
                            tmetrics.accuracy(class_id=i).item()
                            for i in range(self.n_classes)
                        ],
                        "precision": [None]
                        + [
                            tmetrics.precision(class_id=i).item()
                            for i in range(self.n_classes)
                        ],
                        "recall": [None]
                        + [
                            tmetrics.recall(class_id=i).item()
                            for i in range(self.n_classes)
                        ],
                        "dice": [None]
                        + [
                            tmetrics.dice(class_id=i).item()
                            for i in range(self.n_classes)
                        ],
                        "f1": [None]
                        + [
                            tmetrics.f1(class_id=i).item()
                            for i in range(self.n_classes)
                        ],
                    }
                ),
            }
        ).to_csv(self.csv_path)

        return tmetrics
