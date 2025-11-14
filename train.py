import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import Metrics


class Trainer:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        scheduler,
        train_dl: DataLoader,
        val_dl: DataLoader,
        device,
        model_name: str,
        run_name: str,
        callbacks: dict = {},
        csv_path: str = "",
        pyplot_path: str = "",
        checkpoint_path: str = "",
        n_classes: int = 1,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_vloss = 1_000_000
        self.best_model = model
        self.metrics = None
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device
        self.callbacks = callbacks
        self.run_name = run_name
        self.model_name = model_name
        self.csv_path = csv_path
        self.pyplot_path = pyplot_path
        self.checkpoint_path = checkpoint_path
        self.n_classes = n_classes

    def train_one_epoch(self, epoch):
        metrics = Metrics(
            loss_fn=self.loss_fn,
            set_size=len(self.train_dl),
            device=self.device,
            n_classes=self.n_classes,
        )

        for batch in tqdm(self.train_dl):
            inputs, labels, masks = None, None, None
            if "image" in batch:
                inputs = batch["image"]
            if "label" in batch:
                labels = batch["label"]
            if "mask" in batch:
                masks = batch["masks"]

            # Every data instance is an input + label pair
            inputs, labels = (
                inputs.to(self.device),
                labels.to(self.device),
            )

            if masks is not None:
                masks = masks.to(self.device)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model.forward(inputs)

            # Compute the loss and its gradients
            if masks is not None:
                metrics.update(outputs, labels, masks)
            else:
                metrics.update(outputs, labels)

            metrics.current_loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # self.scheduler.step_update(num_updates=num_updates)

            # torch.cuda.synchronize()
            # del inputs, labels, outputs
            # torch.cuda.empty_cache()

        return metrics

    def val_one_epoch(self):
        vmetrics = Metrics(
            loss_fn=self.loss_fn,
            set_size=len(self.val_dl),
            device=self.device,
            n_classes=self.n_classes,
        )

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for batch in self.val_dl:
                vinputs, vlabels, vmasks = None, None, None
                if "image" in batch:
                    vinputs = batch["image"]
                if "label" in batch:
                    vlabels = batch["label"]
                if "mask" in batch:
                    vmasks = batch["masks"]

                vinputs, vlabels = (
                    vinputs.to(self.device),
                    vlabels.to(self.device),
                )

                if vmasks is not None:
                    vmasks = vmasks.to(self.device)

                voutputs = self.model.forward(vinputs)

                if vmasks is not None:
                    vmetrics.update(voutputs, vlabels, vmasks)
                else:
                    vmetrics.update(voutputs, vlabels)

                # del vinputs, vlabels, voutputs
                # torch.cuda.empty_cache()

        return vmetrics

    def train(self, epochs):
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting

        metrics = {
            "epoch": [],
            "train_loss": [],
            "train_macro_accuracy": [],
            "train_macro_precision": [],
            "train_macro_recall": [],
            "train_macro_dice": [],
            "train_macro_f1": [],
            "train_micro_accuracy": [],
            "train_micro_precision": [],
            "train_micro_recall": [],
            "train_micro_dice": [],
            "train_micro_f1": [],
            "val_loss": [],
            "val_macro_accuracy": [],
            "val_macro_precision": [],
            "val_macro_recall": [],
            "val_macro_dice": [],
            "val_macro_f1": [],
            "val_micro_accuracy": [],
            "val_micro_precision": [],
            "val_micro_recall": [],
            "val_micro_dice": [],
            "val_micro_f1": [],
            "elapsed_time": [],
        }

        for epoch in tqdm(range(epochs), total=epochs):
            # print('EPOCH {}:'.format(epoch + 1))

            self.model.train(True)

            epoch_start = time.time()
            train_metrics = self.train_one_epoch(epoch)
            epoch_end = time.time()

            metrics["epoch"].append(epoch)
            metrics["train_loss"].append(train_metrics.loss().item())
            metrics["train_macro_accuracy"].append(
                train_metrics.macro_accuracy().item()
            )
            metrics["train_macro_precision"].append(
                train_metrics.macro_precision().item()
            )
            metrics["train_macro_recall"].append(train_metrics.macro_recall().item())
            metrics["train_macro_dice"].append(train_metrics.macro_dice().item())
            metrics["train_macro_f1"].append(train_metrics.macro_f1().item())
            metrics["train_micro_accuracy"].append(
                train_metrics.micro_accuracy().item()
            )
            metrics["train_micro_precision"].append(
                train_metrics.micro_precision().item()
            )
            metrics["train_micro_recall"].append(train_metrics.micro_recall().item())
            metrics["train_micro_dice"].append(train_metrics.micro_dice().item())
            metrics["train_micro_f1"].append(train_metrics.micro_f1().item())
            metrics["elapsed_time"].append((epoch_end - epoch_start))

            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()
            val_metrics = self.val_one_epoch()

            metrics["val_loss"].append(val_metrics.loss().item())
            metrics["val_macro_accuracy"].append(val_metrics.macro_accuracy().item())
            metrics["val_macro_precision"].append(val_metrics.macro_precision().item())
            metrics["val_macro_recall"].append(val_metrics.macro_recall().item())
            metrics["val_macro_dice"].append(val_metrics.macro_dice().item())
            metrics["val_macro_f1"].append(val_metrics.macro_f1().item())
            metrics["val_micro_accuracy"].append(val_metrics.micro_accuracy().item())
            metrics["val_micro_precision"].append(val_metrics.micro_precision().item())
            metrics["val_micro_recall"].append(val_metrics.micro_recall().item())
            metrics["val_micro_dice"].append(val_metrics.micro_dice().item())
            metrics["val_micro_f1"].append(val_metrics.micro_f1().item())

            metrics_df = pd.DataFrame(metrics)

            if self.csv_path != "":
                metrics_df.to_csv(self.csv_path, index=False)

            if self.pyplot_path != "":
                plt.figure()
                plt.plot(metrics_df["epoch"], metrics_df["train_loss"], label="train")
                plt.plot(metrics_df["epoch"], metrics_df["val_loss"], label="val")
                plt.title(f"Learning curve - {self.run_name}")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig(self.pyplot_path)

            # Log the running loss averaged per batch
            # for both training and validation

            # Track best performance, and save the model's state
            if self.checkpoint_path != "":
                if val_metrics.loss() < self.best_vloss:
                    self.best_vloss = val_metrics.loss()

                    model_path = f"{self.checkpoint_path}/best_model.pth"

                    os.makedirs(self.checkpoint_path, exist_ok=True)
                    torch.save(self.model.state_dict(), model_path)
                    self.best_model = self.model

            # Early stop when there has been past x epochs without increase in val_loss
            if "early_stopper" in self.callbacks.keys():
                if self.callbacks["early_stopper"].early_stop(val_metrics.loss()):
                    print("early stopping halting the training")
                    break

            # del train_metrics, val_metrics, metrics_df
            # torch.cuda.empty_cache()

        return self.model

    def get_best_model(self):
        return self.best_model
