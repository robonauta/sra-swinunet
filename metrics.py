import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image


class Metrics:
    def __init__(self, loss_fn, set_size, device, n_classes, eps=torch.Tensor([1e-5])):
        self.device = device
        self.eps = eps.to(device)
        self.loss_fn = loss_fn
        self.n_classes = n_classes
        self.current_loss = torch.Tensor([1_000_000]).to(self.device)
        self.acc_loss = torch.Tensor([0]).to(self.device)
        self.set_size = set_size
        self.tp = {k: 0 for k in range(self.n_classes)}
        self.fp = {k: 0 for k in range(self.n_classes)}
        self.fn = {k: 0 for k in range(self.n_classes)}
        self.tn = {k: 0 for k in range(self.n_classes)}

        if self.n_classes <= 1:
            self.activation = torch.nn.Sigmoid()
        else:
            self.activation = torch.nn.Softmax(dim=1)

    def update(self, output, target, masks=None):
        output = self.activation(output)
        target_one_hot = F.one_hot(target.squeeze(), num_classes=self.n_classes)
        if len(target_one_hot.shape) < 4:
            target_one_hot = target_one_hot.unsqueeze(dim=0)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        if masks is not None:
            self.current_loss = self.loss_fn(output * masks, target_one_hot * masks)
        else:
            self.current_loss = self.loss_fn(output, target_one_hot)

        # self.current_loss = self.loss_fn(output, target.squeeze())

        if self.n_classes > 1:
            pred = torch.argmax(self.activation(output), 1, keepdim=True)

            for cls in range(self.n_classes):
                pred_cls = pred == cls
                target_cls = target == cls
                tp = torch.sum(pred_cls & target_cls)
                fp = torch.sum(pred_cls & (~target_cls))
                fn = torch.sum((~pred_cls) & target_cls)
                tn = torch.sum((~pred_cls) & (~target_cls))
                # Accumulate per class (store as arrays)
                self.tp[cls] += tp
                self.fp[cls] += fp
                self.fn[cls] += fn
                self.tn[cls] += tn
        else:
            output = (self.activation(output) >= 0.5).to(torch.int32)

            tp = torch.sum(output * target)  # TP
            fp = torch.sum(output * (1 - target))  # FP
            fn = torch.sum((1 - output) * target)  # FN
            tn = torch.sum((1 - output) * (1 - target))  # TN

            self.tp += tp.to(self.device)
            self.fp += fp.to(self.device)
            self.fn += fn.to(self.device)
            self.tn += tn.to(self.device)

        self.acc_loss += self.current_loss.item()

    def accuracy(self, class_id=0) -> torch.Tensor:
        return (self.tp[class_id] + self.tn[class_id] + self.eps) / (
            self.tp[class_id]
            + self.tn[class_id]
            + self.fp[class_id]
            + self.fn[class_id]
            + self.eps
        )

    def precision(self, class_id=0) -> torch.Tensor:
        return (self.tp[class_id] + self.eps) / (
            self.tp[class_id] + self.fp[class_id] + self.eps
        )

    def recall(self, class_id=0) -> torch.Tensor:
        return (self.tp[class_id] + self.eps) / (
            self.tp[class_id] + self.fn[class_id] + self.eps
        )

    def dice(self, class_id=0) -> torch.Tensor:
        return (2 * self.tp[class_id] + self.eps) / (
            2 * self.tp[class_id] + self.fp[class_id] + self.fn[class_id] + self.eps
        )

    def f1(self, class_id=0) -> torch.Tensor:
        p = self.precision(class_id)
        r = self.recall(class_id)
        return (2 * p * r) / (p + r)

    def macro_accuracy(self, ignore_index=None) -> torch.Tensor:
        accs = []
        for cls in range(self.n_classes):
            if cls == ignore_index:
                continue
            accs.append(
                (self.tp[cls] + self.tn[cls] + self.eps)
                / (self.tp[cls] + self.tn[cls] + self.fp[cls] + self.fn[cls] + self.eps)
            )
        return torch.mean(torch.FloatTensor(accs))

    def macro_precision(self, ignore_index=None) -> torch.Tensor:
        ps = []
        for cls in range(self.n_classes):
            if cls == ignore_index:
                continue
            ps.append(
                (self.tp[cls] + self.eps) / (self.tp[cls] + self.fp[cls] + self.eps)
            )

        return torch.mean(torch.FloatTensor(ps))

    def macro_recall(self, ignore_index=None) -> torch.Tensor:
        rs = []
        for cls in range(self.n_classes):
            if cls == ignore_index:
                continue
            rs.append(
                (self.tp[cls] + self.eps) / (self.tp[cls] + self.fn[cls] + self.eps)
            )
        return torch.mean(torch.FloatTensor(rs))

    def macro_dice(self, ignore_index=None) -> torch.Tensor:
        ds = []
        for cls in range(self.n_classes):
            if cls == ignore_index:
                continue
            ds.append(
                (2 * self.tp[cls] + self.eps)
                / (2 * self.tp[cls] + self.fp[cls] + self.fn[cls] + self.eps)
            )
        return torch.mean(torch.FloatTensor(ds))

    def macro_f1(self, ignore_index=None) -> torch.Tensor:
        f1s = []
        for cls in range(self.n_classes):
            if cls == ignore_index:
                continue
            p = (self.tp[cls] + self.eps) / (self.tp[cls] + self.fp[cls] + self.eps)
            r = (self.tp[cls] + self.eps) / (self.tp[cls] + self.fn[cls] + self.eps)
            f1s.append((2 * p * r) / (p + r))
        return torch.mean(torch.FloatTensor(f1s))

    def get_confusion_matrix_summed(self, ignore_index=None):
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for cls in range(self.n_classes):
            if ignore_index is not None and cls == ignore_index:
                continue
            tp += self.tp[cls]
            tn += self.tn[cls]
            fp += self.fp[cls]
            fn += self.fn[cls]

        return tp, tn, fp, fn

    def micro_accuracy(self, ignore_index=None) -> torch.Tensor:
        tp, tn, fp, fn = self.get_confusion_matrix_summed(ignore_index)
        return (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)

    def micro_precision(self, ignore_index=None) -> torch.Tensor:
        tp, tn, fp, fn = self.get_confusion_matrix_summed(ignore_index)
        return (tp + self.eps) / (tp + fp + self.eps)

    def micro_recall(self, ignore_index=None) -> torch.Tensor:
        tp, tn, fp, fn = self.get_confusion_matrix_summed(ignore_index)
        return (tp + self.eps) / (tp + fn + self.eps)

    def micro_dice(self, ignore_index=None) -> torch.Tensor:
        tp, tn, fp, fn = self.get_confusion_matrix_summed(ignore_index)
        return (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)

    def micro_f1(self, ignore_index=None) -> torch.Tensor:
        p = self.micro_precision(ignore_index)
        r = self.micro_recall(ignore_index)
        return (2 * p * r) / (p + r)

    def loss(self) -> torch.Tensor:
        return self.acc_loss / self.set_size
