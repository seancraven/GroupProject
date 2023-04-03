import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from typing import Tuple, List


def evaluate_IoU(
    model: nn.Module,
    data: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    model = model.to(device)
    seen_images = 0

    IoUs = torch.zeros((2,)).to(device)

    for images, labels in data:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).argmax(dim=-1)

        # We're only interested in the binary case where we predict 0s and 1s
        for i in range(2):
            intersection = (torch.logical_and(predictions == i, labels == i)).sum()
            union = (torch.logical_or(predictions == i, labels == i)).sum()
            IoUs[i] += (intersection / union) * images.shape[0]  # Weight by batch size

        seen_images += images.shape[0]

    return (IoUs / seen_images).mean().item()


def evaluate_acc(
    model: nn.Module,
    data: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    model = model.to(device)
    seen_images = 0

    acc = 0.0

    for images, labels in data:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).argmax(dim=-1)

        # We're only interested in the binary case where we predict 0s and 1s
        acc = (predictions == labels).sum().item()
        acc += acc
    return acc / len(DataLoader.dataset)


def watched_evaluate_IoU(
    model: nn.Module,
    data: DataLoader,
    num_samples: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[float, List[Tuple[int, float]], List[Tuple[int, float]]]:
    model = model.to(device)
    score = 0
    seen_images = 0
    best_img: List[Tuple[int, float]] = []
    worst_img: List[Tuple[int, float]] = []

    batch_start_index: int = 0
    for images, labels in data:
        batch_end_index = batch_start_index + images.shape[0]
        images, labels = images.to(device), labels.to(device)

        predictions = model(images).argmax(dim=-1)
        intersection = (torch.logical_and(predictions == 1, labels == 1)).sum(dim=(-1))
        union = (torch.logical_or(predictions == 1, labels == 1)).sum(dim=(-1))
        IoU = intersection / union  # Got this proportion correct on this batch
        score += IoU.sum() * images.shape[0]
        seen_images += images.shape[0]

        worst_batch_best_val = (
            [ind_val_tup[1] for ind_val_tup in worst_img]
            + IoU.tolist()
            + [ind_val_tup[1] for ind_val_tup in best_img]
        )
        worst_batch_best_idx = (
            [ind_val_tup[0] for ind_val_tup in worst_img]
            + [i for i in range(batch_start_index, batch_end_index)]
            + [ind_val_tup[0] for ind_val_tup in best_img]
        )
        worst_batch_best = [
            (idx, val) for idx, val in zip(worst_batch_best_idx, worst_batch_best_val)
        ]
        worst_batch_best.sort(key=lambda x: x[1])

        best_img = worst_batch_best[-num_samples:]
        worst_img = worst_batch_best[:num_samples]

    return (score / seen_images).item(), best_img, worst_img  # type: ignore
