"""
This module contains functions for evaluating a model on a dataloader.

They share an interface as much as possible.
function(model, dataloader, device, *args, **kwargs)
"""
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from typing import Tuple, List


def evaluate_IoU(
    model: nn.Module,
    data: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    """IoU criterion for evaluating a model on a dataset.

    Args:
        model: The model to evaluate.
        data: The data to evaluate the model on, in a DataLoader.
        device: The device to run the model on.

    Returns:
        Mean IoU across the dataset.
    """

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
    """Accuracy criterion for evaluating a model on a dataset.
    This is not used anywhere.

    Args:
        model: The model to evaluate.
        data: The data to evaluate the model on, in a DataLoader.
        device: The device to run the model on.
    
    Returns:
        Mean accuracy across the dataset.
    """
    model = model.to(device)
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
    """
    Evaluates the IoU of a model on a dataset, and returns the best and worst
    perfoming images in that dataset.

    Args:
        model: The model to evaluate.
        data: The data to evaluate the model on, in a DataLoader.
        num_samples: The number of samples to return.
        device: The device to run the model on.

    Returns:
        Mean IoU across the datase.
        A list of tuples of the form (index, IoU) for the best performing images.
        A list of tuples of the form (index, IoU) for the worst performing images.
    """
    model = model.to(device)
    score = 0
    seen_images = 0
    best_img: List[Tuple[int, float]] = []
    worst_img: List[Tuple[int, float]] = []

    batch_start_index: int = 0
    for images, labels in data:
        batch_end_index = batch_start_index + images.shape[0]
        images, labels = images.to(device), labels.to(device)

        ## Get the IoU for each image in the batch
        predictions = model(images).argmax(dim=-1)
        intersection = (torch.logical_and(predictions == 1, labels == 1)).sum(dim=(-1))
        union = (torch.logical_or(predictions == 1, labels == 1)).sum(dim=(-1))
        IoU = intersection / union  # Got this proportion correct on this batch
        score += IoU.sum() * images.shape[0]
        seen_images += images.shape[0]

        ## Update the best and worst images
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
