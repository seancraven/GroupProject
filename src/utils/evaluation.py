import torch
import torch.nn as nn

from torch.utils.data import DataLoader

def evaluate_IoU(
    model: nn.Module,
    data: DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> float:
    model = model.to(device)
    score = 0
    seen_images = 0

    for images, labels in data:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).argmax(dim=-1)
        intersection = (torch.logical_and(predictions == 1, labels == 1)).sum()
        union = (torch.logical_or(predictions == 1, labels == 1)).sum()
        IoU = intersection / union  # Got this proportion correct on this batch
        score += IoU * images.shape[0]
        seen_images += images.shape[0]
    
    return (score / seen_images).item()