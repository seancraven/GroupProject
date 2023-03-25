"""
This module contains temporary functions that are used for plotting.

This file needs to:
    - Load a U-Net model from a file
    - Load pass it and the test data to an evaluation.
"""
import matplotlib.pyplot as plt
import matplotlib
import torch
from torch import nn
from src.models.UNet import UNet
from src.models.LSD import LSD
from typing import List, Callable, Union, Any, Tuple
from torch.utils.data import Dataset
from src.utils.evaluation import evaluate_IoU
from torch.utils.data import DataLoader

matplotlib.style.use("seaborn")


def model_from_file(file_path: str, model_class: Any) -> Union[nn.Module, None]:
    """Tries to load a model from a file. if it fails, returns None."""
    try:
        model = model_class()
        model.load_state_dict(torch.load(file_path))
    except:
        model = None
    return model


def evaluate_models(
    model_f_names: List[str], criterion: Callable, test_data: Dataset
) -> Tuple[List[float], List[str]]:
    """Evaluates a list of models and prints the results.
    Args:
        model_f_names: List of file names of models to evaluate.
        criterion: A function that takes a model and a data loader and returns a loss.
        test_data: The test data to evaluate the models on.
    Returns:
        A list of losses for each model.
    """
    actual_model_f_names = []
    losses: List[float] = []
    test_loader = DataLoader(test_data, num_workers=10, batch_size=64)
    for model_f_name in model_f_names:
        # Hardcode all of possible classes here. not ideal.
        for model_class in [UNet, LSD]:
            model = model_from_file(model_f_name, model_class)
            if model is not None:
                break
        if model is None:
            Warning(f"Could not load model from file {model_f_name}")
            continue
        actual_model_f_names.append(model_f_name)
        model.eval()
        with torch.no_grad():
            loss = criterion(model, test_loader)
            losses.append(loss)
    return losses, actual_model_f_names


def models_bar(model_f_names: List[str], losses: List[float], criterion_name: str):
    """Plots a bar chart of the models and their losses."""
    clean_names = clean_file_names(model_f_names)
    min_y, max_y = min(losses), max(losses)

    fig, ax = plt.subplots()
    ax.bar(clean_names, losses, color="black")
    ax.set_ylim(min_y - 0.1 * (max_y - min_y), max_y + 0.1 * (max_y - min_y))
    ax.set_ylabel(criterion_name)
    ax.tick_params(axis="x", labelrotation=35)
    plt.tight_layout()
    plt.show()


def clean_file_names(file_names: List[str]) -> List[str]:
    """Removes the file path and extension from the file names."""
    return [file_name.split("/")[-1].split(".")[0] for file_name in file_names]
