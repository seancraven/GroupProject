"""
This module contains temporary functions that are used for plotting.

This file needs to:
    - Load a U-Net model from a file
    - Load pass it and the test data to an evaluation.
"""
import os
import matplotlib.pyplot as plt
import matplotlib
import torch
from torch import nn
from src.models.UNet import UNet
from src.models.LSD import LSD
from typing import List, Callable, Union, Any, Tuple
from torch.utils.data import Dataset
from src.utils.evaluation import watched_evaluate_IoU
from torch.utils.data import DataLoader

matplotlib.style.use("seaborn")

MODEL_CLASSES = [UNet, LSD]


def _model_from_file(file_path: str, model_class: Any) -> Union[nn.Module, None]:
    """Tries to load a model from a file. if it fails, returns None."""
    try:
        model = model_class()
        model.load_state_dict(torch.load(file_path))
    except:
        model = None
    return model


def model_from_file(file_path: str) -> Union[nn.Module, None]:
    for m_class in MODEL_CLASSES:
        model = _model_from_file(file_path, m_class)
        if model is not None:
            return model
    if file_path[-2:] == "pt":
        Warning(
            "Cant load a model of this type, add the model class to ./src/plotting/temporary_plot_utils.py"
        )
    return None


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
        model = model_from_file(model_f_name)
        if model is not None:
            actual_model_f_names.append(model_f_name)
            model.eval()
            with torch.no_grad():
                loss = criterion(model, test_loader)
                losses.append(loss)
    return losses, actual_model_f_names


def models_bar(
    model_f_names: List[str],
    losses: List[float],
    criterion_name: str,
    plot_name: str = "test.png",
):
    """Plots a bar chart of the models and their losses."""
    clean_names = clean_file_names(model_f_names)
    min_y, max_y = min(losses), max(losses)

    fig, ax = plt.subplots()
    ax.bar(clean_names, losses, color="black")
    ax.set_ylim(min_y - 0.1 * (max_y - min_y), max_y + 0.1 * (max_y - min_y))
    ax.set_ylabel(criterion_name)
    ax.tick_params(axis="x", labelrotation=35)
    plt.tight_layout()
    fig.savefig(plot_name)
    plt.close()


def models_matshow_best_worst_img(
    model_f_names: List[str],
    watched_criterion: Callable,
    test_dataset: Dataset,
    num_samples: int = 10,
    files_save_path: str = "",
):
    for model_file in model_f_names:
        if model_file is not None:
            matshow_best_worst_img(
                model_file,
                watched_criterion,
                test_dataset,
                num_samples,
                files_save_path,
            )


def matshow_best_worst_img(
    model_f_name: str,
    watched_criterion: Callable,
    test_dataset: Dataset,
    num_samples: int = 10,
    save_dir="",
):
    """
    Makes 2 girids of predictions vs labels, for the
    best and worst models of the dataset.

    Args:
        model_f_name: The file name of the model to evaluate.
        watched_criterion: A function that takes a model and a data loader and returns
            (losses, best_index, worst_index), where best_index and worst_index
            are list's of indecies from a dataset.
        test_dataset: The test data to evaluate the models on.
        num_samples: The number of samples to plot.
        file_save_path: The path to save the images to.
    """
    model = model_from_file(model_f_name)
    if not model:
        return
    model.eval()  # type :ignore

    test_loader = DataLoader(test_dataset, batch_size=64)
    _, bests, worsts = watched_criterion(model, test_loader, num_samples)
    model.to("cpu")
    # List of all our interesting datapoints.
    ind = [tup[0] for tup in worsts] + [tup[0] for tup in bests]

    images = torch.stack([test_dataset[i][0] for i in ind])
    lab = torch.stack([test_dataset[i][1] for i in ind])
    lab = lab.reshape((-1, 256, 256))
    lab.permute((1, 2, 0))

    with torch.no_grad():
        out = model(images) > 0.5
    out = out[:, :, 1].reshape((-1, 256, 256))
    out.permute((1, 2, 0))

    worst_preds, worst_lab = (
        out[:num_samples].numpy(),
        lab[:num_samples].numpy(),
    )
    best_preds, best_lab = (
        out[-num_samples:].numpy(),
        lab[-num_samples:].numpy(),
    )

    w_fig, w_ax = plt.subplots(2, num_samples)
    b_fig, b_ax = plt.subplots(2, num_samples)

    for (fig, ax, pred, lab, name,) in zip(
        [w_fig, b_fig],
        [w_ax, b_ax],
        [worst_preds, best_preds],
        [worst_lab, best_lab],
        ["_worst", "_best"],
    ):
        for i in range(num_samples):
            ax[0, i].matshow(pred[i, :, :])
            ax[1, i].matshow(lab[i, :, :])

            ax[0, i].axis("off")
            ax[1, i].axis("off")

        fig.suptitle(name[1].upper() + name[2:] + " Predictions", fontsize=20)
        fig.supylabel("Ground Truth Labels     Model Predictions", fontsize=16)
        fig.tight_layout()
        ## This doesn't work
        fig.savefig(
            os.path.join(save_dir, clean_file_names([model_f_name])[0] + name + ".png")
        )
        plt.close()


def clean_file_names(file_names: List[str]) -> List[str]:
    """Removes the file path and extension from the file names.

    Note Solution is delicate only works with .pt files.
    """
    return [file_name.split("/")[-1][:-3] for file_name in file_names]
