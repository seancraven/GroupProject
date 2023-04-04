"""

This file needs to:
    - Load a U-Net model from a file
    - Load pass it and the test data to an evaluation.
"""
from typing import List, Callable, Tuple, Any
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils.loading import model_from_file

mpl.style.use("seaborn-ticks")
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.serif"] = "Computer Modern"
mpl.rcParams["text.usetex"] = True


def evaluate_models(
    model_f_names: List[str], criterion: Callable, test_dataset: Dataset
) -> Tuple[List[float], List[str]]:
    """Evaluates a list of models and prints the results.
    Args:
        model_f_names: List of file names of models to evaluate.
        criterion: A function that takes a model and a data loader and
        returns a loss.
        test_dataset: The test data to evaluate the models on.
    Returns:
        A list of losses for each model.
    """
    actual_model_f_names = []
    model_f_names = [i for i in model_f_names if i.endswith(".pt")]
    model_f_names.sort()
    losses: List[float] = []
    test_loader = DataLoader(test_dataset, num_workers=10, batch_size=64)
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
    criterion: Callable,
    test_dataset: Dataset,
    criterion_name: str,
    file_save_path: str = "",
):
    """Plots a bar chart of the models and their evaluation on a criterion.

    Args:
        model_f_names: List of file names of models to evaluate.
        criterion: A function that takes a model and a dataloader and returns a loss.
        This is what the models are evaluated on.
        criterion_name: The name of the criterion.
    """
    losses, model_f_names = evaluate_models(
        model_f_names, criterion, test_dataset
    )

    models_bar_from_list(losses, model_f_names)


def models_bar_from_list(
    losses: List[float],
    names: List[str],
    criterion_name: str = "",
    file_save_path: str = "",
):
    clean_names = _clean_file_names(names)
    min_y, max_y = min(losses), max(losses)

    fig, ax = plt.subplots()
    ax.bar(clean_names, losses, color="black")
    ax.set_ylim(min_y - 0.1 * (max_y - min_y), max_y + 0.1 * (max_y - min_y))
    ax.set_ylabel(criterion_name)
    ax.tick_params(axis="x", labelrotation=35)
    plt.tight_layout()
    fig.savefig(file_save_path)
    plt.close()


def model_line_from_list(
    multiple_losses: List[List[float]],
    names: List[Any],
    criterion_name: str = "",
    x_label: str = "",
    label: str = "",
    file_save_path: str = "",
):
    fig, ax = plt.subplots()
    for line_loss in multiple_losses:
        ax.plot(names, line_loss, c="black")
        ax.scatter(names, line_loss, c="black", marker="x", label=label)
        ax.set_ylabel(criterion_name)
        ax.set_xticks(names)
        ax.set_xlabel(x_label)
    fig.savefig(file_save_path)
    return fig, ax


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
        fig.supylabel(
            "Ground Truth Labels     Model Predictions", fontsize=16
        )
        fig.tight_layout()
        # This doesn't work
        fig.savefig(
            os.path.join(
                save_dir, _clean_file_names([model_f_name])[0] + name + ".png"
            )
        )
        plt.close()


def _clean_file_names(file_names: List[str]) -> List[str]:
    """Removes the file path and extension from the file names.

    Note Solution is delicate only works with .pt files.
    """
    return [file_name.split("/")[-1][:-3] for file_name in file_names]
