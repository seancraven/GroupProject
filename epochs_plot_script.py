"""
Short plotting script.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.utils.evaluation import evaluate_IoU
from src.pet_3.data import PetsDataFetcher
from src.plotting.temporary_plot_utils import (
    evaluate_models,
)

mpl.style.use("default")
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = "Computer Modern"
mpl.rcParams["text.usetex"] = True


if __name__ == "__main__":
    epochs = [5, 10, 20, 30]
    baseline_file = os.path.join("eval_data", "baseline_loss.npy")
    dmt_file = os.path.join("eval_data", "dmt_loss_epoch.npy")
    save_file = os.path.join("final_figs", "epoch_experiment.png")

    if os.path.isfile(dmt_file) and os.path.isfile(baseline_file):
        loss = np.load(dmt_file)
        baselines_loss = np.load(baseline_file)
    else:
        # Load
        models_dir = os.path.join("models", "vary_dmt_epochs")
        baseline_dir = os.path.join("models", "baselines")
        data = PetsDataFetcher("src/pet_3/").get_test_data()
        dmt_models_list = os.listdir(models_dir)
        baseline_models_list = os.listdir(baseline_dir)
        baseline_models_list = [
            os.path.join(baseline_dir, f_name) for f_name in baseline_models_list
        ]
        dmt_models_list = [
            os.path.join(models_dir, f_name) for f_name in dmt_models_list
        ]

        # Eval
        baseline_models_list.sort()
        dmt_models_list.sort()

        baselines_loss, _ = evaluate_models(baseline_models_list, evaluate_IoU, data)
        np.save(baseline_file, baselines_loss)

        loss, _ = evaluate_models(dmt_models_list, evaluate_IoU, data)
        np.save(dmt_file, loss)

    # Logic
    baseline_same_label = baselines_loss[14:19]  # 0.1 label fraction
    baseline_val = np.mean(baseline_same_label)
    baseline_ste = 2 * np.std(baseline_same_label) / 5**0.5
    lb = baseline_val - baseline_ste
    ub = baseline_val + baseline_ste

    ## Plotting
    plot_range = np.arange(0, 40, 5)
    fig, ax = plt.subplots()
    ax.plot(epochs, loss, color="black", marker="x")
    ax.fill_between(
        plot_range,
        [lb for _ in plot_range],
        [ub for _ in plot_range],
        color="grey",
        alpha=0.2,
        label="Baseline $\pm 2 SE$",
    )
    ax.plot(
        plot_range, [baseline_val for _ in plot_range], color="grey", linestyle="--"
    )
    ax.set_xlim(4, 31)
    ax.set_xlabel("Epochs Between \n Student Teacher Reversal", fontsize=20)
    ax.set_ylabel("IoU", fontsize=20)
    ax.set_xticks(epochs, labels=[str(i) for i in epochs], fontsize=14)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_yticks(ax.get_yticks(), [f"{i:.2}" for i in ax.get_yticks()], fontsize=14)

    ax.legend()
    fig.show()
    fig.tight_layout()
    fig.savefig(save_file, dpi=300)
