"""
Short plotting script for the Epoch varying experiment.

This should not be called before all experiments have been run.
However if eval_data is populated it can also be run. Without the experiments

Requires the VaryDMTepochs().run() class to be run first.
Requires the TrainBaselines().run() to be evaluated first.
Requires the PseudoLabel().run() to be evaluated first.
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
    plabel_file = os.path.join("eval_data", "plabel_loss.npy")
    save_file = os.path.join("final_figs", "epoch_experiment.png")

    if (
        os.path.isfile(dmt_file)
        and os.path.isfile(baseline_file)
        and os.path.isfile(plabel_file)
    ):
        loss = np.load(dmt_file)
        baselines_loss = np.load(baseline_file)
        plabel_loss = np.load(plabel_file)
    else:
        # Load
        models_dir = os.path.join("models", "vary_dmt_epochs")
        baseline_dir = os.path.join("models", "baselines")
        plabel_dir = os.path.join("models", "plabel_default")
        data = PetsDataFetcher("src/pet_3/").get_test_data()
        dmt_models_list = os.listdir(models_dir)
        baseline_models_list = os.listdir(baseline_dir)
        plabel_models_list = os.listdir(plabel_dir)
        baseline_models_list = [
            os.path.join(baseline_dir, f_name) for f_name in baseline_models_list
        ]
        dmt_models_list = [
            os.path.join(models_dir, f_name) for f_name in dmt_models_list
        ]
        plabel_models_list = [
            os.path.join(plabel_dir, f_name) for f_name in plabel_models_list
        ]

        # Eval
        baseline_models_list.sort()
        dmt_models_list.sort()
        plabel_models_list.sort()

        baselines_loss, _ = evaluate_models(baseline_models_list, evaluate_IoU, data)
        np.save(baseline_file, baselines_loss)

        loss, _ = evaluate_models(dmt_models_list, evaluate_IoU, data)
        np.save(dmt_file, loss)

        plabel_loss, _ = evaluate_models(plabel_models_list, evaluate_IoU, data)
        np.save(plabel_file, plabel_loss)
    # Logic
    baseline_same_label = baselines_loss[15:20]  # 0.1 label fraction
    baseline_val = np.mean(baseline_same_label)

    baseline_ste = 2 * np.std(baseline_same_label) / 5**0.5
    lb = baseline_val - baseline_ste
    ub = baseline_val + baseline_ste
    plabel_mean = np.mean(plabel_loss)
    plabel_ste = 2 * np.std(plabel_loss) / 5**0.5

    ## Plotting
    plot_range = np.arange(0, 40, 5)
    fig, ax = plt.subplots()
    ax.plot(epochs, loss, color="black", marker="x", linestyle=" ")
    ax.fill_between(
        plot_range,
        [0.823 - 0.005 for _ in plot_range],
        [0.823 + 0.005 for _ in plot_range],
        color="black",
        alpha=0.2,
        label="Default DMT $\pm 2 SE$",
    )
    ax.fill_between(
        plot_range,
        [lb for _ in plot_range],
        [ub for _ in plot_range],
        color="grey",
        alpha=0.2,
        label="Baseline $\pm 2 SE$",
    )
    ax.fill_between(
        plot_range,
        [plabel_mean - plabel_ste for _ in plot_range],
        [plabel_mean + plabel_ste for _ in plot_range],
        color="navy",
        alpha=0.2,
        label="Pseudo Label $\pm 2 SE$",
    )
    ax.plot(
        plot_range,
        [0.823 for _ in plot_range],
        color="black",
        linestyle="--",
    )
    ax.errorbar([0.1], [0.823], yerr=[0.005], color="black", capsize=5.0, capthick=1)
    ax.plot(
        plot_range, [baseline_val for _ in plot_range], color="grey", linestyle="--"
    )
    ax.plot(
        plot_range,
        [plabel_mean for _ in plot_range],
        color="navy",
        linestyle="--",
        alpha=0.2,
    )
    ax.set_xlim(4, 31)
    ax.set_xlabel("Epochs Between \n Student Teacher Reversal", fontsize=20)
    ax.set_ylabel("IoU", fontsize=20)
    ax.set_xticks(epochs, labels=[str(i) for i in epochs], fontsize=14)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_yticks(ax.get_yticks(), [f"{i:.3f}" for i in ax.get_yticks()], fontsize=14)

    ax.legend()
    fig.show()
    fig.tight_layout()
    fig.savefig(save_file, dpi=300)
