"""
Short plotting script for the DMS varying experiment.

This should not be called before all experiments have been run.
Experiments.run_all()
However if eval_data is populated it can also be run. Without the experiments

Requires the VaryDifferenceMaximization().run() class to be run first.
Requires the TrainBaselines().run() to be evaluated first.
Requires the PLabelDefault().run() to be evaluated first.
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
    dms_props = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    baseline_file = os.path.join("eval_data", "baseline_loss.npy")
    dms_file = os.path.join("eval_data", "dmt_loss_dms.npy")
    plabel_file = os.path.join("eval_data", "plabel_loss.npy")
    save_file = os.path.join("final_figs", "dms_experiment.png")
    if os.path.isfile(dms_file) and os.path.isfile(baseline_file):
        loss = np.load(dms_file)
        baselines_loss = np.load(baseline_file)
        plabel_loss = np.load(plabel_file)

    else:
        baseline_dir = os.path.join("models", "baselines")
        models_dir = os.path.join("models", "vary_difference_maximization")
        plabel_dir = os.path.join("models", "plabel_vary_label_proportion")

        data = PetsDataFetcher("src/pet_3/").get_test_data()
        baseline_models_list = os.listdir(baseline_dir)
        baseline_models_list = [
            os.path.join(baseline_dir, f_name) for f_name in baseline_models_list
        ]
        baseline_models_list.sort()

        dmt_models_list = os.listdir(models_dir)
        dmt_models_list = [
            os.path.join(models_dir, f_name) for f_name in dmt_models_list
        ]
        dmt_models_list.sort()

        plabel_models_list = os.listdir(plabel_dir)
        plabel_models_list = [
            os.path.join(plabel_dir, f_name) for f_name in plabel_models_list
        ]

        plabel_models_list.sort()

        plabel_loss, _ = evaluate_models(plabel_models_list, evaluate_IoU, data)
        baselines_loss, _ = evaluate_models(baseline_models_list, evaluate_IoU, data)
        loss, _ = evaluate_models(dmt_models_list, evaluate_IoU, data)

        np.save(dms_file, loss)
        np.save(baseline_file, baselines_loss)

    ## Logic
    baseline_same_label = baselines_loss[15:20]  # 0.1 label fraction
    baseline_val = np.mean(baseline_same_label)
    baseline_ste = 2 * np.std(baseline_same_label) / len(baseline_same_label) ** 0.5
    lb = baseline_val - baseline_ste
    ub = baseline_val + baseline_ste
    plabel_mean = np.mean(plabel_loss)
    plabel_ste = 2 * np.std(plabel_loss) / 5**0.5

    ## Plotting
    plot_range = np.linspace(0.45, 1.02, 5)
    fig, ax = plt.subplots()
    ax.plot(dms_props, loss, color="black", marker="x", linestyle=" ")
    ax.fill_between(
        plot_range,
        [0.823 - 0.005 for _ in plot_range],
        [0.823 + 0.005 for _ in plot_range],
        color="black",
        alpha=0.2,
        label="Defualt DMT $\pm 2 SE$",
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
        [plabel_mean for _ in plot_range],
        color="navy",
        linestyle="--",
        alpha=0.2,
    )
    ax.fill_between(
        plot_range,
        [lb for _ in plot_range],
        [ub for _ in plot_range],
        color="grey",
        alpha=0.2,
        label="Baseline $\pm 2 SE$",
    )
    ax.plot(
        plot_range,
        [0.823 for _ in plot_range],
        color="black",
        linestyle="--",
    )
    ax.plot(
        plot_range,
        [baseline_val for _ in plot_range],
        color="grey",
        linestyle="--",
    )

    ax.set_xlabel("DMS Proportion", fontsize=20)
    ax.set_ylabel("IoU", fontsize=20)
    ax.set_xticks(dms_props, labels=[str(i) for i in dms_props], fontsize=14)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_yticks(ax.get_yticks(), [f"{i:.3f}" for i in ax.get_yticks()], fontsize=14)
    ax.set_xlim(0.48, 1.02)

    ax.legend()
    fig.show()
    fig.tight_layout()
    fig.savefig(save_file, dpi=300)
