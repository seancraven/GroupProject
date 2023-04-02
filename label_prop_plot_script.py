"""
Short plotting script.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.plotting.temporary_plot_utils import evaluate_models

from src.utils.evaluation import evaluate_IoU
from src.pet_3.data import PetsDataFetcher

mpl.style.use("default")
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = "Computer Modern"
mpl.rcParams["text.usetex"] = True


if __name__ == "__main__":
    label_fractions = [0.01, 0.02, 0.05, 0.1, 0.5, 0.8, 1.0]
    baseline_file = os.path.join("eval_data", "baseline_loss.npy")
    dmt_file = os.path.join("eval_data", "dmt_loss.npy")
    save_file = os.path.join("final_figs", "label_proportion_experiment.png")

    if os.path.isfile(baseline_file) and os.path.isfile(dmt_file):
        baselines_loss = np.load(baseline_file)
        loss = np.load(dmt_file)

    else:
        # Block only works on sean's machine
        models_dir = os.path.join("models", "vary_label_proportion")
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

        baseline_models_list.sort()
        dmt_models_list.sort()
        print(dmt_models_list)

        baselines_loss, _ = evaluate_models(baseline_models_list, evaluate_IoU, data)
        loss, _ = evaluate_models(dmt_models_list, evaluate_IoU, data)

        np.save(baseline_file, baselines_loss)
        np.save(dmt_file, loss)

    mean_baseline_loss = [sum(baselines_loss[i : i + 5]) / 5 for i in range(0, 35, 5)]
    std_baseline_loss = [
        2 * np.std(baselines_loss[i : i + 5]) / 5**0.5 for i in range(0, 35, 5)
    ]

    ## Plotting
    fig, ax = plt.subplots()
    ax.plot(label_fractions, loss, label="DMT", color="black", marker="x")
    ax.errorbar(
        label_fractions,
        mean_baseline_loss,
        yerr=std_baseline_loss,
        color="grey",
        label="Baseline",
        capsize=5.0,
        capthick=1,
    )
    ax.set_xlabel("Label Fraction", fontsize=20)
    ax.set_ylabel("IoU", fontsize=20)
    ax.semilogx(subs=label_fractions)
    ax.set_xticks(
        label_fractions, labels=[str(i) for i in label_fractions], fontsize=14
    )
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_yticks(ax.get_yticks(), [f"{i:.2}" for i in ax.get_yticks()], fontsize=14)

    ax.legend()
    fig.show()
    fig.tight_layout()
    fig.savefig(save_file, dpi=300)
