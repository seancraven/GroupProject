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
    plabel_file = os.path.join("eval_data", "plabel_loss.npy")
    save_file = os.path.join("final_figs", "label_proportion_experiment.png")

    if (
        os.path.isfile(baseline_file)
        and os.path.isfile(dmt_file)
        and os.path.isfile(plabel_file)
    ):
        baselines_loss = np.load(baseline_file)
        loss = np.load(dmt_file)
        plabel_loss = np.load(plabel_file)

    else:
        data = PetsDataFetcher("src/pet_3/").get_test_data()
        # Block only works on sean's machine
        models_dir = os.path.join("models", "vary_label_proportion")
        baseline_dir = os.path.join("models", "baselines")
        plabel_dir = os.path.join("models", "plabel_vary_label_proportion")

        baseline_models_list = os.listdir(baseline_dir)
        baseline_models_list = [
            os.path.join(baseline_dir, f_name)
            for f_name in baseline_models_list
        ]

        dmt_models_list = os.listdir(models_dir)
        dmt_models_list = [
            os.path.join(models_dir, f_name) for f_name in dmt_models_list
        ]

        plabel_models_list = os.listdir(plabel_dir)
        plabel_models_list = [
            os.path.join(plabel_dir, f_name) for f_name in plabel_models_list
        ]

        plabel_models_list.sort()
        baseline_models_list.sort()
        dmt_models_list.sort()

        baselines_loss, _ = evaluate_models(
            baseline_models_list, evaluate_IoU, data
        )
        print(_)
        loss, _ = evaluate_models(dmt_models_list, evaluate_IoU, data)
        plabel_loss, _ = evaluate_models(
            plabel_models_list, evaluate_IoU, data
        )

        np.save(baseline_file, baselines_loss)
        np.save(dmt_file, loss)
        np.save(plabel_file, plabel_loss)

    mean_baseline_loss = [
        sum(baselines_loss[i : i + 5]) / 5 for i in range(0, 35, 5)
    ]
    ste_baseline_loss = [
        2 * np.std(baselines_loss[i : i + 5]) / 5**0.5
        for i in range(0, 35, 5)
    ]
    print(mean_baseline_loss)
    print(ste_baseline_loss)
    # std_baseline_loss = [2 * np.std(baselines_loss[i : i + 5]) for i in range(0, 35, 5)]

    ## Plotting
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(
        label_fractions,
        loss,
        label="DMT",
        color="black",
        marker="x",
        linestyle=" ",
    )

    ax.plot(
        label_fractions,
        plabel_loss,
        color="navy",
        marker="x",
        label="Pseudo Label",
        linestyle=" ",
    )
    ax.errorbar(
        label_fractions,
        mean_baseline_loss,
        yerr=ste_baseline_loss,
        color="grey",
        label="Baseline",
        capsize=5.0,
        capthick=1,
        linestyle=" ",
    )

    ax.errorbar(
        [0.1], [0.823], yerr=[0.005], color="black", capsize=5.0, capthick=1
    )
    ax.set_xlabel("Label Fraction", fontsize=20)
    ax.set_ylabel("IoU", fontsize=20)
    ax.semilogx(subs=label_fractions)
    ax.set_xticks(
        label_fractions, labels=[str(i) for i in label_fractions], fontsize=14
    )
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_yticks(
        ax.get_yticks(), [f"{i:.2}" for i in ax.get_yticks()], fontsize=14
    )

    ax.legend()
    fig.show()
    fig.tight_layout()
    fig.savefig(save_file, dpi=300)
