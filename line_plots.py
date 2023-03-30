"""
Short plotting script.
"""
import numpy as np
import os
from src.plotting.temporary_plot_utils import (
    evaluate_models,
    model_line_from_list,
)
from src.utils.evaluation import evaluate_IoU
from src.pet_3.data import PetsDataFetcher

if __name__ == "__main__":
    data = PetsDataFetcher("src/pet_3/").get_test_data()
    dmt_models_list = os.listdir("./models/vary_label_proportion")
    baseline_models_list = os.listdir("./models/baselines")
    baseline_models_list = [
        os.path.join("models", "baselines", f_name) for f_name in baseline_models_list
    ]
    dmt_models_list = [
        os.path.join("models", "vary_label_proportion", f_name)
        for f_name in dmt_models_list
    ]

    if os.path.isfile("baseline_loss.npy"):
        baselines_loss = np.load("baseline_loss.npy")

    else:
        baselines_loss, _ = evaluate_models(baseline_models_list, evaluate_IoU, data)
        baselines_loss = np.array(baselines_loss)
        np.save("baseline_loss.npy", baselines_loss)
    label_fractions = [0.01, 0.02, 0.05, 0.1, 0.5, 0.8]
    many_baselines = [
        [baselines_loss[i + j] for i in range(0, 30, 5)] for j in range(5)
    ]
    mean_baseline_loss = [sum(baselines_loss[i : i + 5]) / 5 for i in range(0, 30, 5)]
    std_baseline_loss = [
        2 * np.std(baselines_loss[i : i + 5]) / 5**0.5 for i in range(0, 30, 5)
    ]

    if os.path.isfile("dmt_loss.npy"):
        loss = np.load("dmt_loss.npy")
    else:
        loss, _ = evaluate_models(dmt_models_list, evaluate_IoU, data)
        np.save("dmt_loss.npy", loss)
    loss = np.array(loss)
    fig, ax = model_line_from_list(
        [loss], label_fractions, file_save_path="line.png", label="DMT"
    )
    ax.errorbar(
        label_fractions,
        mean_baseline_loss,
        yerr=std_baseline_loss,
        color="purple",
        label="Baseline",
        capsize=5.0,
        capthick=1,
    )
    ax.set_xlabel("Label fraction", fontsize=20)
    ax.set_ylabel("IoU", fontsize=20)
    ax.semilogx(subs=label_fractions)
    ax.set_xticks(
        label_fractions, labels=[str(i) for i in label_fractions], fontsize=14
    )
    ax.legend()
    fig.show()
    fig.savefig("line.png", dpi=300)
