"""
Short plotting script.
"""
import os
from src.plotting.temporary_plot_utils import (
    models_matshow_best_worst_img,
    models_bar,
    evaluate_models,
)
from src.utils.evaluation import watched_evaluate_IoU, evaluate_IoU
from src.pet_3.data import PetsDataFetcher

if __name__ == "__main__":
    data = PetsDataFetcher("src/pet_3/").get_test_data()
    dmt_models_list = os.listdir("./models/dmt")
    baseline_models_list = os.listdir("./models/baselines")
    # Each baseline model has five copies
    baseline_models_list = [
        os.path.join("models", "baseline", f_name) for f_name in baseline_models_list
    ]
    dmt_models_list = [
        os.path.join("models", "dmt", f_name) for f_name in dmt_models_list
    ]
    dmt_models_list.sort()
    baseline_models_list.sort()

    baseline_losses, baseline_models_list = evaluate_models(
        baseline_models_list, evaluate_IoU, data
    )
    dmt_losses, dmt_models_list = evaluate_models(dmt_models_list, evaluate_IoU, data)
    list_loss = [sum(dmt_losses[i : i + 5]) / 5 for i in range(0, 25, 5)]
    list_name = [dmt_models_list[i] for i in range(0, 25, 5)]
    models_bar(list_name, list_loss, "IoU")
