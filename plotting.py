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
    baseline_models_list = os.listdir("./models/baseline")
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
    joint_list_names = []
    joint_list_losses = []
    for dmtl, dmtn, bl, bn in zip(
        dmt_losses, dmt_models_list, baseline_losses, baseline_models_list
    ):
        joint_list_names.append(dmtn)
        joint_list_names.append(bn)
        joint_list_losses.append(dmtl)
        joint_list_losses.append(bl)
    print(joint_list_names)
    print(joint_list_losses)

    models_matshow_best_worst_img(
        dmt_models_list, watched_evaluate_IoU, data, files_save_path="images/dmt"
    )
    models_matshow_best_worst_img(
        baseline_models_list,
        watched_evaluate_IoU,
        data,
        files_save_path="images/baseline",
    )
    models_bar(baseline_models_list, baseline_losses, "IoU")
    models_bar(dmt_models_list, dmt_losses, "IoU")
    models_bar(joint_list_names, joint_list_losses, "IoU")
