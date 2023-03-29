"""
Short plotting script.
"""
import os
from src.plotting.temporary_plot_utils import (
    models_matshow_best_worst_img,
    models_bar,
    evaluate_models,
    models_bar_from_list,
)
from src.utils.evaluation import watched_evaluate_IoU, evaluate_IoU
from src.pet_3.data import PetsDataFetcher

if __name__ == "__main__":
    data = PetsDataFetcher("src/pet_3/").get_test_data()
    dmt_models_list = os.listdir("./models/dmt")
    baseline_models_list = os.listdir("./models/baselines")
    baseline_models_list = [
        os.path.join("models", "baselines", f_name) for f_name in baseline_models_list
    ]
    dmt_models_list = [
        os.path.join("models", "dmt", f_name) for f_name in dmt_models_list
    ]
    dmt_models_list.sort()
    baseline_models_list.sort()

    losses, model_list = evaluate_models(baseline_models_list, evaluate_IoU, data)
    dm_loss, dm_models = evaluate_models(dmt_models_list, evaluate_IoU, data)
    print(f"len baselines {len(model_list)}")

    loss = [sum(losses[i : i + 5]) / 5 for i in range(0, 30, 5)]
    names = [model_list[i] for i in range(0, 30, 5)]

    tnames = []
    tloss = []
    for bn, mbl, dmn, dml in zip(names, loss, dm_models, dm_loss):
        tnames.append(bn)
        tnames.append(dmn)
        tloss.append(mbl)
        tloss.append(dml)

    models_bar_from_list(tloss, tnames, file_save_path="bars.png")

    # baseline_losses, baseline_models_list = evaluate_models(
    #     baseline_models_list, evaluate_IoU, data
    # )
    # dmt_losses, dmt_models_list = evaluate_models(dmt_models_list, evaluate_IoU, data)
    # joint_list_names = []
    # joint_list_losses = []
    # for dmtl, dmtn, bl, bn in zip(
    #     dmt_losses, dmt_models_list, baseline_losses, baseline_models_list
    # ):
    #     joint_list_names.append(dmtn)
    #     joint_list_names.append(bn)
    #     joint_list_losses.append(dmtl)
    #     joint_list_losses.append(bl)
    # print(joint_list_names)
    # print(joint_list_losses)
    #
    # models_matshow_best_worst_img(
    #     dmt_models_list, watched_evaluate_IoU, data, files_save_path="images/dmt"
    # )
    # models_matshow_best_worst_img(
    #     baseline_models_list,
    #     watched_evaluate_IoU,
    #     data,
    #     files_save_path="images/baseline",
    # )
    # models_bar(baseline_models_list, baseline_losses, "IoU")
    # models_bar(dmt_models_list, dmt_losses, "IoU")
    # models_bar(joint_list_names, joint_list_losses, "IoU")
