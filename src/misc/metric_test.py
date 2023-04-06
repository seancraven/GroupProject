# pylint: disable-all
import torch

from src.pet_3.deprocated_data import Pets
from src.testing.model_testing_utils import ModelMetrics

from src.models.LSD import LSD

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = LoadedModel("./models/u_net_supervised/Mean Squared Error_20.pt")
    model = LSD()
    testdataset = Pets("./src/pet_3", "test", binary_labels=True)
    model_metrics = ModelMetrics(model, testdataset)
    print(
        "LSD accuracy: {:.5f}".format(model_metrics.test_accuracy),
    )
    print(
        "LSD loss: {:.5f}".format(model_metrics.test_loss),
    )
    print("LSD IOU: {:.5f}".format(model_metrics.test_iou))
