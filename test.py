import os

from src.plotting.temporary_plot_utils import model_from_file, evaluate_models, models_bar
from src.utils.evaluation import evaluate_IoU
from src.pet_3.data import PetsDataFetcher

if __name__ == "__main__":
    model_fnames = os.listdir("models")
    model_fnames = [os.path.join("models", fname) for fname in model_fnames]
    test_data = PetsDataFetcher(root='src/pet_3', ).get_test_data()
    losses, model_fnames = evaluate_models(model_fnames, evaluate_IoU, test_data)
    models_bar(model_fnames, losses, "IoU")

