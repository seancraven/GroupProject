import gc
import os
import torch
import torch.nn as nn

from abc import ABC, abstractmethod, abstractproperty
from torch.utils.data import DataLoader
from typing import Type, Optional, Dict

from src.models.DMT import DMT
from src.models.UNet import UNet
from src.pet_3.data import PetsDataFetcher
from src.utils.evaluation import evaluate_IoU, watched_evaluate_IoU
from src.plotting.temporary_plot_utils import models_matshow_best_worst_img, models_bar


class Experiments:
    REGISTRY: Dict[str, "BaseExperiment"] = {}

    @staticmethod
    def register(experiment: Type["BaseExperiment"]) -> None:
        Experiments.REGISTRY[experiment.__name__] = experiment

    @staticmethod
    def run_all() -> None:
        for name, experiment in Experiments.REGISTRY.items():
            try:
                experiment().run()
            except Exception as exc:
                print(f"!!! Failed to run experiment {name} !!!")
                print(exc)
            finally:
                torch.cuda.empty_cache()
                gc.collect()

    @staticmethod
    def plot_all() -> None:
        for name, experiment in Experiments.REGISTRY.items():
            try:
                experiment().plot()
            except Exception as exc:
                print(f"!!! Failed to plot experiment {name} !!!")
                print(exc)


class BaseExperiment(ABC):
    _SEED = 0
    _ROOT = "src/pet_3"

    BATCH_SIZE = 32
    LABEL_PROPORTION = 0.1
    VALIDATION_PROPORTION = 0.1
    DIFFERENCE_MAXIMIZED_PROPORTION = 0.7
    PERCENTILES = (0.2, 0.4, 0.6, 0.8, 1.0)
    NUM_DMT_EPOCHS = 10
    GAMMA_1 = 3
    GAMMA_2 = 3

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        Experiments.register(cls)

    @abstractproperty
    def model_folder(self) -> str:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    def plot(self) -> None:
        model_fnames = os.listdir(self.model_folder)
        model_fnames = [
            os.path.join(self.model_folder, fname) for fname in model_fnames
        ]
        test_data = PetsDataFetcher(root=self._ROOT).get_test_data()

        models_bar(
            model_fnames,
            evaluate_IoU,
            test_data,
            "IoU",
            f"{os.path.join(self.model_folder, 'IoU_bar.png')}",
        )
        models_matshow_best_worst_img(
            model_fnames, watched_evaluate_IoU, test_data, 4, f"{self.model_folder}"
        )

    def _base_run(
        self,
        *,
        batch_size: int = BATCH_SIZE,
        label_proportion: float = LABEL_PROPORTION,
        validation_proportion: float = VALIDATION_PROPORTION,
        difference_maximized_proportion: float = DIFFERENCE_MAXIMIZED_PROPORTION,
        percentiles: tuple = PERCENTILES,
        num_dmt_epochs: int = NUM_DMT_EPOCHS,
        gamma_1: int = GAMMA_1,
        gamma_2: int = GAMMA_2,
        seed: int = _SEED,
        baseline_fname: Optional[str] = None,
        best_model_fname: Optional[str] = None,
    ) -> None:
        unet_a = UNet()
        unet_b = UNet()
        baseline = UNet()

        fetcher = PetsDataFetcher(root=self._ROOT)
        labeled, validation, unlabeled = fetcher.get_train_data(
            label_proportion, validation_proportion, seed=seed, class_balance=True
        )
        dmt = DMT(
            unet_a,
            unet_b,
            labeled,
            unlabeled,
            validation,
            max_batch_size=batch_size,
            gamma_1=gamma_1,
            gamma_2=gamma_2,
            baseline=baseline,
        )
        dmt.wandb_init(
            percentiles=percentiles,
            num_epochs=num_dmt_epochs,
            batch_size=batch_size,
            label_ratio=label_proportion,
            difference_maximized_proportion=difference_maximized_proportion,
            gamma_1=gamma_1,
            gamma_2=gamma_2,
        )
        dmt.pretrain(max_epochs=10_000, proportion=difference_maximized_proportion)
        dmt.dynamic_train(percentiles=percentiles, max_epochs=num_dmt_epochs)

        baseline_IoU = self.test(dmt.baseline)
        best_model_IoU = self.test(dmt.best_model)
        dmt.wandb_log(
            {
                "Baseline test IoU": baseline_IoU,
                "Best model test IoU": best_model_IoU,
            }
        )

        if baseline_fname is not None:
            fname = os.path.join(self.model_folder, baseline_fname)
            dmt.save_baseline(fname)

        if best_model_fname is not None:
            fname = os.path.join(self.model_folder, best_model_fname)
            dmt.save_best_model(fname)

    @staticmethod
    def test(model: nn.Module) -> float:
        fetcher = PetsDataFetcher(root="src/pet_3")
        test_data = fetcher.get_test_data()
        test_loader = DataLoader(test_data, batch_size=BaseExperiment.BATCH_SIZE)
        test_IoU = evaluate_IoU(model, test_loader)
        return test_IoU
