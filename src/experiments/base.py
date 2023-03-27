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
from src.utils.evaluation import evaluate_IoU


class Experiments:
    REGISTRY: Dict[str, 'BaseExperiment'] = {}
    
    @staticmethod
    def register(experiment: Type['BaseExperiment']) -> None:
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
    _ROOT = 'src/pet_3'

    BATCH_SIZE = 32
    LABEL_PROPORTION = 0.1
    ALL_LABEL_PROPORTIONS = (0.01, 0.02, 0.05, 0.1, 0.5, 0.8, 1.0)
    VALIDATION_PROPORTION = 0.1
    DIFFERENCE_MAXIMIZED_PROPORTION = 0.7
    PERCENTILES = (0.2, 0.4, 0.6, 0.8, 1.0)
    NUM_DMT_EPOCHS = 10
    MAX_PRETRAIN_EPOCHS = 10_000
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

    @abstractmethod
    def plot(self) -> None:
        pass

    def _train_baseline_only(
        self,
        *,
        batch_size: int=BATCH_SIZE,
        label_proportion: float=LABEL_PROPORTION,
        validation_proportion: float=VALIDATION_PROPORTION,
        seed: int=_SEED,
        max_pretrain_epochs: int=MAX_PRETRAIN_EPOCHS,
        baseline_fname: Optional[str]=None,
    ):
        baseline = UNet()

        fetcher = PetsDataFetcher(root=self._ROOT)
        labeled, validation, unlabeled = fetcher.get_train_data(
            label_proportion, validation_proportion, seed=seed, class_balance=True
        )
        dmt = DMT(  # Instantiate DMT with a lot of dummy arguments
            None,
            None,
            labeled,
            unlabeled,
            validation,
            max_batch_size=batch_size,
            gamma_1=0,
            gamma_2=0,
            baseline=baseline
        )
        dmt.pretrain_baseline(max_epochs=max_pretrain_epochs)
        if baseline_fname is not None:
            fname = os.path.join(self.model_folder, baseline_fname)
            dmt.save_baseline(fname)


    def _base_run(
        self,
        *,
        batch_size: int=BATCH_SIZE,
        label_proportion: float=LABEL_PROPORTION,
        validation_proportion: float=VALIDATION_PROPORTION,
        difference_maximized_proportion: float=DIFFERENCE_MAXIMIZED_PROPORTION,
        percentiles: tuple=PERCENTILES,
        num_dmt_epochs: int=NUM_DMT_EPOCHS,
        max_pretrain_epochs: int=MAX_PRETRAIN_EPOCHS,
        gamma_1: int=GAMMA_1,
        gamma_2: int=GAMMA_2,
        seed: int=_SEED,
        baseline_fname: Optional[str]=None,
        best_model_fname: Optional[str]=None,
    ) -> None:
        unet_a = UNet()
        unet_b = UNet()
        baseline = UNet() if baseline_fname is not None else None

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
            baseline=baseline
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
        dmt.pretrain(max_epochs=max_pretrain_epochs, proportion=difference_maximized_proportion)
        dmt.dynamic_train(percentiles=percentiles, max_epochs=num_dmt_epochs)

        best_model_IoU = self.test(dmt.best_model)
        dmt.wandb_log({"Best model test IoU": best_model_IoU})
        if baseline is not None:
            baseline_IoU = self.test(dmt.baseline)
            dmt.wandb_log({"Baseline test IoU": baseline_IoU})
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
