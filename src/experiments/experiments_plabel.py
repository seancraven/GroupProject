import gc
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.DMT import DMT
from src.models.UNet import UNet
from src.models.PLabel import PLabel
from src.pet_3.data import PetsDataFetcher
from src.plotting.temporary_plot_utils import (
    models_bar,
    models_matshow_best_worst_img,
)
from src.utils.evaluation import evaluate_IoU, watched_evaluate_IoU


class Experiments:
    REGISTRY: Dict[str, "BaseExperiment"] = {}

    @staticmethod
    def register(experiment: Type["BaseExperiment"]) -> None:
        Experiments.REGISTRY[experiment.__name__] = experiment

    @staticmethod
    def run_all() -> None:
        for name, Experiment in Experiments.REGISTRY.items():
            try:
                experiment = Experiment()
                print(f"Running experiment: {experiment.description}")
                experiment.create_model_folder()
                experiment.run()
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

    BATCH_SIZE = 32  # For poor Michael's computer
    LABEL_PROPORTION = 0.1
    ALL_LABEL_PROPORTIONS = (0.01, 0.02, 0.05, 0.1, 0.5, 0.8, 0.95)
    VALIDATION_PROPORTION = 0.05
    DIFFERENCE_MAXIMIZED_PROPORTION = 0.7
    PERCENTILES = (0.2, 0.4, 0.6, 0.8, 1.0)
    NUM_DMT_EPOCHS = 10
    MAX_PRETRAIN_EPOCHS = 10000  # Will be 10_000
    GAMMA_1 = 3
    GAMMA_2 = 3

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        Experiments.register(cls)

    @property
    def model_folder(self) -> str:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    def plot(self) -> None:
        """Plot the results of the experiment.
        Files are saved in the model folder.
        """
        model_fnames = os.listdir(self.model_folder)
        print(f"Found {len(model_fnames)} models in {self.model_folder}")

        model_fnames = [
            os.path.join(self.model_folder, fname) for fname in model_fnames
        ]
        model_fnames = os.listdir(self.model_folder)
        model_fnames = [
            os.path.join(self.model_folder, fname) for fname in model_fnames
        ]
        model_fnames.sort()
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

    @property
    def description(self) -> str:
        pass

    def create_model_folder(self):
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

    def _train_baseline_only(
        self,
        *,
        batch_size: int = BATCH_SIZE,
        label_proportion: float = LABEL_PROPORTION,
        validation_proportion: float = VALIDATION_PROPORTION,
        seed: int = _SEED,
        max_pretrain_epochs: int = MAX_PRETRAIN_EPOCHS,
        baseline_fname: Optional[str] = None,
    ):
        fetcher = PetsDataFetcher(root=self._ROOT)
        labeled, validation, unlabeled = fetcher.get_train_data(
            label_proportion, validation_proportion, seed=seed, class_balance=True
        )
        # Instantiate DMT, but only use baseline
        dmt = DMT(
            model_a=UNet(),
            model_b=UNet(),
            labeled_dataset=labeled,
            unlabeled_dataset=unlabeled,
            validation_dataset=validation,
            max_batch_size=batch_size,
            gamma_1=0,
            # Doesn't matter since were only doing the baseline
            gamma_2=0,
            baseline=UNet(),
        )
        dmt.pretrain_baseline(max_epochs=max_pretrain_epochs)
        if baseline_fname is not None:
            fname = os.path.join(self.model_folder, baseline_fname)
            dmt.save_baseline(fname)

    def _base_run(
        self,
        *,
        batch_size: int = BATCH_SIZE,
        label_proportion: float = LABEL_PROPORTION,
        validation_proportion: float = VALIDATION_PROPORTION,
        difference_maximized_proportion: float = DIFFERENCE_MAXIMIZED_PROPORTION,
        percentiles: tuple = PERCENTILES,
        num_dmt_epochs: int = NUM_DMT_EPOCHS,
        max_pretrain_epochs: int = MAX_PRETRAIN_EPOCHS,
        gamma_1: int = GAMMA_1,
        gamma_2: int = GAMMA_2,
        seed: int = _SEED,
        baseline_fname: Optional[str] = None,
        best_model_fname: Optional[str] = None,
        # plabel name
        plabel_fname: Optional[str] = None,
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
        dmt.pretrain(
            max_epochs=max_pretrain_epochs, proportion=difference_maximized_proportion
        )
        dmt.dynamic_train(percentiles=percentiles, num_epochs=num_dmt_epochs)

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

    # PLABEL RUN   
    def _plabel_run(
        self,
        *,
        batch_size: int = BATCH_SIZE,
        label_proportion: float = LABEL_PROPORTION,
        validation_proportion: float = VALIDATION_PROPORTION,
        num_epochs: int = NUM_DMT_EPOCHS,
        max_pretrain_epochs: int = MAX_PRETRAIN_EPOCHS,
        seed: int = _SEED,
        baseline_fname: Optional[str] = None,
        model_fname: Optional[str] = None,
    ) -> None:
        unet_p = UNet()
        baseline = UNet() if baseline_fname is not None else None

        fetcher = PetsDataFetcher(root=self._ROOT)
        labeled, validation, unlabeled = fetcher.get_train_data(
            label_proportion, validation_proportion, seed=seed, class_balance=True
        )

        # PLabel stuff
        unet_p = UNet()
        plabel = PLabel(
            unet_p,
            labeled,
            unlabeled,
            validation,
            max_batch_size=batch_size,
            baseline=baseline,
        )
        plabel.wandb_init(
            num_epochs=num_epochs,
            batch_size=batch_size,
            label_ratio=label_proportion,
        )
        plabel.pretrain(
            max_epochs=max_pretrain_epochs,
        )
        plabel.train(num_epochs=num_epochs)

        plabel_IoU = self.test(plabel.model)
        plabel.wandb_log({"Model test IoU": plabel_IoU})
        if baseline is not None:
            baseline_IoU = self.test(plabel.baseline)
            plabel.wandb_log({"Baseline test IoU": baseline_IoU})
            fname = os.path.join(self.model_folder, baseline_fname)
            plabel.save_baseline(fname)

        if model_fname is not None:
            fname = os.path.join(self.model_folder, model_fname)
            plabel.save_model(fname)

    @staticmethod
    def test(model: nn.Module) -> float:
        fetcher = PetsDataFetcher(root="src/pet_3")
        test_data = fetcher.get_test_data()
        test_loader = DataLoader(test_data, batch_size=BaseExperiment.BATCH_SIZE)
        test_IoU = evaluate_IoU(model, test_loader)
        return test_IoU


class TrainBaselines(BaseExperiment):
    @property
    def model_folder(self) -> str:
        return "models/baselines"

    @property
    def description(self) -> str:
        return "Train baselines for all label proportions"

    def run(self) -> None:
        pass

        NO_RUNS_PER_BASELINE = 5
        for proportion in [self.ALL_LABEL_PROPORTIONS[-1]]:
            for i in range(NO_RUNS_PER_BASELINE):
                fname = "baseline_{}_{}.pt".format(proportion, i + 1)
                self._train_baseline_only(
                    label_proportion=proportion, baseline_fname=fname
                )


class VaryDifferenceMaximization(BaseExperiment):
    PROPORTIONS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    @property
    def model_folder(self) -> str:
        return "models/vary_difference_maximization"

    @property
    def description(self) -> str:
        return "Try different proportions of difference maximization"

    def run(self) -> None:
        for proportion in self.PROPORTIONS:
            self._base_run(
                difference_maximized_proportion=proportion,
                best_model_fname=f"dmt_{proportion}.pt",
            )


class VaryDMTEpochs(BaseExperiment):
    EPOCHS = (5, 10, 20, 30)

    @property
    def model_folder(self) -> str:
        return "models/vary_dmt_epochs"

    @property
    def description(self) -> str:
        return "Try different numbers of DMT epochs"

    def run(self) -> None:
        for num_epochs in self.EPOCHS:
            self._base_run(
                num_dmt_epochs=num_epochs, best_model_fname=f"dmt_{num_epochs}.pt"
            )


class VaryLabelProportion(BaseExperiment):
    @property
    def model_folder(self) -> str:
        return "models/vary_label_proportion"

    @property
    def description(self) -> str:
        return "Try different label proportions"

    def run(self) -> None:
        for proportion in self.ALL_LABEL_PROPORTIONS:
            self._base_run(
                label_proportion=proportion, best_model_fname=f"dmt_{proportion}.pt"
            )

# PLABEL EXPERIMENTS
class PLabelVaryLabelProportion(BaseExperiment):
    @property
    def model_folder(self) -> str:
        return "models/plabel_vary_label_proportion"

    @property
    def description(self) -> str:
        return "Try different label proportions"

    def run(self) -> None:
        for proportion in self.ALL_LABEL_PROPORTIONS:
            self._plabel_run(
                label_proportion=proportion, model_fname=f"plabel_{proportion}.pt"
            )