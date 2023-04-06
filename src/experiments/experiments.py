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
    """
    Global registry of experiments.
    To add a new experiment, simply create a new class that inherits from BaseExperiment.
    BaseExperiment will automatically register the experiment in the registry.

    Then, to run all experiments, simply call Experiments.run_all()

    """

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
    """
    Base class for experiments.

    To create a new experiment, simply create a new class that inherits from BaseExperiment.
    The run method is an abstract method that must be implemented by the subclass.
    The BaseExperiment contains all the common functionality for running experiments.
    These can be provided as default values in the constructor, or overridden by the subclass.

    Example:
    class MyExperiment(BaseExperiment):
        def run(self):
            # Do something

    Args:
        BATCH_SIZE (int): Batch size for training
        LABEL_PROPORTION (float): Proportion of labeled data in the training set
        ALL_LABEL_PROPORTIONS (tuple): All label proportions to run
        DIFFERENCE_MAXIMIZED_PROPORTION (float): Proportion of validation data
        PERCENTILES (tuple): Percentiles to use for DMT
        NUM_DMT_EPOCHS (int): Number of epochs to train DMT
        MAX_PRETRAIN_EPOCHS (int): Maximum number of epochs to train baseline
        GAMMA_1 (int): Gamma 1 for DMT
        GAMMA_2 (int): Gamma 2 for DMT
    """

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
        # Register the experiment in the global registry
        Experiments.register(cls)

    @property
    def model_folder(self) -> str:
        pass

    # This is an abstract method that must be implemented by the subclass
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
            model_fnames,
            watched_evaluate_IoU,
            test_data,
            4,
            f"{self.model_folder}",
        )

    # this property is a description of the experiment
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
        """
        Training the baseline model only. Used for comparison with DMT.

        Args:
            batch_size (int): Batch size for training
            label_proportion (float): Proportion of labeled data
            validation_proportion (float): Proportion of validation data
            seed (int): Seed for random number generator
            max_pretrain_epochs (int): Maximum number of epochs to train baseline
            baseline_fname (Optional[str]): Baseline model filename

        """
        fetcher = PetsDataFetcher(root=self._ROOT)
        labeled, validation, unlabeled = fetcher.get_train_data(
            label_proportion,
            validation_proportion,
            seed=seed,
            class_balance=True,
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
    ) -> None:
        """
        Base run method for all experiments. This method is called by the
        subclass run method. This method trains the DMT model and saves the
        best model.

        Args:
            batch_size (int): Batch size for training
            label_proportion (float): Proportion of labeled data
            validation_proportion (float): Proportion of validation data
            difference_maximized_proportion (float): Proportion of data to be
                                                     difference maximized
            percentiles (tuple): Percentiles to be used for assigning pseudo
                                 labels to unlabeled data for DMT
            num_dmt_epochs (int): Number of epochs to train DMT
            max_pretrain_epochs (int): Maximum number of epochs to train models
                                       in DMT
            gamma_1 (int): Exponent 1 for the dynamic loss
            gamma_2 (int): Exponent 2 for the dynamic loss
            seed (int): Seed for random number generator
            baseline_fname (Optional[str]): Baseline model filename
            best_model_fname (Optional[str]): Best model filename
        """
        # initialize models
        unet_a = UNet()
        unet_b = UNet()
        # initialize baseline model if provided
        baseline = UNet() if baseline_fname is not None else None

        # get data
        fetcher = PetsDataFetcher(root=self._ROOT)
        labeled, validation, unlabeled = fetcher.get_train_data(
            label_proportion,
            validation_proportion,
            seed=seed,
            class_balance=True,
        )
        # instantiate DMT
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
        # initialize wandb
        dmt.wandb_init(
            percentiles=percentiles,
            num_epochs=num_dmt_epochs,
            batch_size=batch_size,
            label_ratio=label_proportion,
            difference_maximized_proportion=difference_maximized_proportion,
            gamma_1=gamma_1,
            gamma_2=gamma_2,
        )
        # pre train models for DMT
        dmt.pretrain(
            max_epochs=max_pretrain_epochs,
            proportion=difference_maximized_proportion,
        )
        # run DMT
        dmt.dynamic_train(percentiles=percentiles, num_epochs=num_dmt_epochs)

        # logging and saving
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
        # We run DMT for each percentile, so we need to multiply by the number of percentiles here
        num_epochs: int = len(PERCENTILES) * NUM_DMT_EPOCHS,
        max_pretrain_epochs: int = MAX_PRETRAIN_EPOCHS,
        seed: int = _SEED,
        baseline_fname: Optional[str] = None,
        model_fname: Optional[str] = None,
    ) -> None:
        """
        Pseudo-label run method for all experiments. This method is called
        for pseudo label experiments for subclasses. This method trains the
        PLabel model and saves the best model (See PLabel class for details).
        If a baseline model is provided, it is also trained and saved.

        Args:
            batch_size (int): Batch size for training
            label_proportion (float): Proportion of labeled data
            validation_proportion (float): Proportion of validation data
            num_epochs (int): Number of epochs to train PLabel
            max_pretrain_epochs (int): Maximum number of epochs to train model
            seed (int): Seed for random number generator
            baseline_fname (Optional[str]): Baseline model filename
            model_fname (Optional[str]): Best model filename
        """
        # initialize models
        unet_p = UNet()
        baseline = UNet() if baseline_fname is not None else None
        # get data
        fetcher = PetsDataFetcher(root=self._ROOT)
        labeled, validation, unlabeled = fetcher.get_train_data(
            label_proportion,
            validation_proportion,
            seed=seed,
            class_balance=True,
        )

        # PLabel instantiation
        unet_p = UNet()
        plabel = PLabel(
            unet_p,
            labeled,
            unlabeled,
            validation,
            max_batch_size=batch_size,
            baseline=baseline,
        )
        # initialize wandb
        plabel.wandb_init(
            num_epochs=num_epochs,
            batch_size=batch_size,
            label_ratio=label_proportion,
        )
        # pre train model
        plabel.pretrain(
            max_epochs=max_pretrain_epochs,
        )
        # train PLabel
        plabel.train(num_epochs=num_epochs)

        # logging and saving
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
        """
        Test method for all experiments.

        Args:
            model (nn.Module): Model to test
        Returns:
            float: Test IoU
        """
        fetcher = PetsDataFetcher(root="src/pet_3")
        test_data = fetcher.get_test_data()
        test_loader = DataLoader(test_data, batch_size=BaseExperiment.BATCH_SIZE)
        test_IoU = evaluate_IoU(model, test_loader)
        return test_IoU


class TrainBaselines(BaseExperiment):
    """
    Trains all baselines for the experiments.
    The number of runs per baseline is 5.
    The proportion used is set to x which is x/0.95 of the train data.
    This is due to .05 of the train data being used for validation.
    """

    @property
    def model_folder(self) -> str:
        return "models/baselines"

    @property
    def description(self) -> str:
        return "Train baselines for all label proportions"

    def run(self) -> None:

        NO_RUNS_PER_BASELINE = 5
        for proportion in [self.ALL_LABEL_PROPORTIONS[-1]]:
            for i in range(NO_RUNS_PER_BASELINE):
                fname = "baseline_{}_{}.pt".format(proportion, i + 1)
                self._train_baseline_only(
                    label_proportion=proportion, baseline_fname=fname
                )


class VaryDifferenceMaximization(BaseExperiment):
    """
    Trains DMT with different proportions of for the
    subsets given to model A and model B (alpha from the paper).
    Model A gets the alpha proportion of the data
    Model B gets the (1 - alpha) proportion of the data

    The proportions for pretraining are set to:
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0 (as a percentage)

    All other parameters are set to the default values.

    """

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
    """
    Varying the number of epochs for DMT.
    The epochs determine how long we train
    the student model on the teacher model's predictions
    for each swap.

    The epochs in the experiment are: 5, 10, 20, 30.

    All the other parameters are set to the default values.
    """

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
                num_dmt_epochs=num_epochs,
                best_model_fname=f"dmt_{num_epochs}.pt",
            )


class VaryLabelProportion(BaseExperiment):
    """
    Varying the label proportion for the experiments.
    The label proportion is the proportion of the data that is labeled.

    The proportions in the experiment are:
    0.01, 0.02, 0.05, 0.1, 0.5, 0.8, 0.95 (as a proportion of x/0.95 the train data)
    This is due to .05 of the train data being used for validation.
    """

    @property
    def model_folder(self) -> str:
        return "models/vary_label_proportion"

    @property
    def description(self) -> str:
        return "Try different label proportions"

    def run(self) -> None:
        for proportion in self.ALL_LABEL_PROPORTIONS:
            self._base_run(
                label_proportion=proportion,
                best_model_fname=f"dmt_{proportion}.pt",
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
        self.create_model_folder()
        for proportion in self.ALL_LABEL_PROPORTIONS:
            self._plabel_run(
                label_proportion=proportion,
                model_fname=f"plabel_{proportion}.pt",
            )


class PlabelDefault(BaseExperiment):
    """
    The default pseudo label experiment.
    The label proportion is 0.1 and the number of runs is 4.
    It trains the U-Net model until convergence, and then
    trains the model on the unlabeled data using the pseudo labels.

    See PLabel class in src/models/PLabel.py for more details.
    """

    @property
    def model_folder(self) -> str:
        return "models/plabel_default"

    @property
    def description(self) -> str:
        return "Try different label proportions"

    def run(self) -> None:
        self.create_model_folder()
        for i in range(4):
            proportion = 0.1
            self._plabel_run(
                label_proportion=proportion, model_fname=f"plabel_{proportion}_{i+2}.pt"
            )
