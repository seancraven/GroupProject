import time
from typing import Tuple, Optional
from functools import partial
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import wandb

from src.utils.misc import ReporterMixin
from src.utils.datasets import balanced_minibatch_sizes
from src.utils.evaluation import evaluate_IoU
from src.utils.training import PreTrainer, FineTuner as FT


class PLabel(nn.Module, ReporterMixin):

    """Psuedo-Label self-training model

    After pretraining on a labelled training set,
     this model generates pseudolabels on the unlabelled training set
     as the class of maximal probability for each pixel. These pseudolabels
     are used to generate an unlabelled loss via crossentropy with the
    model's predictions: optimising with respect to this unlabelled loss
    has the effect of entropy minimisation over unseen predictions.
    We then combine labelled and unlabelled loss at each step with a
    factor alpha, which increments linearly through training until
    it reaches a maximum value.

    """

    def __init__(
        self,
        model: nn.Module,
        labeled_dataset: Dataset,
        unlabeled_dataset: Dataset,
        validation_dataset: Dataset,
        max_batch_size: int,
        baseline: Optional[nn.Module] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbosity: int = 2,
        t1: int = 100,
        t2: int = 600,
        max_alpha: float = 3.0,
    ):
        super().__init__()
        self.model = model.to(device)
        self.baseline = baseline.to(device) if baseline is not None else None

        # Linear Scheduler Params
        self.t = 0
        self.t1 = t1
        self.t2 = t2
        self.alpha = 0
        self.max_alpha = max_alpha

        # We access dataloaders more than datasets, so create them here and we
        # can access the underlying data later if needed
        labeled_batch_size, unlabeled_batch_size = balanced_minibatch_sizes(
            labeled_dataset, unlabeled_dataset, max_batch_size
        )
        self.labeled_loader, self.unlabeled_loader = map(
            lambda data, batch_size: DataLoader(
                data, batch_size, shuffle=True, num_workers=4
            ),
            (labeled_dataset, unlabeled_dataset),
            (labeled_batch_size, unlabeled_batch_size),
        )
        self.validation_loader = DataLoader(
            validation_dataset, max_batch_size, shuffle=True, num_workers=4
        )

        self.max_batch_size = max_batch_size
        self.device = device
        self.verbosity = verbosity

        # Create convenience functions to get the train IoU and validation IoU
        self.train_IoU, self.validation_IoU = map(
            lambda loader: partial(evaluate_IoU, data=loader, device=self.device),
            (self.labeled_loader, self.validation_loader),
        )

    # Linear Scheduler
    @staticmethod
    def calculate_alpha(t, t1, t2, max_alpha):
        schedule_ratio = (t - t1) / (t2 - t1) * max_alpha
        return min(max(0, schedule_ratio), max_alpha)

    def alpha_schedule_step(self):
        self.t += 1
        self.alpha = self.calculate_alpha(self.t, self.t1, self.t2, self.max_alpha)

    def compute_pseudolabels(
        self, confidences: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes pseudolabels for the unlabeled data. We choose the class that
        has the maximum predictive probability

        Args:
            confidences (torch.Tensor): Tensor of shape (B, W*H, C) containing
                the confidences of the model for each pixel in each
                image in the batch, for each class.
        """
        pseudolabels = torch.argmax(confidences, dim=-1)
        return pseudolabels

    @staticmethod
    def compute_loss(
        confidences: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        criterion = nn.CrossEntropyLoss(reduction="mean")
        logits = torch.log(confidences).permute(0, 2, 1)
        return criterion(logits, labels)

    def pretrain_baseline(self, max_epochs: int) -> None:
        if self.baseline is None:
            return
        trainer = PreTrainer(
            self.baseline,
            self.labeled_loader,
            self.validation_loader,
            name="Baseline",
            device=self.device,
        )
        trainer.train(max_epochs)

    def pretrain(self, max_epochs: int) -> None:
        """
        Pretrain on the entire labeled dataset until convergence
        """
        # pretrain on entire labeled dataset
        subset = self.labeled_loader.dataset

        loader = DataLoader(subset, batch_size=self.max_batch_size, shuffle=True)

        trainer = PreTrainer(
            self.model,
            loader,
            self.validation_loader,
            name="Pseudo-Labeling Model",
            device=self.device,
        )
        trainer.train(max_epochs)

        # Train baseline on full labeled dataset
        self.pretrain_baseline(max_epochs)

    def train(self, num_epochs: int) -> None:
        self.model_IoU = self.validation_IoU(self.model)
        self.best_model_IoU = self.model_IoU
        self.best_model = self.model
        self.baseline_IoU = (
            self.validation_IoU(self.baseline) if self.baseline is not None else None
        )
        self.model.train()

        fine_tuner = FT(
            self.model, num_epochs, self.labeled_loader, self.unlabeled_loader
        )
        opt, scheduler = fine_tuner.optimizer, fine_tuner.scheduler

        for epoch in range(num_epochs):
            epoch_labeled_loss = 0.0
            epoch_unlabeled_loss = 0.0
            epoch_n = 0

            tic = time.time()
            for unlabeled, (labeled, labels) in zip(
                self.unlabeled_loader, self.labeled_loader
            ):
                unlabeled, labeled, labels = map(
                    lambda x: x.to(self.device), (unlabeled, labeled, labels)
                )

                self.alpha_schedule_step()
                opt.zero_grad()

                # predict on labeled data
                conf_labeled = self.model(labeled)
                # predict on unlabeled data
                conf_unlabeled = self.model(unlabeled)

                labeled_loss = self.compute_loss(conf_labeled, labels)

                # Compute Pseudolabels
                with torch.no_grad():
                    pseudolabels = conf_unlabeled.argmax(-1)

                # use pseudo-labels from previous time step
                unlabeled_loss = self.compute_loss(conf_unlabeled, pseudolabels)

                total_loss = labeled_loss + self.alpha * unlabeled_loss
                total_loss.backward()
                opt.step()

                # update epoch losses
                epoch_labeled_loss += labeled_loss.item()
                epoch_unlabeled_loss += unlabeled_loss.item()
                epoch_n += unlabeled.shape[0]

            scheduler.step()
            toc = time.time()

            # Bookkeeping
            # note this part is
            epoch_mean_labeled_loss = epoch_labeled_loss / (epoch_n)
            epoch_mean_unlabeled_loss = epoch_unlabeled_loss / (epoch_n)
            epoch_mean_loss = epoch_mean_labeled_loss + epoch_mean_unlabeled_loss

            # I0U
            train_accuracy = self.train_IoU(self.model)
            val_accuracy = self.validation_IoU(self.model)

            if val_accuracy > self.best_model_IoU:
                self.best_model_IoU = val_accuracy
                self.best_model_dict = self.model.state_dict().copy()

            # Everything below here is just logging
            self.wandb_log(
                {
                    "Epoch time": toc - tic,
                    "Epoch mean labeled loss": epoch_mean_labeled_loss,
                    "Epoch mean unlabeled loss": epoch_mean_unlabeled_loss,
                    "Epoch mean loss": epoch_mean_loss,
                }
            )
            self.wandb_log_named(
                {
                    "Train IoU": train_accuracy,
                    "Validation IoU": val_accuracy,
                },
                "Consistency PLabel",
            )
            if self.baseline is not None:
                self.wandb_log_named(
                    {"Best validation IoU": self.baseline_IoU}, "Baseline"
                )

    @staticmethod
    def wandb_init(
        num_epochs,
        batch_size,
        label_ratio,
    ):
        try:
            wandb.init(
                project="PLabel model",
                config={
                    "Number of epochs": num_epochs,
                    "Batch size": batch_size,
                    "Label ratio": label_ratio,
                },
            )
        except:
            pass

    def save_model(self, filename: str) -> None:
        """Save the model to the filename specified"""
        if self.best_model is None:
            raise ValueError("Model has not been trained yet")
        torch.save(self.best_model.state_dict(), filename)

    def save_baseline(self, filename: str) -> None:
        """Save the baseline model to the filename specified"""
        if not self.baseline:
            raise ValueError("No baseline model was specified")
        torch.save(self.baseline.state_dict(), filename)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model(x)
