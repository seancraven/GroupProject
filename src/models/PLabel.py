import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from typing import Tuple, Iterable, Optional

from src.utils.datasets import balanced_minibatch_sizes
from src.utils.evaluation import evaluate_IoU
from src.utils.training import PreTrainer, FineTuner as FT
from functools import partial

import time

import wandb


class PLabel(nn.Module):
    """Psuedo-Label"""
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
    ):
        super().__init__()
        self.model = model.to(device)
        self.baseline = baseline.to(device) if baseline is not None else None

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
        self.model_IoU = -torch.inf

    def compute_pseudolabels(
        self, confidences: torch.Tensor, alpha: float
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
        self,
        confidences: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        criterion = nn.CrossEntropyLoss(reduction="none")
        logits = torch.log(confidences).permute(0, 2, 1)
        ce = criterion(logits, labels)
        loss = ce.mean()
        return loss
    
    def pretrain_baseline(self, max_epochs: int) -> None:
        if self.baseline is None:
            return
        trainer = PreTrainer(
            self.baseline, self.labeled_loader, self.validation_loader, name="Baseline", device=self.device
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
            self.model, loader, self.validation_loader, name="Pseudo-Labeling Model", device=self.device
        )
        trainer.train(max_epochs)

        # Train baseline on full labeled dataset
        self.pretrain_baseline(max_epochs)

    def train(self, num_epochs: int) -> None:
        self.model_IoU = self.validation_IoU(self.model)
        self.baseline_IoU = (
            self.validation_IoU(self.baseline) if self.baseline is not None else None
        )
        self.model.train()

        fine_tuner = FT(self.model, num_epochs, self.labeled_loader, self.unlabeled_loader)
        opt, scheduler = fine_tuner.optimizer, fine_tuner.scheduler

        total_batches = min(len(self.labeled_loader), len(self.unlabeled_loader))

        for epoch in range(num_epochs):

            epoch_labeled_loss = 0.0
            epoch_unlabeled_loss = 0.0
            
            tic = time.time()
            for t, (unlabeled, (labeled, labels)) in enumerate(
                zip(self.unlabeled_loader, self.labeled_loader)
            ):
                unlabeled, labeled, labels = map(
                    lambda x: x.to(self.device), (unlabeled, labeled, labels)
                )

                opt.zero_grad()

                # predict on labeled data
                conf_labeled = self.model(labeled)
                # predict on unlabeled data
                conf_unlabeled = self.model(unlabeled)
                pseudolabels = self.compute_pseudolabels(conf_unlabeled)
                labeled_loss += self.compute_loss(conf_labeled, labels).sum() 
                unlabeled_loss += self.compute_loss(conf_unlabeled, pseudolabels).sum()
                total_loss = labeled_loss + unlabeled_loss
                total_loss.backward()
                opt.step()

                # update epoch losses
                epoch_labeled_loss += labeled_loss.item()
                epoch_unlabeled_loss += unlabeled_loss.item()
            
            scheduler.step()
            toc = time.time()

            # Bookkeeping
            epoch_mean_labeled_loss = epoch_labeled_loss / (t + 1)
            epoch_mean_unlabeled_loss = epoch_unlabeled_loss / (t + 1)
            epoch_mean_loss = epoch_mean_labeled_loss + epoch_mean_unlabeled_loss

            # I0U
            train_accuracy = evaluate_IoU(
                self.model, self.labeled_loader, self.device
            )
            val_accuracy = evaluate_IoU(
                self.model, self.validation_loader, self.device
            )

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
                'Consitency PLabel',
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
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        torch.save(self.model.state_dict(), filename)

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