import copy
import time
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Union

from src.utils.evaluation import evaluate_IoU
from src.utils.misc import ReporterMixin


class EarlyStopping(ReporterMixin):
    """ Quick and dirty implementation of early stopping using validation metric. """
    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        """
        Args:
            patience: Number of epochs to wait without seeing any improvement
                before stopping.
            min_delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.bad_epochs = 0
        self.best_validation: Union[float, torch.Tensor] = -torch.inf
        self.best_parameters: Union[None, dict] = None

    def monitor(self, model: nn.Module, validation: float | torch.Tensor) -> None:
        """
        Monitors the model's validation performance and updates the best validation
        metric/parameters if necessary.

        Args:
            model: The model to monitor.
            validation: The validation metric to use for early stopping.
        """
        if validation >= self.best_validation + self.min_delta:
            self.best_validation = validation
            # Has to be a deep copy, or you get shallow copies of the 
            # state dict and the best parameters just track the model's
            # current parameters
            self.best_parameters = copy.deepcopy(model.state_dict())
            self.bad_epochs = 0
            self.debug(f'Best validation {validation:.4f}, updated best params')
        else:
            self.bad_epochs += 1

    @property
    def should_stop(self) -> bool:
        """ Returns True if the training should stop, False otherwise. """
        return self.bad_epochs >= self.patience

    def restore_best_parameters(self, model: nn.Module) -> None:
        """
        Restores the model's parameters to the best parameters seen during training
        as judged by the validation metric.

        Args:
            model: The model to restore.
        """
        if self.best_parameters is None:
            raise RuntimeError("No best parameters to restore")
        model.load_state_dict(self.best_parameters)


class PreTrainer(ReporterMixin):
    """ Pre-trains a model until the validation IoU stops improving """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        name = '',
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        verbosity: int = 2
    ) -> None:
        super().__init__()
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.name = name
        self.device = device
        self.verbosity = verbosity
        
        # The DMT paper doesn't say how to pretrain models, they just use models
        # pretrained elsewhere. We'll use Adam with default hyperparameters,
        # as well as a learning rate scheduler and early stopping.
        # Note that patience of the scheduler should be smaller than the patience
        # of early stopping, since otherwise the scheduler never gets chance
        # to reduce the learning rate.
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss_scheduler = ReduceLROnPlateau(self.optimizer, patience=5, verbose=True)
        self.IoU_scheduler = ReduceLROnPlateau(self.optimizer, mode='max', patience=10, verbose=True)
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(patience=20)


    def train(self, max_epochs: int) -> None:
        """
        Trains self.model for at most max_epochs, stopping early if the IoU on
        the validation set stops improving.

        Args:
            max_epochs: Maximum number of epochs to train for.
        
        Returns:
            None (but the model is modified by reference)
        """
        self.info(f'\n===== Pretraining {self.name} =====')
        for epoch in range(max_epochs):
            # Perform a loop over the training set
            self.model.train()
            epoch_loss = 0.0
            tic = time.time()
            for i, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)  # Of shape (B, W*H, C)
                # Cross-entropy needs logits, and inputs of shape (B, C, W*H)
                # and targets has shape (B, 1, W*H)
                logits = torch.log(outputs).permute(0,2,1)
                targets = targets.squeeze(1)
                loss = self.criterion(logits, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
            toc = time.time()

            # Evaluate metrics for monitoring
            epoch_mean_loss = epoch_loss / len(self.train_loader)
            self.model.eval()
            validation_IoU = evaluate_IoU(self.model, self.val_loader, self.device)
            train_IoU = evaluate_IoU(self.model, self.train_loader, self.device)

            # Change the learning rate if needed
            epoch_mean_loss = epoch_loss / len(self.train_loader)
            self.loss_scheduler.step(epoch_mean_loss)
            self.IoU_scheduler.step(validation_IoU)


            self.early_stopping.monitor(self.model, validation_IoU)

            self.info(f'Epoch {epoch+1}/{max_epochs}')
            self.wandb_log_named({
                    "Epoch time": toc-tic,
                    "Epoch mean loss": epoch_mean_loss,
                    "Validation IoU": validation_IoU,
                    "Train IoU": train_IoU,
                    "Best validation IoU": self.early_stopping.best_validation,
                },
                self.name
            )

            # Check if we should stop early, and if so restore the best parameters
            if self.early_stopping.should_stop:
                self.info(
                    f"Stopping early after {epoch+1} epochs; no improvement in "
                    f"validation IoU after {self.early_stopping.patience} epochs."
                )
                break
        else:
            # Only gets executed if we didn't break out of the for loop
            self.info(f"Finished training after {max_epochs} epochs.")
        
        # End of training; restore the best parameters of the model
        # according to the validation IoU
        self.info("Restored best parameters.")
        self.early_stopping.restore_best_parameters(self.model)


class FineTuner:
    """
    Helper class to create the optimizer and learning rate scheduler for the
    'fine-tuning' stage of DMT
    """
    # The values below are from the DMT paper
    DEFAULT_LR = 4e-3
    DEFAULT_MOMENTUM = 0.9
    def __init__(
        self,
        model: nn.Module,
        no_epochs: int,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader
    ) -> None:
        # In the paper they just use SGD with learning rate 4e-3 and momentum 0.9
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.DEFAULT_LR,
            momentum=self.DEFAULT_MOMENTUM
        )
        no_minibatches = min(len(labeled_loader), len(unlabeled_loader)) * no_epochs
        # This is the scheduler they use in the DMT code in the official repo
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda x: (1 - x / no_minibatches) ** 0.9
        )