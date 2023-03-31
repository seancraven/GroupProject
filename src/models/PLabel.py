import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from typing import Tuple, Iterable, Optional

from src.utils.datasets import balanced_minibatch_sizes
from src.utils.evaluation import evaluate_IoU
from src.utils.training import PreTrainer, FineTuner as FT
from functools import partial


class PLabel(nn.Module):
    """Psuedo-Label"""

    """
    To do:
    - complete init done
    - compute_pseudolabels() done
    - compute_loss()
    - pretrain() done
    - train()
    - wandb logging
    
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
            epoch_loss = 0.0
            
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
                epoch_loss += (1/len(self.labeled_loader))*self.compute_loss(conf_labeled, labels).sum() 
                epoch_loss += (1/len(self.unlabeled_loader))*self.compute_loss(conf_unlabeled, pseudolabels).sum()



   def _train_from_teacher(
        self,
        alpha: float,
        num_epochs: int,
        teacher: nn.Module,
        student: nn.Module,
        student_name: str = "Student",
    ) -> None:
        """
        TODO docstring
        """
        teacher.eval()
        student.train()

        # Create the optimizer and scheduler as specified in the paper
        fine_tuner = FT(student, num_epochs, self.labeled_loader, self.unlabeled_loader)
        opt, scheduler = fine_tuner.optimizer, fine_tuner.scheduler

        total_batches = min(len(self.labeled_loader), len(self.unlabeled_loader))
        def _dynamic_gamma(gamma: float, t: int) -> float:
            return gamma
            total_train_steps = num_epochs * total_batches
            return gamma * math.exp(5*(1 - (t / total_train_steps))**2)

        for epoch in range(num_epochs):
            epoch_dynamic_loss = 0.0
            epoch_standard_loss = 0.0

            tic = time.time()
            for t, (unlabeled, (labeled, labels)) in enumerate(
                zip(self.unlabeled_loader, self.labeled_loader)
            ):
                unlabeled, labeled, labels = map(
                    lambda x: x.to(self.device), (unlabeled, labeled, labels)
                )

                opt.zero_grad()
                # Compute the dynamic loss
                teacher_confidences = teacher(unlabeled)
                pseudolabels, mask = self.compute_pseudolabels(
                    teacher_confidences, alpha
                )
                student_confidences = student(unlabeled)
                with torch.no_grad():
                    # Pretty sure this needs no_grad, because otherwise the model
                    # seems to just learn to disagree with the teacher with high
                    # confidence in order to minimise the weights
                    weights = self.compute_weights(
                        pseudolabels,
                        mask,
                        teacher_confidences,
                        student_confidences,
                        gamma_1=_dynamic_gamma(
                            self.gamma_1_max,
                            epoch * total_batches + t
                        ),
                        gamma_2=_dynamic_gamma(
                            self.gamma_2_max,
                            epoch * total_batches + t
                        )
                    )
                # These lines do a sanity check
                student_labels = torch.argmax(student_confidences, dim=-1)
                DMT.sanity_check(
                    unlabeled,
                    pseudolabels,
                    student_labels,
                    mask,
                    weights,
                    0,
                    "test1.png",
                )
                DMT.sanity_check(
                    unlabeled,
                    pseudolabels,
                    student_labels,
                    mask,
                    weights,
                    1,
                    "test2.png",
                )

                dynamic_loss = self.compute_dynamic_loss(
                    student_confidences, pseudolabels, weights
                )
                # Compute the standard loss
                student_predictions = student(labeled)
                standard_loss = self.compute_standard_loss(student_predictions, labels)
                # Total loss and update
                total_loss = dynamic_loss +  standard_loss
                total_loss.backward()
                opt.step()

                epoch_dynamic_loss += dynamic_loss
                epoch_standard_loss += standard_loss
            scheduler.step()
            toc = time.time()

            # Bookkeeping
            epoch_mean_dynamic_loss = epoch_dynamic_loss / (t + 1)
            epoch_mean_standard_loss = epoch_standard_loss / (t + 1)
            student_train_accuracy = evaluate_IoU(
                student, self.labeled_loader, self.device
            )
            student_val_accuracy = evaluate_IoU(
                student, self.validation_loader, self.device
            )

            if student is self.model_a:
                if student_val_accuracy >= self.best_model_a_IoU:
                    self.best_model_a_IoU = student_val_accuracy
                    self.best_model_a_parameters = copy.deepcopy(student.state_dict())
            elif student is self.model_b:
                if student_val_accuracy >= self.best_model_b_IoU:
                    self.best_model_b_IoU = student_val_accuracy
                    self.best_model_b_parameters = copy.deepcopy(student.state_dict())

            # Everything below here is just logging
            debug_msg = "Epoch {}/{} of percentile {} completed in {:2f} secs."
            debug_msg_args = (epoch + 1, num_epochs, alpha, toc - tic)
            self.debug(debug_msg.format(*debug_msg_args))
            self.wandb_log(
                {
                    "Epoch time": toc - tic,
                    "Epoch mean dynamic loss": epoch_mean_dynamic_loss,
                    "Epoch mean standard loss": epoch_mean_standard_loss,
                }
            )
            self.wandb_log_named(
                {
                    "Train IoU": student_train_accuracy,
                    "Validation IoU": student_val_accuracy,
                },
                student_name,
            )
            self.wandb_log_named(
                {"Best validation IoU": self.best_model_a_IoU}, "Model A"
            )
            self.wandb_log_named(
                {"Best validation IoU": self.best_model_b_IoU}, "Model B"
            )
            if self.baseline is not None:
                self.wandb_log_named(
                    {"Best validation IoU": self.baseline_IoU}, "Baseline"
                )


        # TODO: maybe Load best parameters each model found on the validation set?
        loader = DataLoader(
            ConcatDataset(
                [self.labeled_loader.dataset, self.validation_loader.dataset]
            ),
            batch_size=self.max_batch_size,
            shuffle=True,
        )

        self.best_model = max(
            (self.model_a, self.model_b),
            key=lambda model: evaluate_IoU(model, loader, self.device),
        )