import copy
from typing import Tuple, Iterable, Optional
from functools import partial
import math
import time
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset

try:
    import wandb
except ImportError:
    pass


from src.utils.datasets import (
    balanced_minibatch_sizes,
    difference_maximized_sampling,
)
from src.utils.evaluation import evaluate_IoU
from src.utils.mixin import ReporterMixin
from src.utils.training import PreTrainer, FineTuner as FT


class DMT(nn.Module, ReporterMixin):
    """
    Dynamic Mixup Training (DMT) as described in the paper:
    https://arxiv.org/abs/2004.08514

    The DMT class is a wrapper around two models, model_a and model_b, which
    are trained in an alternating fashion. First the models are pre-trained on
    labeled data using the Difference Maximized Sampling (DMS). Then the models
    are running the DMT algorithm, which alternates between training model_a
    and model_b on a mix of labeled and unlabeled data. The models generate
    psuedo labels for the unlabeled data using the other model, and the
    psuedo labels are used to train the model on the unlabeled data and labeled.

    Args:
        model_a (nn.Module): The teacher first then student model
        model_b (nn.Module): The student first then teacher model
        labeled_dataset (Dataset): The labeled dataset
        unlabeled_dataset (Dataset): The unlabeled dataset
        validation_dataset (Dataset): The validation dataset
        max_batch_size (int): The maximum batch size to use for training
        gamma_1 (float): The exponent for weighting the unlabeled loss when models agree
        gamma_2 (float): The weight of the labeled loss when models disagree and teacher is more
                         confident than the student in the pseudo label
        baseline (Optional[nn.Module], optional): The baseline model to use for the DMS pre-training.
                                                  (Defaults to None).
        device (str, optional): The device to use for training. (Defaults to cuda if available).
        verbosity (int, optional): The verbosity level for training. (Defaults to 2).
    """

    def __init__(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        labeled_dataset: Dataset,
        unlabeled_dataset: Dataset,
        validation_dataset: Dataset,
        max_batch_size: int,
        gamma_1: float,
        gamma_2: float,
        baseline: Optional[nn.Module] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbosity: int = 2,
    ):
        super().__init__()
        self.model_a = model_a.to(device)
        self.model_b = model_b.to(device)
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
        self.gamma_1_max = gamma_1
        self.gamma_2_max = gamma_2
        self.device = device
        self.verbosity = verbosity

        # Create convenience functions to get the train IoU and validation IoU
        self.train_IoU, self.validation_IoU = map(
            lambda loader: partial(evaluate_IoU, data=loader, device=self.device),
            (self.labeled_loader, self.validation_loader),
        )
        # Set up the best model parameters
        self.best_model_a_IoU = -torch.inf
        self.best_model_b_IoU = -torch.inf
        self.best_model_a_parameters = None
        self.best_model_b_parameters = None

    def compute_pseudolabels(
        self, confidences: torch.Tensor, alpha: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes pseudolabels for the unlabeled data given confidences from a teacher model.

        Args:
            confidences (torch.Tensor): Tensor of shape (B, W*H, C) containing
                the confidences of the teacher model for each pixel in each
                image in the batch, for each class.
            alpha (float): percentage to use for computing the class thresholds.
                A pseudolabel is only assigned if the confidence of the predicted
                class is in the upper alpha% (e.g. the top 20%) of confidences 
                for that class across the batch.
        
        Returns:
            pseudolabels (torch.Tensor): Tensor of shape (B,W*H) containing the
                pseudolabels for each pixel in each image in the batch.
            mask (torch.Tensor): Tensor of shape (B,W*H) containing a boolean
                mask indicating for which pixels the pseudolabels are valid,
                based on the confidence threshold.
        """
        # Torch.quantile has to be done on-device
        quantile = torch.tensor(1 - alpha).to(self.device)
        # After flattening, confidences will be of shape (B*W*H, C), so
        # computing the quantile along the first dimension will give us per-class thresholds
        class_thresholds = torch.quantile(confidences.flatten(0, -2), quantile, dim=0)
        pseudolabels = torch.argmax(confidences, dim=-1)
        max_confidences, _ = torch.max(confidences, dim=-1)
        # Indexing into class_thresholds in this way finds the threshold for each
        # pixel based on the class it was assigned
        mask = max_confidences >= class_thresholds[pseudolabels]
        return pseudolabels, mask

    @staticmethod
    def compute_weights(
        pseudolabels: torch.Tensor,
        mask: torch.Tensor,
        student_class_confidences: torch.Tensor,
        teacher_class_confidences: torch.Tensor,
        gamma_1: float,
        gamma_2: float,
    ) -> torch.Tensor:
        """
        Computes the weights for the unlabeled loss based on the pseudo labels.
        Weights are determined by three cases:
            1. The models agree on the pseudolabel and the teacher is more confident
                than the student in the pseudolabel. The weight is probability of student predictiong
                the same class as the teacher, raised to the power gamma_1.
            2. The models disagree on the pseudolabel and the teacher is more confident. The weight is
                probability of student predictiong the same class as the teacher, raised to the power gamma_2.
            3. The models disagree on the pseudolabel and the student is more confident. The weight is 0.

        Args:
            pseudolabels (torch.Tensor): Tensor of shape (B, W*H) containing the pseudolabels.
            mask (torch.Tensor): Tensor of shape (B, W*H) containing a mask of pixels that should be
                                 assigned a weight. Pixels that are not assigned a pseudolabel are
                                 masked out. (case 3)
            student_class_confidences (torch.Tensor): Tensor of shape (B, W*H, C) containing the confidences
            teacher_class_confidences (torch.Tensor): Tensor of shape (B, W*H, C) containing the confidences
            gamma_1 (float): power to raise the student probability to for case 1.
            gamma_2 (float): power to raise the student probability to for case 2.

        Returns:
            weights (torch.Tensor): Tensor of shape (B, W*H) containing the weights for each pixel.
        """
        # Student labels are y_B in the paper
        student_labels = torch.argmax(student_class_confidences, dim=-1)
        # p_B is the probability the student assigns to the teacher's label
        # but it's tricky to get out by indexing so we do this grim masking thing
        teacher_max_confidences, _ = torch.max(
            teacher_class_confidences, dim=-1, keepdim=True
        )
        p_B_mask = teacher_max_confidences == teacher_class_confidences
        student_probs, _ = torch.max(student_class_confidences * p_B_mask, dim=-1)

        student_max_confidences, _ = torch.max(student_class_confidences, dim=-1)

        # Compute the weights
        teacher_max_confidences = teacher_max_confidences.squeeze(-1)
        agree_mask = pseudolabels == student_labels
        confidence_mask = teacher_max_confidences >= student_max_confidences
        weights = (
            torch.pow(student_probs, gamma_1) * agree_mask
            + torch.pow(student_probs, gamma_2) * confidence_mask * (~agree_mask)
        ) * mask

        return weights

    @staticmethod
    def compute_dynamic_loss(
        student_class_confidences: torch.Tensor,
        pseudolabels: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the dynamic loss for the unlabeled data. The loss is the cross entropy (CE)
        between the student's confidences and the pseudolabels, weighted by the weights
        computed in compute_weights.

        Args:
            student_class_confidences (torch.Tensor): Tensor of shape (B, W*H, C) containing the confidences
            pseudolabels (torch.Tensor): Tensor of shape (B, W*H) containing the pseudolabels.
            weights (torch.Tensor): Tensor of shape (B, W*H) containing the weights for each pixel.

        Returns:
            loss (torch.Tensor): The dynamic CE loss for the unlabeled data.
        """
        # The weights are set to zero at the locations we should ignore
        criterion = nn.CrossEntropyLoss(reduction="none")
        # Cross entropy expects logits with shape (B, C, W*H)
        logits = torch.log(student_class_confidences).permute(0, 2, 1)
        ce = criterion(logits, pseudolabels)
        loss = (weights * ce).mean()
        return loss

    @staticmethod
    def compute_standard_loss(
        student_class_confidences: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:

        """
        Computes the standard loss for the labeled data. The loss is the cross entropy
        between the student's confidences and the labels.

        Args:
            student_class_confidences (torch.Tensor): Tensor of shape (B, W*H, C) containing the confidences
            labels (torch.Tensor): Tensor of shape (B, W*H) containing the labels.

        Returns:
            loss (torch.Tensor): The standard CE loss for the labeled data.
        """
        criterion = nn.CrossEntropyLoss(reduction="none")
        logits = torch.log(student_class_confidences).permute(0, 2, 1)
        ce = criterion(logits, labels)
        loss = ce.mean()
        return loss

    def pretrain_baseline(self, max_epochs: int) -> None:
        """
        Trains the baseline model on the labeled data.

        If the baseline is not specified, it returns None.
        Otherwise, it trains the baseline model on the labeled data
        for max epochs or until convergence of IoU on the validation set.

        Args:
            max_epochs (int): The maximum number of epochs to train for.
        """
        if self.baseline is None:
            return
        # Train baseline on labeled data (see training.py)
        trainer = PreTrainer(
            self.baseline,
            self.labeled_loader,
            self.validation_loader,
            name="Baseline",
            device=self.device,
        )
        trainer.train(max_epochs)

    def pretrain(self, max_epochs: int, proportion: float = 0.5) -> None:
        """
        Trains the teacher (model_a) and student (model_b) models on the labeled data.
        The DMS aims to maximize the difference between the subsets (A and B)
        of the labeled data for which the teacher and student are trained.

        Args:
            max_epochs (int): The maximum number of epochs to train for.
            proportion (float): The proportion of the labeled data to use for training the teacher.
        """
        subset_a, subset_b = difference_maximized_sampling(
            self.labeled_loader.dataset, proportion=proportion
        )
        loader_a, loader_b = map(
            lambda x: DataLoader(x, batch_size=self.max_batch_size, shuffle=True),
            (subset_a, subset_b),
        )

        # Train models A/B on subsets A/B
        for model, loader, name in zip(
            (self.model_a, self.model_b),
            (loader_a, loader_b),
            ("Model A", "Model B"),
        ):
            trainer = PreTrainer(
                model,
                loader,
                self.validation_loader,
                name=name,
                device=self.device,
            )
            trainer.train(max_epochs)

        # Train baseline on full labeled dataset
        self.pretrain_baseline(max_epochs)

    def _train_from_teacher(
        self,
        alpha: float,
        num_epochs: int,
        teacher: nn.Module,
        student: nn.Module,
        student_name: str = "Student",
    ) -> None:
        """
        Trains the student model from the teacher model. The student is trained
        on the labeled data and the unlabeled data. The student is trained on
        the unlabeled data using the dynamic loss. The teacher generates the
        pseudo labels for the unlabeled data. The student is trained on the
        labeled data and the pseudo labels using the loss as combination of the
        standard loss and the dynamic loss.

        Args:
            alpha (float): The percentile of the pseudo labels to use for training the student.
            num_epochs (int): The number of epochs to train for.
            teacher (nn.Module): The teacher model.
            student (nn.Module): The student model.
            student_name (str): The name of the student model. Used for WandB logging and
                debug output.
        """
        teacher.eval()
        student.train()

        # Create the optimizer and scheduler as specified in the paper
        fine_tuner = FT(student, num_epochs, self.labeled_loader, self.unlabeled_loader)
        opt, scheduler = fine_tuner.optimizer, fine_tuner.scheduler
        # The number of batches in the unlabeled and labeled loaders
        total_batches = min(len(self.labeled_loader), len(self.unlabeled_loader))

        # dynamic gamma function for powers of weights as in the paper.
        # This is currently not used since it is not used in the source code for
        # the original DMT paper.
        def _dynamic_gamma(gamma: float, t: int) -> float:
            # comment the below return to apply dynamic gamma
            return gamma
            total_train_steps = num_epochs * total_batches
            return gamma * math.exp(5 * (1 - (t / total_train_steps)) ** 2)

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
                # Attain all attributes for the dynamic loss
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
                            self.gamma_1_max, epoch * total_batches + t
                        ),
                        gamma_2=_dynamic_gamma(
                            self.gamma_2_max, epoch * total_batches + t
                        ),
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
                # Compute the dynamic loss
                dynamic_loss = self.compute_dynamic_loss(
                    student_confidences, pseudolabels, weights
                )
                # Compute the standard loss
                student_predictions = student(labeled)
                standard_loss = self.compute_standard_loss(student_predictions, labels)
                # Total loss and update
                total_loss = dynamic_loss + standard_loss
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

    def dynamic_train(self, percentiles: Iterable[float], num_epochs: int) -> None:
        """
        Trains the model following the DMT training procedure.
        See Algorithm 2 in the paper for details (https://arxiv.org/pdf/2004.08514.pdf).

        Args:
            percentiles (Iterable[float]): The percentiles of pseudo labels for
                                           which to use in the training procedure.
                                           Increases the amount of pseudo labels iteratively.
            num_epochs (int): The number of epochs to train for.
        """

        # Potentially swap the models so that model A is the better one, since
        # model A teaches model B first
        model_a_IoU, model_b_IoU = map(
            self.validation_IoU, (self.model_a, self.model_b)
        )
        if model_b_IoU > model_a_IoU:
            self.debug("Swapping models A and B")
            self.model_a, self.model_b = self.model_b, self.model_a
            model_a_IoU, model_b_IoU = model_b_IoU, model_a_IoU
        self.best_model_a_IoU = model_a_IoU
        self.best_model_b_IoU = model_b_IoU
        self.best_model_a_parameters = copy.deepcopy(self.model_a.state_dict())
        self.best_model_b_parameters = copy.deepcopy(self.model_b.state_dict())

        self.baseline_IoU = (
            self.validation_IoU(self.baseline) if self.baseline is not None else None
        )
        # Train the models on increasingly more pseudo labels
        # As models are trained, we expect less pseudo noise, so we can
        # use more pseudo labels without hurting performance.
        for alpha in percentiles:
            self._train_from_teacher(
                alpha=alpha,
                num_epochs=num_epochs,
                teacher=self.model_a,
                student=self.model_b,
                student_name="Model B",
            )
            self._train_from_teacher(
                alpha=alpha,
                num_epochs=num_epochs,
                teacher=self.model_b,
                student=self.model_a,
                student_name="Model A",
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

    @staticmethod
    def wandb_init(
        percentiles,
        num_epochs,
        batch_size,
        label_ratio,
        difference_maximized_proportion,
        gamma_1,
        gamma_2,
    ):
        try:
            wandb.init(
                project="DMT model",
                config={
                    "Percentiles": percentiles,
                    "Number of epochs": num_epochs,
                    "Batch size": batch_size,
                    "Label ratio": label_ratio,
                    "Difference maximized sampling proportion": difference_maximized_proportion,
                    "Gamma 1": gamma_1,
                    "Gamma 2": gamma_2,
                },
            )
        except:
            pass

    @staticmethod
    def sanity_check(
        images: torch.Tensor,
        pseudolabels: torch.Tensor,
        student_labels: torch.Tensor,
        masks: torch.Tensor,
        weights: torch.Tensor,
        idx: int,
        filename: str,
    ) -> None:
        """
        A sanity check to make sure the pseudolabels are being computed correctly.
        Could also prove useful for producing images for the report.

        Args:
            images (torch.Tensor): Tensor of shape (B, C, W, H) containing the images,
                where B is the batch size, C is the number of channels, W is the width,
                and H is the height.
            pseudolabels (torch.Tensor): Tensor of shape (B, W*H) containing the
                pseudolabels for each pixel in each image in the batch, *without*
                mask applied.
            student_labels (torch.Tensor): Tensor of shape (B, W*H) containing the
                student's labels for each pixel in each image in the batch.
            masks (torch.Tensor): Tensor of shape (B, W*H) containing the masks for
                the pseudolabels.
            weights (torch.Tensor): Tensor of shape (B, W*H) containing the weights
                for the pseudolabels.
            idx (int): Index of the image to plot.
            filename (str): Filename to save the plot to.
        """
        # Comment the line below out to actually produce the sanity check images.
        # Disabled since we now believe the pseudolabels are being computed correctly.
        return
        image = images[idx].cpu().detach().permute(1, 2, 0).numpy()
        image_shape = (image.shape[0], image.shape[1])
        # Make pseudolabel take values 0, 1, 2 instead of 0, 1
        pseudolabel = 1 + pseudolabels[idx].cpu().detach().numpy().reshape(image_shape)
        student_label = student_labels[idx].cpu().detach().numpy().reshape(image_shape)
        mask = masks[idx].cpu().detach().numpy().reshape(image_shape)
        pseudolabel_with_mask = pseudolabel * mask
        weight = weights[idx].cpu().detach().numpy().reshape(image_shape)
        fig, ax = plt.subplots(1, 6)
        ax[0].imshow(image)
        ax[0].set_title("Image")
        ax[1].imshow(pseudolabel)
        ax[1].set_title("Pseudolabels")
        ax[2].imshow(pseudolabel_with_mask)
        ax[2].set_title("Masked pseudolabels")
        ax[3].imshow(student_label)
        ax[3].set_title("Student labels")
        ax[4].imshow(weight)
        ax[4].set_title("Weight")
        ax[5].imshow(pseudolabel * weight)
        ax[5].set_title("Weighted pseudolabels")
        for i in range(6):
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        fig.savefig(filename)
        plt.close(fig)

    def save_best_model(self, filename: str) -> None:
        """Save the best model to the filename specified"""
        if self.best_model is None:
            raise ValueError("Model has not been trained yet")
        torch.save(self.best_model.state_dict(), filename)

    def save_baseline(self, filename: str) -> None:
        """Save the baseline model to the filename specified"""
        if not self.baseline:
            raise ValueError("No baseline model was specified")
        torch.save(self.baseline.state_dict(), filename)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the best model"""
        if self.best_model is None:
            raise ValueError("Model has not been trained yet")
        return self.best_model(x)
