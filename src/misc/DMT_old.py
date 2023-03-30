import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.ioff()
import time
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from torch.utils.data import DataLoader, Dataset, Subset
from typing import Tuple, Iterable, Optional


# Networks need to ouput shape (B, W*H, 2) with probabilities in the classes
class DMT(nn.Module):
    """An implementation of dynamic mutual training."""

    def __init__(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        optimizer_a: optim.Optimizer,
        optimizer_b: optim.Optimizer,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        gamma_1: float,
        gamma_2: float,
        use_validation: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbosity: int = 2,
        baseline_model: Optional[nn.Module] = None,
        baseline_optimizer: Optional[optim.Optimizer] = None,
    ):
        """
        Note that the dataloaders should be configured such that they return
        the same number of batches in total (or as close as possible). Each
        training step of DMT constructs a loss from a batch of labeled data
        and a batch of pseudolabeled data, so the number of batches in each
        must be the same or as close as possible.

        Additionally, the models are expected to give outputs of shape (B, W*H, C),
        where B is the batch size, W is the image width, H is the image height,
        and C is the number of classes. For our purposes C should be 2, but I
        think the code should work for any C.

        Args:
            model_a (nn.Module): Model to train on labeled data.
            model_b (nn.Module): Model to train on unlabeled data.
            optimizer_a (optim.Optimizer): Optimizer for model_a.
            optimizer_b (optim.Optimizer): Optimizer for model_b.
            labeled_loader (DataLoader): Dataloader for labeled data.
            unlabeled_loader (DataLoader): Dataloader for unlabeled data. This
                should *just* return the unlabeled data, nothing else.
            gamma_1 (float) : Controls the weight of the dynamic loss.
            gamma_2 (float) : Controls the weight of the dynamic loss.
            device (str): Device to train on. Defaults to cuda if available.
            verbosity (int): Verbosity level. 0: no logging, 1: information, 2: debug.
                Currently just prints to stdout, but could be changed to use a logger.
        """
        super().__init__()
        self.model_a = model_a.to(device)
        self.model_b = model_b.to(device)
        self.optimizer_a = optimizer_a
        self.optimizer_b = optimizer_b

        self.validation_loader = None
        if use_validation:
            labeled_dataset = labeled_loader.dataset
            validation_upper_bound = int(0.05 * len(labeled_dataset))
            validation_dataset = Subset(labeled_dataset, range(validation_upper_bound))
            training_dataset = Subset(
                labeled_dataset, range(validation_upper_bound, len(labeled_dataset))
            )
            self.labeled_loader = DataLoader(
                training_dataset, batch_size=labeled_loader.batch_size
            )
            self.validation_loader = DataLoader(
                validation_dataset, batch_size=labeled_loader.batch_size
            )
        else:
            self.labeled_loader = labeled_loader

        self.unlabeled_loader = unlabeled_loader
        self.gamma_1_max = gamma_1
        self.gamma_2_max = gamma_2
        self.device = device
        self.verbosity = verbosity
        self.baseline_model = baseline_model.to(device)
        self.baseline_optimizer = baseline_optimizer

        self.best_model = None

    def wandb_log(self, *args, **kwargs) -> None:
        """Logs to wandb if it's available."""
        # Wandb statement needs to be a dict, e.g.
        # wandb.log({"loss": loss.item()})
        try:
            wandb.log(*args, **kwargs)
        except:  # Blanket except is bad!
            pass

    def debug(self, msg: str) -> None:
        """Prints a debug message if verbosity is >= 2"""
        if self.verbosity >= 2:
            print(msg)

    def info(self, msg: str) -> None:
        """Prints an info message if verbosity is >= 1"""
        if self.verbosity >= 1:
            print(msg)

    def pretrain(
        self, num_epochs: int, batch_size: int, proportion: float = 0.5
    ) -> None:
        """
        Pretrains the models on the labeled data.

        Args:
            num_epochs (int): Number of epochs to train for.
            batch_size (int): Batch size to use.
            proportion (float): Proportion of the labeled data to use for each model.
        """
        subset_a, subset_b = self.difference_maximized_sampling(
            self.labeled_loader.dataset, proportion=proportion
        )

        train_loader_a, train_loader_b = map(
            lambda subset: DataLoader(subset, batch_size=batch_size, num_workers=2),
            (subset_a, subset_b),
        )

        criterion = nn.CrossEntropyLoss()

        def _pretrain(model, opt, loader):
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                tic = time.time()
                for batch in loader:
                    images, labels = batch
                    images, labels = images.to(self.device), labels.to(self.device)

                    opt.zero_grad()
                    # loss = self.compute_standard_loss(model(images), labels)
                    logits = torch.log(model(images)).permute(0, 2, 1)
                    loss = criterion(logits, labels.flatten(1))
                    loss.backward()
                    opt.step()

                    epoch_loss += loss
                toc = time.time()
                epoch_loss /= len(loader)
                self.debug(
                    f"Epoch {epoch + 1}/{num_epochs} completed in {toc-tic:.2f} secs. Model loss: {epoch_loss}"
                )

        self.debug("Pretraining model A...")
        _pretrain(self.model_a, self.optimizer_a, train_loader_a)
        self.debug("Pretraining model B...")
        _pretrain(self.model_b, self.optimizer_b, train_loader_b)
        if self.baseline_model:
            self.debug("Pretraining baseline model...")
            # Use the whole dataset for this one
            _pretrain(self.baseline_model, self.baseline_optimizer, self.labeled_loader)

    def compute_pseudolabels(
        self, confidences: torch.Tensor, alpha: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes pseudolabels for the unlabeled data given confidences from a teacher model.

        Args:
            confidences (torch.Tensor): Tensor of shape (B, W*H, C) containing the
                confidences of the teacher model for each pixel in each image in the batch.
            alpha (float): percentage to use for computing the class thresholds.
                A pseudolabel is only assigned if the confidence of the predicted
                class is in the upper (1-alpha)% of confidences for that class
                across the batch.
        """
        # Confidences has shape (B, W * H, C)
        quantile = torch.tensor(1 - alpha).to(self.device)
        class_thresholds = torch.quantile(confidences.flatten(0, -2), quantile, dim=0)
        pseudolabels = torch.argmax(confidences, dim=-1)
        max_confidences, _ = torch.max(confidences, dim=-1)
        mask = max_confidences > class_thresholds[pseudolabels]
        return pseudolabels, mask

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

    @staticmethod
    def compute_weights(
        pseudolabels: torch.Tensor,  # Shape (B, W*H)
        mask: torch.Tensor,  # Shape (B, W*H)
        student_per_class_confidences: torch.Tensor,  # Shape (B, W*H, C)
        teacher_per_class_confidences: torch.Tensor,  # Shape (B, W*H, C)
        gamma_1: float,
        gamma_2: float,
    ) -> torch.Tensor:
        """
        Computes the weights for the pseudolabels.

        Args:
            pseudolabels (torch.Tensor): Tensor of shape (B, W*H) containing the
                pseudolabels for each pixel in each image in the batch, *without*
                mask applied.
            mask (torch.Tensor): Tensor of shape (B, W*H) containing the masks for
                the pseudolabels.
            student_per_class_confidences (torch.Tensor): Tensor of shape (B, W*H, C)
                containing the confidences of the student model for each pixel in
                each image in the batch.
            teacher_per_class_confidences (torch.Tensor): Tensor of shape (B, W*H, C)
                containing the confidences of the teacher model for each pixel in
                each image in the batch.
            gamma_1 (float): Weight for when the models agree on the pixel's class.
            gamma_2 (float): Weight for when the teacher model has a higher confidence.

        Returns:
            weights (torch.Tensor): Tensor of shape (B, W*H) containing the weights
                for the pseudolabels.
        """
        # Find teacher and student labels, quantities y_a/y_b in the paper
        teacher_labels = torch.argmax(
            teacher_per_class_confidences, dim=-1
        )  # Shape (B, W*H
        student_labels = torch.argmax(
            student_per_class_confidences, dim=-1
        )  # Shape (B, W*H)
        # Now we have to do some grim stuff to get p_B
        teacher_max_confidences, _ = torch.max(
            teacher_per_class_confidences, dim=-1, keepdim=True
        )  # Shape (B, W*H)
        p_B_mask = teacher_max_confidences == teacher_per_class_confidences
        # The student probs are the probability the student assigns to the
        # teacher's labels
        student_probs, _ = torch.max(student_per_class_confidences * p_B_mask, dim=-1)
        student_max_confidences, _ = torch.max(student_per_class_confidences, dim=-1)

        # Compute the weights
        teacher_max_confidences = teacher_max_confidences.squeeze(-1)
        agree_mask = teacher_labels == student_labels
        confidence_mask = teacher_max_confidences >= student_max_confidences
        weights = (
            torch.zeros_like(pseudolabels)
            + torch.pow(student_probs, gamma_1) * agree_mask
            + torch.pow(student_probs, gamma_2) * confidence_mask * (~agree_mask)
        ) * mask

        return weights

    @staticmethod
    def compute_dynamic_loss(
        student_confidences: torch.Tensor,  # Shape (B, W*H, C)
        teacher_confidences: torch.Tensor,  # Shape (B, W*H, C)
        weights: torch.Tensor,  # Shape (B, W*H)
    ) -> torch.Tensor:
        labels = torch.argmax(teacher_confidences, dim=-1)
        criterion = nn.CrossEntropyLoss(reduction="none")
        logits = torch.log(student_confidences).permute(0, 2, 1)
        ce = criterion(logits, labels.flatten(1))
        loss = (weights * ce).mean()
        return loss

    @staticmethod
    def compute_standard_loss(student_predictions: torch.Tensor, labels: torch.Tensor):
        """
        Computes the standard cross entropy loss.

        Args:
            student_predictions (torch.Tensor): Tensor of shape (B, W*H, C) containing
                the predictions of the student model.
            labels (torch.Tensor): Tensor of shape (B, W*H) containing the labels for
                each pixel in each image in the batch.
        """
        criterion = nn.CrossEntropyLoss(reduction="none")
        logits = torch.log(student_predictions).permute(0, 2, 1)
        ce = criterion(logits, labels.flatten(1))
        loss = ce.mean()
        return loss

    def evaluate_IoU(self, net: nn.Module, data: DataLoader) -> float:
        """
        Evaluates the IoU score of the model on the given data.
        """
        net.eval()
        score = 0
        seen_images = 0
        for images, labels in data:
            images = images.to(self.device)
            labels = labels.to(self.device).flatten(1)
            predictions = net(images).argmax(dim=-1)
            intersection = (torch.logical_and(predictions == 1, labels == 1)).sum()
            union = (torch.logical_or(predictions == 1, labels == 1)).sum()
            IoU = intersection / union  # Got this propoortion correct on this batch
            score += (
                IoU * images.shape[0]
            )  # Needs to be a weighted sum bc the last batch might be smaller
            seen_images += images.shape[0]
        return (score / seen_images).item()

    def train(
        self,
        percentiles: Iterable[float],
        num_epochs: int,
        batch_size: int,
        label_ratio: float,  # Just used for logging to wandb
        skip_pretrain: bool = False,
    ) -> None:
        """
        TODO docstring
        """
        wandb.init(
            project="DMT model",
            config={
                "Percentiles": percentiles,
                "Number of epochs": num_epochs,
                "Batch size": batch_size,
                "Label ratio": label_ratio,
                "Gamma 1": self.gamma_1_max,
                "Gamma 2": self.gamma_2_max,
            },
        )

        if not skip_pretrain:
            self.pretrain(num_epochs=10, batch_size=batch_size, proportion=0.7)
            torch.save(self.model_a.state_dict(), "DMT_model_a.pt")
            torch.save(self.model_b.state_dict(), "DMT_model_b.pt")
            if self.baseline_model:
                torch.save(self.baseline_model.state_dict(), "DMT_baseline.pt")

        if self.validation_loader:
            if self.baseline_model:
                self.debug(
                    f"Baseline accuracy before training: {self.evaluate_IoU(self.baseline_model, self.validation_loader):.4f}"
                )
            self.debug(
                f"Model A accuracy before training: {self.evaluate_IoU(self.model_a, self.validation_loader):.4f}"
            )
            self.debug(
                f"Model B accuracy before training: {self.evaluate_IoU(self.model_b, self.validation_loader):.4f}"
            )

        def _train_from_teacher(
            teacher,
            student,
            opt_student,
            alpha,
            student_name="Student",
            train_baseline=False,
        ):
            # If train_baseline is true, we train the baseline as well just on the labeled data.
            # This is so the baseline 'sees' the labeled data as much as a student does,
            # so we can see if DMT is actually helping.
            teacher.eval()
            student.train()

            self.debug(f"Beginning dynamic mutual training on percentile {alpha}")
            if self.validation_loader:
                self.debug(
                    f"Student accuracy before training: {self.evaluate_IoU(student, self.validation_loader):.4f}"
                )

            for epoch in range(num_epochs):
                epoch_dynamic_loss = 0.0
                epoch_standard_loss = 0.0
                tic = time.time()
                for i, (unlabeled, (labeled, labels)) in enumerate(
                    zip(self.unlabeled_loader, self.labeled_loader)
                ):
                    unlabeled = unlabeled.to(self.device)
                    labeled = labeled.to(self.device)
                    labels = labels.to(self.device)

                    opt_student.zero_grad()
                    # Compute the dynamic loss
                    teacher_confidences = teacher(unlabeled)
                    pseudolabels, mask = self.compute_pseudolabels(
                        teacher_confidences, alpha
                    )
                    student_confidences = student(unlabeled)
                    with torch.no_grad():
                        # This needs to be no_grad, because otherwise
                        # the model learns to disagree with the teacher with high
                        # confidence to get low weights
                        weights = self.compute_weights(
                            pseudolabels,
                            mask,
                            student_confidences,
                            teacher_confidences,
                            # TODO see if we're meant to decrease gamma over time?
                            gamma_1=self.gamma_1_max,
                            gamma_2=self.gamma_2_max,
                        )
                    student_labels = torch.argmax(student_confidences, dim=-1)
                    # The below lines output some images for sanity checks
                    # DMT.sanity_check(unlabeled, pseudolabels, student_labels, mask, weights, 0, 'test1.png')
                    # DMT.sanity_check(unlabeled, pseudolabels, student_labels, mask, weights, 1, 'test2.png')
                    dynamic_loss = self.compute_dynamic_loss(
                        student_confidences, teacher_confidences, weights
                    )
                    # Compute the standard loss
                    student_predictions = student(labeled)
                    standard_loss = self.compute_standard_loss(
                        student_predictions, labels
                    )
                    # Compute total loss and update
                    total_loss = dynamic_loss + standard_loss
                    total_loss.backward()
                    opt_student.step()

                    if train_baseline and self.baseline_model:
                        # Train the baseline only if we're told do and we have a baseline model
                        self.baseline_optimizer.zero_grad()
                        baseline_predictions = self.baseline_model(labeled)
                        baseline_loss = self.compute_standard_loss(
                            baseline_predictions, labels
                        )
                        baseline_loss.backward()
                        self.baseline_optimizer.step()

                    # Epoch bookkeeping
                    epoch_dynamic_loss += dynamic_loss
                    epoch_standard_loss += standard_loss

                toc = time.time()
                # i is the number of batches - 1
                epoch_dynamic_loss /= {i + 1}
                epoch_standard_loss /= {i + 1}

                debug_msg = "Epoch {}/{} completed in {:.2f} secs.\n\tDynamic loss: {:.4f}. Standard loss: {:.4f}."
                debug_msg_args = [
                    epoch + 1,
                    num_epochs,
                    toc - tic,
                    epoch_dynamic_loss,
                    epoch_standard_loss,
                ]

                student_train_accuracy = self.evaluate_IoU(student, self.labeled_loader)
                self.wandb_log(
                    {
                        "Epoch time": toc - tic,
                        "Epoch mean dynamic loss": epoch_dynamic_loss,
                        "Epoch mean standard loss": epoch_standard_loss,
                        f"{student_name} train accuracy": student_train_accuracy,
                    }
                )
                if self.validation_loader:
                    debug_msg += "\n\tStudent validation accuracy: {:.4f}."
                    student_validation_accuracy = self.evaluate_IoU(
                        student, self.validation_loader
                    )
                    debug_msg_args.append(student_validation_accuracy)
                    self.wandb_log(
                        {
                            f"{student_name} validation accuracy": student_validation_accuracy,
                        }
                    )
                    if self.baseline_model:
                        debug_msg += " Baseline validation accuracy: {:.4f}."
                        baseline_validation_accuracy = self.evaluate_IoU(
                            self.baseline_model, self.validation_loader
                        )
                        debug_msg_args.append(baseline_validation_accuracy)
                        if (
                            train_baseline
                        ):  # Only log the baseline accuracy if we trained it
                            self.wandb_log(
                                {
                                    "Baseline validation accuracy": baseline_validation_accuracy,
                                }
                            )

                self.debug(debug_msg.format(*debug_msg_args))

        for alpha in percentiles:
            # Only train the baseline on one of these, since a model only
            # learns from the labeled data when it is the student
            _train_from_teacher(
                teacher=self.model_a,
                student=self.model_b,
                alpha=alpha,
                opt_student=self.optimizer_b,
                student_name="Model B",
                train_baseline=True,
            )
            _train_from_teacher(
                teacher=self.model_b,
                student=self.model_a,
                alpha=alpha,
                opt_student=self.optimizer_a,
                student_name="Model A",
                train_baseline=False,
            )

        # Finally, the best model is the one with the highest IoU on the labeled set
        model_a_accuracy = self.evaluate_IoU(self.model_a, self.labeled_loader)
        model_b_accuracy = self.evaluate_IoU(self.model_b, self.labeled_loader)
        best_model = (
            self.model_a if model_a_accuracy >= model_b_accuracy else self.model_b
        )
        self.best_model = best_model

        try:
            wandb.finish()
        except:
            pass

    def save_best_model(self, filename: str) -> None:
        """Save the best model to the filename specified"""
        if self.best_model is None:
            raise ValueError("No best model found. Did you run train()?")
        torch.save(self.best_model.state_dict(), filename)

    def save_baseline(self, filename: str) -> None:
        """Save the baseline model to the filename specified"""
        if not self.baseline_model:
            raise ValueError("No baseline model was trained!")
        torch.save(self.baseline_model.state_dict(), filename)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the best model"""
        if self.best_model is None:
            raise ValueError("No best model found. Did you run train()?")
        return self.best_model(x)
