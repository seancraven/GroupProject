# pylint: disable-all
from pet_3.data import Pets
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import wandb  # type: ignore
from torch.nn.functional import mse_loss, cross_entropy
import time


# TODO all the below code is just dummy code to test the dynamic mutual training


class DynamicMutualTraining(nn.Module):
    def __init__(self, gamma1=5, gamma2=5):
        super(DynamicMutualTraining, self).__init__()
        self.model_a = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=3,
            out_channels=1,
            init_features=32,
            pretrained=False,
        )
        self.model_b = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=3,
            out_channels=1,
            init_features=32,
            pretrained=False,
        )
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.optimizer_a = torch.optim.Adam(self.model_a.parameters(), lr=1e-3)
        self.optimizer_b = torch.optim.Adam(self.model_b.parameters(), lr=1e-3)

    def difference_maximized_sampling(self, L):
        """
        Sample two subsets from the given dataset using the difference maximized sampling algorithm.
        Aim to find two equal-sized subsets from the given dataset.

        Args:
            L (int): Size of the subset to sample.
        Returns:
            subset1 (torch.utils.data.Subset): Subset of the given dataset.
            subset2 (torch.utils.data.Subset): Subset of the given dataset.
        """
        dataset = Pets("./pet_3", "all_train")
        # get subset 1 and 2 where we sample alpha from uniform(0.500001,0.99999)
        # then subset A is from 0 to L*alpha and subset B is from (1 - alpha)*L to L
        alpha = torch.rand(1).item() * 0.499999 + 0.500001

        subset1 = torch.utils.data.Subset(dataset, range(int(L * alpha)))
        subset2 = torch.utils.data.Subset(dataset, range(int((1 - alpha) * L), L))

        return subset1, subset2

    def train_models(self, L=1000):
        """
        Train the models on the given datasets. The datasets should be seperate subsamples of the Pets dataset.

        Args:
            dataset1 (torch.utils.data.Dataset): Subset of Pets dataset for model_a with only labeled data.
            dataset2 (torch.utils.data.Dataset): Subset of Pets dataset for model_b with only labeled data.
        """
        # TODO manage the size L of the subsets and other hyperparams below

        dataset1, dataset2 = self.difference_maximized_sampling(L=L)

        train_loader1 = DataLoader(dataset1, batch_size=32, num_workers=2)
        train_loader2 = DataLoader(dataset2, batch_size=32, num_workers=2)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_a.to(device)
        self.model_b.to(device)

        for epoch in range(10):
            epoch_start_time = time.time()
            epoch_loss = 0.0

            for img, labels in train_loader1:
                img, labels = img.to(device), labels.to(device)
                self.optimizer_a.zero_grad()
                out = self.model_a(img)
                loss = cross_entropy(out, labels)
                loss.backward()
                self.optimizer_a.step()
            for img, labels in train_loader2:
                img, labels = img.to(device), labels.to(device)
                self.optimizer_b.zero_grad()
                out = self.model_b(img)
                loss = cross_entropy(out, labels)
                loss.backward()
                self.optimizer_b.step()

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            print(f"Epoch {epoch} took {epoch_time} seconds")

    @staticmethod
    def compute_pseudolabel_mask(
        alpha: float, predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes pseudolabels from predictions
        predictions are of shape (batch, no_pixels)
        """
        threshold = torch.quantile(predictions, q=1 - alpha, dim=1, keepdim=True)
        mask = predictions >= threshold
        return mask

    def compute_weights(self, pseudolabel_mask, confidence_a, confidence_b):
        """
        Computes the weights for the loss function.

        Args:
            pseudolabels1 (torch.Tensor): Pseudolabels from model_a.
        """
        weights = torch.zeros_like(confidence_a)
        agreement_mask = torch.round(confidence_a) == torch.round(confidence_b)
        more_confident_mask = confidence_a >= confidence_b
        weights += agreement_mask * torch.pow(
            confidence_b, self.gamma1
        ) + ~agreement_mask * more_confident_mask * torch.pow(confidence_b, self.gamma2)
        return weights * pseudolabel_mask

    def train_step(
        self, img_unlabelled, img_labelled, frozen_model, train_model, alpha
    ):
        """
        Perform one step of the dynamic mutual training algorithm.
        The frozen model produce pseudolabels for the unlabeled data.
        The train model is trained on the labeled data and the pseudolabels.
        Alpha determines the percentile threshold for the pseudolabels.

        Args:
            img_unlabelled (torch.Tensor): Batch of unlabeled images.
            img_labelled (torch.Tensor): Batch of labeled images.
            frozen_model (nn.Module): Model to produce pseudolabels.
            train_model (nn.Module): Model to train on the pseudolabels.
            alpha (float): Percentile threshold for the pseudolabels.

        Returns:
            loss (torch.Tensor): Loss of the train model.
        """
        # pseudo labels and confidence scores from frozen model

        predictions = frozen_model(img_unlabelled).flatten()
        pseudo_labels, pseudolabel_mask = self.compute_pseudolabels(
            predictions, alpha=alpha
        )
        confidence1 = torch.max(frozen_model(img_labelled), dim=1)[0]

        # fine tuning train model
        predictions = train_model(img_unlabelled).flatten()
        confidence2 = torch.max(predictions, dim=1)[0]

        weights = self.compute_weights(
            psuedolabels=pseudo_labels,
            pseudolabel_mask=pseudolabel_mask,
            confidence1=confidence1,
            confidence2=confidence2,
        )

        loss = torch.mean(
            cross_entropy(weight=weights, input=predictions, target=pseudo_labels)
        )

        return loss

    def train(
        self, unlabelled_loader, labelled_loader, percentile=torch.linspace(0.2, 1, 5)
    ):
        """
        Implement the dynamic mutual training algorithm. (still under construction)!
        The weight of the unlabeled data is as follows for the CE loss
        Let p2 be the probability of class from pseudo label by model_a.

        Then:
        w =  pb^gamma_1 if pseudo label = prediction (i.e., model_a and model_b agree on the label)
        w = pb^gamma_2 if pseudo label != prediction and confidence of model 1 >= confidence of model 2
        0 if pseudo label != prediction and confidence of model 1 < confidence of model 2

        Finally, the loss will be:
        1/N w.T * CE_loss(pseudo_labels, prediction).

        Here the gamma values are hyperparameters.
        - high gamma 1 emphasizes entropy minimization
        - high gamma 2 emphasizes  mutual learning
        """

        for alpha in percentile:
            for img_unlabelled, (img_labelled, labels) in zip(
                unlabelled_loader, labelled_loader
            ):
                # TODO also not sure about whether to use the same optimizer for both models or have two different ones

                # Train Model B
                self.model_a.eval()
                self.model_b.train()

                # Unlabelled Loss
                loss_unlabelled = self.train_step(
                    img_unlabelled,
                    img_labelled,
                    frozen_model=self.model_a,
                    train_model=self.model_b,
                    alpha=alpha,
                )

                # Labelled Loss
                labelled_preds = self.model_b(img_labelled)
                loss_labelled = cross_entropy(labelled_preds, labels)

                loss = loss_labelled + loss_unlabelled

                self.optimizer2.zero_grad()
                loss.backward()
                self.optimizer2.step()

                # Train Model A
                self.model_b.eval()
                self.model_a.train()

                # Unlabelled Loss
                loss_unlabelled = self.train_step(
                    img_unlabelled,
                    img_labelled,
                    frozen_model=self.model_b,
                    train_model=self.model_a,
                    alpha=alpha,
                )

                # Labelled Loss
                labelled_preds = self.model_a(img_labelled)
                loss_labelled = cross_entropy(labelled_preds, labels)
                loss = loss_labelled + loss_unlabelled

                self.optimizer1.zero_grad()
                loss.backward()
                self.optimizer1.step()


if __name__ == "__main__":
    TOTAL_BATCH_SIZE = 50
    LABEL_PROPORTION = 0.5

    dmt = DynamicMutualTraining()
    dmt.train_models()
    # not sure what fraction to set for the labeled data, but this is just a test

    train_dataset = Pets(
        "./pet_3", "labeled_unlabeled", labeled_fraction=LABEL_PROPORTION
    )
    train_unlabeled, train_labeled = train_dataset.get_datasets()
    train_unlabeled = DataLoader(
        train_unlabeled,
        batch_size=int((1 - LABEL_PROPORTION) * TOTAL_BATCH_SIZE),
        num_workers=8,
    )
    train_labeled = DataLoader(
        train_labeled,
        batch_size=int(LABEL_PROPORTION * TOTAL_BATCH_SIZE),
        num_workers=8,
    )
    dmt.train(train_unlabeled, train_labeled)

    # loaders
