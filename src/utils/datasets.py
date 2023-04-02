import math

from torch.utils.data import Dataset, Subset
from typing import Tuple


def balanced_minibatch_sizes(
    labeled_dataset: Dataset, unlabeled_dataset: Dataset, max_batch_size: int
) -> Tuple[int, int]:
    """
    Computes batch sizes for the labeled and unlabeled datasets such that they
    each have as close to the same number of minibatches as possible.
    """
    if len(labeled_dataset) >= len(unlabeled_dataset):
        labeled_batch_size = max_batch_size
        unlabeled_batch_size = math.ceil(
            len(unlabeled_dataset) / len(labeled_dataset) * max_batch_size
        )
    else:
        unlabeled_batch_size = max_batch_size
        labeled_batch_size = math.ceil(
            len(labeled_dataset) / len(unlabeled_dataset) * max_batch_size
        )
    return labeled_batch_size, unlabeled_batch_size


def difference_maximized_sampling(
    dataset: Dataset, proportion: float = 0.5
) -> Tuple[Dataset, Dataset]:
    """
    Returns two subsets of the dataset such that the overlap between them is minimized.
    """
    subset_1 = Subset(dataset, range(int(len(dataset) * proportion)))
    subset_2 = Subset(dataset, range(int(1 - proportion) * len(dataset), len(dataset)))

    return subset_1, subset_2
