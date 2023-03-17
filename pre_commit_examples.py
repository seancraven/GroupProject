"""
    Short file with which to test precommit hooks and some docsting stuff
"""
from typing import Any, Iterator
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.pet_3.data import Pets


def example_function(alpha: int, beta: str, gamma: Iterator[Any]) -> list[int]:
    """
    My prefered docstring format is sphinx.
    if you are using vscode:
    1) install the autodocstring extension.
    2) add to settings.json, exclude the single quotes.
    '"autoDocstring.docstringFormat":"sphinx",'

    Args:
        a (int): _description_
        b (str): _description_
        c (Iterator[Any]): _description_

    Returns:
        Tuple[int]: _description_
    """
    print(alpha)
    print(beta)
    print(gamma)
    return [
        1,
        1,
        1,
    ]


if __name__ == "__main__":
    unlabeled, labeled = Pets(
        "./pet_3", "labeled_unlabeled", 0.5, binary_labels=True
    ).get_datasets()
    unlabeled_loader = DataLoader(unlabeled, batch_size=1)
    labeled_loader = DataLoader(labeled, batch_size=1)

    test_label = labeled[40][1]
    plt.matshow(test_label.permute(1, 2, 0).numpy())
    plt.show()
