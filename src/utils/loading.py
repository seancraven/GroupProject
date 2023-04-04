from typing import Union, Any
from torch import nn
from src.models.UNet import UNet
from src.models.LSD import LSD
import torch

MODEL_CLASSES = [UNet, LSD]


def _model_from_file(
    file_path: str, model_class: Any
) -> Union[nn.Module, None]:
    """Tries to load a model from a file. if it fails, returns None."""
    try:
        model = model_class()
        model.load_state_dict(torch.load(file_path))
    except Exception as e:
        print(e)
        model = None
    return model


def model_from_file(file_path: str) -> Union[nn.Module, None]:
    """Tries to load a model from a file. if it fails, returns None."""
    if file_path[-2:] == "pt":
        try:
            for m_class in MODEL_CLASSES:
                model = _model_from_file(file_path, m_class)
                if model is not None:
                    return model
        except Warning(
            "Could not load model from file. Add to MODEL_CLASSES. in src/utils/loading.py"
        ):
            pass
    return None
