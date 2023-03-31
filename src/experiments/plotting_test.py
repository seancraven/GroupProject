from src.experiments.base import BaseExperiment
from torch import nn


class PlotTest(BaseExperiment):
    def __init__(self):
        pass

    def run(self):
        pass

    def test(model: nn.Module) -> float:
        pass

    @property
    def model_folder(self) -> str:
        return "models/baseline"
