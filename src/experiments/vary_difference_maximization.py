from src.experiments.base import BaseExperiment


class VaryDifferenceMaximization(BaseExperiment):
    PROPORTIONS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    @property
    def model_folder(self) -> str:
        return "models/vary_difference_maximization"

    def run(self) -> None:
        for proportion in self.PROPORTIONS:
            self._base_run(
                difference_maximized_proportion=proportion,
                baseline_fname=f"baseline_{proportion}.pt",
                best_model_fname=f"dmt_{proportion}.pt",
            )
