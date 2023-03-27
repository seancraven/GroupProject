from src.experiments.base import BaseExperiment

class VaryLabelProportion(BaseExperiment):
    LABEL_PROPORTIONS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0)

    @property
    def model_folder(self) -> str:
        return 'models/vary_label_proportion'

    def run(self) -> None:
        for i, proportion in enum self.LABEL_PROPORTIONS:
            self._base_run(
                label_proportion=proportion,
                baseline_fname=f'baseline_{proportion}.pt',
                best_model_fname=f'dmt_{proportion}.pt'
            )
