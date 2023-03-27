from src.experiments.base import BaseExperiment

class VaryLabelProportion(BaseExperiment):
    @property
    def model_folder(self) -> str:
        return 'models/vary_label_proportion'
    
    def run(self) -> None:
        for proportion in self.ALL_LABEL_PROPORTIONS:
            self._base_run(
                label_proportion=proportion,
                best_model_fname=f'dmt_{proportion}.pt'
            )

    def plot(self) -> None:
        pass