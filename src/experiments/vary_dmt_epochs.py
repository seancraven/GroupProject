from src.experiments.base import BaseExperiment

class VaryDMTEpochs(BaseExperiment):
    EPOCHS = (5, 10, 20, 30)

    @property
    def model_folder(self) -> str:
        return 'models/vary_dmt_epochs'
    
    def run(self) -> None:
        for num_epochs in self.EPOCHS:
            self._base_run(
                num_dmt_epochs=num_epochs,
                best_model_fname=f'dmt_{num_epochs}.pt'
            )