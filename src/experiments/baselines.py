from src.experiments.base import BaseExperiment

class TrainBaselines(BaseExperiment):
    @property
    def model_folder(self) -> str:
        return 'models/baselines'

    def run(self) -> None:
        NO_RUNS_PER_BASELINE = 5
        for proportion in self.ALL_LABEL_PROPORTIONS:
            for i in range(NO_RUNS_PER_BASELINE):
                fname = 'baseline_{}_{}.pt'.format(proportion, i)
                self._train_baseline_only(
                    label_proportion=proportion,
                    baseline_fname=fname
                )

    def plot(self) -> None:
        pass