from src.experiments.experiments import VaryLabelProportion, TrainBaselines

if __name__ == "__main__":
    TrainBaselines().run()
    VaryLabelProportion().run()
