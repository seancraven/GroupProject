from src.testing.model_testing_utils import LoadedModel
from src.pet_3.data import Pets

if __name__ == "__main__":
    #dave = LoadedModel("models/u_net_supervised/Mean Squared Error Loss_20.pt")
    pass
    test_data = Pets("./src/pet_3/", split="test", binary_labels=True)
    print(test_data.images)