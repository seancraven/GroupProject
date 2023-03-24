import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from src.models.UNet import UNet
from src.models.DMT import DMT
from src.pet_3.michael_data import PetsDataFetcher
from src.utils.evaluation import evaluate_IoU

TOTAL_BATCH_SIZE = 6
LABEL_PROPORTION = 0.02
VALIDATION_PROPORTION = 0.1
DIFFERENCE_MAXIMIZED_PROPORTION = 0.6
PERCENTILES = [0.2, 0.4, 0.6, 0.8, 1.0]
NUM_DMT_EPOCHS = 10
GAMMA_1 = 3
GAMMA_2 = 3

using_pretrained = False
unet_a = UNet()
unet_b = UNet()
baseline = UNet()
# try:
#     unet_a_state = torch.load("DMT_model_a.pt")
#     unet_b_state = torch.load("DMT_model_b.pt")
#     baseline_state = torch.load("DMT_baseline.pt")
#     unet_a.load_state_dict(unet_a_state)
#     unet_b.load_state_dict(unet_b_state)
#     baseline.load_state_dict(baseline_state)
#     print('Using pretrained models.')
#     using_pretrained = True
# except:
#     print('No pretrained models found. Training from scratch.')


fetcher = PetsDataFetcher(root='src/pet_3')
labeled, validation, unlabeled = fetcher.get_train_data(
    LABEL_PROPORTION, VALIDATION_PROPORTION,
    seed = 0
)
unlabeled = Subset(unlabeled, range(100))
print(f'Labeled: {len(labeled)} | Validation: {len(validation)} | Unlabeled: {len(unlabeled)}')


def save_fig(tensor, index, filename, reshape=True):
    image = tensor[index].cpu().detach()
    if reshape:
        image = image.permute(1,2,0)
    image = image.numpy()
    if image.shape[-1] == 65536:
        image = image.reshape((256,256))
    plt.imshow(image)
    plt.savefig(filename)

loader = DataLoader(validation, batch_size=10)

dmt = DMT(
    unet_a,
    unet_b,
    labeled,
    unlabeled,
    validation,
    max_batch_size=TOTAL_BATCH_SIZE,
    gamma_1=GAMMA_1,
    gamma_2=GAMMA_2,
    baseline=baseline
)
dmt.wandb_init(
    percentiles=PERCENTILES,
    num_epochs=NUM_DMT_EPOCHS,
    batch_size=TOTAL_BATCH_SIZE,
    label_ratio=LABEL_PROPORTION,
    difference_maximized_proportion=DIFFERENCE_MAXIMIZED_PROPORTION,
    gamma_1=GAMMA_1,
    gamma_2=GAMMA_2
)
dmt.pretrain(
    max_epochs=10000,
    proportion=DIFFERENCE_MAXIMIZED_PROPORTION
)
dmt.dynamic_train(
    percentiles=PERCENTILES,
    num_epochs=NUM_DMT_EPOCHS
)
test_data = fetcher.get_test_data()
test_loader = DataLoader(test_data, batch_size=TOTAL_BATCH_SIZE)
baseline_test_IoU = evaluate_IoU(dmt.baseline, test_loader)
best_model_test_IoU = evaluate_IoU(dmt.best_model, test_loader)

print("Baseline test IoU: ", baseline_test_IoU)
print("Best model test IoU: ", best_model_test_IoU)

dmt.save_best_model('best_dmt.pt')
dmt.save_baseline('baseline.pt')