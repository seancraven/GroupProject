import wandb
from torch.utils.data import DataLoader, Subset

from src.models.LSD import LSD
from src.models.UNet import UNet
from src.models.PLabel import PLabel
from src.pet_3.data import PetsDataFetcher
from src.utils.evaluation import evaluate_IoU

TOTAL_BATCH_SIZE = 32
LABEL_PROPORTION = 0.01
VALIDATION_PROPORTION = 0.1
NUM_EPOCHS = 50
# GAMMA_1 = 3
# GAMMA_2 = 3

using_pretrained = False
#  model = PLabel()
unet = UNet()
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


fetcher = PetsDataFetcher(root="src/pet_3")
(labeled, validation, unlabeled), name = fetcher.get_train_data_with_name(
    LABEL_PROPORTION, VALIDATION_PROPORTION, seed=1, class_balance=True
)
print(
    f"Labeled: {len(labeled)} | Validation: {len(validation)} | Unlabeled: {len(unlabeled)}"
)

lsd = LSD()
unet = UNet()
plabel = PLabel(
   # unet,
    lsd,
    labeled,
    unlabeled,
    validation,
    max_batch_size=TOTAL_BATCH_SIZE,
    baseline=baseline,
)

plabel.wandb_init(
    num_epochs=NUM_EPOCHS,
    batch_size=TOTAL_BATCH_SIZE,
    label_ratio=LABEL_PROPORTION,
)
plabel.pretrain(max_epochs=10000)
plabel.train(num_epochs=NUM_EPOCHS)
test_data = fetcher.get_test_data()
test_loader = DataLoader(test_data, batch_size=TOTAL_BATCH_SIZE)
baseline_test_IoU = evaluate_IoU(plabel.baseline, test_loader)
model_test_IoU = evaluate_IoU(plabel.model, test_loader)

print("Baseline test IoU: ", baseline_test_IoU)
print("Model test IoU: ", model_test_IoU)
plabel.wandb_log(
    {
        "Baseline test IoU": baseline_test_IoU,
        "Model test IoU": model_test_IoU,
    }
)

plabel.save_model(f"models/model_data_{name}.pt")
plabel.save_baseline(f"models/baseline_data_{name}.pt")

wandb.finish()
