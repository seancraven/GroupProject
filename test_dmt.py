import torch

from torch.utils.data import DataLoader

from src.models.UNet import get_unet
from src.models.DMT import DMT
from src.pet_3.data import Pets

TOTAL_BATCH_SIZE = 8
LABEL_PROPORTION = 0.2

using_pretrained = False
unet_a = get_unet()
unet_b = get_unet()
baseline = get_unet()
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


optimizer_a = torch.optim.Adam(unet_a.parameters(), lr=1e-3)
optimizer_b = torch.optim.Adam(unet_b.parameters(), lr=1e-3)
baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=1e-3)
train_dataset = Pets("src/pet_3", "labeled_unlabeled", labeled_fraction=LABEL_PROPORTION, shuffle=True)
train_unlabeled, train_labeled = train_dataset.get_datasets()
unlabeled_loader = DataLoader(
    train_unlabeled,
    batch_size = int((1-LABEL_PROPORTION) * TOTAL_BATCH_SIZE),
    num_workers=8
)
labeled_loader = DataLoader(
    train_labeled,
    batch_size=int(LABEL_PROPORTION * TOTAL_BATCH_SIZE),
    num_workers=8
)
test_dataset = Pets("src/pet_3", "test")
test_loader = DataLoader(test_dataset, batch_size=TOTAL_BATCH_SIZE, num_workers=8)

dmt = DMT(
    unet_a,
    unet_b,
    optimizer_a,
    optimizer_b,
    labeled_loader,
    unlabeled_loader,
    gamma_1=1,
    gamma_2=1,
    verbosity=2,
    baseline_model=baseline,
    baseline_optimizer=baseline_optimizer,
)
dmt.train(
    percentiles=[0.2,0.4,0.6,0.8,1.0],
    num_epochs=10,
    batch_size=TOTAL_BATCH_SIZE,
    label_ratio=LABEL_PROPORTION,
    skip_pretrain=False
)
dmt.save_best_model('best_dmt.pt')
dmt.save_baseline('baseline.pt')
best_model_test_accuracy = dmt.evaluate_IoU(test_loader, dmt.best_model)
baseline_test_accuracy = dmt.evaluate_IoU(test_loader, dmt.baseline_model)

print('=== Done ===')
print('Best model test accuracy: ', best_model_test_accuracy)
print('Baseline test accuracy: ', baseline_test_accuracy)