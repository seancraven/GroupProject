import torch

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=3,
            out_channels=2,
            init_features=32,
            pretrained=False,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = self.model(x)
        preds = preds.flatten(2,-1)  # Shape (B, C, H*W)
        preds = preds.permute(0, 2, 1)  # Shape (B, H*W, C)
        return preds
    
    @classmethod
    def from_state_dict(cls, state_dict) -> "UNet":
        unet = cls()
        unet.model.load_state_dict(state_dict)
        return unet
