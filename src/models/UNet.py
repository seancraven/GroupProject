"""
Model used in all experiments. 
Model forward is slightly altered to flatten output.
"""
import torch


class UNet(torch.nn.Module):
    """
    A UNet model taken from a brain segmentation paper. The forward pass is
    modified to produce an output of the shape DMT expects, i.e. (B, H*W, C).
    """
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
        preds = preds.flatten(2, -1)  # Shape (B, C, H*W)
        preds = preds.permute(0, 2, 1)  # Shape (B, H*W, C)
        return preds

    @classmethod
    def from_state_dict(cls, state_dict) -> "UNet":
        """ Produces a UNet model from a state dict."""
        unet = cls()
        unet.load_state_dict(state_dict)
        return unet
