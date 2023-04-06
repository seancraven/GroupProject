import torch


class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "fcn_resnet50",
            num_classes=2,
            pretrained=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = self.model(x)["out"]
        preds = preds.flatten(2, -1)  # Shape (B, C, H*W)
        preds = preds.permute(0, 2, 1)  # Shape (B, H*W, C)
        return preds

    @classmethod
    def from_state_dict(cls, state_dict) -> "fcn":
        fcn = cls()
        fcn.load_state_dict(state_dict)
        return fcn
