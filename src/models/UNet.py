import torch

def get_unet():
    """
    Load the U-Net model from the brain-segmentation-pytorch repo
    and modify it to output a flattened tensor of shape (B, H*W, C)
    """
    model = torch.hub.load(
        "mateuszbuda/brain-segmentation-pytorch",
        "unet",
        in_channels=3,
        out_channels=2,
        init_features=32,
        pretrained=False,
    )

    class UNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unet = model

        def forward(self, x):
            preds = self.unet(x)
            preds = preds.flatten(2,-1)  # Shape (B, C, H*W)
            preds = preds.permute(0, 2, 1)  # Shape (B, H*W, C)
            return preds
        
    unet = UNet()

    return unet