import torch
import torch.nn as nn
from .lra_base import LRATask


class PathfinderTask(LRATask):
    """
    Pathfinder: determine whether two dots in an image are connected by a path.
    Input is a flattened grayscale image (resolution x resolution pixels).
    Binary classification (connected / not connected).
    """

    def __init__(self, batch_size=32, resolution=32):
        max_length = resolution * resolution
        super().__init__(task_name="pathfinder", batch_size=batch_size, max_length=max_length)
        self.resolution = resolution
        self.num_classes = 2

    def preprocess_batch(self, batch):
        # Expect batch["image"] as pixel arrays; flatten to (B, L)
        images = torch.tensor(batch["image"], dtype=torch.float32)
        if images.dim() == 4:
            images = images.mean(dim=-1)  # convert to grayscale if multi-channel
        pixels = images.view(images.size(0), -1) / 255.0  # normalize to [0, 1]
        labels = torch.tensor(batch["label"], dtype=torch.long)
        return pixels.to(self.device), labels.to(self.device)


class PathfinderModel(nn.Module):
    """
    Task-specific model: linear pixel projection + ATTD backbone + binary head.
    Each pixel scalar is projected to d_model dimensions.
    """

    def __init__(self, backbone, d_model, num_classes=2):
        super().__init__()
        self.pixel_proj = nn.Linear(1, d_model)
        self.backbone = backbone
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, pixels):
        # pixels: (B, L) where L = resolution^2
        x = pixels.unsqueeze(-1)        # (B, L, 1)
        x = self.pixel_proj(x)          # (B, L, D)
        x = self.backbone(x)            # (B, L, D)
        return self.classifier(x[:, -1, :])
