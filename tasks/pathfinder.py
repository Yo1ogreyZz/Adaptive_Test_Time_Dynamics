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
        if hasattr(self.backbone, "set_task_head"):
            self.backbone.set_task_head(self.classifier)

    def forward(self, pixels, mask=None, return_info=False):
        # pixels: (B, L) where L = resolution^2
        x = pixels.unsqueeze(-1)        # (B, L, 1)
        x = self.pixel_proj(x)          # (B, L, D)

        need_info = return_info or (self.training and getattr(self.backbone, "train_mode", "base") == "controller")
        out = self.backbone(x, return_info=need_info) if need_info else self.backbone(x)

        info = None
        if isinstance(out, tuple):
            x, info = out
        else:
            x = out

        if x is None:
            if hasattr(self.backbone, "backbone"):
                x = self.backbone.backbone(self.pixel_proj(pixels.unsqueeze(-1)))
            else:
                raise RuntimeError("Backbone returned None in PathfinderModel.forward")

        logits = self.classifier(x[:, -1, :])

        if self.training and info is not None and "K_soft" in info:
            return logits, info["K_soft"]

        if return_info:
            return logits, (info if info is not None else {})
        return logits
