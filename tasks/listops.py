import torch
import torch.nn as nn
from .lra_base import LRATask


class ListOpsTask(LRATask):
    """
    ListOps: classify the result of nested logical operations (0-9).
    Actual token lengths are ~50, so max_length=256 is sufficient.
    """

    def __init__(self, batch_size=32, max_length=256, data_name="listops"):
        super().__init__(task_name=data_name, batch_size=batch_size, max_length=max_length)
        self.vocab = {
            "<PAD>": 0, "(": 1, ")": 2, "[": 3, "]": 4,
            "0": 5, "1": 6, "2": 7, "3": 8, "4": 9,
            "5": 10, "6": 11, "7": 12, "8": 13, "9": 14,
            "MAX": 15, "MIN": 16, "MED": 17, "FIRST": 18,
            "LAST": 19, "SM": 20,
        }
        self.vocab_size = len(self.vocab)
        self.num_classes = 10

    def tokenize(self, text):
        tokens = (text.replace("(", " ( ").replace(")", " ) ")
                      .replace("[", " [ ").replace("]", " ] ").split())
        ids = [self.vocab.get(t, 0) for t in tokens]
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        else:
            ids = ids + [0] * (self.max_length - len(ids))
        return ids

    def preprocess_batch(self, batch):
        input_ids = [self.tokenize(text) for text in batch["Source"]]
        labels = [int(label) for label in batch["Target"]]
        input_tensor = torch.tensor(input_ids).to(self.device)
        # Compute mask for non-PAD positions (PAD token id = 0)
        mask = (input_tensor != 0)
        return (
            input_tensor,
            torch.tensor(labels).to(self.device),
            mask,
        )


class ListOpsModel(nn.Module):
    """
    Task-specific model: token embedding + ATTD backbone + 10-class head.
    Uses mean pooling over non-PAD positions for classification.
    """

    def __init__(self, backbone, vocab_size, d_model, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.backbone = backbone
        self.classifier = nn.Linear(d_model, num_classes)
        # Let backbone access classifier for pseudo-label TTT
        if hasattr(self.backbone, 'set_task_head'):
            self.backbone.set_task_head(self.classifier)

    def _pool(self, x, mask=None):
        if mask is None:
            return x.mean(dim=1)
        mask_f = mask.unsqueeze(-1).float()
        return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

    def forward(self, input_ids, mask=None, return_info=False):
        x = self.embedding(input_ids)   # (B, L, D)

        need_info = return_info or (self.training and getattr(self.backbone, "train_mode", "base") == "controller")
        out = self.backbone(x, return_info=need_info) if need_info else self.backbone(x)

        info = None
        if isinstance(out, tuple):
            x, info = out
        else:
            x = out

        if x is None:
            if hasattr(self.backbone, "backbone"):
                x = self.backbone.backbone(self.embedding(input_ids))
            else:
                raise RuntimeError("Backbone returned None in ListOpsModel.forward")

        pooled = self._pool(x, mask=mask)
        logits = self.classifier(pooled)

        if self.training and info is not None and "K_soft" in info:
            return logits, info["K_soft"]

        if return_info:
            return logits, (info if info is not None else {})
        return logits
