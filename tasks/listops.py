import torch
import torch.nn as nn
from .lra_base import LRATask


class ListOpsTask(LRATask):
    """
    ListOps: classify the result of nested logical operations (0-9).
    Sequence length: up to 2048 tokens.
    """

    def __init__(self, batch_size=32, max_length=2048):
        super().__init__(task_name="listops", batch_size=batch_size, max_length=max_length)
        self.vocab = {
            "<PAD>": 0, "(": 1, ")": 2, "[": 3, "]": 4,
            "0": 5, "1": 6, "2": 7, "3": 8, "4": 9,
            "5": 10, "6": 11, "7": 12, "8": 13, "9": 14,
            "MAX": 15, "MIN": 16, "MED": 17, "FIRST": 18,
            "LAST": 19, "SUM_MOD": 20,
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
        return (
            torch.tensor(input_ids).to(self.device),
            torch.tensor(labels).to(self.device),
        )


class ListOpsModel(nn.Module):
    """
    Task-specific model: token embedding + ATTD backbone + 10-class head.
    """

    def __init__(self, backbone, vocab_size, d_model, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.backbone = backbone
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)   # (B, L, D)
        x = self.backbone(x)            # (B, L, D)
        return self.classifier(x[:, -1, :])
