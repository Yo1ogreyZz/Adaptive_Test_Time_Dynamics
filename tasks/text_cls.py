import torch
import torch.nn as nn
from .lra_base import LRATask


class TextClsTask(LRATask):
    """
    Text classification (IMDB sentiment): byte-level encoding, binary output.
    Sequence length: up to 4096 bytes.
    """

    def __init__(self, batch_size=32, max_length=4096):
        super().__init__(task_name="text", batch_size=batch_size, max_length=max_length)
        # Byte-level: 256 values + PAD at index 0
        self.vocab_size = 257
        self.num_classes = 2

    def preprocess_batch(self, batch):
        input_ids = []
        for text in batch["Source"]:
            # +1 offset so that PAD = 0
            ids = [b + 1 for b in text.encode("utf-8")][:self.max_length]
            ids = ids + [0] * (self.max_length - len(ids))
            input_ids.append(ids)
        labels = [int(label) for label in batch["Target"]]
        return (
            torch.tensor(input_ids).to(self.device),
            torch.tensor(labels).to(self.device),
        )


class TextClsModel(nn.Module):
    """
    Task-specific model: byte embedding + ATTD backbone + binary classification head.
    """

    def __init__(self, backbone, vocab_size, d_model, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.backbone = backbone
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)   # (B, L, D)
        x = self.backbone(x)            # (B, L, D)
        return self.classifier(x[:, -1, :])
