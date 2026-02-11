import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class AGNewsCSVDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"{csv_path} must contain columns: text,label")
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class TextClsTask:
    """
    AG News text classification task.
    Expects:
      data_dir/train.csv
      data_dir/val.csv
      data_dir/test.csv
    with columns: text,label
    """

    def __init__(self, batch_size=32, max_length=512, dataset_name="agnews", data_dir="data/agnews", num_workers=2):
        self.task_name = "text"
        self.batch_size = batch_size
        self.max_length = max_length
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.num_workers = num_workers

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = 257  # byte-level [0..255] + PAD=0 with +1 shift
        if self.dataset_name == "agnews":
            self.num_classes = 4
        elif self.dataset_name == "imdb":
            self.num_classes = 2
        else:
            raise ValueError(f"Unsupported text dataset: {self.dataset_name}")

    def _encode_text(self, text: str):
        ids = [b + 1 for b in text.encode("utf-8")][: self.max_length]
        mask = [1] * len(ids)
        if len(ids) < self.max_length:
            pad_len = self.max_length - len(ids)
            ids.extend([0] * pad_len)
            mask.extend([0] * pad_len)
        return ids, mask

    def _collate_fn(self, batch):
        input_ids, labels, masks = [], [], []
        for text, label in batch:
            ids, m = self._encode_text(text)
            input_ids.append(ids)
            masks.append(m)
            labels.append(int(label))
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(masks, dtype=torch.long),
        )

    def get_dataloader(self, split="train"):
        split_to_file = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}
        if split not in split_to_file:
            raise ValueError(f"Unsupported split: {split}")

        csv_path = os.path.join(self.data_dir, split_to_file[split])
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing split file: {csv_path}")

        ds = AGNewsCSVDataset(csv_path)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self._collate_fn,
        )

    def preprocess_batch(self, batch):
        input_ids, labels, mask = batch
        return (
            input_ids.to(self.device, non_blocking=True),
            labels.to(self.device, non_blocking=True),
            mask.to(self.device, non_blocking=True),
        )


class TextClsModel(nn.Module):
    def __init__(self, backbone, vocab_size, d_model, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.backbone = backbone
        self.classifier = nn.Linear(d_model, num_classes)

        # expose task head to ATTD for TTT loss
        if hasattr(self.backbone, "set_task_head"):
            self.backbone.set_task_head(self.classifier)

    def _pool(self, x, mask=None):
        if mask is None:
            return x.mean(dim=1)
        m = mask.unsqueeze(-1).float()
        denom = m.sum(dim=1).clamp_min(1.0)
        return (x * m).sum(dim=1) / denom

    def forward(self, input_ids, mask=None, return_info=False):
        x = self.embedding(input_ids)  # (B, L, D)

        need_info = return_info or (self.training and getattr(self.backbone, "train_mode", "base") == "controller")
        out = self.backbone(x, return_info=need_info) if need_info else self.backbone(x)

        info = None
        if isinstance(out, tuple):
            x, info = out
        else:
            x = out

        # Defensive fallback: ATTD wrapper should return tensor, but if None appears, fallback to raw backbone.
        if x is None:
            if hasattr(self.backbone, "backbone"):
                x = self.backbone.backbone(self.embedding(input_ids))
            else:
                raise RuntimeError("Backbone returned None in TextClsModel.forward")

        pooled = self._pool(x, mask=mask)
        logits = self.classifier(pooled)

        if self.training and info is not None and "K_soft" in info:
            return logits, info["K_soft"]

        if return_info:
            return logits, (info if info is not None else {})
        return logits
