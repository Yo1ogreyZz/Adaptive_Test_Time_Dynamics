import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


class TokenBlockDataset(Dataset):
    def __init__(self, token_ids: List[int], block_size: int):
        self.block_size = block_size
        usable = (len(token_ids) // (block_size + 1)) * (block_size + 1)
        token_ids = token_ids[:usable]
        self.inputs = []
        self.labels = []
        for i in range(0, len(token_ids), block_size + 1):
            chunk = token_ids[i : i + block_size + 1]
            if len(chunk) < block_size + 1:
                continue
            self.inputs.append(chunk[:-1])
            self.labels.append(chunk[1:])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class WikiTextLMTask:
    """
    Byte-level causal LM on WikiText-103 (or local txt fallback).
    Returns batches for next-token prediction.
    """

    def __init__(self, batch_size=16, block_size=256, dataset_name="wikitext-103-raw-v1", data_dir=None, num_workers=2,
                 max_train_samples=None, data_fraction=None):
        self.task_name = "lm"
        self.batch_size = batch_size
        self.block_size = block_size
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.max_train_samples = max_train_samples
        self.data_fraction = data_fraction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = 257
        self.num_classes = self.vocab_size
        self._datasets: Dict[str, TokenBlockDataset] = {}

    @staticmethod
    def _encode_text(text: str):
        return [b + 1 for b in text.encode("utf-8")]

    def _read_local_split(self, split: str):
        if self.data_dir is None:
            return None
        split_map = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}
        txt_path = os.path.join(self.data_dir, split_map[split])
        if not os.path.exists(txt_path):
            return None
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_split_text(self, split: str):
        local = self._read_local_split(split)
        if local is not None:
            return local

        hf_split = "validation" if split == "val" else split
        ds = load_dataset("wikitext", self.dataset_name, split=hf_split)
        return "\n".join([t for t in ds["text"] if t is not None and len(t) > 0])

    def _build_split(self, split: str):
        text = self._load_split_text(split)
        token_ids = self._encode_text(text)
        ds = TokenBlockDataset(token_ids, self.block_size)

        if split == "train":
            n = len(ds)
            limit = n
            if self.data_fraction is not None:
                limit = min(limit, int(n * self.data_fraction))
            if self.max_train_samples is not None:
                limit = min(limit, self.max_train_samples)
            if limit < n:
                ds.inputs = ds.inputs[:limit]
                ds.labels = ds.labels[:limit]
                print(f"[WikiTextLMTask] train subset: {limit}/{n} samples "
                      f"(fraction={self.data_fraction}, max={self.max_train_samples})")

        return ds

    def get_dataloader(self, split="train"):
        if split not in self._datasets:
            self._datasets[split] = self._build_split(split)

        ds = self._datasets[split]
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def preprocess_batch(self, batch):
        input_ids, labels = batch
        mask = (input_ids != 0).long()
        return (
            input_ids.to(self.device, non_blocking=True),
            labels.to(self.device, non_blocking=True),
            mask.to(self.device, non_blocking=True),
        )


class WikiTextLMModel(nn.Module):
    def __init__(self, backbone, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.backbone = backbone
        self.lm_head = nn.Linear(d_model, vocab_size)

        if hasattr(self.backbone, "set_task_head"):
            self.backbone.set_task_head(self.lm_head, mode="token_lm")

    def forward(self, input_ids, mask=None, return_info=False):
        x = self.embedding(input_ids)

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
                raise RuntimeError("Backbone returned None in WikiTextLMModel.forward")

        token_logits = self.lm_head(x)

        if self.training and info is not None and "K_soft" in info:
            return token_logits, info["K_soft"]

        if return_info:
            return token_logits, (info if info is not None else {})
        return token_logits
