import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


class SQuADv2ByteQADataset(Dataset):
    """
    Byte-level extractive QA dataset for SQuAD v2.

    Token IDs:
      PAD=0, CLS=1, SEP=2, bytes are shifted by +3.
    Sequence format:
      [CLS] question [SEP] context [SEP]
    """

    PAD_ID = 0
    CLS_ID = 1
    SEP_ID = 2
    BYTE_SHIFT = 3

    def __init__(self, split_ds, max_length=512):
        self.max_length = max_length
        self.samples = []
        for item in split_ds:
            self.samples.append(self._encode_item(item))

    @staticmethod
    def _to_byte_ids(text: str):
        return [b + SQuADv2ByteQADataset.BYTE_SHIFT for b in text.encode("utf-8")]

    @staticmethod
    def _char_to_byte_idx(text: str, char_idx: int):
        return len(text[:char_idx].encode("utf-8"))

    def _encode_item(self, item):
        question = str(item.get("question", ""))
        context = str(item.get("context", ""))
        answers = item.get("answers", {"text": [], "answer_start": []})

        q_ids = self._to_byte_ids(question)
        c_bytes = list(context.encode("utf-8"))
        c_ids = [b + self.BYTE_SHIFT for b in c_bytes]

        # Build sequence [CLS] Q [SEP] C [SEP]
        input_ids = [self.CLS_ID] + q_ids + [self.SEP_ID]
        context_start = len(input_ids)
        input_ids.extend(c_ids)
        input_ids.append(self.SEP_ID)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]

        attention_mask = [1] * len(input_ids)
        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            input_ids += [self.PAD_ID] * pad_len
            attention_mask += [0] * pad_len

        # Default: unanswerable or truncated answer -> CLS position
        start_pos = 0
        end_pos = 0

        ans_texts = answers.get("text", []) if isinstance(answers, dict) else []
        ans_starts = answers.get("answer_start", []) if isinstance(answers, dict) else []
        if len(ans_texts) > 0 and len(ans_starts) > 0 and len(ans_texts[0]) > 0:
            ans_text = str(ans_texts[0])
            ans_char_start = int(ans_starts[0])
            ans_char_end = ans_char_start + len(ans_text)
            ans_b_start = self._char_to_byte_idx(context, ans_char_start)
            ans_b_end = self._char_to_byte_idx(context, ans_char_end) - 1

            # Map context byte positions into sequence indices
            seq_ctx_start = context_start
            seq_ctx_end = min(context_start + len(c_ids), self.max_length - 1) - 1

            if seq_ctx_start <= seq_ctx_end:
                for idx in range(seq_ctx_start, seq_ctx_end + 1):
                    ctx_b_idx = idx - context_start
                    if ctx_b_idx == ans_b_start:
                        start_pos = idx
                    if ctx_b_idx == ans_b_end:
                        end_pos = idx

                # If only one end is found (partial truncation), reset to CLS.
                if (start_pos == 0) != (end_pos == 0):
                    start_pos = 0
                    end_pos = 0

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "start_positions": torch.tensor(start_pos, dtype=torch.long),
            "end_positions": torch.tensor(end_pos, dtype=torch.long),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SQuADv2Task:
    def __init__(self, batch_size=16, max_length=512, num_workers=2, dataset_id="rajpurkar/squad_v2"):
        self.task_name = "qa"
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.dataset_id = dataset_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab_size = 260  # PAD/CLS/SEP + 256 byte values + spare
        self.num_classes = None
        self._datasets = {}

    def _build_split(self, split):
        ds = load_dataset(self.dataset_id, split=split)
        return SQuADv2ByteQADataset(ds, max_length=self.max_length)

    def get_dataloader(self, split="train"):
        split_map = {"train": "train", "val": "validation", "test": "validation"}
        if split not in split_map:
            raise ValueError(f"Unsupported split: {split}")
        mapped = split_map[split]

        if mapped not in self._datasets:
            self._datasets[mapped] = self._build_split(mapped)

        ds = self._datasets[mapped]

        def collate_fn(batch):
            return {
                "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
                "attention_mask": torch.stack([x["attention_mask"] for x in batch], dim=0),
                "start_positions": torch.stack([x["start_positions"] for x in batch], dim=0),
                "end_positions": torch.stack([x["end_positions"] for x in batch], dim=0),
            }

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
        )

    def preprocess_batch(self, batch):
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        labels = torch.stack(
            [
                batch["start_positions"],
                batch["end_positions"],
            ],
            dim=-1,
        ).to(self.device, non_blocking=True)
        return input_ids, labels, attention_mask


class SQuADv2QAModel(nn.Module):
    def __init__(self, backbone, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.backbone = backbone
        self.qa_head = nn.Linear(d_model, 2)

        if hasattr(self.backbone, "set_task_head"):
            self.backbone.set_task_head(self.qa_head, mode="token_qa")

    def forward(self, input_ids, mask=None, return_info=False):
        x = self.embedding(input_ids)
        need_info = return_info or (
            self.training and getattr(self.backbone, "train_mode", "base") == "controller"
        )
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
                raise RuntimeError("Backbone returned None in SQuADv2QAModel.forward")

        logits = self.qa_head(x)
        start_logits = logits[..., 0]
        end_logits = logits[..., 1]

        if self.training and info is not None and "K_soft" in info:
            return (start_logits, end_logits), info["K_soft"]

        if return_info:
            return (start_logits, end_logits), (info if info is not None else {})
        return (start_logits, end_logits)