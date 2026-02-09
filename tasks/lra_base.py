import os
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


class LRATask:
    """
    Base class for LRA (Long Range Arena) benchmark tasks.
    Handles dataset loading and a generic evaluation loop.
    """

    def __init__(self, task_name, batch_size=32, max_length=2048):
        self.task_name = task_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        dataset_path = os.path.join(DATA_DIR, task_name)
        if not os.path.isdir(dataset_path):
            raise ValueError(f"Dataset not found at {dataset_path}")

        print(f"Loading LRA task: {task_name} (from {dataset_path})...")
        self.dataset = load_from_disk(dataset_path)

    def get_dataloader(self, split="train"):
        return DataLoader(
            self.dataset[split],
            batch_size=self.batch_size,
            shuffle=(split == "train"),
        )

    def evaluate(self, model, split="test"):
        model.eval()
        dataloader = self.get_dataloader(split)
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {self.task_name}"):
                inputs, labels = self.preprocess_batch(batch)
                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"{self.task_name} {split} Accuracy: {accuracy:.4f}")
        return accuracy

    def preprocess_batch(self, batch):
        raise NotImplementedError
