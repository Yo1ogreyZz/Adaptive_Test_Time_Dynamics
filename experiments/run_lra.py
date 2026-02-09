import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_factory import build_attd_backbone
from tasks.listops import ListOpsTask, ListOpsModel
from tasks.text_cls import TextClsTask, TextClsModel
from tasks.pathfinder import PathfinderTask, PathfinderModel

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        predictions = torch.argmax(outputs, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=loss.item(), acc=correct/total)

    return total_loss / num_batches, correct / total

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
    return correct / total

def main():
    parser = argparse.ArgumentParser(description="Run LRA tasks with ATTD-Mamba")
    parser.add_argument("--task", type=str, default="listops", choices=["listops", "text", "pathfinder"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--no_attd", action="store_true", help="Disable ATTD (pure Mamba)")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Task
    if args.task == "listops":
        task_manager = ListOpsTask(batch_size=args.batch_size)
        vocab_size = task_manager.vocab_size
        num_classes = task_manager.num_classes
    elif args.task == "text":
        task_manager = TextClsTask(batch_size=args.batch_size)
        vocab_size = task_manager.vocab_size
        num_classes = task_manager.num_classes
    elif args.task == "pathfinder":
        task_manager = PathfinderTask(batch_size=args.batch_size)
        num_classes = task_manager.num_classes
    
    # 2. Build Model
    config = {
        "n_layers": args.n_layers,
        "d_model": args.d_model,
        "d_state": 16,
        "expand": 2,
        "rank": 8,
        "k_max": 5 if not args.no_attd else 0,
        "inner_lr": 1e-3
    }
    
    backbone = build_attd_backbone(config)
    
    if args.task == "listops":
        model = ListOpsModel(backbone, vocab_size, args.d_model, num_classes)
    elif args.task == "text":
        model = TextClsModel(backbone, vocab_size, args.d_model, num_classes)
    elif args.task == "pathfinder":
        model = PathfinderModel(backbone, args.d_model, num_classes)
        
    model.to(device)
    
    # 3. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training Loop
    train_loader = task_manager.get_dataloader(split="train")
    test_loader = task_manager.get_dataloader(split="test")
    
    print(f"Starting training on {args.task} (ATTD: {not args.no_attd})...")
    for epoch in range(args.epochs):
        # We need to wrap the dataloader to match our train_one_epoch expectations
        # because task_manager.preprocess_batch is needed
        def wrapped_loader(loader):
            for batch in loader:
                yield task_manager.preprocess_batch(batch)
        
        train_loss, train_acc = train_one_epoch(model, wrapped_loader(train_loader), optimizer, criterion, device)
        test_acc = evaluate(model, wrapped_loader(test_loader), device)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()
