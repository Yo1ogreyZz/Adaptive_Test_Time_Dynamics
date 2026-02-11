import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys


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
    is_controller = getattr(model.backbone, "train_mode", "base") == "controller"
    for batch in pbar:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        mask = batch[2].to(device) if len(batch) > 2 else None
        optimizer.zero_grad()
        outputs = model(inputs, mask=mask) if mask is not None else model(inputs)
        if is_controller and isinstance(outputs, tuple):
            logits, K_soft = outputs
            loss = criterion(logits, labels) + model.backbone.lambda_cost * K_soft.mean()
            outputs = logits
        else:
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
        inputs, labels = batch[0].to(device), batch[1].to(device)
        mask = batch[2].to(device) if len(batch) > 2 else None

        outputs = model(inputs, mask=mask) if mask is not None else model(inputs)
        predictions = torch.argmax(outputs, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return correct / total

def main():
    parser = argparse.ArgumentParser(description="Run LRA tasks with ATTD-Mamba")
    parser.add_argument("--task", type=str, default="listops_small", choices=["listops", "listops_small", "text", "pathfinder"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--no_attd", action="store_true"); parser.add_argument("--train_mode", type=str, default="base"); parser.add_argument("--inner_lr", type=float, default=0.1)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.task in ("listops", "listops_small"):
        task_manager = ListOpsTask(batch_size=args.batch_size, data_name=args.task)
        vocab_size = task_manager.vocab_size
        num_classes = task_manager.num_classes
    elif args.task == "text":
        task_manager = TextClsTask(batch_size=args.batch_size)
        vocab_size = task_manager.vocab_size
        num_classes = task_manager.num_classes
    elif args.task == "pathfinder":
        task_manager = PathfinderTask(batch_size=args.batch_size)
        num_classes = task_manager.num_classes
    
    config = {
        "n_layers": args.n_layers,
        "d_model": args.d_model,
        "d_state": 16,
        "expand": 2,
        "rank": 8,
        "k_max": 5 if not args.no_attd else 0,
        "inner_lr": args.inner_lr, "train_mode": args.train_mode
    }
    
    backbone = build_attd_backbone(config)
    
    if args.task in ("listops", "listops_small"):
        model = ListOpsModel(backbone, vocab_size, args.d_model, num_classes)
    elif args.task == "text":
        model = TextClsModel(backbone, vocab_size, args.d_model, num_classes)
    elif args.task == "pathfinder":
        model = PathfinderModel(backbone, args.d_model, num_classes)
        
    model.to(device)
    
    if args.train_mode == "controller": optimizer = optim.AdamW(model.backbone.controller.parameters(), lr=args.lr)
    else: optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = os.path.join(args.ckpt_dir, args.task)
    os.makedirs(ckpt_dir, exist_ok=True)
    start_epoch = 0
    best_test_acc = 0.0

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_test_acc = ckpt.get("best_test_acc", 0.0)
        print(f"Resumed at epoch {start_epoch}, best_test_acc={best_test_acc:.4f}")

    # 5. Training Loop
    train_loader = task_manager.get_dataloader(split="train")
    test_loader = task_manager.get_dataloader(split="test")

    print(f"Starting training on {args.task} (ATTD: {not args.no_attd})...")
    for epoch in range(start_epoch, args.epochs):
        def wrapped_loader(loader):
            for batch in loader:
                yield task_manager.preprocess_batch(batch)

        train_loss, train_acc = train_one_epoch(model, wrapped_loader(train_loader), optimizer, criterion, device)
        test_acc = evaluate(model, wrapped_loader(test_loader), device)

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

        ckpt_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "best_test_acc": best_test_acc,
            "config": config,
            "args": vars(args),
        }
        torch.save(ckpt_state, os.path.join(ckpt_dir, "latest.pt"))

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            ckpt_state["best_test_acc"] = best_test_acc
            torch.save(ckpt_state, os.path.join(ckpt_dir, "best.pt"))
            print(f"New best test acc: {best_test_acc:.4f}, saved to {ckpt_dir}/best.pt")

    print(f"Training complete. Best test acc: {best_test_acc:.4f}")

if __name__ == "__main__":
    main()
