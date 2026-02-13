import argparse
import math
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_factory import build_attd_backbone
from tasks.listops import ListOpsTask, ListOpsModel
from tasks.text_cls import TextClsTask, TextClsModel
from tasks.pathfinder import PathfinderTask, PathfinderModel
from tasks.lm_wikitext import WikiTextLMTask, WikiTextLMModel
from tasks.qa_squad_v2 import SQuADv2Task, SQuADv2QAModel


def _is_qa_output(logits, labels):
    return isinstance(logits, (tuple, list)) and len(logits) == 2 and labels.dim() == 2 and labels.size(-1) == 2

def _compute_ce_loss(logits, labels):
    if _is_qa_output(logits, labels):
        start_logits, end_logits = logits
        start_labels = labels[:, 0]
        end_labels = labels[:, 1]
        return 0.5 * (F.cross_entropy(start_logits, start_labels) + F.cross_entropy(end_logits, end_labels))
    if logits.dim() == 3 and labels.dim() == 2:
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=0)
    return F.cross_entropy(logits, labels)


def _compute_accuracy(logits, labels):
    if _is_qa_output(logits, labels):
        start_logits, end_logits = logits
        start_preds = torch.argmax(start_logits, dim=-1)
        end_preds = torch.argmax(end_logits, dim=-1)
        exact = ((start_preds == labels[:, 0]) & (end_preds == labels[:, 1])).sum().item()
        total = labels.size(0)
        return exact, total
    if logits.dim() == 3 and labels.dim() == 2:
        preds = torch.argmax(logits, dim=-1)
        mask = (labels != 0)
        correct = ((preds == labels) & mask).sum().item()
        total = mask.sum().item()
        return correct, total
    preds = torch.argmax(logits, dim=-1)
    return (preds == labels).sum().item(), labels.size(0)

def _compute_controller_target(logits, model):
    if isinstance(logits, (tuple, list)) and len(logits) == 2:
        start_logits, end_logits = logits
        start_probs = F.softmax(start_logits, dim=-1)
        end_probs = F.softmax(end_logits, dim=-1)
        ent_start = -(start_probs * torch.log(start_probs.clamp_min(1e-8))).sum(dim=-1)
        ent_end = -(end_probs * torch.log(end_probs.clamp_min(1e-8))).sum(dim=-1)
        ent_norm = 0.5 * (
            ent_start / math.log(start_logits.size(-1)) + ent_end / math.log(end_logits.size(-1))
        )
        return ent_norm * model.backbone.controller.k_max

    probs = F.softmax(logits, dim=-1)
    ent = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
    ent_norm = ent / math.log(logits.size(-1))
    return ent_norm * model.backbone.controller.k_max

def train_one_epoch(model, dataloader, optimizer, criterion, device, clip_grad=1.0):
    model.train()
    total_loss, correct, total, num_batches = 0.0, 0, 0, 0

    pbar = tqdm(dataloader, desc="Training", miniters=10)
    is_controller = getattr(model.backbone, "train_mode", "base") == "controller"

    for batch in pbar:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        mask = batch[2].to(device) if len(batch) > 2 else None

        optimizer.zero_grad()
        outputs = model(inputs, mask=mask) if mask is not None else model(inputs)

        if is_controller and isinstance(outputs, tuple):
            logits, K_soft = outputs
            ce_loss = _compute_ce_loss(logits, labels)

            with torch.no_grad():
                k_target = _compute_controller_target(logits, model)

            ctrl_loss = F.mse_loss(K_soft.float(), k_target.float())
            loss = ce_loss + model.backbone.lambda_cost * K_soft.mean() + model.backbone.lambda_ctrl * ctrl_loss
            out_for_acc = logits
        else:
            if isinstance(outputs, tuple) and not _is_qa_output(outputs, labels):
                outputs = outputs[0]
            loss = _compute_ce_loss(outputs, labels)
            out_for_acc = outputs

        loss.backward()
        if clip_grad is not None and clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        c, t = _compute_accuracy(out_for_acc, labels)
        correct += c
        total += t

        pbar.set_postfix(loss=loss.item(), acc=correct / max(total, 1))

    return total_loss / max(num_batches, 1), correct / max(total, 1)


def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0

    k_soft_all = []
    k_hard_all = []

    for batch in tqdm(dataloader, desc="Evaluating", miniters=10):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        mask = batch[2].to(device) if len(batch) > 2 else None

        outputs = model(inputs, mask=mask, return_info=True) if mask is not None else model(inputs, return_info=True)

        info = {}
        if isinstance(outputs, tuple):
            logits, info = outputs
        else:
            logits = outputs

        c, t = _compute_accuracy(logits, labels)
        correct += c
        total += t

        if isinstance(info, dict):
            if "K_soft" in info and info["K_soft"] is not None:
                k_soft = info["K_soft"]
                if isinstance(k_soft, torch.Tensor):
                    k_soft_all.append(k_soft.detach().float().cpu())
            if "k_dyn" in info:
                k_hard_all.append(float(info["k_dyn"]))

    acc = correct / max(total, 1)

    stats = {}
    if len(k_soft_all) > 0:
        k_soft_cat = torch.cat(k_soft_all)
        stats["avg_k_soft"] = float(k_soft_cat.mean().item())
        stats["p90_k_soft"] = float(torch.quantile(k_soft_cat, 0.9).item())
    if len(k_hard_all) > 0:
        k_hard_t = torch.tensor(k_hard_all, dtype=torch.float32)
        stats["avg_k_hard"] = float(k_hard_t.mean().item())

    return acc, stats


def main():
    parser = argparse.ArgumentParser(description="Run LRA/ATTD tasks with Mamba")

    parser.add_argument("--task", type=str, default="text", choices=["listops", "listops_small", "text", "pathfinder", "lm", "qa"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--clip_grad", type=float, default=1.0)

    parser.add_argument("--k_max", type=int, default=5)
    parser.add_argument("--inner_lr", type=float, default=0.1)
    parser.add_argument("--train_mode", type=str, default="base")
    parser.add_argument("--no_attd", action="store_true")

    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--lambda_cost", type=float, default=1e-3)
    parser.add_argument("--lambda_ctrl", type=float, default=1.0)
    parser.add_argument("--entropy_weight", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=0.2)
    parser.add_argument("--confidence_tau", type=float, default=0.0)
    parser.add_argument("--adapter_grad_clip", type=float, default=1.0)
    parser.add_argument("--max_delta_norm", type=float, default=0.5)

    parser.add_argument("--text_dataset", type=str, default="agnews", choices=["agnews", "imdb"])
    parser.add_argument("--data_dir", type=str, default="data/agnews")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--lm_dataset", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--qa_dataset", type=str, default="rajpurkar/squad_v2")
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume_weights_only", action="store_true")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.no_attd and args.train_mode == "controller":
        raise ValueError("--no_attd is incompatible with --train_mode controller")

    if args.task in ("listops", "listops_small"):
        task_manager = ListOpsTask(batch_size=args.batch_size, data_name=args.task)
        vocab_size, num_classes = task_manager.vocab_size, task_manager.num_classes
    elif args.task == "text":
        task_manager = TextClsTask(
            batch_size=args.batch_size,
            max_length=args.max_length,
            dataset_name=args.text_dataset,
            data_dir=args.data_dir,
            num_workers=args.num_workers,
        )
        vocab_size, num_classes = task_manager.vocab_size, task_manager.num_classes
    elif args.task == "pathfinder":
        task_manager = PathfinderTask(batch_size=args.batch_size)
        num_classes = task_manager.num_classes
        vocab_size = None
    elif args.task == "lm":
        task_manager = WikiTextLMTask(
            batch_size=args.batch_size,
            block_size=args.block_size,
            dataset_name=args.lm_dataset,
            data_dir=args.data_dir,
            num_workers=args.num_workers,
        )
        vocab_size, num_classes = task_manager.vocab_size, task_manager.num_classes
    elif args.task == "qa":
        task_manager = SQuADv2Task(
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=args.num_workers,
            dataset_id=args.qa_dataset,
        )
        vocab_size, num_classes = task_manager.vocab_size, task_manager.num_classes
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    config = {
        "n_layers": args.n_layers,
        "d_model": args.d_model,
        "d_state": 16,
        "expand": 2,
        "rank": 8,
        "dropout": args.dropout,
        "k_max": args.k_max,
        "inner_lr": args.inner_lr,
        "train_mode": args.train_mode,
        "tau": args.tau,
        "lambda_cost": args.lambda_cost,
        "lambda_ctrl": args.lambda_ctrl,
        "entropy_weight": args.entropy_weight,
        "kl_weight": args.kl_weight,
        "confidence_tau": args.confidence_tau,
        "adapter_grad_clip": args.adapter_grad_clip,
        "max_delta_norm": args.max_delta_norm,
    }

    ckpt = None
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        saved_config = ckpt.get("config", {})
        for key in ["n_layers", "d_model", "d_state", "expand", "rank", "dropout"]:
            if key in saved_config:
                config[key] = saved_config[key]
        config["k_max"] = args.k_max
        config["train_mode"] = args.train_mode

    backbone = build_attd_backbone(config)

    if args.task in ("listops", "listops_small"):
        model = ListOpsModel(backbone, vocab_size, args.d_model, num_classes)
    elif args.task == "text":
        model = TextClsModel(backbone, vocab_size, args.d_model, num_classes)
    elif args.task == "lm":
        model = WikiTextLMModel(backbone, vocab_size, args.d_model)
    elif args.task == "qa":
        model = SQuADv2QAModel(backbone, vocab_size, args.d_model)
    else:
        model = PathfinderModel(backbone, args.d_model, num_classes)

    model.to(device)

    if args.train_mode == "controller":
        optimizer = optim.AdamW(model.backbone.controller.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()
    ckpt_dir = os.path.join(args.ckpt_dir, args.task)
    os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch = 0
    best_val_acc = 0.0

    if ckpt is not None:
        state = ckpt["model_state_dict"]

        if args.train_mode == "controller":
            state = {k: v for k, v in state.items() if not k.startswith("backbone.controller.")}

        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"load_state_dict(strict=False): missing={len(missing)}, unexpected={len(unexpected)}")

        best_val_acc = ckpt.get("best_val_acc", ckpt.get("best_test_acc", 0.0))

        if not args.resume_weights_only:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt["epoch"] + 1
                print(f"Resumed full state at epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")
            except Exception as e:
                print(f"Warning: optimizer resume failed ({e}); fallback to weights-only.")
                start_epoch = 0
        else:
            print("Resumed weights only.")

    if args.no_attd:
        model.backbone.disable_attd = True

    print(
        f"Run config: task={args.task}, train_mode={args.train_mode}, no_attd={args.no_attd}, "
        f"k_max={config['k_max']}, inner_lr={config['inner_lr']}"
    )

    train_loader = task_manager.get_dataloader(split="train")
    val_loader = task_manager.get_dataloader(split="val")
    test_loader = task_manager.get_dataloader(split="test")

    def wrapped_loader(loader):
        for batch in loader:
            yield task_manager.preprocess_batch(batch)

    if args.epochs == 0:
        test_acc, test_stats = evaluate(model, wrapped_loader(test_loader), device)
        print(
            f"Test Acc: {test_acc:.4f}, "
            f"avg_k_soft: {test_stats.get('avg_k_soft', -1):.3f}, "
            f"p90_k_soft: {test_stats.get('p90_k_soft', -1):.3f}, "
            f"avg_k_hard: {test_stats.get('avg_k_hard', -1):.3f}"
        )
        return

    print(f"Starting training on {args.task} ...")
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, wrapped_loader(train_loader), optimizer, criterion, device, clip_grad=args.clip_grad
        )
        val_acc, val_stats = evaluate(model, wrapped_loader(val_loader), device)

        print(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
            f"avg_k_soft: {val_stats.get('avg_k_soft', -1):.3f}, "
            f"p90_k_soft: {val_stats.get('p90_k_soft', -1):.3f}, "
            f"avg_k_hard: {val_stats.get('avg_k_hard', -1):.3f}"
        )

        ckpt_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "config": config,
            "args": vars(args),
        }
        torch.save(ckpt_state, os.path.join(ckpt_dir, "latest.pt"))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_state["best_val_acc"] = best_val_acc
            torch.save(ckpt_state, os.path.join(ckpt_dir, "best.pt"))
            print(f"New best val acc: {best_val_acc:.4f}")

    best_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"], strict=False)

    final_test_acc, test_stats = evaluate(model, wrapped_loader(test_loader), device)
    print(
        f"Final Test Acc (selected by val): {final_test_acc:.4f}, "
        f"avg_k_soft: {test_stats.get('avg_k_soft', -1):.3f}, "
        f"p90_k_soft: {test_stats.get('p90_k_soft', -1):.3f}, "
        f"avg_k_hard: {test_stats.get('avg_k_hard', -1):.3f}"
    )
    print(f"Training complete. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
