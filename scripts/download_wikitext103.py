#!/usr/bin/env python3
"""
Download WikiText-103 and export plain text splits to local directory.

Usage:
  python scripts/download_wikitext103.py \
      --output_dir data/wikitext103 \
      --dataset_config wikitext-103-raw-v1

This creates:
  output_dir/train.txt
  output_dir/val.txt
  output_dir/test.txt
"""

import argparse
import json
import os
from datasets import load_dataset


def _split_to_text(ds_split):
    lines = []
    for x in ds_split["text"]:
        if x is None:
            continue
        s = str(x)
        if len(s) == 0:
            continue
        lines.append(s)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Download WikiText-103 into txt files")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/wikitext103",
        help="Destination directory for train.txt/val.txt/test.txt",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-103-raw-v1",
        choices=["wikitext-103-raw-v1", "wikitext-103-v1"],
        help="HuggingFace WikiText config",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading WikiText ({args.dataset_config}) from HuggingFace...")
    train_ds = load_dataset("wikitext", args.dataset_config, split="train")
    val_ds = load_dataset("wikitext", args.dataset_config, split="validation")
    test_ds = load_dataset("wikitext", args.dataset_config, split="test")

    print("Converting splits to plain text...")
    train_txt = _split_to_text(train_ds)
    val_txt = _split_to_text(val_ds)
    test_txt = _split_to_text(test_ds)

    train_path = os.path.join(args.output_dir, "train.txt")
    val_path = os.path.join(args.output_dir, "val.txt")
    test_path = os.path.join(args.output_dir, "test.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        f.write(train_txt)
    with open(val_path, "w", encoding="utf-8") as f:
        f.write(val_txt)
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(test_txt)

    meta = {
        "dataset": "wikitext",
        "config": args.dataset_config,
        "num_train_rows": len(train_ds),
        "num_val_rows": len(val_ds),
        "num_test_rows": len(test_ds),
        "files": {
            "train": train_path,
            "val": val_path,
            "test": test_path,
        },
    }

    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Saved: {test_path}")
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()
