#!/usr/bin/env python3
"""
Download WikiText-103 (streaming) and export plain text splits to local directory.

Usage:
  export XXX
  export XXX 
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


def write_split(ds_split, out_path):
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in ds_split:
            x = item.get("text", None)
            if x is None:
                continue
            s = str(x)
            if not s:
                continue
            f.write(s + "\n")
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description="Download WikiText-103 into txt files")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="XXX",
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

    print(f"Loading WikiText ({args.dataset_config}) from HuggingFace (streaming)...")
    train_ds = load_dataset("wikitext", args.dataset_config, split="train", streaming=True)
    val_ds = load_dataset("wikitext", args.dataset_config, split="validation", streaming=True)
    test_ds = load_dataset("wikitext", args.dataset_config, split="test", streaming=True)

    train_path = os.path.join(args.output_dir, "train.txt")
    val_path = os.path.join(args.output_dir, "val.txt")
    test_path = os.path.join(args.output_dir, "test.txt")

    print("Writing splits to plain text (streaming)...")
    n_train = write_split(train_ds, train_path)
    print(f"  train: {n_train} lines -> {train_path}")
    n_val = write_split(val_ds, val_path)
    print(f"  val:   {n_val} lines -> {val_path}")
    n_test = write_split(test_ds, test_path)
    print(f"  test:  {n_test} lines -> {test_path}")

    meta = {
        "dataset": "wikitext",
        "config": args.dataset_config,
        "num_train_rows": n_train,
        "num_val_rows": n_val,
        "num_test_rows": n_test,
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
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()
