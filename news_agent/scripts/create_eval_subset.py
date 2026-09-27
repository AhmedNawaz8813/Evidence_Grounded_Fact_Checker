#!/usr/bin/env python3
"""
Create a stratified evaluation subset from the LIAR dataset.

Creates a balanced subset of 200 examples (100 true, 100 false) from
the validation set, using a fixed random seed for reproducibility.

Usage:
    python scripts/create_eval_subset.py --dataset liar --split val --n 200
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter


def create_stratified_subset(
    input_path: Path,
    output_path: Path,
    n_per_class: int = 100,
    seed: int = 42,
):
    """
    Create a stratified subset with equal class representation.

    Args:
        input_path: Path to full JSONL dataset
        output_path: Path to save subset
        n_per_class: Number of examples per class
        seed: Random seed for reproducibility
    """
    # Load all examples
    examples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples from {input_path}")

    # Group by label
    by_label = {}
    for ex in examples:
        label = ex.get("label", "unknown")
        by_label.setdefault(label, []).append(ex)

    print(f"Label distribution: {dict(Counter(ex['label'] for ex in examples))}")

    # Sample from each class
    rng = random.Random(seed)
    subset = []

    for label in sorted(by_label.keys()):
        pool = by_label[label]
        n = min(n_per_class, len(pool))
        sampled = rng.sample(pool, n)
        subset.extend(sampled)
        print(f"  {label}: sampled {n}/{len(pool)}")

    # Shuffle to mix classes
    rng.shuffle(subset)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in subset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(subset)} examples to {output_path}")
    print(f"Final distribution: {dict(Counter(ex['label'] for ex in subset))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create stratified evaluation subset")
    parser.add_argument("--dataset", default="liar", help="Dataset name")
    parser.add_argument("--split", default="val", help="Dataset split")
    parser.add_argument("--proc-dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--n", type=int, default=100, help="Number of examples per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", default=None, help="Output path")

    args = parser.parse_args()

    input_path = Path(args.proc_dir) / f"{args.dataset}_{args.split}.jsonl"
    output_path = Path(args.out or f"data/processed/{args.dataset}_{args.split}_subset{args.n * 2}.jsonl")

    if not input_path.exists():
        print(f"Error: Dataset not found: {input_path}")
        exit(1)

    create_stratified_subset(
        input_path=input_path,
        output_path=output_path,
        n_per_class=args.n,
        seed=args.seed,
    )
