"""Shared helpers: seeds, paths, I/O, and small string utilities."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def find_workspace_root() -> Path:
    """Locate the folder that contains dataset/ and output/."""
    here = Path(__file__).resolve()
    candidates = [
        Path.cwd(),
        here.parents[3],  # src -> package -> code -> workspace
        here.parents[2],
        here.parents[1],
        Path.cwd().parent,
    ]
    for root in candidates:
        if (root / "dataset" / "train" / "train_source1.tsv").exists():
            return root
        if (root / "dataset" / "test" / "test_source1.tsv").exists():
            return root
    return here.parents[3] if len(here.parents) >= 4 else Path.cwd()


def workspace_paths() -> dict[str, Path]:
    root = find_workspace_root()
    return {
        "root": root,
        "train_dir": root / "dataset" / "train",
        "test_dir": root / "dataset" / "test",
        "output_dir": root / "output",
        "utils_dir": root / "utils",
    }


def parse_id_list(value) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    seen = set()
    out = []
    for part in text.split(","):
        item = part.strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def format_id_list(ids) -> str:
    seen = set()
    out = []
    for item in ids:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return ",".join(out)


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [tok for tok in text.split() if tok]
