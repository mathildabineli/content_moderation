"""Dataset preparation utilities for content moderation.

Provides helper functions to load, split, and tokenize datasets
for multi-label text classification tasks.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset as HFDataset
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from .config import CFG


def resolve_text_col(df: pd.DataFrame, hint: str = "prompt_clean") -> str:
    """Resolve the main text column in a dataframe.

    Tries the hint first, then searches for likely text fields.
    """
    if hint in df.columns:
        return hint
    for c in ("text", "content", "message", "body", "input", "user_input", "prompt"):
        if c in df.columns:
            return c
    obj = df.select_dtypes(include="object")
    lengths = obj.apply(lambda s: s.astype(str).str.len().mean())
    return lengths.sort_values(ascending=False).index[0]


def detect_labels(df: pd.DataFrame, exclude: list[str]) -> List[str]:
    """Detect binary label columns in the dataset.

    Excludes non-label columns and returns numeric/binary ones.
    """
    cand = df.columns.difference(list(exclude))
    num = df[cand].select_dtypes(include=[np.number, bool]).columns
    sub = df[num].dropna()
    is_bin = (sub.max(0) <= 1) & (sub.min(0) >= 0)
    labs = list(num[is_bin])
    return labs or list(num)


def split_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], str]:
    """Split dataset into train, validation, and test sets.

    Uses multilabel stratified sampling for balanced splits.
    """
    df = pd.read_csv(CFG.data_csv)
    text_col = resolve_text_col(df)
    labels = detect_labels(df, exclude=[text_col])
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0].reset_index(drop=True)
    df[labels] = (df[labels] > 0.5).astype(np.int8)

    s1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.20, random_state=CFG.seed
    )
    trp_idx, te_idx = next(s1.split(df[[text_col]], df[labels]))
    train_pool, test_df = df.iloc[trp_idx], df.iloc[te_idx]

    s2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.10, random_state=CFG.seed + 1
    )
    tr_idx, va_idx = next(s2.split(train_pool[[text_col]], train_pool[labels]))
    train_df = train_pool.iloc[tr_idx].reset_index(drop=True)
    val_df = train_pool.iloc[va_idx].reset_index(drop=True)
    return train_df, val_df, test_df, labels, text_col


def build_hf_dataset(tokenizer, df: pd.DataFrame, text_col: str, labels: List[str]):
    """Convert a pandas dataframe into a HuggingFace Dataset.

    Tokenizes texts and attaches multi-label targets.
    """
    enc = tokenizer(
        df[text_col].tolist(),
        return_tensors="np",
        truncation=True,
        padding=True,
        max_length=CFG.max_len,
    )
    return HFDataset.from_dict(
        {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": df[labels].values.astype(np.float32),
        }
    )
