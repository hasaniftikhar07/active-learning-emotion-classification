"""
Utility functions for loading and splitting data for emotion classification.

This module provides simple helpers for reading a CSV dataset, performing train–test
splits and selecting an initial labelled set for active learning.  The dataset
should contain at least two columns: `text` and `label`.  Additional columns will
be ignored by the active learning loop.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path: str) -> pd.DataFrame:
    """Load a dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.  The file must contain at least two columns: `text`
        and `label`.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the dataset.
    """
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            "Dataset must contain 'text' and 'label' columns. Found: "
            + ", ".join(df.columns)
        )
    return df[["text", "label"]]


def train_test_split_dataset(
    df: pd.DataFrame, initial_size: int = 50, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into labelled, unlabelled (pool) and test sets.

    A small initial set of labelled examples is taken from the training portion of
    the dataset; the remainder of the training data becomes the unlabelled pool
    used in active learning.  The initial labelled set is selected randomly.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the full dataset with `text` and `label` columns.
    initial_size : int, default 50
        Number of labelled samples to return for the initial training set.
    test_size : float, default 0.2
        Fraction of the dataset to reserve for testing.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    tuple of (labelled_df, pool_df, test_df)
        * `labelled_df` – DataFrame of initially labelled samples.
        * `pool_df` – DataFrame of unlabelled samples to be queried in active learning.
        * `test_df` – DataFrame reserved for evaluating the model.
    """
    # First perform a train–test split
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )

    # Then select initial labelled samples from the training set
    if initial_size >= len(train_df):
        raise ValueError(
            f"initial_size ({initial_size}) must be smaller than the size of the training set ({len(train_df)})."
        )
    labelled_df = train_df.sample(n=initial_size, random_state=random_state)
    pool_df = train_df.drop(labelled_df.index).reset_index(drop=True)
    labelled_df = labelled_df.reset_index(drop=True)
    return labelled_df, pool_df, test_df.reset_index(drop=True)
