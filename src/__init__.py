"""
Active Learning for Emotion Classification.

This package contains modules for loading data, building a baseline model and running
an active learning loop.  Use the `main.py` script as the entry point for running
experiments.
"""

# Expose high‑level classes/functions to the package namespace
from .data_utils import load_dataset  # noqa: F401
from .model import build_model, train_model, evaluate_model  # noqa: F401
from .active_learning import ActiveLearningLoop  # noqa: F401
