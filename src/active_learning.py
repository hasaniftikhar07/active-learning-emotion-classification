"""
Implementation of a basic active learning loop for emotion classification.

The `ActiveLearningLoop` class encapsulates the logic for iteratively training a
model on a growing labelled dataset, querying unlabelled samples based on an
uncertainty measure and evaluating the model on a held‑out test set.

The uncertainty sampling strategy implemented here selects the samples for which
the model has the lowest maximum class probability.  This encourages the model
to ask for labels for examples it is least confident about.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .model import train_model, evaluate_model


class ActiveLearningLoop:
    """Basic active learning loop using uncertainty sampling."""

    def __init__(
        self,
        model,
        labelled_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        test_df: pd.DataFrame,
        sample_size: int = 10,
        verbose: bool = True,
    ) -> None:
        """
        Initialise the active learning loop.

        Parameters
        ----------
        model : sklearn pipeline or similar
            The base classifier.  Must implement `fit` and `predict_proba`.
        labelled_df : pd.DataFrame
            DataFrame containing the initial labelled samples.
        pool_df : pd.DataFrame
            DataFrame containing unlabelled samples available for querying.
        test_df : pd.DataFrame
            DataFrame containing the test set for evaluation.
        sample_size : int
            Number of samples to query in each iteration.
        verbose : bool
            Whether to print progress and evaluation metrics.
        """
        self.model = model
        self.labelled_df = labelled_df.copy()
        self.pool_df = pool_df.copy()
        self.test_df = test_df.copy()
        self.sample_size = sample_size
        self.verbose = verbose

    def run(self, num_iterations: int = 10) -> None:
        """Run the active learning loop for a given number of iterations."""
        for iteration in range(1, num_iterations + 1):
            if self.verbose:
                print(f"\n=== Iteration {iteration}/{num_iterations} ===")
                print(f"Labelled samples: {len(self.labelled_df)}, Pool size: {len(self.pool_df)}")

            # Train the model on current labelled data
            self.model = train_model(
                self.model,
                self.labelled_df["text"],
                self.labelled_df["label"],
            )

            # Evaluate current model on test set
            acc, report = evaluate_model(
                self.model, self.test_df["text"], self.test_df["label"]
            )
            if self.verbose:
                print(f"Test accuracy: {acc:.4f}")
                print(report)

            # If pool is empty or we have completed all iterations, stop
            if len(self.pool_df) == 0 or iteration == num_iterations:
                if self.verbose:
                    print("No more samples to query or reached maximum iterations.")
                break

            # Query the most uncertain samples from the pool
            query_indices = self._query()
            self._label_and_update(query_indices)

    def _query(self) -> np.ndarray:
        """Select indices of unlabelled samples to query using uncertainty sampling."""
        # Predict class probabilities for all pool samples
        probas = self.model.predict_proba(self.pool_df["text"])
        # Compute uncertainty as 1 - max predicted probability per sample
        uncertainties = 1.0 - np.max(probas, axis=1)
        # Get indices of the most uncertain samples
        query_indices = np.argsort(uncertainties)[-self.sample_size :]
        return query_indices

    def _label_and_update(self, query_indices: np.ndarray) -> None:
        """Simulate labelling the queried samples and add them to the labelled set."""
        # In a real scenario, you would ask an annotator to label these samples.
        # Here we assume the true labels are available in the pool_df for simulation.
        queried_samples = self.pool_df.iloc[query_indices]
        # Add queried samples to labelled set
        self.labelled_df = pd.concat([self.labelled_df, queried_samples], ignore_index=True)
        # Remove them from the pool
        self.pool_df = self.pool_df.drop(queried_samples.index).reset_index(drop=True)

    def evaluate(self) -> None:
        """Evaluate the trained model on the test set and print a summary."""
        acc, report = evaluate_model(
            self.model, self.test_df["text"], self.test_df["label"]
        )
        if self.verbose:
            print("\nFinal evaluation on test set:")
            print(f"Accuracy: {acc:.4f}")
            print(report)
