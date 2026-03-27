#!/usr/bin/env python3
"""
Main entry point for running active learning for emotion classification.

This script loads a dataset, builds a baseline model and runs an active learning loop
to iteratively select samples for annotation.  At the end of the run it reports
evaluation metrics for the final model.

The dataset is expected to be a CSV file containing at least two columns:
```
text,label
some text expressing an emotion,joy
another example,sadness
```

You can run this script from the command line, for example:
```
python src/main.py --dataset_path data/my_dataset.csv --iterations 10 --sample_size 10
```
"""
import argparse
from pathlib import Path

from .data_utils import load_dataset, train_test_split_dataset
from .model import build_model
from .active_learning import ActiveLearningLoop



def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments for the main script."""
    parser = argparse.ArgumentParser(
        description="Active Learning for Emotion Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the CSV dataset file with columns 'text' and 'label'.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of active learning iterations to perform.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10,
        help="Number of samples to query for annotation in each iteration.",
    )
    parser.add_argument(
        "--initial_size",
        type=int,
        default=50,
        help="Number of labelled samples to use for the initial training set.",
    )
    return parser.parse_args()



def main() -> None:
    """Execute the active learning experiment."""
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # Load the full dataset (pandas DataFrame)
    df = load_dataset(str(dataset_path))

    # Split the dataset into a small labelled set and a pool of unlabelled samples
    labelled_df, pool_df, test_df = train_test_split_dataset(
        df, initial_size=args.initial_size
    )

    # Build a baseline model
    model = build_model()

    # Create and run the active learning loop
    al_loop = ActiveLearningLoop(
        model=model,
        labelled_df=labelled_df,
        pool_df=pool_df,
        test_df=test_df,
        sample_size=args.sample_size,
    )
    al_loop.run(num_iterations=args.iterations)

    # Evaluate the final model
    al_loop.evaluate()


if __name__ == "__main__":
    main()
