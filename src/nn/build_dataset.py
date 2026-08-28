"""Builds question/answer quality-profile CSVs used to train QAPredictorNN.

Walks a dataset directory (one subfolder per benchmark, each containing a
test split file), profiles every question and its gold answer with the
metrics in metrics.profile, and writes two CSVs: one row per
record, one column per metric, for questions and answers respectively.

Run directly as: python src/nn/build_dataset.py --dataset-dir ... --q-df-out ... --a-df-out ...
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from logger import get_logger
from metrics import profile as profile_metrics

logger = get_logger(__name__)

METRIC_COLS: List[str] = profile_metrics.__all__


def _build_metric_instances() -> Dict[str, Any]:
    """Instantiate one metric object per name in METRIC_COLS."""
    return {name: getattr(profile_metrics, name)() for name in METRIC_COLS}


def profile_dataset(
    test_file_path: str,
    metric_instances: Dict[str, Any],
    max_records: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Profile every question/answer pair in one test split file.

    Args:
        test_file_path: Path to a JSONL file with "question"/"answer" fields.
        metric_instances: Mapping of metric name to a callable metric instance.
        max_records: Maximum number of records to read from the file.

    Returns:
        A (question_profile_df, answer_profile_df) pair, each with columns
        ['record_id', 'text'] + METRIC_COLS.
    """
    with open(test_file_path, "r", encoding="utf-8") as infile:
        content = infile.read().strip()

    records = [json.loads(line) for line in content.split("\n") if line.strip()]
    if not records:
        return pd.DataFrame(), pd.DataFrame()

    q_rows, a_rows = [], []
    for i, record in enumerate(records[:max_records]):
        question = str(record.get("question", ""))
        answer = str(record.get("answer", ""))

        q_metrics = {name: metric(question) for name, metric in metric_instances.items()}
        a_metrics = {name: metric(answer) for name, metric in metric_instances.items()}

        q_rows.append([i, question] + list(q_metrics.values()))
        a_rows.append([i, answer] + list(a_metrics.values()))

    columns = ["record_id", "text"] + METRIC_COLS
    return pd.DataFrame(q_rows, columns=columns), pd.DataFrame(a_rows, columns=columns)


def build_dataset(
    dataset_dir: str,
    q_df_out_path: str,
    a_df_out_path: str,
    test_filename: str = "test.jsonl",
    max_records: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Profile every benchmark under dataset_dir and save combined CSVs.

    Args:
        dataset_dir: Directory containing one subfolder per benchmark.
        q_df_out_path: Output CSV path for question profiles.
        a_df_out_path: Output CSV path for answer profiles.
        test_filename: Name of the split file to profile in each subfolder.
        max_records: Max records to profile per benchmark.

    Returns:
        The combined (question_profile_df, answer_profile_df) across all
        benchmarks found.

    Raises:
        FileNotFoundError: If dataset_dir does not exist.
        RuntimeError: If no test_filename files are found under dataset_dir.
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    metric_instances = _build_metric_instances()
    all_q_dfs, all_a_dfs = [], []

    for root, _, files in os.walk(dataset_dir):
        if test_filename not in files:
            continue

        dataset_name = os.path.basename(root)
        test_file_path = os.path.join(root, test_filename)
        q_df, a_df = profile_dataset(test_file_path, metric_instances, max_records)

        if q_df.empty:
            logger.warning(f"No records profiled for {dataset_name}")
            continue

        q_df["dataset"] = dataset_name
        a_df["dataset"] = dataset_name
        all_q_dfs.append(q_df)
        all_a_dfs.append(a_df)
        logger.info(f"Profiled {len(q_df)} records for {dataset_name}")

    if not all_q_dfs:
        raise RuntimeError(f"No {test_filename} files found under {dataset_dir}")

    final_q_df = pd.concat(all_q_dfs, ignore_index=True)
    final_a_df = pd.concat(all_a_dfs, ignore_index=True)

    os.makedirs(os.path.dirname(q_df_out_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(a_df_out_path) or ".", exist_ok=True)
    final_q_df.to_csv(q_df_out_path, index=False)
    final_a_df.to_csv(a_df_out_path, index=False)
    logger.info(f"Saved: Q={final_q_df.shape}, A={final_a_df.shape}")

    return final_q_df, final_a_df


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone dataset building."""
    parser = argparse.ArgumentParser(description="Build QAPredictorNN training CSVs.")
    parser.add_argument("--dataset-dir", required=True, help="Directory with one subfolder per benchmark.")
    parser.add_argument("--q-df-out", required=True, help="Output path for the question profile CSV.")
    parser.add_argument("--a-df-out", required=True, help="Output path for the answer profile CSV.")
    parser.add_argument("--test-filename", default="test.jsonl")
    parser.add_argument("--max-records", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_dataset(
        dataset_dir=args.dataset_dir,
        q_df_out_path=args.q_df_out,
        a_df_out_path=args.a_df_out,
        test_filename=args.test_filename,
        max_records=args.max_records,
    )
