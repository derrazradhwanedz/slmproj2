"""Add task accuracy to the combined evaluation results.

The comparison metrics stored by main.py measure overlap with the reference
answer; none of them says whether the task was solved. This script extracts
the final answer from each response with a per-dataset rule and compares it
with the reference, adding two columns:

    answer_extracted  1 if a final answer of the expected form was found
    accuracy          1 if that answer matches the reference, else 0

Extraction rules (src: analysis_v2.score_record):
    asdiv, gsm8k, MultiArith, SVAMP  last number in the response
    aqua                             option letter after an answer marker, else the last "(X)"
    StrategyQA                       first yes/no/true/false after the last answer marker
    QASports                         last plausibility verdict
    clutrr                           last kinship term
    date                             last MM/DD/YYYY date
    saycan                           the sequence of action calls, matched against any accepted plan

Run from the repository root as: python compute_accuracy.py
"""

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from analysis_v2 import score_record  # noqa: E402

INPUT_PATH = os.path.join(ROOT, "results", "evaluation", ".2ndrun", "combined_ALL_models_profiled.csv")
OUTPUT_PATH = INPUT_PATH  # the accuracy columns are added in place


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    scored = [score_record(dataset, str(answer) if pd.notna(answer) else "", str(gold))
              for dataset, answer, gold in zip(df["dataset"], df["final_answer"], df["gold_answer"])]
    scores = pd.DataFrame(scored, index=df.index)
    df["accuracy"] = scores["accuracy"]
    df["answer_extracted"] = scores["answered"]
    df.to_csv(OUTPUT_PATH, index=False)

    summary = df.pivot_table(index="model", columns="mechanism", values="accuracy").mul(100).round(1)
    print(f"Saved {OUTPUT_PATH}: {len(df)} rows")
    print("\nTask accuracy (%), all datasets:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
