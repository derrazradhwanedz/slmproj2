"""Add answer-side quality profiles to the combined evaluation results.

MGCoT prompts carry 11 target quality metrics (target_* columns), predicted
by the DNN from the question alone. To check whether generated answers
actually move toward those targets, this script measures the same 11
metrics (src/metrics/profile, the exact implementations used to build the
DNN training data and the MGCoT targets) on:

    gold_*    the reference answer
    actual_*  the model's generated answer (final_answer)

Gold-answer cleaning (applied before measuring, so the metrics see prose):
    - SayCan gold is a JSON list of acceptable plans; the first plan is used.
    - GSM8K calculator annotations "<<...>>" are removed.
    - The "####" final-answer marker is removed, keeping the text around it.
Most gold answers are a single number/label ('9', 'True', 'A'), so gold_*
is only informative where gold answers are prose (GSM8K, SayCan).

Empty texts (e.g. deepseek-r1:8b answers stripped of their <think> block)
get NaN rather than a formula value computed on zero words.

Run from the repository root as: python profile_answers.py
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from metrics import profile as profile_metrics  # noqa: E402

INPUT_PATH = os.path.join(ROOT, "results", "evaluation", ".2ndrun", "combined_ALL_models.csv")
OUTPUT_PATH = os.path.join(ROOT, "results", "evaluation", ".2ndrun", "combined_ALL_models_profiled.csv")
METRIC_NAMES = profile_metrics.__all__  # same order as the target_* columns


def clean_gold(gold: str) -> str:
    """Turn a raw gold answer into the prose the profile metrics should see."""
    text = gold.strip()
    if text.startswith("["):
        try:
            plans = json.loads(text)
            if isinstance(plans, list) and plans:
                text = str(plans[0])
        except json.JSONDecodeError:
            pass
    text = re.sub(r"<<[^>]*>>", "", text)
    text = text.replace("####", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def profile(text: str, metrics: dict) -> dict:
    """All 11 profile metrics for one text; NaN for empty text."""
    if not text.strip():
        return {name: np.nan for name in METRIC_NAMES}
    return {name: metrics[name](text) for name in METRIC_NAMES}


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    metrics = {name: getattr(profile_metrics, name)() for name in METRIC_NAMES}

    # each gold answer repeats across models and mechanisms: measure it once
    gold_clean = df["gold_answer"].fillna("").astype(str).map(clean_gold)
    gold_cache = {g: profile(g, metrics) for g in gold_clean.unique()}
    gold_cols = pd.DataFrame([gold_cache[g] for g in gold_clean], index=df.index).add_prefix("gold_")

    answers = df["final_answer"].fillna("").astype(str)
    actual_cols = pd.DataFrame([profile(a, metrics) for a in answers], index=df.index).add_prefix("actual_")

    out = pd.concat([df, gold_cols, actual_cols], axis=1)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}: {out.shape[0]} rows, {out.shape[1]} columns "
          f"(+{gold_cols.shape[1]} gold_*, +{actual_cols.shape[1]} actual_*)")


if __name__ == "__main__":
    main()
