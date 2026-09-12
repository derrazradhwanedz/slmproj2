"""SCoT vs MGCoT analysis of the second evaluation run (results/evaluation/.2ndrun).

Compares the two mechanisms along two axes - cross-dataset (pooled over
models) and cross-model (pooled over datasets) - for response quality and
hardware cost. The design is paired: every record is answered by the same
model under both mechanisms, so all tests are paired (Wilcoxon signed-rank
for continuous metrics, exact McNemar for task accuracy), with bootstrap 95%
CIs, matched-pairs rank-biserial effect sizes, and Benjamini-Hochberg FDR
correction within each family of tests.

Run from the repository root as: python analysis_v2.py
Outputs go to results/analysis/2ndrun/.
"""

import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from metrics import profile as profile_metrics  # noqa: E402

RUN_DIR = os.path.join(ROOT, "results", "evaluation", ".2ndrun")
OUT_DIR = os.path.join(ROOT, "results", "analysis", "2ndrun")
A_SCALER_PATH = os.path.join(ROOT, "results", "weights", "a_scaler.pkl")

# num_predict=300 and the profiler sleeps sample_interval=0.5 s after every
# streamed chunk, so a response that hit the token cap takes >= 150 s.
TOKEN_CAP_SECONDS = 150.0
# deepseek-r1:8b spends the whole 300-token budget inside <think>, which the
# client strips, leaving ~97% empty answers - not a valid comparison.
EXCLUDED_MODELS = {"deepseek-r1_8b"}
SEED = 42
N_BOOT = 5000
ALPHA = 0.05

DATASET_ORDER = ["aqua", "asdiv", "clutrr", "date", "gsm8k", "MultiArith", "QASports", "saycan", "StrategyQA", "SVAMP"]
# QASports questions reach the model without the "is this sentence plausible?"
# framing, and SayCan without the robot's action/object inventory, so both are
# ill-posed as prompted; they are reported but kept out of the headline pools.
ILL_POSED_DATASETS = {"QASports", "saycan"}

# (column, higher_is_better, binary)
QUALITY_METRICS = [
    ("accuracy", True, True),
    ("semantic_similarity", True, False),
    ("bert_score_f1", True, False),
    ("rougeL_f1", True, False),
    ("token_f1", True, False),
    ("char_f1", True, False),
]
COST_METRICS = [
    ("output_words", False, False),
    ("truncated", False, True),
    ("delta_time_seconds", False, False),
    ("cpu_usage_avg_percent", False, False),
    ("gpu_usage_avg_percent", False, False),
    ("gpu_memory_avg_mb", False, False),
    ("ram_usage_avg_mb", False, False),
]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_results() -> pd.DataFrame:
    """Load every combined_*-ALL.csv in RUN_DIR into one frame with a model column."""
    frames = []
    for path in sorted(glob.glob(os.path.join(RUN_DIR, "combined_*-ALL.csv"))):
        model = os.path.basename(path).split("-", 1)[1].rsplit("-ALL", 1)[0]
        frames.append(pd.read_csv(path).assign(model=model))
    df = pd.concat(frames, ignore_index=True)
    df["final_answer"] = df["final_answer"].fillna("").astype(str)
    df["gold_answer"] = df["gold_answer"].fillna("").astype(str)
    df["output_words"] = df["final_answer"].str.split().str.len()
    df["truncated"] = (df["delta_time_seconds"] >= TOKEN_CAP_SECONDS).astype(float)
    df["empty_answer"] = (df["final_answer"].str.strip() == "").astype(float)
    return df


# --------------------------------------------------------------------------
# Task-accuracy answer extraction
# --------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
_ANSWER_MARKER_RE = re.compile(r"answer", re.I)
_KIN_SYNONYMS = {
    "granddad": "grandfather", "grandpa": "grandfather", "grandma": "grandmother",
    "granny": "grandmother", "dad": "father", "mom": "mother", "mum": "mother",
}
_KIN_RE = re.compile(
    r"\b(grand(?:father|mother|son|daughter)|(?:father|mother|son|daughter|brother|sister)-in-law"
    r"|father|mother|son|daughter|brother|sister|uncle|aunt|nephew|niece|husband|wife|cousin"
    r"|granddad|grandpa|grandma|granny|dad|mom|mum)s?\b",
    re.I,
)
_SAYCAN_CALL_RE = re.compile(r"\b(find|pick|put|go|done)\s*\(\s*([^)]*?)\s*\)", re.I)


def _gold_tail(gold: str) -> str:
    return gold.split("####")[-1].strip()


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _after_last_marker(text: str) -> str:
    """Text after the last 'answer' marker, or '' if there is none."""
    matches = list(_ANSWER_MARKER_RE.finditer(text))
    return text[matches[-1].end():] if matches else ""


_FINAL_MARKER_RE = re.compile(
    r"final answer|answer is|answer:|\bstate\b|therefore|thus|in total|in summary", re.I)


def extract_number(text: str) -> Optional[float]:
    """Numeric answer, following the zero-shot CoT answer-trigger convention.

    If the response has a final-answer marker, read the first sentence after
    the last marker: the last number if that sentence is an equation (has
    '='), else the first number ("John is 475 miles from home after 4
    hours" -> 475). Without a marker, fall back to the last number overall.
    """
    markers = list(_FINAL_MARKER_RE.finditer(text))
    if markers:
        segment = re.split(r"\n|(?<=[.!?])\s", text[markers[-1].end():].lstrip(" *:#\n"), maxsplit=1)[0]
        nums = _NUMBER_RE.findall(segment)
        if nums:
            return _to_float(nums[-1] if "=" in segment else nums[0])
    nums = _NUMBER_RE.findall(text)
    if nums:
        return _to_float(nums[-1])
    words = re.findall(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", text, re.I)
    return float(_NUMBER_WORDS[words[-1].lower()]) if words else None


_NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}


def _aqua_options(question: str) -> Dict[str, str]:
    """Parse "# Answer option: ['A)...', ...]" into {'A': '...'}."""
    m = re.search(r"Answer option:\s*(\[.*\])", question, re.S)
    if not m:
        return {}
    try:
        items = json.loads(m.group(1).replace("'", '"'))
    except json.JSONDecodeError:
        items = re.findall(r"'([A-E]\).*?)'", m.group(1))
    return {s[0].upper(): s[2:].strip() for s in items if len(s) > 2 and s[1] == ")"}


def extract_choice(text: str, question: str = "") -> Optional[str]:
    """AQUA option letter.

    Order: an explicit 'answer/option is X'; else the last '(X)' / 'X)';
    else the option whose numeric value equals the response's numeric
    answer (models often state "39000 Rupees" without the letter).
    """
    marked = re.findall(r"(?:answer|option|choice)\s*(?:is|:)?\s*[\*\(\[\s'\"]*([A-E])\b", text, re.I)
    if marked:
        return marked[-1].upper()
    loose = re.findall(r"\(([A-E])\)|\b([A-E])\)", text)
    if loose:
        a, b = loose[-1]
        return (a or b).upper()
    value = extract_number(text)
    if value is not None:
        for letter, option in _aqua_options(question).items():
            nums = _NUMBER_RE.findall(option)
            if len(nums) == 1 and _to_float(nums[0]) is not None and abs(_to_float(nums[0]) - value) < 1e-6:
                return letter
    return None


def extract_boolean(text: str) -> Optional[str]:
    """StrategyQA True/False: first yes/no after the last 'answer' marker, else the opening word, else the last one."""
    pattern = re.compile(r"\b(yes|no|true|false)\b", re.I)
    tail = _after_last_marker(text)
    m = pattern.search(tail) if tail else None
    if m is None:
        m = pattern.match(text.strip())
    if m is None:
        found = pattern.findall(text)
        if not found:
            return None
        word = found[-1]
    else:
        word = m.group(1)
    return "True" if word.lower() in ("yes", "true") else "False"


def extract_plausibility(text: str) -> Optional[str]:
    """QASports plausible(1)/implausible(0): last plausibility verdict in the response."""
    pattern = re.compile(
        r"\b(implausible|not plausible|unlikely|not likely|incorrect|false|no|plausible|likely|correct|true|yes)\b",
        re.I,
    )
    found = pattern.findall(text)
    if not found:
        return None
    word = found[-1].lower()
    return "0" if word in ("implausible", "not plausible", "unlikely", "not likely", "incorrect", "false", "no") else "1"


def extract_kinship(text: str) -> Optional[str]:
    """CLUTRR relation: last kinship term in the response."""
    found = _KIN_RE.findall(text)
    if not found:
        return None
    term = found[-1].lower()
    return _KIN_SYNONYMS.get(term, term)


def extract_date(text: str) -> Optional[str]:
    """Date: last MM/DD/YYYY in the response, zero-padded."""
    found = re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text)
    if not found:
        return None
    mm, dd, yyyy = found[-1]
    return f"{int(mm):02d}/{int(dd):02d}/{yyyy}"


def _saycan_plan(text: str) -> tuple:
    return tuple((verb.lower(), arg.lower().strip()) for verb, arg in _SAYCAN_CALL_RE.findall(text))


def score_record(dataset: str, answer: str, gold: str, question: str = "") -> Dict[str, float]:
    """Return {'accuracy': 0/1, 'answered': 0/1} for one response."""
    if dataset in ("asdiv", "gsm8k", "MultiArith", "SVAMP"):
        pred, target = extract_number(answer), _to_float(_gold_tail(gold))
        correct = pred is not None and target is not None and abs(pred - target) <= 1e-4 * max(1.0, abs(target))
    elif dataset == "aqua":
        pred = extract_choice(answer, question)
        correct = pred == _gold_tail(gold).upper()
    elif dataset == "StrategyQA":
        pred = extract_boolean(answer)
        correct = pred == _gold_tail(gold)
    elif dataset == "QASports":
        pred = extract_plausibility(answer)
        correct = pred == _gold_tail(gold)
    elif dataset == "clutrr":
        pred = extract_kinship(answer)
        correct = pred == _gold_tail(gold).lower()
    elif dataset == "date":
        pred = extract_date(answer)
        correct = pred == _gold_tail(gold)
    elif dataset == "saycan":
        pred_plan = _saycan_plan(answer)
        pred = pred_plan or None
        try:
            gold_plans = {_saycan_plan(p) for p in json.loads(gold)}
        except json.JSONDecodeError:
            gold_plans = {_saycan_plan(gold)}
        correct = bool(pred_plan) and pred_plan in gold_plans
    else:
        raise ValueError(f"Unknown dataset {dataset!r}")
    return {"accuracy": float(bool(correct)), "answered": float(pred is not None)}


_METRIC_LEAK_RE = re.compile(
    r"\b(?:readability|coherence|specificity|engagement|conciseness|concise target|zipf|hapax|perplexity"
    r"|entropy|target metric|length target|quality metric)s?\b", re.I)


def add_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    scored = [score_record(d, a, g, q) for d, a, g, q in
              zip(df["dataset"], df["final_answer"], df["gold_answer"], df["question"])]
    df = pd.concat([df, pd.DataFrame(scored, index=df.index)], axis=1)
    # the answer talks about the prompt's quality metrics instead of (or besides) the task
    df["metric_leak"] = df["final_answer"].str.contains(_METRIC_LEAK_RE).astype(float)
    return df


# --------------------------------------------------------------------------
# MGCoT target adherence (manipulation check)
# --------------------------------------------------------------------------

def add_target_adherence(df: pd.DataFrame) -> pd.DataFrame:
    """Measure each answer's quality profile and its distance to the MGCoT targets.

    Targets depend only on the question, so the SCoT answer to the same
    record is scored against the same targets - that is the baseline an
    effective MGCoT prompt must beat. Distance is the mean absolute error
    over the 11 metrics, each divided by the answer scaler's fitted range.
    """
    names = profile_metrics.__all__
    metric_fns = {n: getattr(profile_metrics, n)() for n in names}
    a_scaler = joblib.load(A_SCALER_PATH)
    ranges = dict(zip(names, np.where(a_scaler.data_range_ > 0, a_scaler.data_range_, 1.0)))

    for n in names:
        df[f"actual_{n}"] = [metric_fns[n](a) if a.strip() else np.nan for a in df["final_answer"]]

    key = ["model", "dataset", "record_id"]
    targets = df[df["mechanism"] == "MGCoT"][key + [f"target_{n}" for n in names]]
    df = df.drop(columns=[f"target_{n}" for n in names]).merge(targets, on=key, how="left")

    for n in names:
        df[f"abs_err_{n}"] = (df[f"actual_{n}"] - df[f"target_{n}"]).abs() / ranges[n]
    df["target_error"] = df[[f"abs_err_{n}" for n in names]].mean(axis=1)
    return df


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

_rng = np.random.default_rng(SEED)


def _bootstrap_ci(diff: np.ndarray) -> tuple:
    idx = _rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_test(scot: np.ndarray, mgcot: np.ndarray, binary: bool) -> Dict[str, float]:
    """Paired MGCoT-vs-SCoT comparison on aligned arrays (NaN pairs dropped)."""
    keep = ~(np.isnan(scot) | np.isnan(mgcot))
    scot, mgcot = scot[keep], mgcot[keep]
    diff = mgcot - scot
    n = len(diff)
    out = {"n": n, "scot": np.nan, "mgcot": np.nan, "delta": np.nan, "ci_low": np.nan,
           "ci_high": np.nan, "rel_change_pct": np.nan, "p": np.nan, "effect_rb": np.nan}
    if n == 0:
        return out
    out.update(scot=scot.mean(), mgcot=mgcot.mean(), delta=diff.mean())
    out["rel_change_pct"] = 100 * diff.mean() / abs(scot.mean()) if scot.mean() != 0 else np.nan
    out["ci_low"], out["ci_high"] = _bootstrap_ci(diff)

    if binary:
        b = int(((scot == 1) & (mgcot == 0)).sum())  # only SCoT correct / positive
        c = int(((scot == 0) & (mgcot == 1)).sum())  # only MGCoT correct / positive
        out["discordant_scot_only"], out["discordant_mgcot_only"] = b, c
        out["p"] = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
        out["effect_rb"] = (c - b) / (b + c) if b + c else 0.0
    else:
        nz = diff[diff != 0]
        if len(nz) == 0:
            out["p"], out["effect_rb"] = 1.0, 0.0
        else:
            out["p"] = stats.wilcoxon(nz).pvalue
            ranks = stats.rankdata(np.abs(nz))
            out["effect_rb"] = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
    return out


def bh_adjust(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR-adjusted p-values."""
    p = p.astype(float)
    mask = p.notna()
    vals = p[mask].to_numpy()
    order = np.argsort(vals)
    ranked = vals[order] * len(vals) / (np.arange(len(vals)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = pd.Series(np.nan, index=p.index)
    out[mask] = adj[np.argsort(order)]
    return out


def paired_frame(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per (model, dataset, record) with SCoT and MGCoT values side by side."""
    wide = df.pivot_table(index=["model", "dataset", "record_id"], columns="mechanism", values=metric)
    return wide.reindex(columns=["SCoT", "MGCoT"])


def compare(df: pd.DataFrame, metrics: List[tuple], group: Optional[str]) -> pd.DataFrame:
    """Run paired tests for every metric, per level of `group` (or pooled if None)."""
    rows = []
    for metric, higher_better, binary in metrics:
        wide = paired_frame(df, metric).reset_index()
        groups = [("ALL", wide)] if group is None else list(wide.groupby(group))
        for level, sub in groups:
            res = paired_test(sub["SCoT"].to_numpy(float), sub["MGCoT"].to_numpy(float), binary)
            favours = np.nan
            if not np.isnan(res["delta"]) and res["delta"] != 0:
                favours = "MGCoT" if (res["delta"] > 0) == higher_better else "SCoT"
            rows.append({"group": level, "metric": metric, "higher_is_better": higher_better,
                         **res, "favours": favours})
    out = pd.DataFrame(rows)
    out["p_fdr"] = out.groupby("metric")["p"].transform(bh_adjust)
    out["significant"] = out["p_fdr"] < ALPHA
    return out


def cell_win_table(df: pd.DataFrame, metrics: List[tuple]) -> pd.DataFrame:
    """Per metric: in how many (model, dataset) cells MGCoT beats SCoT, with a sign test."""
    rows = []
    for metric, higher_better, _ in metrics:
        cells = df.groupby(["model", "dataset", "mechanism"])[metric].mean().unstack("mechanism").dropna()
        diff = (cells["MGCoT"] - cells["SCoT"]) * (1 if higher_better else -1)
        wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
        p = stats.binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
        rows.append({"metric": metric, "cells": len(diff), "mgcot_better": wins,
                     "scot_better": losses, "ties": len(diff) - wins - losses, "sign_test_p": p})
    return pd.DataFrame(rows)


def scan_all_levels(df: pd.DataFrame, metrics: List[tuple]) -> pd.DataFrame:
    """Paired test for every metric at every level: overall, dataset, model, model x dataset.

    BH-FDR is applied once across the whole scan (all metrics and levels
    together), so a result reported as significant survives the full search
    rather than a hand-picked subset. Both directions are kept.
    """
    parts = []
    for level, keys in [("overall", None), ("dataset", ["dataset"]), ("model", ["model"]),
                        ("model x dataset", ["model", "dataset"])]:
        for metric, higher_better, binary in metrics:
            wide = paired_frame(df, metric).reset_index()
            groups = [("ALL", wide)] if keys is None else list(wide.groupby(keys))
            for name, sub in groups:
                res = paired_test(sub["SCoT"].to_numpy(float), sub["MGCoT"].to_numpy(float), binary)
                favours = np.nan
                if not np.isnan(res["delta"]) and res["delta"] != 0:
                    favours = "MGCoT" if (res["delta"] > 0) == higher_better else "SCoT"
                label = name if isinstance(name, str) else " / ".join(name)
                parts.append({"level": level, "group": label, "metric": metric, **res, "favours": favours})
    out = pd.DataFrame(parts)
    out["p_fdr"] = bh_adjust(out["p"])
    out["significant"] = out["p_fdr"] < ALPHA
    return out


def delta_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Model x dataset matrix of mean(MGCoT) - mean(SCoT)."""
    cells = df.groupby(["model", "dataset", "mechanism"])[metric].mean().unstack("mechanism")
    mat = (cells["MGCoT"] - cells["SCoT"]).unstack("dataset")
    return mat.reindex(columns=[d for d in DATASET_ORDER if d in mat.columns])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)

    raw = add_accuracy(load_results())

    # --- data validity -----------------------------------------------------
    validity = raw.groupby(["model", "mechanism"]).agg(
        rows=("record_id", "size"),
        empty_answer_rate=("empty_answer", "mean"),
        truncated_rate=("truncated", "mean"),
        answer_extracted_rate=("answered", "mean"),
        metric_leak_rate=("metric_leak", "mean"),
        mean_delta_s=("delta_time_seconds", "mean"),
    ).round(3)
    validity.to_csv(os.path.join(OUT_DIR, "00_validity_by_model.csv"))
    timing_fit = {}
    for model, g in raw[~raw["model"].isin(EXCLUDED_MODELS)].groupby("model"):
        slope, intercept = np.polyfit(g["output_words"], g["delta_time_seconds"], 1)
        timing_fit[model] = {"sec_per_word": slope, "intercept_s": intercept,
                             "r": g[["output_words", "delta_time_seconds"]].corr().iloc[0, 1]}
    pd.DataFrame(timing_fit).T.round(3).to_csv(os.path.join(OUT_DIR, "00_timing_vs_length.csv"))

    df = add_target_adherence(raw[~raw["model"].isin(EXCLUDED_MODELS)].copy())
    df.to_csv(os.path.join(OUT_DIR, "00_scored_rows.csv"), index=False)
    core = df[~df["dataset"].isin(ILL_POSED_DATASETS)]

    all_metrics = QUALITY_METRICS + COST_METRICS
    adherence_metrics = [("target_error", False, False), ("answered", True, True), ("metric_leak", False, True)]

    # --- overall, axis 1 (datasets), axis 2 (models) ------------------------
    overall_core = compare(core, all_metrics + adherence_metrics, None).assign(scope="8 well-posed datasets")
    overall_all = compare(df, all_metrics + adherence_metrics, None).assign(scope="all 10 datasets")
    pd.concat([overall_core, overall_all]).to_csv(os.path.join(OUT_DIR, "01_overall.csv"), index=False)

    by_dataset = compare(df, all_metrics + adherence_metrics, "dataset")
    by_dataset["group"] = pd.Categorical(by_dataset["group"], DATASET_ORDER, ordered=True)
    by_dataset = by_dataset.sort_values(["metric", "group"])
    by_dataset.to_csv(os.path.join(OUT_DIR, "02_axis1_by_dataset.csv"), index=False)

    by_model = compare(core, all_metrics + adherence_metrics, "model").sort_values(["metric", "group"])
    by_model.to_csv(os.path.join(OUT_DIR, "03_axis2_by_model.csv"), index=False)

    wins = cell_win_table(core, all_metrics + adherence_metrics)
    wins.to_csv(os.path.join(OUT_DIR, "04_cell_wins.csv"), index=False)

    for metric in ["accuracy", "semantic_similarity", "bert_score_f1", "output_words", "truncated",
                   "delta_time_seconds", "gpu_usage_avg_percent", "cpu_usage_avg_percent", "target_error"]:
        delta_matrix(df, metric).round(4).to_csv(os.path.join(OUT_DIR, f"05_delta_matrix_{metric}.csv"))

    # --- per-metric adherence and its link to correctness -------------------
    names = profile_metrics.__all__
    adherence_per_metric = compare(core, [(f"abs_err_{n}", False, False) for n in names], None)
    adherence_per_metric.to_csv(os.path.join(OUT_DIR, "06_target_adherence_per_metric.csv"), index=False)
    mg = core[core["mechanism"] == "MGCoT"]
    rho, p_rho = stats.spearmanr(mg["target_error"], mg["accuracy"], nan_policy="omit")
    # efficiency: generated words per correct answer (lower = cheaper correct answers)
    eff = core.groupby(["model", "mechanism"]).agg(words=("output_words", "sum"), correct=("accuracy", "sum"),
                                                  seconds=("delta_time_seconds", "sum")).reset_index()
    eff["words_per_correct"] = eff["words"] / eff["correct"].replace(0, np.nan)
    eff["seconds_per_correct"] = eff["seconds"] / eff["correct"].replace(0, np.nan)
    eff.round(2).to_csv(os.path.join(OUT_DIR, "07_cost_per_correct_answer.csv"), index=False)

    # --- full scan: every metric x every level, one FDR family --------------
    scan_metrics = QUALITY_METRICS + [("output_words", False, False), ("truncated", False, True),
                                      ("delta_time_seconds", False, False), ("target_error", False, False)]
    scan = scan_all_levels(df, scan_metrics)
    scan["ill_posed_dataset"] = scan["group"].str.contains("|".join(ILL_POSED_DATASETS))
    scan.to_csv(os.path.join(OUT_DIR, "08_scan_all_levels.csv"), index=False)

    # --- console summary ----------------------------------------------------
    cols = ["group", "metric", "n", "scot", "mgcot", "delta", "ci_low", "ci_high", "rel_change_pct",
            "effect_rb", "p_fdr", "favours", "significant"]
    print("=== DATA VALIDITY ===")
    print(validity)
    print("\n=== timing vs output length ===")
    print(pd.DataFrame(timing_fit).T.round(3))
    print("\n=== OVERALL (6 models, 8 well-posed datasets) ===")
    print(overall_core[cols].round(4).to_string(index=False))
    print("\n=== OVERALL (6 models, all 10 datasets) ===")
    print(overall_all[cols].round(4).to_string(index=False))
    print("\n=== CELL WINS (60 model x dataset cells minus ill-posed = 48) ===")
    print(wins.round(4).to_string(index=False))
    for metric in ["accuracy", "semantic_similarity", "bert_score_f1", "output_words", "truncated",
                   "delta_time_seconds", "cpu_usage_avg_percent", "gpu_usage_avg_percent", "gpu_memory_avg_mb",
                   "ram_usage_avg_mb", "target_error"]:
        print(f"\n=== AXIS 1 by dataset: {metric} ===")
        print(by_dataset[by_dataset.metric == metric][cols].round(4).to_string(index=False))
        print(f"=== AXIS 2 by model: {metric} ===")
        print(by_model[by_model.metric == metric][cols].round(4).to_string(index=False))
    print("\n=== TARGET ADHERENCE PER METRIC (lower error = closer to MGCoT target) ===")
    print(adherence_per_metric[cols].round(4).to_string(index=False))
    print(f"\nSpearman(target_error, accuracy) within MGCoT: rho={rho:.3f}, p={p_rho:.3g}")
    print("\n=== COST PER CORRECT ANSWER ===")
    print(eff.round(2).to_string(index=False))

    sig = scan[scan["significant"]]
    print(f"\n=== FULL SCAN: {len(scan)} tests, {len(sig)} significant after FDR ===")
    print(sig.pivot_table(index=["level", "metric"], columns="favours", values="p", aggfunc="size",
                          fill_value=0).to_string())
    wins = sig[(sig["favours"] == "MGCoT") & ~sig["ill_posed_dataset"]].sort_values(["metric", "level", "group"])
    print("\n=== SIGNIFICANT MGCoT WINS (well-posed datasets) ===")
    print(wins[["level", "group", "metric", "n", "scot", "mgcot", "delta", "ci_low", "ci_high",
                "effect_rb", "p_fdr"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
