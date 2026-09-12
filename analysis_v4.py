import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

PATH = r"D:\SAMs\slmproj2\results\evaluation\.2ndrun\combined_ALL_models_profiled.csv"

N_DATSETS = 10
N_MODELS = 7
N_SAMPLES = 50
N_MECHANISMS = 2

df = pd.read_csv(PATH)
# Display the shape of the DataFrame
print(df.shape)
print(N_DATSETS * N_MODELS * N_SAMPLES * N_MECHANISMS)

# print DataFrame column names
# print("DataFrame columns:")
# print(df.columns)
print("DataFrame columns length:")
print(len(df.columns))

SAVE_PATH = r"D:\SAMs\slmproj2\results\analysis"

MODEL_NAMES = ['mistral_7b', 'phi3_mini', 'llama3.2_1b', 'gemma2_2b',
               'qwen2_1.5b', 'openchat_7b', 'deepseek-r1_8b']

DATASET_NAMES = ['aqua', 'asdiv', 'clutrr', 'date', 'gsm8k',
                 'MultiArith', 'QASports', 'saycan', 'StrategyQA',
                    'SVAMP']

BASE_COLUMNS = ['model', 'record_id', 'dataset', 'question', 
                'gold_answer', 'mechanism', 'final_answer']


ACRONYMS = {
    "delta_time_seconds": "Time (min)",
    "cpu_usage_avg_percent": "CPU (%)",
    "ram_usage_avg_mb": "RAM (GB)",
    "ram_usage_avg_percent": "RAM (%)",
    "gpu_usage_avg_percent": "PGPU (%)",
    "gpu_memory_avg_mb": "MGPU (GB)",
    # 
    'rouge1_f1': 'RF1',
    'rouge2_f1': 'R2',
    'rougeL_f1': 'RLF1',
    'bleu': 'BLEU',
    'exact_match': 'EM',
    'token_precision': 'TP',
    'token_recall': 'TR',
    'token_f1': 'TF1',
    'token_overlap_count': 'TOC',
    'normalized_edit_distance': 'NED',
    'char_f1': 'CF1',
    'bert_score_f1': 'BERT-F1',
    'semantic_similarity': 'SS',
    'accuracy': 'ACC',
}

HARDWARE_COLUMNS = ['delta_time_seconds', 'cpu_usage_avg_percent',
       'ram_usage_avg_mb', 'ram_usage_avg_percent', 'gpu_usage_avg_percent',
       'gpu_memory_avg_mb']

PERFORMANCE_COLUMNS = ['rouge1_f1', 'rouge2_f1', 'rougeL_f1', 'bleu',
       'exact_match', 'token_precision', 'token_recall', 'token_f1',
       'token_overlap_count', 'normalized_edit_distance', 'char_f1',
       'bert_score_f1', 'semantic_similarity', 'accuracy']

PERFORMANCE_EXCLUDED_COLUMNS = ['exact_match',           # always 0: whole-string match against a one-word gold
                                'bleu',                  # never above 0.05, at the floor for label-type golds
                                'rouge2_f1',             # 0.00 on most datasets: one-word golds have no bigrams
                                'rouge1_f1',             # duplicates RLF1 almost exactly with one-word golds
                                'token_overlap_count']   # raw count that depends on gold length; TR is its proportion


PERFOMRANCE_RULES = {
    'rouge1_f1': {
        "range": (0, 1),
        "remaks": "Higher values indicate better performance.",
        "description": "ROUGE-1 F1 score, measures the overlap of unigrams between the generated and reference text."
        },
    'rouge2_f1': {
        "range": (0, 1),
        "remaks": "Higher values indicate better performance.",
        "description": "ROUGE-2 F1 score, measures the overlap of bigrams between the generated and reference text."
        },
    'rougeL_f1': {
        "range": (0, 1),
        "remaks": "Higher values indicate better performance.",
        "description": "ROUGE-L F1 score, measures the longest common subsequence between the generated and reference text."
        },
    'bleu': {
        "range": (0, 1),
        "remaks": "Higher values indicate better performance.",
        "description": "Sentence BLEU, geometric mean of 1-4-gram precision with a brevity penalty for short outputs."
        },
    'exact_match': {
        "range": (0, 1),
        "remaks": "Binary (0 or 1). Higher values indicate better performance.",
        "description": "1 if the whole generated text equals the reference after stripping whitespace, else 0."
        },
    'token_precision': {
        "range": (0, 1),
        "remaks": "Higher values indicate better performance.",
        "description": "Fraction of distinct generated tokens that also appear in the reference."
        },
    'token_recall': {
        "range": (0, 1),
        "remaks": "Higher values indicate better performance.",
        "description": "Fraction of distinct reference tokens that also appear in the generated text."
        },
    'token_f1': {
        "range": (0, 1),
        "remaks": "Higher values indicate better performance.",
        "description": "Harmonic mean of token precision and token recall."
        },
    'token_overlap_count': {
        "range": (0, float("inf")),
        "remaks": "Integer count, unbounded. Higher values indicate better performance.",
        "description": "Number of distinct tokens shared between the generated and reference text."
        },
    'normalized_edit_distance': {
        "range": (0, 1),
        "remaks": "Despite the name, this is a similarity. Higher values indicate better performance.",
        "description": "1 - Levenshtein distance / length of the longer text; 1 means identical strings."
        },
    'char_f1': {
        "range": (0, 1),
        "remaks": "Higher values indicate better performance.",
        "description": "Harmonic mean of precision and recall over the sets of distinct characters in both texts."
        },
    'bert_score_f1': {
        "range": (0, 1),
        "remaks": "Higher values indicate better performance.",
        "description": "BERTScore F1, token-level cosine similarity of contextual BERT embeddings between generated and reference text."
        },
    'semantic_similarity': {
        "range": (-1, 1),
        "remaks": "Higher values indicate better performance.",
        "description": "Cosine similarity between sentence embeddings (all-MiniLM-L6-v2) of the generated and reference text."
        },
}


QUALITY_COLUMNS = ['target_Readability',
       'target_Coherence', 'target_Relevance', 'target_Specificity',
       'target_Engagement', 'target_Concise', 'target_Length', 'target_Zipf',
       'target_Hapax', 'target_Entropy', 'target_Perplexity',
       'gold_Readability', 'gold_Coherence', 'gold_Relevance',
       'gold_Specificity', 'gold_Engagement', 'gold_Concise', 'gold_Length',
       'gold_Zipf', 'gold_Hapax', 'gold_Entropy', 'gold_Perplexity',
       'actual_Readability', 'actual_Coherence', 'actual_Relevance',
       'actual_Specificity', 'actual_Engagement', 'actual_Concise',
       'actual_Length', 'actual_Zipf', 'actual_Hapax', 'actual_Entropy',
       'actual_Perplexity']

MODEL_NAMES = ['mistral_7b', 'phi3_mini', 'llama3.2_1b', 'gemma2_2b',
               'qwen2_1.5b', 'openchat_7b', 'deepseek-r1_8b']
DATASET_NAMES = ['aqua', 'asdiv', 'clutrr', 'date', 'gsm8k',
                 'MultiArith', 'QASports', 'saycan', 'StrategyQA', 'SVAMP']


REF_COLUMNS = ["model", "mechanism", "dataset"]
COMPARISON_COLUMNS = REF_COLUMNS + HARDWARE_COLUMNS + PERFORMANCE_COLUMNS
# 
df_comp = df[COMPARISON_COLUMNS].copy()
df_comp_mean = df_comp.groupby(REF_COLUMNS).mean().reset_index().round(3)
df_comp_std = df_comp.groupby(REF_COLUMNS).std().reset_index().round(3)

df_comp_mean = df_comp_mean.to_csv(SAVE_PATH + r"\comparison_results_mean.csv", index=False)
df_comp_std = df_comp_std.to_csv(SAVE_PATH + r"\comparison_results_std.csv", index=False)
# print(df_comp)

# ---------------------------------------------------------------------------
# Results tables: one workbook per section (hardware, syntactic, semantic),
# one sheet per model; rows are dataset x mechanism, cells "mean ± std".
# Note: deepseek-r1_8b exhausted the 300-token budget on hidden reasoning in
# 97-98% of runs (empty answers), so its syntactic/semantic scores are ~0.
# ---------------------------------------------------------------------------
PER_MODEL_DIR = SAVE_PATH + r"\per_model"
TABLE_MODELS = MODEL_NAMES
TABLE_METRICS = HARDWARE_COLUMNS + PERFORMANCE_COLUMNS
MECHANISMS = ["SCoT", "MGCoT"]
# syntactic and semantic metrics share one "response quality" table per model;
# PERFORMANCE_EXCLUDED_COLUMNS drops the metrics that stay at the floor for
# label-type gold answers or duplicate a metric that is kept.
QUALITY_COLUMNS_KEPT = [c for c in PERFORMANCE_COLUMNS if c not in PERFORMANCE_EXCLUDED_COLUMNS]
SECTIONS = {
    "hardware": HARDWARE_COLUMNS,
    "quality": QUALITY_COLUMNS_KEPT,
}


def mean_std_table(model, model_df):
    """Rows: dataset (plus 'All datasets'), mechanism; columns: one per metric as 'mean ± std'."""
    rows, raw_means = [], []
    groups = [(d, model_df[model_df["dataset"] == d]) for d in DATASET_NAMES] + [("All datasets", model_df)]
    for name, dataset_df in groups:
        for mech in MECHANISMS:  # SCoT row then MGCoT row for each dataset
            g = dataset_df[dataset_df["mechanism"] == mech]
            row = {"dataset": name, "mechanism": mech}
            for metric in TABLE_METRICS:
                row[metric] = f"{g[metric].mean():.2f} ± {g[metric].std():.2f}"
            rows.append(row)
            raw_means.append(g[TABLE_METRICS].mean())
    # unrounded means, row-aligned with the table, used only to rank cells for shading
    return pd.DataFrame(rows), pd.DataFrame(raw_means).reset_index(drop=True)


# Excel shading: in each metric column, the single best mean across all
# dataset x mechanism rows gets dark gray and the single worst gets light gray.
# Ranking uses unrounded means; "All datasets" rows are not ranked, and an
# extreme shared exactly by several rows (e.g. many 0.0 BLEU) is left unshaded.
# Lower is better for hardware columns, higher for every performance column.
DARK_GRAY = PatternFill("solid", fgColor="BFBFBF")
LIGHT_GRAY = PatternFill("solid", fgColor="F2F2F2")
LOWER_IS_BETTER = set(HARDWARE_COLUMNS)


def shade_sheet(ws, table, raw_means):
    """Shade the best (dark) and worst (light) cell of each metric column in one sheet."""
    ranked = table["dataset"] != "All datasets"
    for col_idx, metric in enumerate(table.columns, start=1):
        ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = 16
        if metric not in TABLE_METRICS:
            continue
        means = raw_means.loc[ranked, metric]
        low, high = means.min(), means.max()
        best, worst = (low, high) if metric in LOWER_IS_BETTER else (high, low)
        for value, fill in ((best, DARK_GRAY), (worst, LIGHT_GRAY)):
            rows_at_value = means.index[means == value]
            if low != high and len(rows_at_value) == 1:
                ws.cell(rows_at_value[0] + 2, col_idx).fill = fill


def save_table_xlsx(table, raw_means, columns, path):
    """One table per file: the section's columns, ACRONYMS headers, best/worst shading."""
    keep = ["dataset", "mechanism"] + columns
    table[keep].rename(columns=ACRONYMS).to_excel(path, index=False)
    wb = load_workbook(path)
    shade_sheet(wb.active, table[keep], raw_means)
    wb.save(path)


# seconds -> minutes, MB -> GB (column names keep their source names; ACRONYMS carries the new units)
df_tables = df.assign(delta_time_seconds=df["delta_time_seconds"] / 60,
                      ram_usage_avg_mb=df["ram_usage_avg_mb"] / 1024,
                      gpu_memory_avg_mb=df["gpu_memory_avg_mb"] / 1024)

os.makedirs(PER_MODEL_DIR, exist_ok=True)
model_tables = {m: mean_std_table(m, df_tables[df_tables["model"] == m]) for m in TABLE_MODELS}
# numbered in results order: hardware 1-7, syntactic 8-14, semantic 15-21
table_jobs = [(section, columns, model) for section, columns in SECTIONS.items() for model in model_tables]
for i, (section, columns, model) in enumerate(table_jobs):
    table, raw_means = model_tables[model]
    path = os.path.join(PER_MODEL_DIR, f"table.{i + 1}_{section}_{model}.xlsx")
    save_table_xlsx(table, raw_means, columns, path)
    print(f"Saved {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# Pattern extraction: one fact sheet per table, so every sentence written in
# the results section traces back to a computed number.
#   gain  = MGCoT - SCoT, signed so that positive always means MGCoT is better
#   p     = paired Wilcoxon signed-rank test over the 50 questions of a dataset
#           (500 for "All datasets"); p_fdr = Benjamini-Hochberg within a table
#   better/worse/tie per dataset follows the 2-decimal means shown in the table
# ---------------------------------------------------------------------------
PATTERNS_PATH = os.path.join(SAVE_PATH, "patterns.json")


def paired_p(scot, mgcot):
    """Paired p-value: exact McNemar for 0/1 outcomes, Wilcoxon signed-rank otherwise."""
    pair = np.stack([scot, mgcot])
    pair = pair[:, ~np.isnan(pair).any(axis=0)]
    if pair.size and np.isin(pair, (0.0, 1.0)).all():        # binary outcome, e.g. task accuracy
        scot_only = int(((pair[0] == 1) & (pair[1] == 0)).sum())
        mgcot_only = int(((pair[0] == 0) & (pair[1] == 1)).sum())
        discordant = scot_only + mgcot_only
        return float(stats.binomtest(min(scot_only, mgcot_only), discordant, 0.5).pvalue) if discordant else 1.0
    diff = mgcot - scot
    diff = diff[~np.isnan(diff) & (diff != 0)]
    return float(wilcoxon(diff).pvalue) if len(diff) else 1.0


def bh_adjust(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


SEED, N_BOOT = 42, 5000
_rng = np.random.default_rng(SEED)


def bootstrap_ci(diff):
    """95% percentile bootstrap CI for the mean paired difference."""
    if len(diff) == 0:
        return (np.nan, np.nan)
    means = diff[_rng.integers(0, len(diff), size=(N_BOOT, len(diff)))].mean(axis=1)
    return (round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4))


def rank_biserial(diff, sign):
    """Matched-pairs rank-biserial correlation, signed so positive favours MGCoT."""
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    effect = (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()
    return round(float(sign * effect), 4)


def describe(scot, mgcot, metric):
    """Means, stds, signed gain, % change, effect size, bootstrap CI and paired p."""
    sign = -1 if metric in LOWER_IS_BETTER else 1
    s_mean, m_mean = np.nanmean(scot), np.nanmean(mgcot)
    scot, mgcot = np.asarray(scot, float), np.asarray(mgcot, float)
    diff = (mgcot - scot)[~(np.isnan(scot) | np.isnan(mgcot))]
    ci_low, ci_high = bootstrap_ci(diff)
    return {
        "effect_rank_biserial": rank_biserial(diff, sign),
        "ci_low": ci_low, "ci_high": ci_high,
        "scot_mean": round(float(s_mean), 4), "scot_std": round(float(np.nanstd(scot, ddof=1)), 4),
        "mgcot_mean": round(float(m_mean), 4), "mgcot_std": round(float(np.nanstd(mgcot, ddof=1)), 4),
        "gain": round(float(sign * (m_mean - s_mean)), 4),
        "pct_change": round(float(100 * (m_mean - s_mean) / abs(s_mean)), 1) if s_mean else None,
        "p": paired_p(np.asarray(scot, float), np.asarray(mgcot, float)),
        "outcome": ("tie" if round(s_mean, 2) == round(m_mean, 2)
                    else "MGCoT better" if sign * (m_mean - s_mean) > 0 else "MGCoT worse"),
    }


def table_patterns(model_df, columns, raw_means, table):
    """Fact sheet for one table: per-metric overall result, per-dataset results and patterns."""
    wide = model_df.pivot_table(index=["dataset", "record_id"], columns="mechanism", values=columns)
    facts, all_rows = {}, []
    for metric in columns:
        per_dataset = []
        for d in DATASET_NAMES:
            sub = wide.loc[d]
            per_dataset.append({"dataset": d, **describe(sub[(metric, "SCoT")].to_numpy(),
                                                          sub[(metric, "MGCoT")].to_numpy(), metric)})
        overall = describe(wide[(metric, "SCoT")].to_numpy(), wide[(metric, "MGCoT")].to_numpy(), metric)
        all_rows += per_dataset + [overall]

        ranked = table["dataset"] != "All datasets"
        means = raw_means.loc[ranked, metric]
        best_idx = means.idxmin() if metric in LOWER_IS_BETTER else means.idxmax()
        worst_idx = means.idxmax() if metric in LOWER_IS_BETTER else means.idxmin()
        by_gain = sorted(per_dataset, key=lambda r: r["gain"], reverse=True)
        facts[metric] = {
            "label": ACRONYMS[metric],
            "lower_is_better": metric in LOWER_IS_BETTER,
            "overall": overall,
            "better_on": [r["dataset"] for r in per_dataset if r["outcome"] == "MGCoT better"],
            "worse_on": [r["dataset"] for r in per_dataset if r["outcome"] == "MGCoT worse"],
            "tie_on": [r["dataset"] for r in per_dataset if r["outcome"] == "tie"],
            "top_gains": [(r["dataset"], r["gain"]) for r in by_gain[:2] if r["gain"] > 0],
            "top_losses": [(r["dataset"], r["gain"]) for r in by_gain[::-1][:2] if r["gain"] < 0],
            "best_cell": [table.loc[best_idx, "dataset"], table.loc[best_idx, "mechanism"],
                          round(float(means[best_idx]), 4)],
            "worst_cell": [table.loc[worst_idx, "dataset"], table.loc[worst_idx, "mechanism"],
                           round(float(means[worst_idx]), 4)],
            "mgcot_lower_std_on": sum(r["mgcot_std"] < r["scot_std"] for r in per_dataset),
            "per_dataset": per_dataset,
        }
    # FDR within the table (all datasets x metrics + overall rows)
    for row, p_fdr in zip(all_rows, bh_adjust([r["p"] for r in all_rows])):
        row["p_fdr"] = round(float(p_fdr), 4)
        row["significant"] = bool(p_fdr < 0.05)
    for f in facts.values():
        f["significant_better_on"] = [r["dataset"] for r in f["per_dataset"]
                                      if r["significant"] and r["outcome"] == "MGCoT better"]
        f["significant_worse_on"] = [r["dataset"] for r in f["per_dataset"]
                                     if r["significant"] and r["outcome"] == "MGCoT worse"]
    return facts


patterns = []
for i, (section, columns, model) in enumerate(table_jobs):
    table, raw_means = model_tables[model]
    patterns.append({"table": i + 1, "section": section, "model": model,
                     "metrics": table_patterns(df_tables[df_tables["model"] == model], columns,
                                               raw_means, table)})
with open(PATTERNS_PATH, "w", encoding="utf-8") as f:
    json.dump(patterns, f, indent=2)
print(f"Saved {PATTERNS_PATH}")

# ---------------------------------------------------------------------------
# Supplementary table: every paired test behind the "significant" statements,
# one row per table x metric x dataset (plus the all-datasets aggregate).
# ---------------------------------------------------------------------------
SUPPLEMENTARY_PATH = os.path.join(SAVE_PATH, "supplementary_tests.xlsx")

rows = []
for table in patterns:
    for metric, facts in table["metrics"].items():
        for result in facts["per_dataset"] + [dict(facts["overall"], dataset="All datasets")]:
            rows.append({
                "table": table["table"], "section": table["section"], "model": table["model"],
                "metric": facts["label"], "dataset": result["dataset"],
                "scot_mean": result["scot_mean"], "scot_std": result["scot_std"],
                "mgcot_mean": result["mgcot_mean"], "mgcot_std": result["mgcot_std"],
                "difference_mgcot_minus_scot": round(result["mgcot_mean"] - result["scot_mean"], 4),
                "ci95_low": result["ci_low"], "ci95_high": result["ci_high"],
                "pct_change": result["pct_change"], "favours": result["outcome"],
                "effect_rank_biserial": result["effect_rank_biserial"],
                "p_wilcoxon": result["p"], "p_bh_corrected": result["p_fdr"],
                "significant_at_0.05": result["significant"],
            })
pd.DataFrame(rows).to_excel(SUPPLEMENTARY_PATH, index=False)
print(f"Saved {SUPPLEMENTARY_PATH}: {len(rows)} tests")
    

    
    