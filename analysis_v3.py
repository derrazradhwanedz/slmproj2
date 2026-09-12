import numpy as np
import pandas as pd

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

BASE_COLUMNS = ['model', 'record_id', 'dataset', 'question', 
                'gold_answer', 'mechanism', 'final_answer']

HARDWARE_COLUMNS = ['delta_time_seconds', 'cpu_usage_avg_percent',
       'ram_usage_avg_mb', 'ram_usage_avg_percent', 'gpu_usage_avg_percent',
       'gpu_memory_avg_mb']

PERFORMANCE_COLUMNS = ['rouge1_f1', 'rouge2_f1', 'rougeL_f1', 'bleu',
       'exact_match', 'token_precision', 'token_recall', 'token_f1',
       'token_overlap_count', 'normalized_edit_distance', 'char_f1',
       'bert_score_f1', 'semantic_similarity']

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

# Check NaN values in the DataFrame
nan_counts = df.isna().sum()
print("NaN counts in each column:")
# print(nan_counts)

# encode the 'mechanism' column to numeric values
df_encoded = df.copy()
codes = {}
for col in df_encoded.select_dtypes(exclude="number").columns:
    df_encoded[col], codes[col] = pd.factorize(df_encoded[col])

# SPlit the DataFrame into two DataFrames based on the 'mechanism' column
df_scot = df_encoded[df['mechanism'] == "SCoT"]
df_mgcot = df_encoded[df['mechanism'] == "MGCoT"]

# get max correlation between columns in df_scot
scot_corr = df_scot.corr().round(2)
mgcot_corr = df_mgcot.corr().round(2)

THRESHOLD = 0.7
GROUP_OF = {col: name for name, cols in [("base", BASE_COLUMNS), ("hardware", HARDWARE_COLUMNS),
                                         ("performance", PERFORMANCE_COLUMNS), ("quality", QUALITY_COLUMNS)]
            for col in cols}

def high_corr_pairs(corr, threshold=THRESHOLD):
    """Unique cross-set variable pairs with |corr| >= threshold, strongest first."""
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().rename("corr").reset_index()
    pairs.columns = ["var1", "var2", "corr"]
    pairs["set1"], pairs["set2"] = pairs["var1"].map(GROUP_OF), pairs["var2"].map(GROUP_OF)
    pairs = pairs[pairs["set1"] != pairs["set2"]]
    return pairs[pairs["corr"].abs() >= threshold].sort_values("corr", key=abs, ascending=False)

scot_high = high_corr_pairs(scot_corr)
mgcot_high = high_corr_pairs(mgcot_corr)
print("SCoT highly correlated pairs:\n", scot_high)
print("MGCoT highly correlated pairs:\n", mgcot_high)

# ---------------------------------------------------------------------------
# SCoT vs MGCoT per model and dataset: mean and std of every hardware and
# performance column, and which mechanism wins on the mean.
# ---------------------------------------------------------------------------
OUT_PATH = r"D:\SAMs\slmproj2\results\analysis\2ndrun\mean_std_by_model_dataset.csv"
LOWER_IS_BETTER = set(HARDWARE_COLUMNS) | {"normalized_edit_distance"}
COMPARE_COLUMNS = HARDWARE_COLUMNS + PERFORMANCE_COLUMNS

stats = df.groupby(["model", "dataset", "mechanism"])[COMPARE_COLUMNS].agg(["mean", "std"])
rows = []
for (model, dataset), g in stats.groupby(level=["model", "dataset"]):
    g = g.droplevel(["model", "dataset"])
    for col in COMPARE_COLUMNS:
        scot_mean, mgcot_mean = g.loc["SCoT", (col, "mean")], g.loc["MGCoT", (col, "mean")]
        if scot_mean == mgcot_mean:
            winner = "tie"
        elif (mgcot_mean < scot_mean) == (col in LOWER_IS_BETTER):
            winner = "MGCoT"
        else:
            winner = "SCoT"
        rows.append({"model": model, "dataset": dataset,
                     "set": "hardware" if col in HARDWARE_COLUMNS else "performance", "metric": col,
                     "scot_mean": scot_mean, "scot_std": g.loc["SCoT", (col, "std")],
                     "mgcot_mean": mgcot_mean, "mgcot_std": g.loc["MGCoT", (col, "std")],
                     "winner": winner})
comparison = pd.DataFrame(rows).round(4)
comparison.to_csv(OUT_PATH, index=False)

# how many metrics each mechanism wins, per model and dataset
wins = comparison.pivot_table(index=["model", "dataset"], columns=["set", "winner"],
                              values="metric", aggfunc="count", fill_value=0)
print(wins)
print(f"Saved: {OUT_PATH}")
