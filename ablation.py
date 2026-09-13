"""Ablation of the MGCoT quality predictor, without re-running any language model.

Answers Reviewer 1: "No ablation study is provided to isolate the contribution
of each metric or the DNN predictor."

Part 1 - the predictor itself (500 question/answer profile pairs from
results/training, same MinMax scaling, 80/20 split and seed as src/nn/train.py):
    a) the trained network against trivial predictors (the mean answer profile,
       a copy of the question profile), linear regression, and four non-neural
       regressors (random forest, gradient boosting, k-nearest neighbours,
       support vector regression) tuned by 5-fold cross-validation on the
       training split only;
    b) leave-one-metric-out: the network is retrained with each input metric
       removed, and the rise in held-out error measures that metric's
       contribution.
    Networks are retrained over several seeds, so differences are reported with
    their spread rather than from a single run.

Part 2 - whether predicted targets steer the generated answers (existing
MGCoT generations, deepseek-r1_8b excluded as budget-incompatible):
    a) own vs shuffled vs constant targets: distance of each MGCoT answer to the
       targets predicted for its own question, to targets predicted for other
       questions of the same model and dataset, and to the mean target;
    b) per metric: whether MGCoT answers are closer to the target than the SCoT
       answer to the same question.

Distances are mean absolute differences in the predictor's scaled [0, 1] space.
Tests are paired Wilcoxon signed-rank with Benjamini-Hochberg correction and
5,000-sample bootstrap confidence intervals.

Run from the repository root as: python ablation.py
Outputs go to results/ablation/.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from metrics import profile as profile_metrics  # noqa: E402
from nn.model import QAPredictorNN, QAPredictorConfig  # noqa: E402
from utils import load_yaml  # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "ablation")
Q_PATH = os.path.join(ROOT, "results", "training", "final_q_df.csv")
A_PATH = os.path.join(ROOT, "results", "training", "final_a_df.csv")
MODEL_PATH = os.path.join(ROOT, "results", "weights", "model.pth")
A_SCALER_PATH = os.path.join(ROOT, "results", "weights", "a_scaler.pkl")
GEN_PATH = os.path.join(ROOT, "results", "evaluation", ".2ndrun", "combined_ALL_models_profiled.csv")

METRICS = profile_metrics.__all__
CFG = load_yaml("nn.yaml")
SEEDS = [42, 43, 44, 45, 46]
N_BOOT = 5000
N_SHUFFLES = 50
EXCLUDED_MODELS = {"deepseek-r1_8b"}
rng = np.random.default_rng(42)


# --------------------------------------------------------------------------
# Shared statistics
# --------------------------------------------------------------------------

def bootstrap_ci(diff):
    diff = np.asarray(diff, float)
    means = diff[rng.integers(0, len(diff), size=(N_BOOT, len(diff)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_p(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    return float(stats.wilcoxon(d).pvalue) if len(d) else 1.0


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    out = np.empty_like(adj)
    out[order] = adj
    return out


# --------------------------------------------------------------------------
# Part 1: predictor ablation
# --------------------------------------------------------------------------

class MLP(nn.Module):
    """Same architecture as QAPredictorNN, with separate input and output sizes."""

    def __init__(self, n_in, n_out):
        super().__init__()
        m = CFG["model"]
        layers, d = [], n_in
        for h, drop, bn in zip(m["hidden_dims"], m["dropout_rates"], m["use_batchnorm"]):
            layers += [nn.Linear(d, h), nn.ReLU()] + ([nn.BatchNorm1d(h)] if bn else []) + [nn.Dropout(drop)]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(x_tr, y_tr, seed):
    """Train with the hyperparameters of src/config/nn.yaml."""
    t = CFG["training"]
    torch.manual_seed(seed)
    model = MLP(x_tr.shape[1], y_tr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    x, y = torch.FloatTensor(x_tr), torch.FloatTensor(y_tr)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(t["epochs"]):
        model.train()
        for idx in torch.randperm(len(x), generator=gen).split(t["batch_size"]):
            if len(idx) < 2:          # BatchNorm needs more than one sample
                continue
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x[idx]), y[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip_norm"])
            opt.step()
    model.eval()
    return model


def ml_baselines(x_tr, y_tr, x_te):
    """Non-neural regressors, each tuned by 5-fold cross-validation on the training split only."""
    cv = dict(cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    candidates = {
        "Random forest": GridSearchCV(RandomForestRegressor(random_state=42),
                                      {"n_estimators": [200, 500], "max_depth": [None, 10], "min_samples_leaf": [1, 5]}, **cv),
        "Gradient boosting": GridSearchCV(MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
                                          {"estimator__n_estimators": [100, 300], "estimator__max_depth": [2, 3],
                                           "estimator__learning_rate": [0.05, 0.1]}, **cv),
        "k-nearest neighbours": GridSearchCV(KNeighborsRegressor(),
                                             {"n_neighbors": [3, 5, 10, 20], "weights": ["uniform", "distance"]}, **cv),
        "Support vector regression": GridSearchCV(MultiOutputRegressor(SVR()),
                                                  {"estimator__C": [0.1, 1, 10], "estimator__gamma": ["scale", 0.1, 1],
                                                   "estimator__epsilon": [0.01, 0.1]}, **cv),
    }
    return {name: model.fit(x_tr, y_tr).predict(x_te) for name, model in candidates.items()}


def per_sample_errors(pred, y):
    mse = ((pred - y) ** 2).mean(axis=1)
    cos = (pred * y).sum(axis=1) / (np.linalg.norm(pred, axis=1) * np.linalg.norm(y, axis=1) + 1e-12)
    return mse, cos


def predictor_ablation():
    q = pd.read_csv(Q_PATH)[METRICS].fillna(0).values
    a = pd.read_csv(A_PATH)[METRICS].fillna(0).values
    q_s, a_s = MinMaxScaler().fit_transform(q), MinMaxScaler().fit_transform(a)   # as in train.py
    t = CFG["training"]
    x_tr, x_te, y_tr, y_te = train_test_split(q_s, a_s, test_size=t["test_size"], random_state=t["seed"])

    preds = {
        "Mean answer profile": np.tile(y_tr.mean(axis=0), (len(y_te), 1)),
        "Copy of question profile": x_te,
        "Linear regression": LinearRegression().fit(x_tr, y_tr).predict(x_te),
    }
    preds.update(ml_baselines(x_tr, y_tr, x_te))
    saved = QAPredictorNN(QAPredictorConfig(num_metrics=len(METRICS)))
    saved.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    saved.eval()
    with torch.no_grad():
        preds["Trained DNN (study model)"] = saved(torch.FloatTensor(x_te)).numpy()
    retrained = [train_mlp(x_tr, y_tr, s) for s in SEEDS]
    with torch.no_grad():
        dnn_runs = [m(torch.FloatTensor(x_te)).numpy() for m in retrained]
    preds["DNN retrained (mean of 5 seeds)"] = np.mean(dnn_runs, axis=0)

    ref_mse, _ = per_sample_errors(preds["Trained DNN (study model)"], y_te)
    rows = []
    for name, p in preds.items():
        mse, cos = per_sample_errors(p, y_te)
        row = {"predictor": name, "heldout_mse": mse.mean(), "heldout_cosine": cos.mean()}
        if name != "Trained DNN (study model)":
            lo, hi = bootstrap_ci(mse - ref_mse)
            row.update(mse_minus_dnn=(mse - ref_mse).mean(), ci_low=lo, ci_high=hi, p=paired_p(mse, ref_mse))
        rows.append(row)
    comparison = pd.DataFrame(rows)
    mask = comparison["p"].notna()
    comparison.loc[mask, "p_fdr"] = bh(comparison.loc[mask, "p"])

    # leave-one-metric-out on the inputs
    base = [per_sample_errors(r, y_te)[0] for r in dnn_runs]
    base_mse = np.mean(base, axis=0)
    feat_rows = []
    for j, metric in enumerate(METRICS):
        keep = [k for k in range(len(METRICS)) if k != j]
        runs = []
        for s in SEEDS:
            m = train_mlp(x_tr[:, keep], y_tr, s)
            with torch.no_grad():
                runs.append(per_sample_errors(m(torch.FloatTensor(x_te[:, keep])).numpy(), y_te)[0])
        mse = np.mean(runs, axis=0)
        lo, hi = bootstrap_ci(mse - base_mse)
        feat_rows.append({"removed_input_metric": metric, "heldout_mse": mse.mean(),
                          "mse_increase": (mse - base_mse).mean(),
                          "pct_increase": 100 * (mse - base_mse).mean() / base_mse.mean(),
                          "ci_low": lo, "ci_high": hi, "p": paired_p(mse, base_mse)})
    features = pd.DataFrame(feat_rows)
    features["p_fdr"] = bh(features["p"])
    return comparison, features.sort_values("mse_increase", ascending=False)


# --------------------------------------------------------------------------
# Part 2: do the targets steer the answers?
# --------------------------------------------------------------------------

def target_steering():
    df = pd.read_csv(GEN_PATH)
    df = df[~df["model"].isin(EXCLUDED_MODELS)]
    scaler = joblib.load(A_SCALER_PATH)
    lo, rng_ = scaler.data_min_, np.where(scaler.data_range_ > 0, scaler.data_range_, 1.0)
    scale = lambda frame, prefix: np.clip((frame[[f"{prefix}_{m}" for m in METRICS]].values - lo) / rng_, 0, 1)

    mg = df[df["mechanism"] == "MGCoT"].dropna(subset=[f"actual_{m}" for m in METRICS] + [f"target_{m}" for m in METRICS]).reset_index(drop=True)
    actual, target = scale(mg, "actual"), scale(mg, "target")
    mean_target = target.mean(axis=0)

    own = np.abs(actual - target).mean(axis=1)
    constant = np.abs(actual - mean_target).mean(axis=1)
    shuffled = np.zeros(len(mg))
    for (_, _), g in mg.groupby(["model", "dataset"]):
        idx = g.index.to_numpy()
        acc = np.zeros(len(idx))
        for _ in range(N_SHUFFLES):
            perm = rng.permutation(idx)
            while len(idx) > 1 and np.any(perm == idx):       # every answer gets another question's target
                perm = rng.permutation(idx)
            acc += np.abs(actual[idx] - target[perm]).mean(axis=1)
        shuffled[idx] = acc / N_SHUFFLES

    rows = []
    for scope, sel in [("All models", np.ones(len(mg), bool))] + [(m, (mg["model"] == m).to_numpy()) for m in sorted(mg["model"].unique())]:
        for label, other in (("shuffled targets", shuffled), ("constant mean target", constant)):
            d = other[sel] - own[sel]
            ci = bootstrap_ci(d)
            rows.append({"scope": scope, "comparison": f"own targets vs {label}", "n": int(sel.sum()),
                         "distance_own": own[sel].mean(), "distance_other": other[sel].mean(),
                         "other_minus_own": d.mean(), "ci_low": ci[0], "ci_high": ci[1],
                         "p": paired_p(other[sel], own[sel])})
    steering = pd.DataFrame(rows)
    steering["p_fdr"] = bh(steering["p"])

    # per metric: MGCoT answer vs SCoT answer, distance to the same question's target
    sc = df[df["mechanism"] == "SCoT"][["model", "dataset", "record_id"] + [f"actual_{m}" for m in METRICS]]
    pair = mg.merge(sc, on=["model", "dataset", "record_id"], suffixes=("", "_scot")).dropna()
    t_s = scale(pair, "target")
    a_mg = scale(pair, "actual")
    a_sc = np.clip((pair[[f"actual_{m}_scot" for m in METRICS]].values - lo) / rng_, 0, 1)
    metric_rows = []
    for j, m in enumerate(METRICS):
        e_mg, e_sc = np.abs(a_mg[:, j] - t_s[:, j]), np.abs(a_sc[:, j] - t_s[:, j])
        ci = bootstrap_ci(e_sc - e_mg)
        metric_rows.append({"metric": m, "n": len(pair), "scot_distance": e_sc.mean(), "mgcot_distance": e_mg.mean(),
                            "reduction": (e_sc - e_mg).mean(), "ci_low": ci[0], "ci_high": ci[1],
                            "p": paired_p(e_sc, e_mg)})
    per_metric = pd.DataFrame(metric_rows)
    per_metric["p_fdr"] = bh(per_metric["p"])
    return steering, per_metric


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)

    comparison, features = predictor_ablation()
    steering, per_metric = target_steering()

    comparison.round(4).to_csv(os.path.join(OUT_DIR, "1a_predictor_vs_baselines.csv"), index=False)
    features.round(4).to_csv(os.path.join(OUT_DIR, "1b_leave_one_metric_out.csv"), index=False)
    steering.round(4).to_csv(os.path.join(OUT_DIR, "2a_own_vs_shuffled_targets.csv"), index=False)
    per_metric.round(4).to_csv(os.path.join(OUT_DIR, "2b_per_metric_target_approach.csv"), index=False)

    print("=== 1a. Predictor vs baselines (held-out, lower MSE / higher cosine is better) ===")
    print(comparison.round(4).to_string(index=False))
    print("\n=== 1b. Leave-one-metric-out (rise in held-out MSE when the input metric is removed) ===")
    print(features.round(4).to_string(index=False))
    print("\n=== 2a. Distance of MGCoT answers to own vs other targets (positive = closer to own) ===")
    print(steering.round(4).to_string(index=False))
    print("\n=== 2b. Per metric: SCoT minus MGCoT distance to the question's target (positive = MGCoT closer) ===")
    print(per_metric.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
