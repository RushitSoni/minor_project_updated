"""
Bootstrap confidence intervals for the 3-class (LOW / MEDIUM / HIGH) classifier.

Input : survey_with_label3_predictions.csv
        (uses columns: label3 = true label, predicted_label = model prediction)
Output: folder `bootstrap_results/` containing
          - summary_metrics.csv        (point estimate + 95% CI for every metric)
          - per_class_f1.csv           (per-class F1 with 95% CI)
          - per_class_report.csv       (precision / recall / F1 / support per class)
          - confusion_matrix.csv       (counts)
          - baseline_comparison.csv    (model vs always-LOW baseline, paired bootstrap + McNemar;
                                        now also Weighted F1 and per-class F1 rows)
          - bootstrap_samples.csv      (all resampled metric values, for re-plotting)
          - results.json               (everything above in one file)
          - report.txt                 (human-readable summary, same as console output)
          - bootstrap_distributions.png (histograms of key metrics with CIs)

Run:
    pip install numpy pandas scikit-learn scipy matplotlib
    python bootstrap_ci_survey.py
    python bootstrap_ci_survey.py --csv path/to/file.csv --n_boot 10000 --seed 42
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# ------------------------- CONFIG -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="survey_with_label3_predictions.csv")
parser.add_argument("--true_col", default="label3")
parser.add_argument("--pred_col", default="predicted_label")
parser.add_argument("--n_boot", type=int, default=10000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out_dir", default="bootstrap_results")
args = parser.parse_args()

CLASS_NAMES = ["LOW", "MEDIUM", "HIGH"]   # ordinal order (matters for QWK)
CLASS_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
ALPHA = 0.05
os.makedirs(args.out_dir, exist_ok=True)

# ------------------------- LOAD DATA -------------------------
df = pd.read_csv(args.csv)
if "prediction_status" in df.columns:
    df = df[df["prediction_status"] == "ok"]          # drop failed predictions, if any
df = df.dropna(subset=[args.true_col, args.pred_col]).reset_index(drop=True)

y_true = df[args.true_col].str.strip().str.upper().map(CLASS_IDX).to_numpy()
y_pred = df[args.pred_col].str.strip().str.upper().map(CLASS_IDX).to_numpy()
assert not np.isnan(y_true.astype(float)).any(), "Unknown label in true column"
assert not np.isnan(y_pred.astype(float)).any(), "Unknown label in predicted column"
y_true, y_pred = y_true.astype(int), y_pred.astype(int)
n = len(y_true)
y_base = np.zeros(n, dtype=int)                       # baseline: always predict LOW
labels = [0, 1, 2]

# ------------------------- POINT ESTIMATES -------------------------
def all_metrics(yt, yp):
    return {
        "accuracy": float(np.mean(yt == yp)),
        "macro_f1": f1_score(yt, yp, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(yt, yp, average="weighted", labels=labels, zero_division=0),
        "qwk": cohen_kappa_score(yt, yp, weights="quadratic", labels=labels),
    }

point = all_metrics(y_true, y_pred)
point_pc = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
base_point = all_metrics(y_true, y_base)
base_pc = f1_score(y_true, y_base, average=None, labels=labels, zero_division=0)   # NEW

# ------------------------- BOOTSTRAP -------------------------
rng = np.random.default_rng(args.seed)
boot = {k: np.empty(args.n_boot) for k in point}
boot_pc = np.empty((args.n_boot, 3))
boot_gain_macro = np.empty(args.n_boot)
boot_gain_acc = np.empty(args.n_boot)
boot_gain_qwk = np.empty(args.n_boot)
boot_gain_wf1 = np.empty(args.n_boot)                 # NEW
boot_gain_pc = np.empty((args.n_boot, 3))             # NEW

for b in range(args.n_boot):
    idx = rng.integers(0, n, n)                       # resample WITH replacement
    yt, yp, yb = y_true[idx], y_pred[idx], y_base[idx]
    m = all_metrics(yt, yp)
    mb = all_metrics(yt, yb)                          # same idx -> paired comparison
    for k in boot:
        boot[k][b] = m[k]
    boot_pc[b] = f1_score(yt, yp, average=None, labels=labels, zero_division=0)
    boot_gain_macro[b] = m["macro_f1"] - mb["macro_f1"]
    boot_gain_acc[b] = m["accuracy"] - mb["accuracy"]
    boot_gain_qwk[b] = m["qwk"] - mb["qwk"]
    # NEW: paired gains for weighted F1 and per-class F1
    boot_gain_wf1[b] = m["weighted_f1"] - mb["weighted_f1"]
    boot_gain_pc[b] = boot_pc[b] - f1_score(yt, yb, average=None, labels=labels, zero_division=0)

def ci(arr):
    lo, hi = np.percentile(arr, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(lo), float(hi)

# ------------------------- TABLES -------------------------
rows = []
for k, label in [("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1"),
                 ("qwk", "Quadratic weighted kappa"), ("accuracy", "Accuracy")]:
    lo, hi = ci(boot[k])
    rows.append({"metric": label, "point_estimate": point[k],
                 "boot_mean": boot[k].mean(), "boot_std": boot[k].std(ddof=1),
                 "ci_low": lo, "ci_high": hi})
summary_df = pd.DataFrame(rows)

pc_rows = []
for c, name in enumerate(CLASS_NAMES):
    lo, hi = ci(boot_pc[:, c])
    pc_rows.append({"class": name, "support": int(np.sum(y_true == c)),
                    "f1_point": point_pc[c], "boot_mean": boot_pc[:, c].mean(),
                    "ci_low": lo, "ci_high": hi, "ci_width": hi - lo})
per_class_df = pd.DataFrame(pc_rows)

p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
report_df = pd.DataFrame({"class": CLASS_NAMES, "precision": p, "recall": r, "f1": f, "support": s})

cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in CLASS_NAMES],
                     columns=[f"pred_{c}" for c in CLASS_NAMES])

# Baseline comparison (paired bootstrap on the difference) + McNemar
model_ok, base_ok = (y_pred == y_true), (y_base == y_true)
b_cnt = int(np.sum(model_ok & ~base_ok))              # model right, baseline wrong
c_cnt = int(np.sum(~model_ok & base_ok))              # model wrong, baseline right
if b_cnt + c_cnt > 0:
    mcnemar_stat = (abs(b_cnt - c_cnt) - 1) ** 2 / (b_cnt + c_cnt)   # continuity-corrected
    mcnemar_p = float(1 - chi2.cdf(mcnemar_stat, df=1))
else:
    mcnemar_stat, mcnemar_p = float("nan"), float("nan")

# NEW: extended comparison list (original 3 + weighted F1 + per-class F1)
comparisons = [
    ("Macro F1", point["macro_f1"], base_point["macro_f1"], boot_gain_macro),
    ("Accuracy", point["accuracy"], base_point["accuracy"], boot_gain_acc),
    ("Quadratic weighted kappa", point["qwk"], base_point["qwk"], boot_gain_qwk),
    ("Weighted F1", point["weighted_f1"], base_point["weighted_f1"], boot_gain_wf1),
] + [(f"F1 ({c})", point_pc[i], base_pc[i], boot_gain_pc[:, i])
     for i, c in enumerate(CLASS_NAMES)]

base_rows = []
for name, pt_model, pt_base, gain in comparisons:
    lo, hi = ci(gain)
    base_rows.append({"metric": name, "model": pt_model, "baseline_always_LOW": pt_base,
                      "gain_point": pt_model - pt_base, "gain_ci_low": lo, "gain_ci_high": hi,
                      "ci_excludes_zero": bool(lo > 0 or hi < 0)})
baseline_df = pd.DataFrame(base_rows)
baseline_df["mcnemar_b_model_right_base_wrong"] = b_cnt
baseline_df["mcnemar_c_model_wrong_base_right"] = c_cnt
baseline_df["mcnemar_chi2"] = mcnemar_stat
baseline_df["mcnemar_p"] = mcnemar_p

# ------------------------- SAVE -------------------------
out = args.out_dir
summary_df.to_csv(f"{out}/summary_metrics.csv", index=False)
per_class_df.to_csv(f"{out}/per_class_f1.csv", index=False)
report_df.to_csv(f"{out}/per_class_report.csv", index=False)
cm_df.to_csv(f"{out}/confusion_matrix.csv")
baseline_df.to_csv(f"{out}/baseline_comparison.csv", index=False)

samples_df = pd.DataFrame({
    "macro_f1": boot["macro_f1"], "weighted_f1": boot["weighted_f1"],
    "qwk": boot["qwk"], "accuracy": boot["accuracy"],
    "f1_LOW": boot_pc[:, 0], "f1_MEDIUM": boot_pc[:, 1], "f1_HIGH": boot_pc[:, 2],
    "gain_macro_f1_vs_baseline": boot_gain_macro,
    "gain_weighted_f1_vs_baseline": boot_gain_wf1,    # NEW
    "gain_f1_LOW_vs_baseline": boot_gain_pc[:, 0],    # NEW
    "gain_f1_MEDIUM_vs_baseline": boot_gain_pc[:, 1], # NEW
    "gain_f1_HIGH_vs_baseline": boot_gain_pc[:, 2],   # NEW
})
samples_df.to_csv(f"{out}/bootstrap_samples.csv", index=False)

results = {
    "run_info": {"timestamp": datetime.now().isoformat(timespec="seconds"),
                 "input_csv": args.csv, "true_col": args.true_col, "pred_col": args.pred_col,
                 "n_samples": n, "n_boot": args.n_boot, "seed": args.seed,
                 "ci_method": "percentile bootstrap, 95%", "baseline": "always predict LOW",
                 "class_order": CLASS_NAMES},
    "summary_metrics": summary_df.to_dict(orient="records"),
    "per_class_f1": per_class_df.to_dict(orient="records"),
    "per_class_report": report_df.to_dict(orient="records"),
    "confusion_matrix": {"rows_true_cols_pred": CLASS_NAMES, "counts": cm.tolist()},
    "baseline_comparison": baseline_df.to_dict(orient="records"),
}
with open(f"{out}/results.json", "w") as fh:
    json.dump(results, fh, indent=2, default=float)

# ------------------------- REPORT TEXT -------------------------
L = []
L.append(f"Bootstrap evaluation  |  n = {n}  |  resamples = {args.n_boot}  |  seed = {args.seed}")
L.append("95% percentile bootstrap confidence intervals\n")
L.append("CONFUSION MATRIX (rows = true, cols = predicted)")
L.append(cm_df.to_string() + "\n")
L.append("OVERALL METRICS")
for _, rw in summary_df.iterrows():
    L.append(f"  {rw['metric']:26s}: {rw['point_estimate']:.3f}  95% CI [{rw['ci_low']:.3f}, {rw['ci_high']:.3f}]")
L.append("\nPER-CLASS F1")
for _, rw in per_class_df.iterrows():
    L.append(f"  {rw['class']:7s} (n={rw['support']:3d}): {rw['f1_point']:.3f}  95% CI [{rw['ci_low']:.3f}, {rw['ci_high']:.3f}]")
L.append("\nMODEL vs ALWAYS-LOW BASELINE (paired bootstrap of the difference)")
for _, rw in baseline_df.iterrows():
    verdict = "excludes 0 -> significant" if rw["ci_excludes_zero"] else "includes 0 -> NOT significant"
    L.append(f"  {rw['metric']:26s}: model {rw['model']:.3f} vs baseline {rw['baseline_always_LOW']:.3f} | "
             f"gain {rw['gain_point']:.3f}  95% CI [{rw['gain_ci_low']:.3f}, {rw['gain_ci_high']:.3f}]  ({verdict})")
L.append(f"\nMcNemar test (model vs baseline correctness): b={b_cnt}, c={c_cnt}, "
         f"chi2={mcnemar_stat:.3f}, p={mcnemar_p:.4g}")
report_txt = "\n".join(L)
with open(f"{out}/report.txt", "w") as fh:
    fh.write(report_txt + "\n")
print(report_txt)

# ------------------------- PLOT (optional) -------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1"),
              ("qwk", "QWK"), ("gain_macro_f1_vs_baseline", "Macro F1 gain vs baseline")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    for ax, (col, title) in zip(axes, panels):
        vals = samples_df[col].to_numpy()
        lo, hi = ci(vals)
        ax.hist(vals, bins=50, color="#4C72B0", alpha=0.8)
        ax.axvline(lo, color="red", ls="--"); ax.axvline(hi, color="red", ls="--")
        ax.axvline(vals.mean(), color="black")
        if col.startswith("gain"):
            ax.axvline(0, color="green", lw=2)
        ax.set_title(f"{title}\n[{lo:.3f}, {hi:.3f}]")
    plt.tight_layout()
    plt.savefig(f"{out}/bootstrap_distributions.png", dpi=150)
    plt.close()
except ImportError:
    print("\n(matplotlib not installed - skipped plot)")

print(f"\nAll results saved in: {os.path.abspath(out)}/")