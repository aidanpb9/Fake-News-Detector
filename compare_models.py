#!/usr/bin/env python3
"""
compare_models.py — Unified model comparison visualization suite
Generates:
  1. ROC Curves       → results/roc_curves.png
  2. Confusion Matrices (2×2 grid) → results/confusion_matrices.png
  3. Metrics Bar Chart             → results/metrics_comparison.png
  4. Summary CSV                   → results/model_comparison_summary.csv

Run AFTER all models have been trained (logreg.py, svm.py, randomforest.py, bert.py).
BERT probabilities are cached to results/bert_probabilities.csv after the first run
so re-running is fast.
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
MODELS_DIR  = "models"
DATA_DIR    = "data"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Consistent colours across every plot ───────────────────────────────────────
MODEL_COLORS = {
    "Logistic Regression": "#4C72B0",
    "SVM":                 "#DD8452",
    "Random Forest":       "#55A868",
    "BERT":                "#C44E52",
}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load test data
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading test data …")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv").dropna(subset=["text", "label"])
y_true  = test_df["label"].values
print(f"  {len(test_df):,} samples")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Classical models — Logistic Regression, SVM, Random Forest
# ═══════════════════════════════════════════════════════════════════════════════
print("\nLoading TF-IDF vectorizer …")
vectorizer = joblib.load(f"{DATA_DIR}/tfidf_vectorizer.joblib")
X_test_vec = vectorizer.transform(test_df["text"])

CLASSICAL_PATHS = {
    "Logistic Regression": f"{MODELS_DIR}/logreg_model.joblib",
    "SVM":                 f"{MODELS_DIR}/svm_model.joblib",
    "Random Forest":       f"{MODELS_DIR}/rf_tuned.joblib",
}

all_results = {}

for name, path in CLASSICAL_PATHS.items():
    print(f"  Evaluating {name} …")
    clf    = joblib.load(path)
    y_pred = clf.predict(X_test_vec)

    # Prefer calibrated probabilities; fall back to decision scores for SVM
    if hasattr(clf, "predict_proba"):
        try:
            y_score = clf.predict_proba(X_test_vec)[:, 1]
        except Exception:
            y_score = clf.decision_function(X_test_vec)
    else:
        y_score = clf.decision_function(X_test_vec)

    acc          = accuracy_score(y_true, y_pred)
    p, r, f1, _  = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    all_results[name] = dict(
        y_true=y_true, y_pred=y_pred, y_score=y_score,
        accuracy=acc, precision=p, recall=r, f1=f1,
    )
    print(f"    acc={acc:.4f}  f1={f1:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. BERT — load softmax probabilities from bert_probabilities.csv
#    (generated at the end of bert.py — must exist before running this script)
# ═══════════════════════════════════════════════════════════════════════════════
BERT_PROB_PATH = f"{RESULTS_DIR}/bert_probabilities.csv"
assert os.path.exists(BERT_PROB_PATH), (
    f"\n[ERROR] {BERT_PROB_PATH} not found.\n"
    "Run bert.py first — it saves softmax probabilities needed for the ROC curve."
)

print("\nLoading BERT probabilities …")
bert_probs   = pd.read_csv(BERT_PROB_PATH)
bert_y_true  = bert_probs["true_label"].values
y_score_bert = bert_probs["prob_fake"].values      # continuous score for ROC curve
y_pred_bert  = (y_score_bert >= 0.5).astype(int)   # derive hard predictions from probs

acc_b        = accuracy_score(bert_y_true, y_pred_bert)
p_b, r_b, f1_b, _ = precision_recall_fscore_support(
    bert_y_true, y_pred_bert, average="weighted"
)
all_results["BERT"] = dict(
    y_true=bert_y_true, y_pred=y_pred_bert, y_score=y_score_bert,
    accuracy=acc_b, precision=p_b, recall=r_b, f1=f1_b,
)
print(f"  BERT — acc={acc_b:.4f}  f1={f1_b:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT A — ROC Curves
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Plotting ROC curves ─────────────────────────────────────────────────────")
fig_roc, ax_roc = plt.subplots(figsize=(8, 7))

for name, res in all_results.items():
    fpr, tpr, _ = roc_curve(res["y_true"], res["y_score"])
    roc_auc     = auc(fpr, tpr)
    ax_roc.plot(
        fpr, tpr, lw=2.2, color=MODEL_COLORS[name],
        label=f"{name}  (AUC = {roc_auc:.4f})",
    )

ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.45, label="Random Classifier")
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel("False Positive Rate", fontsize=13)
ax_roc.set_ylabel("True Positive Rate",  fontsize=13)
ax_roc.set_title(
    "ROC Curves — Fake News Detector Models",
    fontsize=15, fontweight="bold", pad=12,
)
ax_roc.legend(loc="lower right", fontsize=11, framealpha=0.9)
ax_roc.grid(True, alpha=0.3)

fig_roc.tight_layout()
roc_path = f"{RESULTS_DIR}/roc_curves.png"
fig_roc.savefig(roc_path, dpi=150, bbox_inches="tight")
print(f"  Saved → {roc_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT B — Confusion Matrices (2 × 2 grid)
# ═══════════════════════════════════════════════════════════════════════════════
print("── Plotting confusion matrices ──────────────────────────────────────────────")
fig_cm, axes = plt.subplots(2, 2, figsize=(12, 10))
fig_cm.suptitle(
    "Confusion Matrices — All Models",
    fontsize=16, fontweight="bold", y=1.01,
)

for ax, (name, res) in zip(axes.flat, all_results.items()):
    cm     = confusion_matrix(res["y_true"], res["y_pred"])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"], fontsize=11)
    ax.set_yticklabels(["Real", "Fake"], fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title(
        f"{name}\n(Acc = {res['accuracy']:.4f})",
        fontsize=12, fontweight="bold", color=MODEL_COLORS[name],
    )

    # Annotate each cell with raw count + row-normalised percentage
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm_pct[i, j] > 60 else "black"
            ax.text(
                j, i,
                f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)",
                ha="center", va="center",
                fontsize=11, color=text_color, fontweight="bold",
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
        "Row %", fontsize=9
    )

fig_cm.tight_layout()
cm_path = f"{RESULTS_DIR}/confusion_matrices.png"
fig_cm.savefig(cm_path, dpi=150, bbox_inches="tight")
print(f"  Saved → {cm_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT C — Grouped Metrics Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════
print("── Plotting metrics comparison ──────────────────────────────────────────────")
metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
metric_keys   = ["accuracy", "precision", "recall", "f1"]
model_names   = list(all_results.keys())

# Build value matrix  shape: (n_models, n_metrics)
values = np.array([
    [all_results[n][k] for k in metric_keys]
    for n in model_names
])

x     = np.arange(len(metric_labels))
width = 0.18
fig_bar, ax_bar = plt.subplots(figsize=(12, 6))

for i, (name, color) in enumerate(MODEL_COLORS.items()):
    bars = ax_bar.bar(
        x + i * width, values[i],
        width, label=name, color=color, alpha=0.87, zorder=3,
    )
    for bar in bars:
        h = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.001,
            f"{h:.3f}",
            ha="center", va="bottom",
            fontsize=7.5, rotation=45,
        )

ax_bar.set_xticks(x + width * 1.5)
ax_bar.set_xticklabels(metric_labels, fontsize=12)
ax_bar.set_ylim(0.78, 1.04)
ax_bar.set_ylabel("Score", fontsize=12)
ax_bar.set_title(
    "Model Performance Comparison — Accuracy · Precision · Recall · F1",
    fontsize=14, fontweight="bold", pad=12,
)
ax_bar.legend(fontsize=10, loc="lower right", framealpha=0.9)
ax_bar.yaxis.grid(True, alpha=0.35, zorder=0)
ax_bar.set_axisbelow(True)

fig_bar.tight_layout()
bar_path = f"{RESULTS_DIR}/metrics_comparison.png"
fig_bar.savefig(bar_path, dpi=150, bbox_inches="tight")
print(f"  Saved → {bar_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary CSV
# ═══════════════════════════════════════════════════════════════════════════════
summary_df = pd.DataFrame([
    {
        "model":     name,
        "accuracy":  round(res["accuracy"],  4),
        "precision": round(res["precision"], 4),
        "recall":    round(res["recall"],    4),
        "f1":        round(res["f1"],        4),
    }
    for name, res in all_results.items()
])

summary_path = f"{RESULTS_DIR}/model_comparison_summary.csv"
summary_df.to_csv(summary_path, index=False)

print(f"\n── Summary ──────────────────────────────────────────────────────────────────")
print(summary_df.to_string(index=False))
print(f"\nSaved → {summary_path}")
print("All comparison plots saved to results/")
