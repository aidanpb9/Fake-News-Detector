"""Evaluation script: generates confusion matrices and ROC curves for all 4 models.
Loads prediction CSVs from results/ and saves plots to results/."""

import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

os.makedirs("results", exist_ok=True)

#Load predictions for each model
logreg = pd.read_csv("results/logreg_predictions.csv")
svm    = pd.read_csv("results/svm_predictions.csv")
rf     = pd.read_csv("results/rf_tuned_predictions.csv")
bert   = pd.read_csv("results/bert_predictions.csv")

#Normalize bert column names to match others
bert = bert.rename(columns={"true_label": "label", "predicted": "prediction"})

models = {
    "Logistic Regression": logreg,
    "SVM":                 svm,
    "Random Forest":       rf,
    "BERT":                bert,
}


#Confusion Matrices (one per model)
for name, df in models.items():
    ConfusionMatrixDisplay.from_predictions(
        df["label"], df["prediction"],
        display_labels=["Real", "Fake"]
    )
    plt.title(name)
    filename = name.lower().replace(" ", "_")
    plt.savefig(f"results/{filename}_confusion_matrix.png")
    plt.close()
    print(f"Confusion matrix saved: results/{filename}_confusion_matrix.png")


#ROC Curves (all 4 on one plot)
#Note: ROC requires probability scores, but LinearSVC and RF only give hard predictions.
#We use the prediction as a binary score (0 or 1) which gives a single point on the ROC curve.
#For a proper ROC curve, models would need to output probabilities via decision_function or predict_proba.
fig, ax = plt.subplots(figsize=(8, 6))

for name, df in models.items():
    fpr, tpr, _ = roc_curve(df["label"], df["prediction"])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")

ax.plot([0, 1], [0, 1], 'k--', label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models")
ax.legend()
plt.tight_layout()
plt.savefig("results/roc_curves.png")
plt.close()
print("ROC curves saved: results/roc_curves.png")
