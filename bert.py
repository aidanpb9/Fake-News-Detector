"""
Bert
Bert is a pre-trained deep learning model. It processes an entire sequence of words simultaneously and
learns relationships between them using self-attention mechanisms. For this project, the pre-trained Bert model is
tuned to classify news articles as real or fake.

INPUT: Raw Article Text
OUTPUT: Binary classification of whether a news article is real (0) or fake (1)

RELEVANT PARAMETERS:
Max Sequence Length: threshold to number of tokens for each article. If the article has more, it is truncated. If
It has less, it is padded.

Number of Epochs: number of times the model is allowed to see the full dataset
Learning Rate: controls how much the weights change at each update
Batch Size: represents the number of samples processed per update


"""



import json
import os
import time

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from tqdm.auto import tqdm

# config
MODEL_NAME = "bert-base-uncased"
MAX_LEN    = 256   # BERT max is 512; 256 balances coverage vs. speed
BATCH_SIZE = 16
EPOCHS     = 3
LR         = 2e-5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset ────────────────────────────────────────────────────────────────────
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts.tolist()
        self.labels    = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ── Load preprocessed data ─────────────────────────────────────────────────────
# Run preprocess.py first if data/train.csv and data/test.csv don't exist.
print("Loading data...")
train_df = pd.read_csv("data/train.csv").dropna(subset=["text", "label"])
test_df  = pd.read_csv("data/test.csv").dropna(subset=["text", "label"])

print(f"  Train samples : {len(train_df)}")
print(f"  Test  samples : {len(test_df)}")
print(f"  Label dist (train) — 0=Real, 1=Fake: {train_df['label'].value_counts().to_dict()}")

# ── Tokenizer & model ──────────────────────────────────────────────────────────
print(f"\nLoading {MODEL_NAME}...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model     = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)
print(f"  Device : {DEVICE}")
print(f"  Params : {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# ── DataLoaders ────────────────────────────────────────────────────────────────
train_dataset = FakeNewsDataset(train_df["text"], train_df["label"], tokenizer, MAX_LEN)
test_dataset  = FakeNewsDataset(test_df["text"],  test_df["label"],  tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

# ── Optimizer & scheduler ──────────────────────────────────────────────────────
optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

# ── Train / eval helpers ───────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc="  Training", leave=False)
    for batch in progress:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += outputs.loss.item()
        progress.set_postfix(loss=f"{outputs.loss.item():.4f}")
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

# ── Training loop ──────────────────────────────────────────────────────────────
print("\n── Training ───────────────────────────────────────────────────────────────")
start = time.time()

for epoch in range(1, EPOCHS + 1):
    t0       = time.time()
    avg_loss = train_epoch(model, train_loader, optimizer, scheduler)
    preds, labels = evaluate(model, test_loader)
    acc     = accuracy_score(labels, preds)
    elapsed = time.time() - t0
    print(f"  Epoch {epoch}/{EPOCHS}  loss: {avg_loss:.4f}  val_acc: {acc:.4f}  ({elapsed:.0f}s)")

total_time = time.time() - start

# ── Final evaluation ───────────────────────────────────────────────────────────
print("\n── Final Evaluation ───────────────────────────────────────────────────────")
preds, labels = evaluate(model, test_loader)
acc                        = accuracy_score(labels, preds)
precision, recall, f1, _   = precision_recall_fscore_support(labels, preds, average="weighted")

print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  Train time: {total_time:.1f}s\n")
print(classification_report(labels, preds, target_names=["Real", "Fake"]))

# ── Save model ─────────────────────────────────────────────────────────────────
os.makedirs("models/bert-fake-news", exist_ok=True)
model.save_pretrained("models/bert-fake-news")
tokenizer.save_pretrained("models/bert-fake-news")
print("Model saved → models/bert-fake-news/")

# ── Save results CSV for comparison with smaller model ─────────────────────────
results = pd.DataFrame([{
    "model":        MODEL_NAME,
    "accuracy":     round(acc, 4),
    "precision":    round(precision, 4),
    "recall":       round(recall, 4),
    "f1":           round(f1, 4),
    "train_time_s": round(total_time, 1),
    "epochs":       EPOCHS,
    "max_len":      MAX_LEN,
    "batch_size":   BATCH_SIZE,
}])
results.to_csv("data/bert_results.csv", index=False)
print("Results saved → data/bert_results.csv")

# ── Confusion Matrix ───────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
ConfusionMatrixDisplay.from_predictions(labels, preds, display_labels=["Real", "Fake"])
plt.title("BERT")
plt.savefig("results/bert_confusion_matrix.png")
plt.show()
print("Confusion matrix saved → results/bert_confusion_matrix.png")

# ── Save softmax probabilities for ROC curve in compare_models.py ──────────────
print("\nGenerating softmax probabilities for ROC curve …")
model.eval()
all_probs = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="  Probability inference"):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        logits         = model(input_ids=input_ids, attention_mask=attention_mask).logits
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

probs = np.vstack(all_probs)
pd.DataFrame({
    "true_label": labels,
    "prob_real":  probs[:, 0],
    "prob_fake":  probs[:, 1],
}).to_csv("results/bert_probabilities.csv", index=False)
print("Probabilities saved → results/bert_probabilities.csv")
