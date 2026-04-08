"""Logistic Regression model for Fake News Detector.
Load preprocessed data and TF-IDF vectorizer.
Train LR with GridSearchCV hyperparameter tuning and save predictions."""

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

#Load Data
train_df = pd.read_csv('data/train.csv')
val_df = pd.read_csv('data/val.csv')
test_df = pd.read_csv('data/test.csv')

X_train, y_train = train_df['text'], train_df['label']
X_val, y_val = val_df['text'], val_df['label']
X_test, y_test = test_df['text'], test_df['label']

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"Label distribution (train):\n{y_train.value_counts()}")


#Vectorize
vectorizer = joblib.load('data/tfidf_vectorizer.joblib')

X_train_vec = vectorizer.transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

print(f"Vectorized shape (train): {X_train_vec.shape}")


# ── Hyperparameter Tuning ─────────────────────────────────────────────────────


# ── Train Best Model ──────────────────────────────────────────────────────────


# ── Evaluate ──────────────────────────────────────────────────────────────────


# ── Save Predictions ──────────────────────────────────────────────────────────
