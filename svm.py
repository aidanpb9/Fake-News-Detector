"""SVM model for Fake News Detector.
Load preprocessed data and TF-IDF vectorizer.
Train SVM with GridSearchCV hyperparameter tuning and save predictions."""

import pandas as pd
import numpy as np
import joblib
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os


#Make dirs if needed
os.makedirs("results", exist_ok=True)


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


#Hyperparameter Tuning
#C=regularization strength, same as LR
#max_iter=iterations for the solver to converge
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'max_iter': [500, 1000, 2000]
}

#LinearSVC is faster than SVC(kernel='linear') for large datasets
#cv=5 splits train into 4 train and 1 test, rotating 5 times
grid_search = GridSearchCV(
    LinearSVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("Running GridSearchCV (takes a few minutes)")
grid_search.fit(X_train_vec, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")


#Train Best Model
best_params = grid_search.best_params_
model = LinearSVC(**best_params)
model.fit(X_train_vec, y_train)
print("Model trained.")
joblib.dump(model, "models/svm_model.joblib")

#Evaluate
val_preds = model.predict(X_val_vec)
test_preds = model.predict(X_test_vec)

print(f"\nValidation Accuracy: {accuracy_score(y_val, val_preds):.4f}")
print(f"Validation Report:\n{classification_report(y_val, val_preds)}")

print(f"Test Accuracy: {accuracy_score(y_test, test_preds):.4f}")
print(f"Test Report:\n{classification_report(y_test, test_preds)}")


#Feature Importance
feature_names = vectorizer.get_feature_names_out()
weights = model.coef_[0]

#get top 20 words for real and fake
top_fake = np.argsort(weights)[-20:][::-1]
top_real = np.argsort(weights)[:20]

print("\nTop 20 fake words:")
for i in top_fake:
    print(f"{feature_names[i]}: {weights[i]:.4f}")

print("\nTop 20 real words:")
for i in top_real:
    print(f"{feature_names[i]}: {weights[i]:.4f}")


#Save Predictions
pd.DataFrame({'label': y_test, 'prediction': test_preds}).to_csv('results/svm_predictions.csv', index=False)
print("Predictions saved to results/svm_predictions.csv")


#Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, test_preds)
plt.title("SVM")
plt.savefig("results/svm_confusion_matrix.png") 
plt.show()