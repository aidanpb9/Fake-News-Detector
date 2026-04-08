Aidan Brinkley
Ethan Sax 
Vincent Cameron


Future work: feature importance


# Project Task Breakdown
CALL these transform functions instead of .fit
import joblib
vectorizer = joblib.load('data/tfidf_vectorizer.joblib')
X_train_tfidf = vectorizer.transform(train_df['text'])
X_val_tfidf   = vectorizer.transform(val_df['text'])
X_test_tfidf  = vectorizer.transform(test_df['text'])


**Aidan** — Logistic Regression + SVM
- Load saved TF-IDF vectorizer → fit LR and SVM → save predictions
- hyperparam Tune LR + SVM: GridSearchCV on C, kernel, max_iter

**Ethan** — Random Forest
- Load saved TF-IDF vectorizer → fit RF → save predictions
- hyperparam Tune Random Forest: GridSearchCV on n_estimators, max_depth

**Vincent** — Fine-tune bert-base-uncased (pre-trained, decided)
- HuggingFace tokenizer → training loop → save model + predictions
- Uses its own tokenizer, not TF-IDF

## Evaluation — Low Effort
**Vincent** — Confusion matrices + ROC curves
- One script that loads all predictions and plots side-by-side

**Joint** — Results comparison table
- Accuracy, Precision, Recall, F1 for all 4 models in one table


## Final Report (revisit these sections later)
- Introduction
- dataset section and description
- Background, motivation,
- Methods section: describe preprocessing + all 4 models
- Results + conclusion section: Interpret results, answer design questions, future work

fix README: Instructions to run each script, requirements.txt, folder structure