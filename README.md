Aidan Brinkley
Ethan Sax 
Vincent Cameron


Future work: feature importance


# Project Task Breakdown

## Data Split — decided
- 70% train / 10% validation / 20% test
- Validation set is used for tuning, test set is only touched at the very end
- Aidan's preprocessing script needs to be updated to output all three splits

## TF-IDF — decided
- Word-level unigrams only (scikit-learn default)
- Fit the vectorizer once on training data, save it with joblib
- Everyone loads the same saved vectorizer — do not re-fit individually
- This ensures all model results are comparable

CALL these transform functions instead of .fit
import joblib
vectorizer = joblib.load('data/tfidf_vectorizer.joblib')
X_train_tfidf = vectorizer.transform(train_df['text'])
X_val_tfidf   = vectorizer.transform(val_df['text'])
X_test_tfidf  = vectorizer.transform(test_df['text'])

---

## Traditional Models — Low Effort

**Aidan** — Logistic Regression + SVM
- Load saved TF-IDF vectorizer → fit LR and SVM → save predictions

**Ethan** — Random Forest
- Load saved TF-IDF vectorizer → fit RF → save predictions

---

## BERT Model — High Effort

**Vincent** — Fine-tune bert-base-uncased (pre-trained, decided)
- HuggingFace tokenizer → training loop → save model + predictions
- Uses its own tokenizer, not TF-IDF

---

## Evaluation — Low Effort

**Vincent** — Confusion matrices + ROC curves
- One script that loads all predictions and plots side-by-side

**Joint** — Results comparison table
- Accuracy, Precision, Recall, F1 for all 4 models in one table

---

## Hyperparameter Tuning — Medium Effort

**Aidan** — Tune LR + SVM
- GridSearchCV on C, kernel, max_iter

**Ethan** — Tune Random Forest
- GridSearchCV on n_estimators, max_depth

---

## Final Report — Medium Effort

**Aidan** — Introduction + dataset section
- Background, motivation, dataset description

**Ethan** — Methods section
- Describe preprocessing + all 4 models

**Vincent** — Results + conclusion section
- Interpret results, answer design questions, future work

---

## Code + README — Low Effort

**Joint** — Clean up repo + write README
- Instructions to run each script, requirements.txt, folder structure