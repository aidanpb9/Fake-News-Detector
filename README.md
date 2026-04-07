Aidan Brinkley
Ethan Sax 
Vincent Cameron


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

---

## Traditional Models — Low Effort

**Aidan** — Logistic Regression + SVM
- Load saved TF-IDF vectorizer → fit LR and SVM → save predictions

**Person 3** — Random Forest
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

**Person 3** — Tune Random Forest
- GridSearchCV on n_estimators, max_depth

---

## Final Report — Medium Effort

**Aidan** — Introduction + dataset section
- Background, motivation, dataset description

**Person 3** — Methods section
- Describe preprocessing + all 4 models

**Vincent** — Results + conclusion section
- Interpret results, answer design questions, future work

---

## Code + README — Low Effort

**Joint** — Clean up repo + write README
- Instructions to run each script, requirements.txt, folder structure