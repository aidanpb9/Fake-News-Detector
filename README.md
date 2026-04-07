Aidan Brinkley
Ethan Sax 
Vincent Cameron

# Project Task Breakdown

## Traditional Models — Low Effort

**Aidan** — Logistic Regression + SVM
- TF-IDF vectorizer → fit LR and SVM → save predictions

**Ethan** — Random Forest
- TF-IDF vectorizer → fit RF → save predictions

---

## BERT Model — High Effort

**Vincent** — Fine-tune bert-base-uncased
- HuggingFace tokenizer → training loop → save model + predictions

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