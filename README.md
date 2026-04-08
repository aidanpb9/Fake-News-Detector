Aidan Brinkley
Ethan Sax 
Vincent Cameron


Notes:
see logreg.py for example for Random Forest

# Tasks

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

