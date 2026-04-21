# Fake News Detector

Aidan Brinkley, Ethan Sax, Vincent Cameron
COMP 5630/6630 Machine Learning, Auburn University, Spring 2026

Binary classification of news articles as real=0 or fake=1 using classical ML models and BERT.

## Folder Structure

```
Fake-News-Detector/
├── data/           #raw data, splits, vectorizer, predictions
├── results/        #confusion matrices, ROC curves, feature importance plots
├── preprocess.py
├── logreg.py
├── svm.py
├── randomforest.py
└── bert.py
```

## How to Run

Download True.csv and Fake.csv from https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset and place them in data/ before running.

Then run scripts in this order:

**1. Preprocess**
```
python3 preprocess.py
```
Cleans and tokenizes raw data, splits into train/val/test, fits and saves TF-IDF vectorizer. 

**2. Logistic Regression**
```
python3 logreg.py
```
Loads TF-IDF vectorizer, tunes hyperparameters with GridSearchCV, trains final model, saves predictions and confusion matrix.

**3. SVM**
```
python3 svm.py
```
Same structure as logreg.py using LinearSVC.

**4. Random Forest**
```
python3 randomforest.py
```
Same structure using RandomForestClassifier.

**5. BERT**
```
python3 bert.py
```
Fine-tunes bert-base-uncased using HuggingFace. Uses its own tokenizer, not TF-IDF.

**6. Compare Models**
```
python3 compare_models.py
```
Loads all saved predictions and model files, generates ROC curves, confusion matrices, metrics bar chart, and summary CSV.

## Requirements

```
pip install -r requirements.txt
```