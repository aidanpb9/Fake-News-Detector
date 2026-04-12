# Fake News Detection

**Aidan Brinkley** — Auburn University — apb0074@auburn.edu  
**Vincent Cameron** — Auburn University — vjc0009@auburn.edu

---

## Abstract

This project aims to evaluate the effectiveness of various machine learning models in detecting fake news using the ISOT dataset. We will compare traditional classifiers (Logistic Regression, Random Forest, SVM) against a state-of-the-art BERT transformer model to determine if the increased computational cost of deep learning provides a significant boost in classification accuracy for misinformation detection.

---

## 1. Project Idea and Design Questions

Fake news poses a significant threat to information integrity. Our project seeks to answer:

- How do traditional NLP feature extraction methods like TF-IDF combined with standard classifiers compare to the contextual embeddings of BERT?
- Which specific linguistic features are most indicative of "fake" news within the ISOT corpus?

---

## 2. Proposed ML Task

Binary classification: predict a label y ∈ {0, 1} (0 = Real, 1 = Fake) given the text content of a news article. Models will be evaluated using Accuracy, Precision, Recall, and F1-score.

---

## 3. Dataset Description

**ISOT Fake News Dataset**

| Property | Details |
|---|---|
| Size | ~45,000 articles |
| Features | Article title, full text, subject, date |
| Real articles | Reuters.com |
| Fake articles | Various flagged unreliable sources |

**Preprocessing:** Tokenization, stop-word removal, and lemmatization for traditional models. BERT-specific tokenizer for the transformer model.

---

## 4. Software and Planned Analysis

Python libraries:
- **Scikit-learn** — Logistic Regression, Random Forest, SVM
- **HuggingFace Transformers** — loading and fine-tuning bert-base-uncased
- **Pandas/NumPy** — data manipulation

We will develop a unified preprocessing pipeline, custom training loops, and a visualization suite for comparing ROC curves and confusion matrices.

---

## 5. Breakdown of Work

| Member | Responsibilities |
|---|---|
| Aidan Brinkley | Data preprocessing pipeline, TF-IDF feature extraction, Logistic Regression and SVM |
| Vincent Cameron | Random Forest, BERT fine-tuning, evaluation/visualization (ROC curves, confusion matrices) |
| Jointly | Hyperparameter tuning, results analysis, report writing |

---

## 6. Tentative Schedule

| Date | Milestone |
|---|---|
| Feb 25 | Submit Proposal |
| Mar 15 | Complete data cleaning and baseline model training |
| Apr 05 | Complete BERT training and results comparison |
| Apr 20 | Final Report and Code Submission |

---

## References

[1] Ahmed H, Traore I, Saad S. (2017) "Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques."

[2] Ahmed, H., Traore, I., Saad, S. (2018). Detecting Opinion Spams and Fake News Using Text Classification. *Security and Privacy*, 1(1).

[3] Devlin, J., Chang, M. W., Lee, K., Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT 2019*, pp. 4171–4186.