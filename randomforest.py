"""Random Forrest Model for Fake News Detection
1. Load preprocessed data and TF-IDF vectorizer
2. Train and save baseline Random Forrest Model
3. Output and save results"""



#Necessary Libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold

#Load Data
train_df = pd.read_csv('data/train.csv')
val_df = pd.read_csv('data/val.csv')
test_df = pd.read_csv('data/test.csv')

#Train, Test, Validation X and Y pairings (text, label)
X_train, y_train = train_df['text'], train_df['label']
X_val, y_val = val_df['text'], val_df['label']
X_test, y_test = test_df['text'], test_df['label']

#Sanity Check
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"Label distribution (train):\n{y_train.value_counts()}")


#Load shared TF-IDF vectorizer that we're using for all classical (baseline) models
vectorizer = joblib.load('data/tfidf_vectorizer.joblib')

#Vectorize
X_train_vec = vectorizer.transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

#Sanity Check
print(f"Vectorized shape (train): {X_train_vec.shape}")


"""
RANDOM FOREST MODEL
Random Forest is a machine learning mechanism that excels in capturing nonlinear relationships where learning is 
achieved through generation of an ensemble of decision trees. For this project, Random Forest is used to classify news 
articles as real or fake.

INPUT: Article Text converted into numerical feature vectors using TF-IDF vectorizer
OUTPUT: Binary classification of whether a news article is real (0) or fake (1) 

RELEVANT PARAMETERS
n_estimators: number of trees in the forrest
criterion: function to measure the quality of a split. Will use default Gini Criterion
max_depth: maximum depth of the tree
min_samples_split: The minimum number of samples required to split an internal node. Will use default of 2
min_samples_leaf: The minimum number of samples required to be at a leaf node.
A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of
the left and right branches. Will use default of 1

min_weight_fraction_leaf: The minimum weighted fraction of the sum total of weights (of all the input samples) required
to be at a leaf node. Not using sample_weight so samples will have equal weights

max_features: The number of features to consider when looking for the best split.
Will use the square root of the number of features

max_leaf_nodes: Best nodes are defined as relative reduction in impurity. Will use default of unlimited # of leaf nodes
min_impurity_decrease: A node will be split if this split induces a decrease of the impurity greater than or equal
to value selected. Not using this

bootstrap: Whether bootstrap samples are used when building trees. Will be using bootstrap samples as opposed to
entire dataset

oob_score: Whether to use out-of-bag samples to estimate the generalization score. Using accuracy_score by default
n_jobs: The number of jobs to run in parallel.
random_state: Controls both the randomness of the bootstrapping of the samples used when building trees
and the sampling of the features to consider when looking for the best split at each node

max_samples: The number of samples to draw from X to train each base estimator when bootstrapping. Will use default
and draw X.shape[0] samples
"""

#Baseline Random Forrest Model (No Hyperparameter Tuning)
rf_baseline = RandomForestClassifier(
    n_estimators=100, #100 trees in forrest
    max_depth=None, #nodes expanded until all leaves are pure
    random_state=10, #reproducibility
    n_jobs=-1, #utilize all CPU cores
    verbose = 1 #progress messages
)


#Train
start_time = time.time() #time tracking
print("Training Random Forest...")
rf_baseline.fit(X_train_vec, y_train)
end_time = time.time() #time tracking
print(f"Training took {(end_time - start_time)/60:.2f} minutes")

#Save Baseline Model
joblib.dump(rf_baseline, 'models/rf_baseline.joblib')
print("Baseline model saved.")


#Validation and Test Computations
val_preds = rf_baseline.predict(X_val_vec)
test_preds = rf_baseline.predict(X_test_vec)

#Clean Outputs
print(f"\nValidation Accuracy: {accuracy_score(y_val, val_preds):.4f}")
print(f"Validation Report:\n{classification_report(y_val, val_preds)}")

print(f"\nTest Accuracy: {accuracy_score(y_test, test_preds):.4f}")
print(f"Test Report:\n{classification_report(y_test, test_preds)}")


#Save Results
validation_report = classification_report(y_val, val_preds)
test_report = classification_report(y_test, test_preds)

#Validation
with open('results/rf_baseline_val_results.txt', 'w') as f:
    f.write("Validation Accuracy: {:.4f}\n".format(
        accuracy_score(y_val, val_preds)
    ))
    f.write(validation_report)

print("Validation results saved.")

#Test
with open('results/rf_baseline_test_results.txt', 'w') as f:
    f.write("Test Accuracy: {:.4f}\n".format(
        accuracy_score(y_test, test_preds)
    ))
    f.write(test_report)

print("Test results saved.")

#Confusion Matrix: Validation Set
ConfusionMatrixDisplay.from_predictions(y_val, val_preds)
plt.title("Random Forest Baseline - Validation")
plt.savefig("results/rf_baseline_val_confusion_matrix.png") #Save matrix
plt.show()

################################################################################################################
"""
HYPERPARAMETER TUNING
***NOTE: Rubric asks for short description of method, inputs, and outputs***

RELEVANT PARAMETERS
scoring: Strategy to evaluate the performance of the cross-validated model on the test set.
n_jobs: The number of jobs to run in parallel.
cv: Determines the cross-validation splitting strategy.
pre-dispatch: Controls the number of jobs that get dispatched during parallel execution. Reducing this number can be 
useful to avoid an explosion of memory consumption when more jobs get dispatched than CPUs can process. Will utilize 
default (2*n_jobs)

param_grid: Dictionary with parameters names as keys and lists of parameter settings to try as values, 
or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are explored. 
This enables searching over any sequence of parameter settings.
"""

#Grid of Tuning Parameters
#3 x 3 x 3 x 2 = 54 total parameter combinations
#3-Fold Cross-Validation means 54 x 3 = 162 total model fits throughout tuning
param_grid = {
    'n_estimators': [100, 200, 300], #more trees can increase model stability
    'max_depth': [None, 20, 40], #acts to control overfitting
    'min_samples_split': [2, 5, 10], #acts to prevent overfitting
    'max_features': ['sqrt', 'log2'] #controls how many features each tree sees. Functions applied to the total number of features
}

#Stratified K-Fold Cross-Validation ---> For binary classifiers
cv_strategy = StratifiedKFold(
    n_splits=3, #folds
    shuffle=True, #addition of robustness using shuffled splits across folds
    random_state=10 #reproducibility
)


#GridSearch
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=10, #reproducibility
                           n_jobs=-1), #utilize all CPU cores
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='f1', #f1 score: harmonic mean between precision and recall requiring both to contribute equally (balance)
    verbose=3,
    n_jobs=-1 #utilize all CPU cores
)

#Hyperparameter Tuning
print("Starting hyperparameter tuning...")
start_time_ht = time.time() #time tracking

grid_search.fit(X_train_vec, y_train)

end_time_ht = time.time() #time tracking
print(f"Tuning took {(end_time_ht - start_time_ht)/60:.2f} minutes")

#Clean Output
print(f"Best parameters: {grid_search.best_params_}") #Parameter setting that gave the best results on the hold out data
print(f"Best CV accuracy: {grid_search.best_score_:.4f}") #Mean cross-validated score of the best_estimator

#Save tuned parameters
with open('results/rf_tuned_params.txt', 'w') as f:
    f.write(str(grid_search.best_params_))

#Tuned Random Forest Model
rf_tuned = grid_search.best_estimator_

#Save Tuned Random Forest Model
joblib.dump(rf_tuned, 'models/rf_tuned.joblib')

#Compare to Baseline Model
#Validation and Test Computations
tuned_val_preds = rf_tuned.predict(X_val_vec)
tuned_test_preds = rf_tuned.predict(X_test_vec)

#Clean Outputs
print(f"\nTuned Validation Accuracy: {accuracy_score(y_val, tuned_val_preds):.4f}")
print(f"Tuned Validation Report:\n{classification_report(y_val, tuned_val_preds)}")

print(f"\nTuned Test Accuracy: {accuracy_score(y_test, tuned_test_preds):.4f}")
print(f"Tuned Test Report:\n{classification_report(y_test, tuned_test_preds)}")

#Save Results
tuned_validation_report = classification_report(y_val, tuned_val_preds)
tuned_test_report = classification_report(y_test, tuned_test_preds)

#Validation
with open('results/rf_tuned_val_results.txt', 'w') as f:
    f.write("Validation Accuracy: {:.4f}\n".format(
        accuracy_score(y_val, tuned_val_preds)
    ))
    f.write(tuned_validation_report)

print("Tuned Validation results saved.")

#Test
with open('results/rf_tuned_test_results.txt', 'w') as f:
    f.write("Test Accuracy: {:.4f}\n".format(
        accuracy_score(y_test, tuned_test_preds)
    ))
    f.write(tuned_test_report)

print("Tuned Test results saved.")

#Confusion Matrix: Validation Set
ConfusionMatrixDisplay.from_predictions(y_val, tuned_val_preds)
plt.title("Random Forest Hyperparameter Tuned - Validation")
plt.savefig("results/rf_tuned_val_confusion_matrix.png") #Save matrix
plt.show()

#Confusion Matrix: Test Set
ConfusionMatrixDisplay.from_predictions(y_test, tuned_test_preds)
plt.title("Random Forest Hyperparameter Tuned - Test")
plt.savefig("results/rf_tuned_test_confusion_matrix.png") #Save matrix
plt.show()

#Save Predictions: Original article text, true test label, prediction label
#For inspection of misclassifications
#manual review of misclassified articles to identify linguistic ambiguities or stylistic overlap between classes
pd.DataFrame({
    'text': X_test,
    'label': y_test,
    'prediction': tuned_test_preds
}).to_csv(
    'results/rf_tuned_predictions.csv',
    index=False
)
print("Predictions saved to results/rf_tuned_predictions.csv")

#GLobal Feature Importance (overall predictive strength)
#Add description later

#recover actual token names from saved vectorizer
feature_names = vectorizer.get_feature_names_out()

#pair feature names with feature importance scores in a dataframe
#tells how much a word contributed to splits across all trees
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_tuned.feature_importances_
})

#Sort by most important features
importance_df = importance_df.sort_values(
    by='importance',
    ascending=False
)

#Plot of top 20 most important features with scores
top_features = importance_df.head(20)

plt.figure(figsize=(10, 6))
plt.barh(
    top_features['feature'],
    top_features['importance']
)
plt.gca().invert_yaxis()
plt.title("Random Forest Global Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("results/rf_tuned_global_feature_importance.png")
plt.show()


#Feature Importance by Class
#Compare average TF-IDF values by class
fake_mask = (y_train == 1).to_numpy()
real_mask = (y_train == 0).to_numpy()
fake_mean = X_train_vec[fake_mask].mean(axis=0)
real_mean = X_train_vec[real_mask].mean(axis=0)

#Convert to arrays
fake_mean = fake_mean.A1
real_mean = real_mean.A1

#Generate dataframe for comparison
class_feature_df = pd.DataFrame({
    'feature': feature_names,
    'fake_mean_tfidf': fake_mean,
    'real_mean_tfidf': real_mean
})

#Mean Difference Score (fake mean - real mean)
class_feature_df['difference'] = (
    class_feature_df['fake_mean_tfidf']
    - class_feature_df['real_mean_tfidf']
)

#Words most associated with fake articles (positive values)
top_fake_words = class_feature_df.sort_values(
    by='difference',
    ascending=False
).head(20)

print(top_fake_words)

#Save Results
with open('results/top_20_fake_words.txt', 'w') as f:
    f.write(top_fake_words.to_string(index=False))

print("Top fake-associated words saved.")

#Words most associated with real articles (negative values)
top_real_words = class_feature_df.sort_values(
    by='difference',
    ascending=True
).head(20)

print(top_real_words)

#Save Results
with open('results/top_20_real_words.txt', 'w') as f:
    f.write(top_real_words.to_string(index=False))

print("Top real-associated words saved.")
