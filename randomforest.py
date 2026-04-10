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


#RANDOM FORREST RELEVANT PARAMETERS
#n_estimators: number of trees in the forrest
#criterion: function to measure the quality of a split. Will use default Gini Criterion
#max_depth: maximum depth of the tree
#min_samples_split: The minimum number of samples required to split an internal node. Will use default of 2
#min_samples_leaf: The minimum number of samples required to be at a leaf node.
#A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of
# the left and right branches. Will use default of 1

#min_weight_fraction_leaf: The minimum weighted fraction of the sum total of weights (of all the input samples) required
#to be at a leaf node. Not using sample_weight so samples will have equal weights

#max_features: The number of features to consider when looking for the best split.
#Will use the square root of the number of features

#max_leaf_nodes: Best nodes are defined as relative reduction in impurity. Will use default of unlimited # of leaf nodes
#min_impurity_decrease: A node will be split if this split induces a decrease of the impurity greater than or equal
#to value selected. Not using this

#bootstrap: Whether bootstrap samples are used when building trees. Will be using bootstrap samples as opposed to
#entire dataset

#oob_score: Whether to use out-of-bag samples to estimate the generalization score. Using accuracy_score by default
#n_jobs: The number of jobs to run in parallel.
#random_state: Controls both the randomness of the bootstrapping of the samples used when building trees
#and the sampling of the features to consider when looking for the best split at each node

#max_samples: The number of samples to draw from X to train each base estimator when bootstrapping. Will use default
#and draw X.shape[0] samples

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

#Hyperparameter Tuning
