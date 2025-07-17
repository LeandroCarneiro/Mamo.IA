import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

from helpers.datasetHelper import get_samples, split_healthy_data
from pyswarm import pso

from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import make_scorer
from sklearn.metrics import cohen_kappa_score

from models import MyXGboost



def run_pso_with_progress(X, Y, estimator, n_features,
                          swarmsize=50, maxiter=10, threshold=0.7):
    lb = [0]*n_features
    ub = [1]*n_features
    progress = []

    # Create a scorer for Cohen's Kappa
    kappa_scorer = make_scorer(cohen_kappa_score)

    def objective_with_progress(weights, est, X_, Y_):
        Xw = X_ * weights
                
        # Use it in cross_val_score
        fit = 1 - cross_val_score(est, Xw, Y_, cv=5, scoring=kappa_scorer).mean()
        progress.append(fit)
        
        if len(progress) % 10 == 0:
            print(f"Eval {len(progress)}: best fitness so far = {min(progress):.4f}")
        return fit

    best_pos, best_fit = pso(
        objective_with_progress,
        lb, ub,
        args=(estimator, X, Y),
        swarmsize=swarmsize,
        maxiter=maxiter
    )

    mask = best_pos > threshold
    selected_features = np.where(mask)[0].tolist()
    return best_pos, best_fit, progress, selected_features


directory_path = './datasets'
data_health = get_samples(os.path.join(directory_path, 'DT.Healthy.csv'))

# Load the PAN-CANCER-TRANSPOSED.csv data
healthy_cases, prebrca_cases, cancer_cases = split_healthy_data(data_health)

# Combine the data into a single dataframe
# Tag each list of cases
healthy_cases = pd.DataFrame(healthy_cases)
healthy_cases['Tag'] = 'HEALTHY'
prebrca_cases = pd.DataFrame(prebrca_cases)
prebrca_cases['Tag'] = 'PRE-BRCA'
cancer_cases = pd.DataFrame(cancer_cases)
cancer_cases['Tag'] = 'BRCA'

print("Data loaded successfully.")

df_cancer = pd.concat([healthy_cases, prebrca_cases, cancer_cases], ignore_index=True) #blood samples
X = df_cancer.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
Y = df_cancer.iloc[:, -1]

feature_names = np.array(data_health[0][:-1])

# Fill missing values with the lowest value of its cpg site
X = X.apply(lambda col: col.fillna(col.min()), axis=0)

print(feature_names)  # Display first 5 feature names for brevity

n_features = X.shape[1]
print(f"Loaded dataset with {n_features} features and {len(Y)} samples")

# Use DecisionTreeClassifier as the estimator
estimator = MyXGboost.DecisionTreeMultiClass()

# 2) Run PSO
# For high-dimensional data (27k features), reduce swarmsize and maxiter for tractability
best_weights, best_fitness, progress, X_selected = run_pso_with_progress(
    X, Y, estimator, n_features,
    swarmsize=10,   # smaller swarm for memory/speed
    maxiter=25,      # fewer iterations to avoid long runtimes
    threshold=0.90  # higher threshold to select only strongest features
)
print(f"Done PSO → best fitness = {best_fitness:.4f}")

print(X_selected) 
print(best_fitness)
print(best_weights.shape)


# select features with PSO weight > 0.95
mask = best_weights > 0.9
X_selected = X.iloc[:, mask]
# how many features we kept
print(f"Threshold=0.9, selected {X_selected.shape[1]} features")
# Show the selected features
print("Selected features:")
selected_feature_names = feature_names[mask]
print(selected_feature_names)


# Use LabelEncoder to encode the target classes
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# 1) evaluate with all features
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X, Y_encoded, test_size=0.2, random_state=42
)
# Apply SMOTE to balance the training instances
smote = SMOTE(sampling_strategy='auto', random_state=None, k_neighbors=10)
X_train_all, y_train_all = smote.fit_resample(X_train_all, y_train_all)

# 2) evaluate with selected features
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, Y_encoded, test_size=0.2, random_state=42
)
# Apply SMOTE to balance the training instances
smote = SMOTE(sampling_strategy='auto', random_state=None, k_neighbors=10)
X_train, y_train = smote.fit_resample(X_train, y_train)

modes = [
    {
        'Name': 'RandomForest100',
        'Model': MyXGboost.RandomForest100()
    },
    {
        'Name': 'RandomForest200',
        'Model': MyXGboost.RandomForest200()
    },
    {
        'Name': 'RandomForest300',
        'Model': MyXGboost.RandomForest300()
    },
    {
        'Name': 'XGBoost',
        'Model': MyXGboost.XGBoostMultiClass()
    },
    {
        'Name': 'LightGBM',
        'Model': MyXGboost.LightGBMMulticlass()
    },
    {
        'Name': 'AdaBoost',
        'Model': MyXGboost.AdaBoost()
    },
    {
        'Name': 'GradientBoosting',
        'Model': MyXGboost.GradientBoosting()
    }
]

for m in modes:
    selector = m['Model'].fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = selector.predict(X_test)
    y_pred_proba = selector.predict_proba(X_test)
    #print(f'Predict probability: {y_pred_proba}')
          
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    # Handle binary and multiclass cases for ROC AUC
    y_pred_proba = selector.predict_proba(X_test)
    if y_pred_proba.shape[1] == 2:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

    print(f"Model: {m['Name']}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

    # Compute Kappa index for all features
    kappa = cohen_kappa_score(y_test, y_pred)
    print("Selected features → Kappa index:", kappa)

    #2) Confusion matrices side by side
    disp = ConfusionMatrixDisplay.from_estimator(selector, X_test, y_test, normalize='true')
    disp.ax_.set_title("Normalized Confusion Matrix")
    plt.show()

    if hasattr(selector, 'feature_importances_'):
        importances = selector.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        print("Top 20 important features:")
        for rank, idx in enumerate(indices, 1):
            print(f"{rank}. {selected_feature_names[idx]}: {importances[idx]:.4f}")
    else:
        print("This model does not provide feature importances.")