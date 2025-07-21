#imports
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, make_scorer, roc_auc_score
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from helpers.datasetHelper import get_samples, split_healthy_data
from imblearn.over_sampling import SMOTE
from sklearn.metrics import cohen_kappa_score

from helpers.ploting import display_confusion_matrix_pink_variants
from helpers.metaheuristics import run_two_stage_pso_simple, run_ga_with_progress
from models import MyXGboost
import numpy as np

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
best_weights, best_fitness, progress, X_selected = run_two_stage_pso_simple(
    X, Y, estimator, n_features,
    stage1_threshold=0.4,
    stage2_threshold=0.7
)
print(f"Done PSO → best fitness = {best_fitness:.4f}")

print(X_selected) 
print(best_fitness)
print(best_weights.shape)


X_selected_pso = X.iloc[:, X_selected]

# Show the selected features
print("Selected features:")
selected_feature_names = feature_names[X_selected]
print(selected_feature_names)

best_weights_ga, best_fitness_ga, progress_ga, X_selected_ga = run_ga_with_progress(
    X, Y, estimator, X.shape[1], 
    pop_size=30, n_generations=500, threshold=0.5
)
print(f"Done GA → best fitness = {best_fitness_ga:.4f}")
print(f"Number of selected features: {len(selected_feature_names)}")
print(f"Selected feature indices: {selected_feature_names[:10]}...")  # Show first 10

# Convert best_weights_ga to numpy array before comparison
best_weights_ga_array = np.array(best_weights_ga)
mask_ga = best_weights_ga_array > 0.9
X_selected_ga = X.iloc[:, mask_ga]

selected_feature_names_ga = feature_names[mask_ga]
print(selected_feature_names_ga)

# Use LabelEncoder to encode the target classes
label_encoder = LabelEncoder()
# Define the order: Healthy -> Pre-BRCA -> BRCA
ordered_labels = ['HEALTHY', 'PRE-BRCA', 'BRCA']
label_encoder.fit(ordered_labels)
Y_encoded = label_encoder.transform(Y)

print(f"Encoded target classes: {label_encoder.classes_}")

# 1) evaluate with all features
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X, Y_encoded, test_size=0.2, random_state=42
)
# Apply SMOTE to balance the training instances
smote = SMOTE(sampling_strategy='auto', random_state=None, k_neighbors=10)
X_train_all, y_train_all = smote.fit_resample(X_train_all, y_train_all)

# 2) evaluate with selected features
X_train_ga, X_test_ga, y_train_ga, y_test_ga = train_test_split(
    X_selected_ga, Y_encoded, test_size=0.2, random_state=42
)
# Apply SMOTE to balance the training instances
smote = SMOTE(sampling_strategy='auto', random_state=None, k_neighbors=10)
X_train_ga, y_train_ga = smote.fit_resample(X_train_ga, y_train_ga)

# 3) evaluate with PSO selected features
X_train_pso, X_test_pso, y_train_pso, y_test_pso = train_test_split(
    X_selected_pso, Y_encoded, test_size=0.2, random_state=42
)
# Apply SMOTE to balance the training instances
smote = SMOTE(sampling_strategy='auto', random_state=None, k_neighbors=10)
X_train_pso, y_train_pso = smote.fit_resample(X_train_pso, y_train_pso)

modes = [
    # {
    #     'Name': 'RandomForest100',
    #     'Model': MyXGboost.RandomForest100()
    # },
    # {
    #     'Name': 'RandomForest200',
    #     'Model': MyXGboost.RandomForest200()
    # },
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
        'Name': 'GradientBoosting',
        'Model': MyXGboost.GradientBoosting()
    }
]

# Print dimensions of different training datasets
print("Training data dimensions:")
print(f"Original data (X_train_all): {X_train_all.shape}")
print(f"GA selected features (X_train_ga): {X_train_ga.shape}")
print(f"PSO selected features (X_train_pso): {X_train_pso.shape}")

for feature_set in [('GA', X_train_ga, X_test_ga, y_train_ga, y_test_ga), 
                   ('PSO', X_train_pso, X_test_pso, y_train_pso, y_test_pso)]:
    method, X_train, X_test, y_train, y_test = feature_set
    print(f"\n=== Results for {method} selected features ===")
    
    for m in modes:
        # Split training data into train and validation sets for early stopping
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        get_best = MyXGboost.get_best(m['Model'], X_train_split, y_train_split, make_scorer(cohen_kappa_score))
        # Use the best model from grid search
        selector = get_best[0]
    
        # Evaluate the model
        y_pred = selector.predict(X_test)
        y_pred_proba = selector.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        # Handle binary and multiclass cases for ROC AUC
        if y_pred_proba.shape[1] == 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

        print(f"\nModel: {m['Name']}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")

        # Compute Kappa index
        kappa = cohen_kappa_score(y_test, y_pred)
        print(f"Kappa index: {kappa:.4f}")

        # Confusion matrix
        display_confusion_matrix_pink_variants(selector, X_test, y_test)
        
        # Feature importance (if available)
        if hasattr(selector, 'feature_importances_'):
            importances = selector.feature_importances_
            indices = np.argsort(importances)[::-1][:20]
            print("\nTop 20 important features:")
            feature_names = selected_feature_names_ga if method == 'GA' else selected_feature_names
            for rank, idx in enumerate(indices, 1):
                print(f"{rank}. {feature_names[idx]}: {importances[idx]:.4f}")
        else:
            print("\nThis model does not provide feature importances.")
        print("-" * 80)