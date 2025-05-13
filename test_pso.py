import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from helpers.datasetHelper import get_samples, split_healthy_data
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml

import matplotlib.pyplot as plt

from MyPSO import MyPSO

def load_methylation_data(file_path):
    """Load DNA methylation data from a file"""
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Extract feature names, exclude first column (sample names) and last column (target)
    feature_names = np.array(data.columns[1:-1])
    
    # Extract data: X contains all columns except the first (sample names) and last (target)
    X = data.iloc[:, 1:-1].values
    
    # Extract target: last column
    y = data.iloc[:, -1].values
    
    return X, y, feature_names

def main(data_path, n_particles=30, max_iter=100, init_prob=0.3, min_features=5):
    """Main function to run the PSO feature selection with XGBoost"""
    # Load methylation data
    print("Loading DNA methylation data...")
    
    data_health = get_samples(data_path)

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
    

    # Load Iris dataset
    # Load a dataset with more features
    
    # Human Activity Recognition dataset with 561 features
    # print("Loading Human Activity Recognition dataset...")
    # har = fetch_openml('har', version=1, as_frame=True)
    # X = har.data.values
    # Y = har.target
    # feature_names = np.array(har.feature_names)

    df_cancer = pd.concat([healthy_cases, prebrca_cases, cancer_cases], ignore_index=True) #blood samples
    # The last column is the target classes
    # Ensure all data is numeric
    X = df_cancer.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
    # Replace NaN values with the minimum value for each column
    for column in X.columns:
        if X[column].isna().any():
            min_value = X[column].min()
            X[column].fillna(min_value, inplace=True)
    # Fill missing values with the lowest value of its cpg site
    # X = X.apply(lambda col: col.fillna(col.min()), axis=0)


    Y = df_cancer.iloc[:, -1]
    # Label encode the target variable
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    print(f"Label encoding mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    feature_names = {index: value for index, value in enumerate(data_health)}

    n_features = X.shape[1]
    print(f"Loaded dataset with {n_features} features and {len(Y)} samples")
    
    # Initialize PSO
    pso = MyPSO(n_particles=n_particles, 
              n_features=n_features, 
              max_iter=max_iter, 
              init_prob=init_prob,
              min_features=min_features)
    
    # Run optimization
    # Run optimization
    best_position, best_fitness = pso.optimize(X, Y)
    
    # Plot best position (selected features)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(best_position)), best_position)
    plt.title(f'Best Feature Selection\nTotal features: {int(sum(best_position))}')
    plt.xlabel('Feature Index')
    plt.ylabel('Selection Status (0/1)')
    
    # Plot best fitness
    plt.subplot(1, 2, 2)
    plt.bar(['Best Fitness'], [best_fitness])
    plt.title(f'Best Fitness: {best_fitness:.4f}')
    plt.tight_layout()
    plt.show()
    
    # Plot progress
    pso.plot_progress()
    
    # Evaluate final model
    pso.evaluate_final_model(X, Y, feature_names)
    
    return pso

if __name__ == "__main__":
    # Example usage
    directory_path = './datasets'
    data_path = os.path.join(directory_path, 'DT.Healthy.csv')

    pso = main(data_path)
    pass