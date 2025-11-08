
# Define the parameter grid for GridSearchCV
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def get_best_knn(X_train, y_train):

    param_grid = {'n_neighbors': np.arange(1, 31)}

    # Initialize the KNeighborsClassifier
    knn = KNeighborsClassifier()

    # Initialize GridSearchCV with the KNeighborsClassifier and parameter grid
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best estimator
    best_params = grid_search.best_params_
    best_knn_model = grid_search.best_estimator_

    # print(f"Best n_neighbors: {best_params['n_neighbors']}")
    score = grid_search.best_score_
    return best_knn_model, best_params, score


def get_best_rf(X_train, y_train):

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 6, 8, 10, 12],
        'criterion': ['gini', 'entropy']
    }

    # Initialize the RandomForestClassifier
    rf = RandomForestClassifier()

    # Initialize GridSearchCV with the RandomForestClassifier and parameter grid
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')

    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best estimator
    best_params = grid_search.best_params_
    best_rf_model = grid_search.best_estimator_

    # print(f"Best parameters: {best_params}")
    score = grid_search.best_score_
    return best_rf_model, best_params, score


def get_best_logistic_regression(X_train, y_train):

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

    # Initialize the LogisticRegression
    lr = LogisticRegression()

    # Initialize GridSearchCV with the LogisticRegression and parameter grid
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy')

    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best estimator
    best_params = grid_search.best_params_
    best_lr_model = grid_search.best_estimator_

    # print(f"Best parameters: {best_params}")
    score = grid_search.best_score_
    return best_lr_model, best_params, score


def get_best_svm(X_train, y_train):

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }

    # Initialize the SVC
    svm = SVC(probability=True)

    # Initialize GridSearchCV with the SVC and parameter grid
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')

    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best estimator
    best_params = grid_search.best_params_
    best_svm_model = grid_search.best_estimator_

    # print(f"Best parameters: {best_params}")
    score = grid_search.best_score_
    return best_svm_model, best_params, score


def get_best_decision_tree(X_train, y_train):
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize the DecisionTreeClassifier
    dt = DecisionTreeClassifier()

    # Initialize GridSearchCV with the DecisionTreeClassifier and parameter grid
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')

    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best estimator
    best_params = grid_search.best_params_
    best_dt_model = grid_search.best_estimator_

    # print(f"Best parameters: {best_params}")
    score = grid_search.best_score_
    return best_dt_model, best_params, score

def get_best_xgboost(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 1, 5],
        'reg_alpha': [0, 0.1, 1]
    }

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)

    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_xgb_model = grid_search.best_estimator_
    score = grid_search.best_score_
    print("Best XGBoost parameters:", best_params)
    print(f"Best XGBoost CV score: {score:.4f}")
    return best_xgb_model, best_params, score