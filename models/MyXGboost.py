from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

def XGBoostMultiClass():
    model = XGBClassifier(
        use_label_encoder=True,
        eval_metric='mlogloss',
        objective='multi:logistic',
        sampling_method='uniform', 
        max_depth=5,              # deeper trees for complex data
        learning_rate=0.1,       # lower learning rate for stability
        n_estimators=1000,         # more trees for better performance
        random_state=42,  
        max_delta_step=1,               # Help with class imbalance
    )

    param_grid = {
        'max_depth': [3, 4, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
        'subsample': [0.3, 0.6, 0.8, 1.0],
        'colsample_bytree': [0.5, 0.8, 1.0],
        'colsample_bylevel': [0.5, 0.8, 1.0],
        'colsample_bynode': [0.5, 0.8, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0],
        'reg_alpha': [0, 0.1, 0.5],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.5],
        'tree_method': ['hist']
    }
    return model, param_grid

def DecisionTreeMultiClass():
    
    # Configured for high dimensionality problems
    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=5,       # Prevent overfitting
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt', # Only consider sqrt(n_features) at each split
        class_weight='balanced'
    )
    return model

def RandomForest300():
    model = RandomForestClassifier(
        n_estimators=1000,
        random_state=42)
    param_grid = {
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    return model, param_grid

def LightGBMMulticlass():
    model = LGBMClassifier(
        objective='multiclass',
        random_state=42,
        verbosity=-1, 
        learning_rate=0.1,  # Lower learning rate for better accuracy
        num_leaves=1000,  # Higher complexity for high-dimensional data
        max_depth=-1,  # No limit on depth for flexibility
        min_data_in_leaf=20,  # Prevent overfitting
        device='cpu',  # Utilize GPU for faster training
        force_col_wise=True
    )
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [63, 127, 255],
        'max_depth': [10, 15, 20],
        'min_data_in_leaf': [10, 20, 30],
        'lambda_l1': [0, 0.1, 0.5],
        'lambda_l2': [0, 0.2, 0.5],
        'min_gain_to_split': [0.01, 0.1, 0.5],
        'feature_fraction': [0.6, 0.8, 1.0],
        'bagging_fraction': [0.6, 0.8, 1.0],
        'bagging_freq': [1, 5, 10]
    }
    return model, param_grid

def GradientBoosting():
    model = GradientBoostingClassifier(
        random_state=42,
        learning_rate=0.1,
        n_estimators=1000
    )
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    return model, param_grid

def AdaBoostMultiClass():
    model = AdaBoostClassifier(
        random_state=42,
        n_estimators=1000,
        learning_rate=0.1
        )
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'algorithm': ['SAMME', 'SAMME.R']
    }
    return model, param_grid

from sklearn.metrics import cohen_kappa_score, make_scorer

def get_best(model, X_train, y_train, cv=5, scoring=None):
    if scoring is None:
        scoring = make_scorer(cohen_kappa_score)

    grid_search = GridSearchCV(estimator=model[0], param_grid=model[1], cv=cv, scoring=scoring, n_jobs=4,pre_dispatch='2*n_jobs', verbose=2, return_train_score=True)
    grid_search.fit(X_train, y_train)
    print("Best parameters found: ", grid_search.best_params_)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
