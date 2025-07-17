from xgboost import Booster, XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier

from sklearn_genetic import GAFeatureSelectionCV
from sklearn_genetic.plots import plot_fitness_evolution
from sklearn.tree import DecisionTreeClassifier

def XGBoostMultiClass():
    # Configured for high-dimensional, 3-class classification
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        objective='multi:softprob',
        num_class=3,
        max_depth=4,              # deeper trees for complex data
        learning_rate=0.05,       # lower learning rate for stability
        n_estimators=50,         # more trees for better performance
        subsample=0.3,            # prevent overfitting
        colsample_bytree=0.8,     # use half of features per tree
        tree_method='hist',       # faster for high-dimensional data 
        random_state=42,
        reg_lambda=1.0,          # L2 regularization to prevent overfitting
        reg_alpha=0.1,           # L1 regularization to enhance feature selection
        booster='gbtree',        # tree-based boosting
        early_stopping_rounds=10,  # early stopping to prevent overfitting
        colsample_bylevel=0.8,      # Feature sampling per level
        colsample_bynode=0.8,     # Feature sampling per node
        min_child_weight=1,    # Minimum sum of weights in child
        gamma=0                    # Minimum loss reduction for split
    )
    # Note: thresholding for multiclass is handled at prediction time, not in the model
    return model

def DecisionTreeMultiClass():
    
    # Configured for high dimensionality problems
    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=15,       # Prevent overfitting
        min_samples_split=2,
        min_samples_leaf=10,
        max_features='sqrt', # Only consider sqrt(n_features) at each split
        class_weight='balanced'
    )
    return model

def XGBoost():
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', objective='binary:logistic')
    return model

def RandomForest100():
    model = RandomForestClassifier(n_estimators=100)
    return model

def RandomForest200():
    model = RandomForestClassifier(n_estimators=200)
    return model

def RandomForest300():
    model = RandomForestClassifier(n_estimators=300)
    return model

def LightGBM():
    model = LGBMClassifier(num_class = 1, objective='binary', metric='binary_logloss', boosting_type='gbdt', num_leaves=31, learning_rate=0.05, n_estimators=20)
    return model

def LightGBMMulticlass():
    # Configured for high-dimensional, 3-class classification
    model = LGBMClassifier(
        objective='multiclass',
        num_class=3,
        max_depth=8,              # deeper trees for complex data
        learning_rate=0.05,       # lower learning rate for stability
        n_estimators=100,         # more trees for better performance
        subsample=0.8,            # prevent overfitting
        colsample_bytree=0.5,     # use half of features per tree
        random_state=42
    )
    return model

def AdaBoostMultiClass():
    # Configured for high-dimensional, 3-class classification
    # Uses SAMME for multiclass, base_estimator with max_depth to handle complexity
    model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=4),  # base estimator for AdaBoost        
        n_estimators=200,           # more estimators for stability
        learning_rate=0.05,         # lower learning rate for stability
        algorithm='SAMME',          # multiclass support
        random_state=42
    )
    return model

def GradientBoosting():
    model = GradientBoostingClassifier()
    return model

def get_best(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
    }
    
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', objective='multi:softprob', booster='gbtree', enable_categorical=True)
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_