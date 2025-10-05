from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

def XGBoostMultiClass(num_classes=3):
    # Configured for high-dimensional, 3-class classification
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        objective='multi:softprob',
        num_class=num_classes,
        max_depth=30,              # deeper trees for complex data
        learning_rate=0.05,       # lower learning rate for stability
        n_estimators=50,         # more trees for better performance
        subsample=0.3,            # prevent overfitting
        colsample_bytree=0.8,     # use half of features per tree
        tree_method='hist',       # faster for high-dimensional data 
        random_state=42,
        reg_lambda=1.0,          # L2 regularization to prevent overfitting
        reg_alpha=0.1,           # L1 regularization to enhance feature selection
        booster='gbtree',        # tree-based boosting
        colsample_bylevel=0.8,      # Feature sampling per level
        colsample_bynode=0.8,     # Feature sampling per node
        min_child_weight=1,    # Minimum sum of weights in child
        gamma=0,     # Minimum loss reduction for split       
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

def XGBoostBinary():
    # Configured for high-dimensional, binary classification
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        objective='binary:logistic',
        max_depth=30,
        learning_rate=0.05,
        n_estimators=50,
        subsample=0.3,
        colsample_bytree=0.8,
        tree_method='hist',
        random_state=42,
        reg_lambda=1.0,
        reg_alpha=0.1,
        booster='gbtree',
        colsample_bylevel=0.8,
        colsample_bynode=0.8,
        min_child_weight=1,
        gamma=0,
        max_delta_step=1,
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
        max_depth=30,       # Prevent overfitting
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt', # Only consider sqrt(n_features) at each split
        class_weight='balanced'
    )
    return model

def RandomForest100():
    model = RandomForestClassifier(n_estimators=100)
    return model

def RandomForest200():
    model = RandomForestClassifier(n_estimators=200)
    return model

def RandomForest300():
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    param_grid = {
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    return model, param_grid

def LightGBMMulticlass(num_classes=3):
    model = LGBMClassifier(objective='multiclass', num_class=num_classes, random_state=42)
    param_grid = {
        'max_depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.5, 0.8, 1.0],
        'num_leaves': [31, 63, 127]
    }
    return model, param_grid

def LightGBMBinary():
    model = LGBMClassifier(objective='binary', random_state=42)
    param_grid = {
        'max_depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.5, 0.8, 1.0],
        'num_leaves': [31, 63, 127]
    }
    return model, param_grid

def GradientBoosting():
    model = GradientBoostingClassifier()
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


def AdaBoostBinary():
    model = AdaBoostClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'algorithm': ['SAMME', 'SAMME.R']
    }
    return model, param_grid

def AdaBoostMultiClass():
    model = AdaBoostClassifier(random_state=42)
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
