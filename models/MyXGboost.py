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
        random_state=42,
        booster='gbtree'
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
        max_depth=15,       # Prevent overfitting
        min_samples_split=2,
        min_samples_leaf=10,
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
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    return model, param_grid

def LightGBMMulticlass():
    model = LGBMClassifier(objective='multiclass', num_class=3, random_state=42)
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

from sklearn.metrics import cohen_kappa_score, make_scorer

def get_best(model, param_grid, X_train, y_train, cv=10, scoring=None):
    if scoring is None:
        scoring = make_scorer(cohen_kappa_score)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best parameters found: ", grid_search.best_params_)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
