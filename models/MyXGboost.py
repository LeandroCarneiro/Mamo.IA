from xgboost import Booster, XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier


def XGBoostMultiClass(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', objective='binary:logistic')
    model.fit(X_train, y_train)
    return model

def XGBoost(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', objective='binary:logistic')
    model.fit(X_train, y_train)
    return model

def RandomForest100(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def RandomForest200(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)
    return model

def RandomForest300(X_train, y_train):
    model = RandomForestClassifier(n_estimators=300)
    model.fit(X_train, y_train)
    return model

def LightGBM(X_train, y_train):
    model = LGBMClassifier(num_class = 1, objective='binary', metric='binary_logloss', boosting_type='gbdt', num_leaves=31, learning_rate=0.05, n_estimators=20)
    model.fit(X_train, y_train)
    return model

def LightGBMMulticlass(X_train, y_train):
    model = LGBMClassifier(num_class = 2)
    model.fit(X_train, y_train)
    return model

def AdaBoost(X_train, y_train):
    model = AdaBoostClassifier()
    model.fit(X_train, y_train)
    return model

def GradientBoosting(X_train, y_train):
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
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