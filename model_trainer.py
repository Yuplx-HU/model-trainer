import random
import numpy as np
from sklearn.base import clone
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    VotingRegressor, VotingClassifier,
    StackingRegressor, StackingClassifier,
    AdaBoostRegressor, AdaBoostClassifier
)
from sklearn.metrics import (
    make_scorer, root_mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from scipy.stats import loguniform


def create_estimator(task_type: str, model_type: str, **fixed_model_params):
    task_type_model_map = {
        "regression": {
            "rf": RandomForestRegressor,
            "ert": ExtraTreesRegressor, 
            "gbdt": GradientBoostingRegressor,
            "svm": SVR,
            "vote": VotingRegressor,
            "adaboost": AdaBoostRegressor,
            "stack": StackingRegressor,
        },
        "classification": {
            "rf": RandomForestClassifier,
            "ert": ExtraTreesClassifier,
            "gbdt": GradientBoostingClassifier, 
            "svm": SVC,
            "vote": VotingClassifier,
            "adaboost": AdaBoostClassifier,
            "stack": StackingClassifier,
        }
    }
    if task_type not in task_type_model_map or model_type not in task_type_model_map[task_type]:
        raise ValueError(f"Unsupported task type '{task_type}' or model type '{model_type}'")
    return task_type_model_map[task_type][model_type](**fixed_model_params)


def create_scorer(task_type: str, scorer_type: str):
    scorer_map = {
        "regression": {
            "rmse": make_scorer(lambda yt, yp: -root_mean_squared_error(yt, yp)),
            "r2": "r2"
        },
        "classification": {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, average='weighted', zero_division=0),
            "recall": make_scorer(recall_score, average='weighted', zero_division=0), 
            "f1": make_scorer(f1_score, average='weighted', zero_division=0)
        }
    }
    if task_type not in scorer_map or scorer_type not in scorer_map[task_type]:
        raise ValueError(f"Unsupported scorer '{scorer_type}' for task '{task_type}'")
    return scorer_map[task_type][scorer_type]


def create_cv(task_type: str, random_state: int = None):
    if random_state is None:
        random_state = random.randint(0, 4294967296)

    if task_type == "classification":
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    else:
        return KFold(n_splits=5, shuffle=True, random_state=random_state)


def create_searcher(
    search_type: str,
    estimator,
    search_model_params: dict = {},
    searcher_params: dict = {}
):
    if search_type == "grid":
        return GridSearchCV(param_grid=search_model_params, estimator=estimator, **searcher_params)
    elif search_type == "random":
        return RandomizedSearchCV(param_distributions=search_model_params, estimator=estimator, **searcher_params)
    else:
        raise ValueError(f"Unsupported search type '{search_type}'. Use: grid/random")


def train_model(task_type: str, searcher, X_train, y_train, X_test, y_test):
    metric_funcs = {
        "regression": {
            "rmse": lambda yt, yp: round(root_mean_squared_error(yt, yp), 4),
            "r2": lambda yt, yp: round(r2_score(yt, yp), 4)
        },
        "classification": {
            "accuracy": lambda yt, yp: round(accuracy_score(yt, yp), 4),
            "precision": lambda yt, yp: round(precision_score(yt, yp, average='macro', zero_division=0), 4),
            "recall": lambda yt, yp: round(recall_score(yt, yp, average='macro', zero_division=0), 4),
            "f1": lambda yt, yp: round(f1_score(yt, yp, average='macro', zero_division=0), 4)
        }
    }
    
    searcher_clone = clone(searcher)
    searcher_clone.fit(X_train, y_train)
    y_pred = searcher_clone.predict(X_test)
    
    return {
        "best_estimator": searcher_clone.best_estimator_,
        "best_parameters": searcher_clone.best_params_,
        "prediction": y_pred,
        "metrics": {name: func(y_test, y_pred) for name, func in metric_funcs[task_type].items()}
    }


def rf_parameters(task_type: str, search_type: str, random_state: int = None):
    if random_state is None:
        random_state = random.randint(0, 4294967296)
    
    fixed_model_params = {
        'random_state': random_state,
        'bootstrap': True
    }
    
    search_model_params = {
        'n_estimators': np.arange(50, 1001, 50).tolist(),
        'max_depth': [None] + np.arange(5, 31, 5).tolist(),
        'min_samples_split': np.concatenate([np.arange(2, 21, 1), [30]]).tolist(),
        'min_samples_leaf': np.concatenate([np.arange(1, 11, 1), [20]]).tolist(),
        'max_features': ['sqrt', 'log2', None] + np.linspace(0.3, 0.9, 7).tolist(),
        'max_samples': [None] + np.linspace(0.6, 1.0, 5).tolist()
    }
    
    searcher_params = {
        "cv": create_cv(task_type, random_state),
        "scoring": create_scorer(task_type, "f1" if task_type == "classification" else "rmse"),
        "n_jobs": -1,
        "verbose": 3,
        'n_iter': 75
    }
    if search_type == "random":
        searcher_params['random_state'] = random_state
    
    return fixed_model_params, search_model_params, searcher_params


def ert_parameters(task_type: str, search_type: str, random_state: int = None):
    if random_state is None:
        random_state = random.randint(0, 4294967296)
    
    fixed_model_params = {
        'random_state': random_state,
        'bootstrap': True
    }
    
    search_model_params = {
        'n_estimators': np.arange(50, 1001, 50).tolist(),
        'max_depth': [None] + np.arange(5, 31, 5).tolist(),
        'min_samples_split': np.concatenate([np.arange(2, 16, 1), [20]]).tolist(),
        'min_samples_leaf': np.concatenate([np.arange(1, 11, 1), [20]]).tolist(),
        'max_features': ['sqrt', 'log2', None] + np.linspace(0.2, 0.9, 7).tolist(),
        'max_samples': [None] + np.linspace(0.6, 1.0, 5).tolist()
    }
    
    searcher_params = {
        "cv": create_cv(task_type, random_state),
        "scoring": create_scorer(task_type, "f1" if task_type == "classification" else "rmse"),
        "n_jobs": -1,
        "verbose": 3,
        'n_iter': 75
    }
    if search_type == "random":
        searcher_params['random_state'] = random_state
    
    return fixed_model_params, search_model_params, searcher_params


def gbdt_parameters(task_type: str, search_type: str, random_state: int = None):
    if random_state is None:
        random_state = random.randint(0, 4294967296)
    
    fixed_model_params = {
        'random_state': random_state
    }
    
    search_model_params = {
        'n_estimators': np.arange(100, 1001, 50).tolist(),
        'learning_rate': loguniform(1e-3, 0.1).rvs(7, random_state=random_state).tolist(),
        'max_depth': np.arange(2, 9, 1).tolist(),
        'min_samples_split': np.arange(2, 21, 1).tolist(),
        'min_samples_leaf': np.arange(1, 11, 1).tolist(),
        'subsample': np.linspace(0.5, 1.0, 6).tolist(),
        'max_features': ['sqrt', None] + np.linspace(0.3, 0.9, 7).tolist()
    }
    
    searcher_params = {
        "cv": create_cv(task_type, random_state),
        "scoring": create_scorer(task_type, "f1" if task_type == "classification" else "rmse"),
        "n_jobs": -1,
        "verbose": 3,
        'n_iter': 100
    }
    if search_type == "random":
        searcher_params['random_state'] = random_state
    
    return fixed_model_params, search_model_params, searcher_params


def svm_parameters(task_type: str, search_type: str, random_state: int = None):
    if random_state is None:
        random_state = random.randint(0, 4294967296)
    
    fixed_model_params = {
        'max_iter': 10000,
        'cache_size': 800,
        'decision_function_shape': 'ovr',
        'random_state': random_state
    }
    
    search_model_params = {
        'C': loguniform(1e-4, 1e4).rvs(10, random_state=random_state).tolist(),
        'gamma': ['scale', 'auto'] + loguniform(1e-5, 1e1).rvs(6, random_state=random_state).tolist(),
        'kernel': ['linear', 'rbf', 'poly'],
        'degree': np.arange(2, 6, 1).tolist(),
        'coef0': np.linspace(-1.0, 2.0, 7).tolist()
    }
    if task_type == "classification":
        search_model_params['class_weight'] = [None, 'balanced']

    searcher_params = {
        "cv": create_cv(task_type, random_state),
        "scoring": create_scorer(task_type, "f1" if task_type == "classification" else "rmse"),
        "n_jobs": -1,
        "verbose": 3,
        'n_iter': 150
    }
    if search_type == "random":
        searcher_params['random_state'] = random_state
    
    return fixed_model_params, search_model_params, searcher_params


def stack_parameters(task_type: str, search_type: str, random_state: int = None):
    if random_state is None:
        random_state = random.randint(0, 4294967296)
    
    is_classifier = task_type == "classification"

    fixed_model_params = {}
    
    search_model_params = {
        'passthrough': [False, True]
    }
    search_model_params = {k: v for k, v in search_model_params.items() if v}

    searcher_params = {
        "cv": create_cv(task_type, random_state),
        "scoring": create_scorer(task_type, "f1" if is_classifier else "rmse"),
        "n_jobs": -1,
        "verbose": 3,
        'n_iter': 100
    }
    if search_type == "random":
        searcher_params['random_state'] = random_state
    
    return fixed_model_params, search_model_params, searcher_params
