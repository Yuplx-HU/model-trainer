import os
import joblib
from tqdm import tqdm

import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.base import clone
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import (
    make_scorer,
    root_mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
)


def create_estimator(task_type, model_type, params):
    model_map = {
        "regression": {
            "RF": RandomForestRegressor,
            "ERT": ExtraTreesRegressor, 
            "GBDT": GradientBoostingRegressor,
            "SVM": SVR
        },
        "classification": {
            "RF": RandomForestClassifier,
            "ERT": ExtraTreesClassifier,
            "GBDT": GradientBoostingClassifier, 
            "SVM": SVC
        }
    }
    
    if task_type not in model_map or model_type not in model_map[task_type]:
        raise ValueError(f"Unsupported task type '{task_type}' or model type '{model_type}'")
    
    model_class = model_map[task_type][model_type]
    
    if task_type == "classification" and model_type == "SVM":
        params = params.copy()
        params.setdefault("decision_function_shape", "ovr")
    
    return model_class(**params)


def create_searcher(estimator, params, search_type, n_splits, shuffle, scoring, task_type, n_iter, random_state, n_jobs):
    scoring_map = {
        "regression": {
            "rmse": make_scorer(lambda yt, yp: -root_mean_squared_error(yt, yp)),
            "r2": "r2"
        },
        "classification": {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, average='macro'),
            "recall": make_scorer(recall_score, average='macro'), 
            "f1": make_scorer(f1_score, average='macro')
        }
    }
    
    if task_type not in scoring_map or scoring not in scoring_map[task_type]:
        supported = list(scoring_map[task_type].keys())
        raise ValueError(f"Unsupported scoring '{scoring}' for task '{task_type}'. Use: {supported}")
    
    cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    common_params = {
        "estimator": estimator,
        "cv": cv,
        "scoring": scoring_map[task_type][scoring],
        "n_jobs": n_jobs,
        "verbose": 0
    }
    
    if search_type == "grid":
        return GridSearchCV(param_grid=params, **common_params)
    elif search_type == "random":
        return RandomizedSearchCV(param_distributions=params, n_iter=n_iter, random_state=random_state, **common_params)
    else:
        raise ValueError(f"Unsupported search type '{search_type}'. Use: grid/random")


def train_model(searcher, X_train, y_train, X_test, y_test, task_type):
    searcher_clone = clone(searcher)
    searcher_clone.fit(X_train, y_train)
    y_pred = searcher_clone.predict(X_test)
    
    metric_funcs = {
        "regression": {
            "rmse": lambda yt, yp: round(root_mean_squared_error(yt, yp), 4),
            "r2": lambda yt, yp: round(r2_score(yt, yp), 4)
        },
        "classification": {
            "accuracy": lambda yt, yp: round(accuracy_score(yt, yp), 4),
            "precision": lambda yt, yp: round(precision_score(yt, yp, average='macro'), 4),
            "recall": lambda yt, yp: round(recall_score(yt, yp, average='macro'), 4),
            "f1": lambda yt, yp: round(f1_score(yt, yp, average='macro'), 4)
        }
    }
    
    metrics = {name: func(y_test, y_pred) for name, func in metric_funcs[task_type].items()}
    
    return {
        "best_estimator": searcher_clone.best_estimator_,
        "best_params": searcher_clone.best_params_,
        "metrics": metrics,
    }


def _get_best_model(models_result, task_type, scoring):
    if task_type == "regression":
        if scoring == "rmse":
            best_model = min(models_result.items(), key=lambda x: x[1]["metrics"]["rmse"])
            return best_model[0], True, "rmse"
        else:
            best_model = max(models_result.items(), key=lambda x: x[1]["metrics"]["r2"])
            return best_model[0], False, "r2"
    else:
        best_model = max(models_result.items(), key=lambda x: x[1]["metrics"].get(scoring, float('-inf')))
        return best_model[0], False, scoring


def _calculate_avg_metrics(best_models_metrics, save_path):
    if not best_models_metrics:
        return
    
    avg_metrics = {
        metric: round(sum(m[metric] for m in best_models_metrics) / len(best_models_metrics), 4)
        for metric in best_models_metrics[0]
    }
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pd.DataFrame([avg_metrics]).to_csv(save_path, index=False)
    
    return avg_metrics


def machine_learning(tasks, config):
    task_type = config["type"]
    models_cfg = config["models"]
    verbose = config["verbose"]
    random_state = config["random_state"]
    scoring = config["scoring"]
    
    best_models_metrics = []

    for task in tqdm(tasks, disable=not verbose, desc="Process tasks"):
        task_name = task["name"]
        models_result = {}
        all_models_params = []
        
        for model_name, model_cfg in tqdm(models_cfg.items(), disable=not verbose, desc=f"Train model [{task_name}]", leave=False):
            estimator_cfg = model_cfg["estimator"]
            estimator = create_estimator(task_type, estimator_cfg["type"], estimator_cfg["params"])
            
            searcher_cfg = model_cfg["searcher"]
            searcher = create_searcher(
                estimator=estimator,
                params=searcher_cfg["params"],
                search_type=searcher_cfg["type"],
                n_splits=searcher_cfg["n_splits"],
                shuffle=searcher_cfg["shuffle"],
                scoring=scoring,
                task_type=task_type,
                n_iter=searcher_cfg["n_iter"],
                random_state=random_state,
                n_jobs=searcher_cfg["n_jobs"]
            )
            
            result = train_model(searcher, task["X_train"], task["y_train"], task["X_test"], task["y_test"], task_type)
            models_result[model_name] = result
            
            all_models_params.append({
                "task_name": task_name,
                "model_name": model_name,
                "best_params": result["best_params"],
                "metrics": result["metrics"],
            })
        
        best_model_name, sort_ascending, sort_key = _get_best_model(models_result, task_type, scoring)
        best_model_result = models_result[best_model_name]
        best_models_metrics.append(best_model_result["metrics"])
        
        save_models_dir = config.get("save_models_dir")
        if save_models_dir:
            os.makedirs(save_models_dir, exist_ok=True)
            joblib.dump(best_model_result["best_estimator"], os.path.join(save_models_dir, f"{task_name}_{best_model_name}.pkl"))
        
            params_df = pd.DataFrame(all_models_params)
            params_df['_temp_sort'] = params_df['metrics'].apply(lambda x: x[sort_key])
            params_df = params_df.sort_values('_temp_sort', ascending=sort_ascending).drop('_temp_sort', axis=1)
            params_df.to_csv(os.path.join(save_models_dir, f"{task_name}_all_model_params.csv"), index=False)
    
    save_avg_results_dir = config.get("save_avg_results_dir")
    if save_avg_results_dir:
        _calculate_avg_metrics(best_models_metrics, save_avg_results_dir)
