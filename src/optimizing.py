import optuna
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from src.resampling import resample_data, resample_df, resample_sequence_data_with_gpt
from src.word2vec_embeddings import df_2_embeddings

def set_seed(seed):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def define_sampling_strategy(max_ratio, min_ratio, original_n_samples):
    sampling_strategy = {}
    major_class_count = max(original_n_samples)
    for label in range(9):
        if label == 5:
            objective_count = min(round(max_ratio * major_class_count), original_n_samples[label])
        else:
            objective_count = max(round(min_ratio * major_class_count), original_n_samples[label])
        sampling_strategy[label] = objective_count

    return sampling_strategy

def hyperparams_grid(
    clf,
    search_space: dict,
    fixed_params: dict,
    x: np.ndarray,
    y: np.ndarray,
    internal_validation=False,
    save_file: str = "hyperparams_optim",
    random_state: int = 42
) -> optuna.study:
    set_seed(random_state)

    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)

    def objective(trial: optuna.trial.Trial):
        params_model = {}
        params_train = {}

        for param_name, values in search_space.get("model", {}).items():
            params_model[param_name] = trial.suggest_categorical(f"model__{param_name}", values)

        for param_name, values in search_space.get("train", {}).items():
            params_train[param_name] = trial.suggest_categorical(f"train__{param_name}", values)

        full_model_params = {**fixed_params.get("model", {}), **params_model}
        full_train_params = {**fixed_params.get("train", {}), **params_train}

        f1_scores = []

        for fold, (train_index, val_index) in enumerate(kf.split(x, y)):
            x_train_fold = x[train_index]
            x_val_fold = x[val_index]
            y_train_fold = y[train_index]
            y_val_fold = y[val_index]

            model = clf(**full_model_params)

            if internal_validation:
                model.fit(x_train_fold, y_train_fold, x_val_fold, y_val_fold, **full_train_params)
            else:
                model.fit(x_train_fold, y_train_fold, **full_train_params)

            y_pred = model.predict(x_val_fold)
            f1_macro = f1_score(y_val_fold, y_pred, average="macro")
            f1_scores.append(f1_macro)

        return np.mean(f1_scores)

    all_space = {
        f"{group}__{k}": v for group in search_space for k, v in search_space[group].items()
    }
    combinations = np.prod([len(v) for v in all_space.values()])

    sampler = optuna.samplers.GridSampler(all_space)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=combinations, show_progress_bar=True)

    resultados = study.trials_dataframe()
    resultados.to_excel(save_file, index=False)

    return study


SEARCH_SPACE = {
    "max": [0.6, 0.7, 0.8, 0.9, 1],
    "min": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
}

def smote_grid(
    clf,
    fixed_params: dict,
    x: np.ndarray,
    y: np.ndarray,
    internal_validation=False,
    save_file: str = "smote_optim",
    random_state: int = 42
) -> optuna.study:
    set_seed(random_state)

    model_params = fixed_params.get("model", {})
    train_params = fixed_params.get("train", {})

    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)

    def objective(trial: optuna.trial.Trial):
        max_ratio = trial.suggest_categorical("max", SEARCH_SPACE["max"])
        min_ratio = trial.suggest_categorical("min", SEARCH_SPACE["min"])

        f1_scores = []

        for fold, (train_index, val_index) in enumerate(kf.split(x, y)):
            x_train_fold = x[train_index]
            x_val_fold = x[val_index]
            y_train_fold = y[train_index]
            y_val_fold = y[val_index]

            _, original_n_samples = np.unique(y_train_fold, return_counts=True)
            sampling_strategy = define_sampling_strategy(max_ratio, min_ratio, original_n_samples)
            x_train_fold, y_train_fold = resample_data(x_train_fold, y_train_fold, sampling_strategy, random_state)

            model = clf(**model_params)
            if internal_validation:
                model.fit(x_train_fold, y_train_fold, x_val_fold, y_val_fold, **train_params)
            else:
                model.fit(x_train_fold, y_train_fold, **train_params)

            y_pred = model.predict(x_val_fold)
            f1_macro = f1_score(y_val_fold, y_pred, average="macro")
            f1_scores.append(f1_macro)

        return np.mean(f1_scores)

    combinations = np.prod([len(v) for v in SEARCH_SPACE.values()])
    sampler = optuna.samplers.GridSampler(SEARCH_SPACE)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(objective, n_trials=combinations, show_progress_bar=True)

    resultados = study.trials_dataframe()
    resultados.to_excel(save_file, index=False)

    return study

LABEL_COLUMN = "my_label"
TEXT_COLUMN = "Title"

def gpt_grid(
    clf,
    fixed_params: dict,
    x_train_val: np.ndarray,
    y_train_val: np.ndarray,
    x_gpt: np.ndarray,
    y_gpt: np.ndarray,
    internal_validation=False,
    save_file: str = "gpt_optim",
    random_state: int = 42
) -> optuna.study:
    set_seed(random_state)
    
    model_params = fixed_params.get("model", {})
    train_params = fixed_params.get("train", {})
    
    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)

    def objective(trial: optuna.trial.Trial):
        max_ratio = trial.suggest_categorical("max", SEARCH_SPACE["max"])
        min_ratio = trial.suggest_categorical("min", SEARCH_SPACE["min"])

        f1_scores = []

        for fold, (train_index, val_index) in enumerate(kf.split(x_train_val, y_train_val)):
            x_train_fold = x_train_val[train_index]
            x_val_fold = x_train_val[val_index]
            y_train_fold = y_train_val[train_index]
            y_val_fold = y_train_val[val_index]

            _, original_n_samples = np.unique(y_train_fold, return_counts=True)
            sampling_strategy = define_sampling_strategy(max_ratio, min_ratio, original_n_samples)
            
            x_train_fold, y_train_fold = resample_sequence_data_with_gpt(
                x_train_fold, y_train_fold,
                x_gpt, y_gpt,
                sampling_strategy=sampling_strategy,
                seed=random_state
            )

            model = clf(**model_params)
            if internal_validation:
                model.fit(x_train_fold, y_train_fold, x_val_fold, y_val_fold, **train_params)
            else:
                model.fit(x_train_fold, y_train_fold, **train_params)

            y_pred = model.predict(x_val_fold)
            f1_macro = f1_score(y_val_fold, y_pred, average="macro")
            f1_scores.append(f1_macro)

        return np.mean(f1_scores)

    combinations = np.prod([len(v) for v in SEARCH_SPACE.values()])
    sampler = optuna.samplers.GridSampler(SEARCH_SPACE)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(objective, n_trials=combinations, show_progress_bar=True)

    resultados = study.trials_dataframe()
    resultados.to_excel(save_file, index=False)

    return study