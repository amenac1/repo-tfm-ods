#Optimizar NN
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.svm import SVC
import sys
import os
sys.path.append("C:/Users/USER/Documents/TFM Arturo Menac/tfm-final")
from src.keras_nn.ann_keras import KerasNN
from src.preprocessing.array_mapping import map_labels
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.keras_nn.ann_keras import KerasNN
from IPython.display import clear_output


optuna.logging.set_verbosity(optuna.logging.WARNING)

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

def optimize_svm_parameters(x_train, y_train, n_iter):

    def objective(trial):
        # Espacio de búsqueda (Bayesiano)
        c = trial.suggest_float('C', 1e-5, 1e4, log=True)
        gamma = trial.suggest_float('gamma', 1e-5, 1e1, log=True)

        f1_scores = []
        
        for train_index, val_index in skf.split(x_train, y_train):
            x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
            
            # Modelo SVM
            svm = SVC(
                kernel='rbf', 
                C=c, 
                gamma=gamma, 
                class_weight='balanced',
                decision_function_shape='ovr'
            )
            svm.fit(x_train_fold, y_train_fold)
            y_val_pred = svm.predict(x_val_fold)

            # Evaluar con F1-macro
            f1_macro = f1_score(y_val_fold, y_val_pred, average='macro')
            f1_scores.append(f1_macro)
            
        
        mean_f1 = np.mean(f1_scores)

        trial.report(mean_f1, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return mean_f1

    # Se usa busqueda bayesiana y pruner para acelerar la busqueda
    sampler = optuna.samplers.TPESampler() 
    pruner = optuna.pruners.HyperbandPruner() # Poda
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_iter, n_jobs=-1, show_progress_bar=True)

    best_parameters = study.best_params
    print("Mejor conjunto de parámetros:")
    print(best_parameters)
    print("Mejor f1 macro")
    print(study.best_value)

    return study

def optimize_rf_parameters(x_train, y_train, n_iter):
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 250, step=25)
        max_depth = trial.suggest_int('max_depth', 3, 50)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 30)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

        f1_scores = []
        
        for train_index, val_index in skf.split(x_train, y_train):
            x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
            # Modelo Random Forest
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                criterion=criterion,
                class_weight='balanced',
                n_jobs=-1
            )

            rf.fit(x_train_fold, y_train_fold)
            y_val_pred = rf.predict(x_val_fold)

            # Evaluar con F1-macro
            f1_macro = f1_score(y_val_fold, y_val_pred, average='macro')
            f1_scores.append(f1_macro)
        
        mean_f1 = np.mean(f1_scores)

        trial.report(mean_f1, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return mean_f1

    # Configurar búsqueda bayesiana y pruner
    sampler = optuna.samplers.TPESampler()  # Búsqueda bayesiana
    pruner = optuna.pruners.HyperbandPruner() # Poda
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_iter, n_jobs=-1, show_progress_bar=True)

    best_parameters = study.best_params
    print("Mejor conjunto de parámetros:")
    print(best_parameters)
    print("Mejor f1 macro")
    print(study.best_value)
    
    return study

# ignorados xgboost
# subsample = trial.suggest_float('subsample', 0.5, 1.0)
# colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
# min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
# gamma = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
def optimize_xgboost_parameters(x_train, y_train_mapped, n_iter):
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 50)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1, log=True)
        reg_lambda = trial.suggest_float('reg_lambda', 1e-5, 10, log=True)
        reg_alpha = trial.suggest_float('reg_alpha', 1e-5, 10, log=True)

        f1_scores = []
        
        for train_index, val_index in skf.split(x_train, y_train_mapped):
            x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
            y_train_fold, y_val_fold = y_train_mapped[train_index], y_train_mapped[val_index]
            
            xgb = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
                eval_metric='logloss',
                n_jobs=16
            )


            xgb.fit(x_train_fold, y_train_fold)
            y_val_pred = xgb.predict(x_val_fold)

            ...
            # Evaluar con F1-macro
            f1_macro = f1_score(y_val_fold, y_val_pred, average='macro')
            f1_scores.append(f1_macro)
        
        mean_f1 = np.mean(f1_scores)

        trial.report(mean_f1, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return mean_f1

    # Configurar búsqueda bayesiana y pruner
    sampler = optuna.samplers.TPESampler()  # Búsqueda bayesiana
    pruner = optuna.pruners.HyperbandPruner() # Poda

    # Crear el estudio de Optuna
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_iter, n_jobs=-1, show_progress_bar=True) # 1 solo hilo, porque XGBoost usa todos

    best_parameters = study.best_params
    print("Mejor conjunto de parámetros:")
    print(best_parameters)
    print("Mejor f1 macro")
    print(study.best_value)

    # Devolver el estudio ya realizado
    return study

def optimize_nn_parameters(x_train, y_train_mapped, n_iter):
    input_shape = x_train.shape[1]
    num_classes = len(np.unique(y_train_mapped))

    def objective(trial):
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256, 512, 1024])
        num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 7)
        dropout = trial.suggest_float('dropout', 0.1, 0.7)
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        epochs = trial.suggest_int('epochs', 3, 20)

        f1_scores = []
        
        for train_index, val_index in skf.split(x_train, y_train_mapped):
            x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
            y_train_fold, y_val_fold = y_train_mapped[train_index], y_train_mapped[val_index]
            
            # Crear el modelo con los hiperparámetros sugeridos
            nn = KerasNN(
                input_shape=input_shape,
                num_classes=num_classes,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                dropout=dropout,
                lr=lr,
                epochs=epochs,
                batch_size=128
            )
            # Entrenar el modelo
            _ = nn.fit(x_train_fold, y_train_fold, x_val_fold, y_val_fold, verbose=0)
            # Predecir sobre el conjunto de validación
            y_val_pred = nn.predict(x_val_fold)
            f1_macro = f1_score(y_val_fold, y_val_pred, average='macro')
            f1_scores.append(f1_macro)

        mean_f1 = np.mean(f1_scores)

        trial.report(mean_f1, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return mean_f1

    # Configurar búsqueda bayesiana y pruner
    sampler = optuna.samplers.TPESampler()  
    pruner = optuna.pruners.HyperbandPruner() 

    # Crear el estudio de Optuna
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_iter, n_jobs=-1, show_progress_bar=True)

    clear_output()

    best_parameters = study.best_params
    print("Mejor conjunto de parámetros:")
    print(best_parameters)
    print("Mejor f1 macro")
    print(study.best_value)

    return study

from src.keras_nn.cnn_keras import train_cnn, predict_cnn
def optimize_cnn_parameters(x_train, y_train, x_val, y_val, n_iter):
    def objective(trial):
        num_filters = trial.suggest_categorical('num_filters', [16, 32, 64, 128])
        dense_size = trial.suggest_categorical('dense_size', [32, 64, 128])
        lr = trial.suggest_float('lr', 0.0005, 0.01, log=True)
        dropout = trial.suggest_float('dropout', 0, 0.3)

        cnn, _ = train_cnn(
            x_train,
            y_train,
            x_train,
            y_train,
            epochs=50,
            batch_size=64,
            num_filters=num_filters,
            kernel_size=3,
            dense_size=dense_size,
            dropout=dropout,
            lr=lr
            )
        
        y_val_predict = predict_cnn(cnn, x_val)
        f1 = f1_score(y_val, y_val_predict, average='macro')

        trial.report(f1, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return f1
    
    # Configurar búsqueda bayesiana y pruner
    sampler = optuna.samplers.TPESampler()  
    pruner = optuna.pruners.HyperbandPruner() 

    # Crear el estudio de Optuna
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_iter, n_jobs=-1, show_progress_bar=True)

    clear_output()

    
    best_parameters = study.best_params
    print("Mejor conjunto de parámetros:")
    print(best_parameters)
    print("Mejor f1 macro")
    print(study.best_value)

    return study

from src.keras_nn.gru_keras import train_gru, predict_gru
def optimize_gru_parameters(x_train, y_train, x_val, y_val, n_iter):
    def objective(trial):
        gru_size = trial.suggest_categorical('gru_size', [16, 32, 64, 128])
        dense_size = trial.suggest_categorical('dense_size', [32, 64, 128])
        lr = trial.suggest_float('lr', 0.0005, 0.01, log=True)
        dropout = trial.suggest_float('dropout', 0, 0.3)

        from sklearn.model_selection import train_test_split

        # Divide el 80% para entrenamiento y el 20% para validación
        x_train_train, x_train_val, y_train_train, y_train_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)


        gru, _ = train_gru(x_train_train,
                        y_train_train,
                        x_train_val,
                        y_train_val,
                        epochs=50,
                        batch_size=64,
                        gru_size=gru_size,
                        dense_size=dense_size,
                        dropout=dropout,
                        lr=lr
        )
        
        y_val_predict = predict_gru(gru, x_val)
        f1 = f1_score(y_val, y_val_predict, average='macro')

        trial.report(f1, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return f1
    
    # Configurar búsqueda bayesiana y pruner
    sampler = optuna.samplers.TPESampler()  
    pruner = optuna.pruners.HyperbandPruner() 

    # Crear el estudio de Optuna
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_iter, n_jobs=-1, show_progress_bar=True)

    clear_output()

    
    best_parameters = study.best_params
    print("Mejor conjunto de parámetros:")
    print(best_parameters)
    print("Mejor f1 macro")
    print(study.best_value)

    return study


if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split
    # Generar X aleatoria de dimensiones (100, 50, 300)
    X = np.random.rand(100, 50, 300)
    # Generar Y aleatoria de tamaño (100) con valores entre 0 y 1 (por ejemplo, para clasificación binaria)
    y = np.random.randint(0, 8, size=(100,))

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    optimize_gru_parameters(x_train, y_train, x_val, y_val, 10)




        
        




