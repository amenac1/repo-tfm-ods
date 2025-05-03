from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from src.metrics import test_model
import numpy as np
def train_and_test_svm(X_train, y_train, X_test, y_test, print_result=False, random_state=42):
    svm = SVC(kernel='rbf', decision_function_shape='ovr', random_state=random_state)  
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy, f1_macro = test_model(y_test, y_pred, print_result=print_result)
    return accuracy, f1_macro


def train_and_test_rf(X_train, y_train, X_test, y_test, print_result=False, random_state=42):
    random_forest = RandomForestClassifier(n_estimators=100, random_state=random_state) 
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    accuracy, f1_macro = test_model(y_test, y_pred, print_result=print_result)
    return accuracy, f1_macro


def train_and_test_xgboost(X_train, y_train, X_test, y_test, print_result=False, random_state=42):
    num_classes = len(np.unique(y_train))
    xgboost_model = xgb.XGBClassifier(objective='multi:softprob', num_class=num_classes, random_state=random_state, eval_metric="mlogloss")
    xgboost_model.fit(X_train, y_train)
    y_pred = xgboost_model.predict(X_test)
    accuracy, f1_macro = test_model(y_test, y_pred, print_result=print_result)
    return accuracy, f1_macro
