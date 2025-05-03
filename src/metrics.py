import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def test_model(model, y_test, x_test, verbose=False):
    y_pred = model.predict(x_test)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0.0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0.0)
    print("F1 macro:", f1_macro)

    if verbose:
        print("F1 weighted:", f1_weighted)
        print(classification_report(y_test, y_pred, zero_division=0.0))
        print(confusion_matrix(y_test, y_pred))

