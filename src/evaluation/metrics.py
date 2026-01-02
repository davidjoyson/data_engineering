import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_classification(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }

def subgroup_metrics(y_true, y_pred, group_series, group_name):
    """Calculates fairness metrics for subgroups."""
    rows = []
    for g in sorted(group_series.dropna().unique()):
        mask = (group_series == g)
        if mask.sum() == 0: continue
        yt = y_true[mask]
        yp = y_pred[mask]
        rows.append({
            group_name: g,
            "n_samples": int(mask.sum()),
            "accuracy": accuracy_score(yt, yp),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
        })
    return pd.DataFrame(rows)