import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from src.config import MODELS_DIR
from src.feature_engineering.features import get_preprocessor, get_academic_preprocessor

def train_models(abt_df):
    """Trains LR, RF, and Regression models."""
    print("--- Starting Model Training ---")


    # Setup Classification Data
    X = abt_df.drop(columns=["G3", "target_pass"])
    y = abt_df["target_pass"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_academic = abt_df[['G1', 'G2']]
    y_academic = abt_df['target_pass']

    X_academic_train, X_academic_test, y_academic_train, y_academic_test = train_test_split(
        X_academic, y_academic, test_size=0.2, random_state=42, stratify=y_academic
    )

    preprocessor = get_preprocessor(X_train)
    academic_preprocessor = get_academic_preprocessor(X_academic_train)

    # 1. Logistic Regression
    lr_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])
    lr_pipeline.fit(X_train, y_train)

    log_reg_academic_clf = Pipeline([
    ("preprocess", academic_preprocessor),
    ("model", LogisticRegression(max_iter=1000))
    ])
    # Fit academic-only classifiers on the academic train split
    log_reg_academic_clf.fit(X_academic_train, y_academic_train)
    
    # 2. Random Forest
    rf_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
    ])
    rf_pipeline.fit(X_train, y_train)

    rf_academic_clf = Pipeline([
    ("preprocess", academic_preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
    ])
    rf_academic_clf.fit(X_academic_train, y_academic_train)

    # 3. Linear Regression (for G3 prediction)
    y_reg = abt_df["G3"]
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    
    reg_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", LinearRegression())
    ])
    reg_pipeline.fit(X_train_reg, y_train_reg)

    # Save Models
    joblib.dump(rf_pipeline, os.path.join(MODELS_DIR, "rf_pass_prediction.pkl"))
    joblib.dump(lr_pipeline, os.path.join(MODELS_DIR, "lr_pass_prediction.pkl"))
    joblib.dump(reg_pipeline, os.path.join(MODELS_DIR, "linear_regression_model.pkl"))
    joblib.dump(log_reg_academic_clf, os.path.join(MODELS_DIR, "log_reg_academic_clf.pkl"))
    joblib.dump(rf_academic_clf, os.path.join(MODELS_DIR, "rf_academic_clf.pkl"))
    
    print("Models saved to models/")
    
    return {
        "lr": lr_pipeline, "rf": rf_pipeline, "reg": reg_pipeline, "lr_academic_clf": log_reg_academic_clf, "rf_academic_clf": rf_academic_clf,
        "data": (X_train, X_test, y_train, y_test, X_academic_train, X_academic_test, y_academic_train, y_academic_test)
    }

def generate_resilience_analysis(abt):
    # 1. Filter for 'At-Risk' group (Parents with Primary education or less)
    df_at_risk = abt[(abt['Medu'] <= 2) | (abt['Fedu'] <= 2)].copy()

    if df_at_risk.empty:
        print("Warning: No students found in the at-risk category. Skipping Fig 2.2.")
        return

    # 2. Define "Resilient" (High Final Grade: G3 >= 14)
    df_at_risk['resilient'] = (df_at_risk['G3'] >= 14).astype(int)
    
    # Feature selection
    features = ['address', 'famsup', 'famrel', 'studytime', 'failures', 'schoolsup', 'internet', 'absences']
    X = df_at_risk[features]
    y = df_at_risk['resilient']
    
    # Preprocessing
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # 3. Model Training (Mini-analysis)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")

    return (y_test, clf.feature_importances_,X.columns)