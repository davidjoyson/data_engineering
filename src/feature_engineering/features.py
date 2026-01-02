import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import DATA_PROCESSED

def create_abt(df):
    """Creates Analytical Base Table with engineered features."""
    print("--- Starting Feature Engineering ---")
    
    # Feature Engineering
    df["avg_prev_grade"] = df[["G1", "G2"]].mean(axis=1)
    df["grade_trend"] = df["G3"] - df["G1"]
    df["high_absence"] = (df["absences"] > df["absences"].median()).astype(int)
    
    # Target definition: Pass (1) if G3 >= 10
    df["target_pass"] = (df["G3"] >= 10).astype(int)

    # Save ABT
    abt_path = os.path.join(DATA_PROCESSED, "abt_student_performance.csv")
    df.to_csv(abt_path, index=False)
    print(f"ABT saved to {abt_path}")
    return df

def get_preprocessor(X_train):
    """Returns a sklearn ColumnTransformer for the given data."""
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return transformer

def get_academic_preprocessor(X_train):
    academic_features_scaler = StandardScaler()
    academic_preprocess_transformer = ColumnTransformer(
        transformers=[
            ("num", academic_features_scaler, ['G1', 'G2'])
            ],
        remainder='passthrough' 
    )

    print("Academic-only feature preprocessor defined.")
    return academic_preprocess_transformer