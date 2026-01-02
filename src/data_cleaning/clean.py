import pandas as pd
import os
from src.config import DATA_CLEANED

def clean_and_merge(mat_path, por_path):
    """Merges datasets, handles missing values and types."""
    print("--- Starting Data Cleaning ---")
    mat = pd.read_csv(mat_path)
    por = pd.read_csv(por_path)

    # 1. Add labels
    mat["course"] = "math"
    por["course"] = "portuguese"

    # 2. Combine
    df = pd.concat([mat, por], ignore_index=True)

    # 3. Treat empty strings
    df.replace("", pd.NA, inplace=True)

    # 4. Handle Missing Values
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna("unknown")

    # Save
    out_path = os.path.join(DATA_CLEANED, "student_performance_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"Cleaned data saved to {out_path}")
    return df