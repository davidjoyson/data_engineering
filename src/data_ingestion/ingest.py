import os
import glob
import pandas as pd
import kagglehub
from src.config import DATA_RAW

def download_data():
    """Downloads dataset from Kaggle and organizes raw files."""
    print("--- Starting Data Ingestion ---")
    download_dir = kagglehub.dataset_download("whenamancodes/student-performance")
    
    # Define sub-paths
    math_path = os.path.join(DATA_RAW, "maths")
    port_path = os.path.join(DATA_RAW, "portuguese")
    os.makedirs(math_path, exist_ok=True)
    os.makedirs(port_path, exist_ok=True)

    # Read and Save to Raw
    mat = pd.read_excel(os.path.join(download_dir, "Maths.csv")) # Note: Source might be CSV or Excel
    por = pd.read_excel(os.path.join(download_dir, "Portuguese.csv"))
    
    mat_out = os.path.join(math_path, "Maths.csv")
    por_out = os.path.join(port_path, "Portuguese.csv")
    
    mat.to_csv(mat_out, index=False)
    por.to_csv(por_out, index=False)
    
    print(f"Raw data saved to {DATA_RAW}")
    return mat_out, por_out