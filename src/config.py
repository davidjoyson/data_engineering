import os

# Project Root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Directories
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_CLEANED = os.path.join(BASE_DIR, "data", "cleaned")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
TABLES_DIR = os.path.join(BASE_DIR, "tables")

# Create directories if they don't exist
for d in [DATA_RAW, DATA_CLEANED, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)