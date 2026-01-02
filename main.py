import pandas as pd
from src.data_ingestion.ingest import download_data
from src.data_cleaning.clean import clean_and_merge
from src.feature_engineering.features import create_abt
from src.modeling.train import train_models
from src.evaluation.metrics import evaluate_classification
from src.evaluation.visuals import generate_rq1_plots, generate_fairness_plots, generate_feature_importance, generate_rq2_plots

def main():
    # 1. Ingestion
    mat_path, por_path = download_data()
    
    # 2. Cleaning
    clean_df = clean_and_merge(mat_path, por_path)
    
    # 3. Feature Engineering
    abt_df = create_abt(clean_df)
    
    # 4. Modeling
    # Note: train_models returns a dictionary with models and split data
    artifacts = train_models(abt_df)
    rf_model = artifacts['rf']
    lr_model = artifacts['lr']
    log_reg_academic_trained = artifacts['lr_academic_clf']
    rf_academic_trained = artifacts['rf_academic_clf']
    X_test = artifacts['data'][1]
    y_test = artifacts['data'][3]
    X_academic_test = artifacts['data'][5]
    y_academic_test = artifacts['data'][7]
    
    # 5. Evaluation & Figures
    
    # Comparison Dataframe
    res_rf = evaluate_classification(rf_model, X_test, y_test, "Multi-Source Random Forest")
    res_lr = evaluate_classification(lr_model, X_test, y_test, "Multi-Source Logistic Regression")
    academic_res_rf = evaluate_classification(rf_academic_trained, X_academic_test, y_academic_test, "Single-Source Random Forest")
    academic_res_lr = evaluate_classification(log_reg_academic_trained, X_academic_test, y_academic_test, "Single-Source Logistic Regression")
    comparison_df = pd.DataFrame([res_rf, academic_res_rf, res_lr, academic_res_lr])
    
    # Generate Plots
    generate_rq1_plots(comparison_df, abt_df)
    generate_rq2_plots(abt_df)
    # generate_fairness_plots(rf_model, X_test, y_test)
    # generate_feature_importance(rf_model, X_test, y_test)

    print("Pipeline completed successfully. Check /figures and /models.")

if __name__ == "__main__":
    main()