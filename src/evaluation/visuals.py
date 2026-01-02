import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.inspection import permutation_importance
from src.config import DATA_CLEANED, FIGURES_DIR, TABLES_DIR
from sklearn.metrics import f1_score
from src.modeling.train import generate_resilience_analysis

def generate_rq1_plots(metrics_df, abt_df):
    """Generates RQ1_Fig1, RQ1_Fig2, 1.3, 1.4"""
    print("Generating RQ1 Figures...")

    # RQ1_Table1 & RQ1_Fig3: Performance Comparison
    generate_rq1_table1_fig3(metrics_df)

    # RQ1_Fig1: Performance Comparison
    melted = metrics_df.melt(id_vars=['model'], value_vars=['accuracy', 'precision', 'recall', 'f1'])
    melted['value'] = melted['value'] * 100
    
    plt.figure(figsize=(12, 7))
    sns.barplot(x='variable', y='value', hue='model', data=melted, palette='husl')
    plt.ylim(80, 100)
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Score (%)', fontsize=12)
    plt.title('RQ1_Fig1 Model Performance Comparison: Multi-Source vs. Single-Source')
    plt.savefig(os.path.join(FIGURES_DIR, "RQ1_Fig1.pdf"))
    plt.close()

    # RQ1_Fig2: Scatter G1 vs G2
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='G1', y='G2', hue='target_pass', data=abt_df, palette='tab10', s=100, alpha=0.7)
    plt.title('RQ1_Fig2 G1 vs G2 Grades')
    plt.xlabel('First Period Grade (G1)', fontsize=12)
    plt.ylabel('Second Period Grade (G2)', fontsize=12)
    plt.savefig(os.path.join(FIGURES_DIR, "RQ1_Fig2.pdf"))
    plt.close()

    # RQ1_Fig4: Scatter G1 vs G2
    df = pd.read_csv(os.path.join(DATA_CLEANED, "student_performance_clean.csv"))
    plt.figure(figsize=(10, 6))
    # Note: hue='studytime' is added to avoid the palette warning in newer Seaborn versions
    sns.boxplot(x='studytime', y='G3', data=df, palette='viridis', hue='studytime', legend=False)
    
    plt.title('RQ1_Fig4 Final Grades by Study Time Categories', fontsize=16)
    plt.xlabel('Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)', fontsize=12)
    plt.ylabel('Final Grade (G3)', fontsize=12)
    
    # Use the centralized FIGURES_DIR path
    boxplot_path = os.path.join(FIGURES_DIR, "RQ1_Fig4.pdf")
    plt.savefig(boxplot_path)
    plt.close() # Important to close the plot to save memory
    
    print(f"Box plot saved to {boxplot_path}")

def generate_fairness_plots(model, X_test, y_test):
    """Generates Fairness figures (RQ3)"""
    print("Generating RQ3 Figures...")
    y_pred = model.predict(X_test)
    
    # Example: Fairness by Sex
    from src.evaluation.metrics import subgroup_metrics
    fair_sex = subgroup_metrics(y_test, y_pred, X_test['sex'], "sex")
    fair_sex.to_csv(os.path.join(TABLES_DIR, "fairness_by_sex.csv"), index=False)
    
    # Figure 3.1 Gap
    overall_f1 = f1_score(y_test, y_pred)
    fair_sex["f1_gap"] = fair_sex["f1"] - overall_f1
    
    plt.figure(figsize=(8,6))
    plt.bar(fair_sex["sex"], fair_sex["f1_gap"])
    plt.title("RQ3_Fig1 Fairness Gap (Sex)")
    plt.savefig(os.path.join(FIGURES_DIR, "RQ3_Fig1.pdf"))
    plt.close()

def generate_feature_importance(model, X_test, y_test):
    """Generates Feature Importance Plots (RQ4)"""
    print("Generating RQ4 Figures...")
    
    # Use a sample for speed
    X_sample = X_test.sample(n=min(300, len(X_test)), random_state=42)
    y_sample = y_test.loc[X_sample.index]
    
    result = permutation_importance(model, X_sample, y_sample, n_repeats=5, random_state=42, n_jobs=-1)
    
    # Get feature names
    try:
        names = model.named_steps['preprocess'].get_feature_names_out()
    except:
        names = [f"feat_{i}" for i in range(X_test.shape[1])]
        
    imp_df = pd.DataFrame({'feature': names[:len(result.importances_mean)], 
                           'importance': result.importances_mean})
    imp_df = imp_df.sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=imp_df, palette='viridis')
    plt.title('RQ4_Fig5 Top Predictive Features')
    plt.savefig(os.path.join(FIGURES_DIR, "RQ4_Fig5.pdf"))
    plt.close()

def generate_rq1_table1_fig3(all_models_comparison_df):
    print("Generating RQ1_Table1")

    # Filter for Random Forest models
    rf_multi = all_models_comparison_df[all_models_comparison_df['model'] == 'Multi-Source Random Forest']
    rf_academic = all_models_comparison_df[all_models_comparison_df['model'] == 'Single-Source Random Forest']

    # Calculate improvement for Random Forest
    rf_improvement = pd.DataFrame({
        'Model Type': ['Random Forest (Multi-Source vs. Single-Source)'],
        'Accuracy Improvement': [rf_multi['accuracy'].values[0] - rf_academic['accuracy'].values[0]],
        'Precision Improvement': [rf_multi['precision'].values[0] - rf_academic['precision'].values[0]],
        'Recall Improvement': [rf_multi['recall'].values[0] - rf_academic['recall'].values[0]],
        'F1-Score Improvement': [rf_multi['f1'].values[0] - rf_academic['f1'].values[0]]
    })

    # Filter for Logistic Regression models
    lr_multi = all_models_comparison_df[all_models_comparison_df['model'] == 'Multi-Source Logistic Regression']
    lr_academic = all_models_comparison_df[all_models_comparison_df['model'] == 'Single-Source Logistic Regression']

    # Calculate improvement for Logistic Regression
    lr_improvement = pd.DataFrame({
        'Model Type': ['Logistic Regression (Multi-Source vs. Single-Source)'],
        'Accuracy Improvement': [lr_multi['accuracy'].values[0] - lr_academic['accuracy'].values[0]],
        'Precision Improvement': [lr_multi['precision'].values[0] - lr_academic['precision'].values[0]],
        'Recall Improvement': [lr_multi['recall'].values[0] - lr_academic['recall'].values[0]],
        'F1-Score Improvement': [lr_multi['f1'].values[0] - lr_academic['f1'].values[0]]
    })

    # Combine into a single DataFrame
    comparison_improvement_df = pd.concat([rf_improvement, lr_improvement], ignore_index=True)
    
    # Scale to percentages
    cols_to_fix = ['Accuracy Improvement', 'Precision Improvement', 'Recall Improvement', 'F1-Score Improvement']
    comparison_improvement_df[cols_to_fix] = comparison_improvement_df[cols_to_fix] * 100
    
    # Save to the configured tables directory
    output_path = os.path.join(TABLES_DIR, "RQ1_Table1.csv")
    comparison_improvement_df.to_csv(output_path, index=False)
    
    print(f"RQ1_Table1 saved to {output_path}")

    # RQ1_Fig3: Improvement Bar Plot
    melted_improvement_df = comparison_improvement_df.melt(
        id_vars=['Model Type'],
        value_vars=['Accuracy Improvement', 'Precision Improvement', 'Recall Improvement', 'F1-Score Improvement'],
        var_name='Metric',
        value_name='Improvement (%)'
    )

    plt.figure(figsize=(12, 7))
    sns.barplot(
        x='Metric', 
        y='Improvement (%)', 
        hue='Model Type', 
        data=melted_improvement_df, 
        palette='viridis'
    )
    
    plt.title('RQ1_Fig3 Model Performance Improvement: Multi-Source vs. Single-Source', fontsize=16)
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Improvement (%)', fontsize=12)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.legend(title='Model Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Use the FIGURES_DIR from config
    figure_path = os.path.join(FIGURES_DIR, "RQ1_Fig3.pdf")
    plt.savefig(figure_path)
    plt.close()
    
    print(f"Saved RQ1_Fig3 to {figure_path}")
    return comparison_improvement_df

def generate_rq2_plots(abt):

    # RQ2_Fig1: Mean G3 by Parental Education
    grouped_g3_mean = abt.groupby(['Medu', 'Fedu', 'famsup', 'Pstatus'])['G3'].mean().reset_index()
    
    mean_g3_by_fedu = grouped_g3_mean.groupby('Fedu')['G3'].mean().reset_index()
    mean_g3_by_medu = grouped_g3_mean.groupby('Medu')['G3'].mean().reset_index()
    
    mean_g3_by_fedu['Education Type'] = 'Paternal'
    mean_g3_by_medu['Education Type'] = 'Maternal'

    # Rename columns for consistency
    mean_g3_by_fedu.rename(columns={'Fedu': 'Education Level'}, inplace=True)
    mean_g3_by_medu.rename(columns={'Medu': 'Education Level'}, inplace=True)
    combined_data = pd.concat([mean_g3_by_fedu, mean_g3_by_medu], ignore_index=True)

    # Create the combined plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Education Level', y='G3', hue='Education Type', data=combined_data, palette='colorblind')
    
    plt.title('RQ2_Fig1 Mean Grade by Parental Education Level', fontsize=14)
    plt.xlabel('Education Level (0: none, 1: primary, 2: 5th to 9th, 3: secondary, 4: higher)', fontsize=11)
    plt.ylabel('Mean Final Grade (G3)', fontsize=11)
    plt.legend(title='Education Type')
    plt.tight_layout()

    # Save using project config path
    figure_path = os.path.join(FIGURES_DIR, "RQ2_Fig1.pdf")
    plt.savefig(figure_path)
    plt.close()
    
    # RQ2_Fig2: 
    clf = generate_resilience_analysis(abt)
    importances = pd.Series(clf[1], index=clf[2]).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette='viridis', hue=importances.index, legend=False)
    plt.title('RQ2_Fig2 Key Drivers of Academic Resilience (Low Parental Ed Group)', fontsize=14)
    plt.xlabel('Importance Score', fontsize=12)
    
    # Save using project config path
    figure_path = os.path.join(FIGURES_DIR, "RQ2_Fig2.pdf")
    plt.savefig(figure_path)
    plt.close()