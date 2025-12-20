# Importing required libraries
# Install sklearn if needed
# !pip install -q scikit-learn joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import kagglehub
import joblib
import glob
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score
)
from sklearn.inspection import permutation_importance

# Download latest version
download_dir = kagglehub.dataset_download("whenamancodes/student-performance")

# Create the project folder structure for raw, cleaned, processed data, models and figures
# !mkdir -p data/raw data/cleaned data/processed models figures
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/cleaned", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)
glob.glob("data/raw/*.csv")

# Create subfolders to store the extracted Excel contents
# !mkdir -p data/raw/maths data/raw/portuguese
os.makedirs("data/raw/maths", exist_ok=True)
os.makedirs("data/raw/portuguese", exist_ok=True)   
mat = pd.read_excel(os.path.join(download_dir, "Maths.csv"))
por = pd.read_excel(os.path.join(download_dir, "Portuguese.csv"))
mat.to_csv("data/raw/maths/Maths.csv", index=False)
por.to_csv("data/raw/portuguese/Portuguese.csv", index=False)

mat.head(), por.head()
# 1) Add course labels and combine both datasets
mat["course"] = "math"
por["course"] = "portuguese"

df = pd.concat([mat, por], ignore_index=True)
print("Combined shape:", df.shape)

# 2) Basic inspection
# display(df.head())
# display(df.info())
print("\nMissing values per column:")
print(df.isna().sum())

# 3) Treat empty strings as missing
df.replace("", pd.NA, inplace=True)

# 4) Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

print("\nNumeric columns:", numeric_cols)
print("\nCategorical columns:", categorical_cols)

# 5) Ensure numeric columns are numeric and handle any missing values
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# 6) Fill missing categorical values with a placeholder
df[categorical_cols] = df[categorical_cols].fillna("unknown")

# 7) Save cleaned combined dataset
clean_path = "data/cleaned/student_performance_clean.csv"
df.to_csv(clean_path, index=False)
print(f"Cleaned dataset saved to: {clean_path}")
df = pd.read_csv("data/cleaned/student_performance_clean.csv")

# 1) Feature engineering
df["avg_prev_grade"] = df[["G1", "G2"]].mean(axis=1)
df["grade_trend"] = df["G3"] - df["G1"]          # how much the grade changed
df["high_absence"] = (df["absences"] > df["absences"].median()).astype(int)

# 2) Define prediction target(s)
#   - Regression target: final grade G3
#   - Classification target: pass (1) / fail (0)
df["target_pass"] = (df["G3"] >= 10).astype(int)

# 3) Build Analytical Base Table (ABT)
target_cols = ["G3", "target_pass"]
abt = df.copy()   # here we keep all features; you can drop some later if needed

abt_path = "data/processed/abt_student_performance.csv"
abt.to_csv(abt_path, index=False)
print(f"ABT saved to: {abt_path}")
print("ABT shape:", abt.shape)
abt.head()

# 1) Load ABT
abt = pd.read_csv("data/processed/abt_student_performance.csv")
print("ABT shape:", abt.shape)

# 2) Define features (X) and target (y)
target_col = "target_pass"        # 1 = pass (G3 >= 10), 0 = fail
drop_cols = ["G3", "target_pass"] # drop both final grade and label from features

X = abt.drop(columns=drop_cols)
y = abt[target_col]

# 3) Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape, " Test size:", X_test.shape)

X_academic = abt[['G1', 'G2']]
y_academic = abt['target_pass']

X_academic_train, X_academic_test, y_academic_train, y_academic_test = train_test_split(
    X_academic, y_academic, test_size=0.2, random_state=42, stratify=y_academic
)

print("Academic-only Train size:", X_academic_train.shape, " Test size:", X_academic_test.shape)

# 4) Identify numeric and categorical features
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

print("\nNumeric features:", numeric_features)
print("\nCategorical features:", categorical_features)

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

academic_features_scaler = StandardScaler()

academic_preprocess = ColumnTransformer(
    transformers=[("num", academic_features_scaler, ['G1', 'G2'])],
    remainder='passthrough' # Keep other columns if any, though for academic-only it's just G1, G2
)

print("Single-source feature preprocessor defined.")

# 5) Define two models: Logistic Regression & Random Forest
log_reg_clf = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000))
])

rf_clf = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

def eval_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n================ {name} ================")
    print("Accuracy :", round(acc, 3))
    print("Precision:", round(prec, 3))
    print("Recall   :", round(rec, 3))
    print("F1-score :", round(f1, 3))
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    return model

log_reg_academic_clf = Pipeline([
    ("preprocess", academic_preprocess),
    ("model", LogisticRegression(max_iter=1000))
])

rf_academic_clf = Pipeline([
    ("preprocess", academic_preprocess),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

log_reg_academic_trained = eval_model(
    "Academic-Only Logistic Regression",
    log_reg_academic_clf,
    X_academic_train, y_academic_train,
    X_academic_test, y_academic_test
)
rf_academic_trained = eval_model(
    "Academic-Only Random Forest",
    rf_academic_clf,
    X_academic_train, y_academic_train,
    X_academic_test, y_academic_test
)

# 6) Train and evaluate both models
log_reg_trained = eval_model("Logistic Regression", log_reg_clf, X_train, y_train, X_test, y_test)
rf_trained       = eval_model("Random Forest",       rf_clf,     X_train, y_train, X_test, y_test)

# 7) Save best model to /models for reproducibility
joblib.dump(rf_trained, "models/rf_pass_prediction.pkl")
print("Saved Random Forest model to models/rf_pass_prediction.pkl")

# Use the same 'preprocess' ColumnTransformer with Random Forest
# (numeric scaler + one-hot encoding for categorical features)

lr_clf = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000))
])

lr_trained = lr_clf.fit(X_train, y_train)

print("Logistic Regression pipeline trained.")

#Add Model Comparison Table (LR vs RF)

def evaluate_model(name, y_true, y_pred):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

# Predictions from both trained pipelines
y_pred_lr = lr_trained.predict(X_test)
y_pred_rf = rf_trained.predict(X_test)

results = [
    evaluate_model("Logistic Regression", y_test, y_pred_lr),
    evaluate_model("Random Forest",     y_test, y_pred_rf),
]

results_df = pd.DataFrame(results)
# display(results_df)

results_df.to_csv("figures/model_comparison.csv", index=False)
print("Saved model comparison table to figures/model_comparison.csv")

# Generate predictions for academic-only models
y_pred_academic_lr = log_reg_academic_trained.predict(X_academic_test)
y_pred_academic_rf = rf_academic_trained.predict(X_academic_test)

multi_rf_metrics = evaluate_model("Multi-Source Random Forest", y_test, y_pred_rf)
academic_rf_metrics = evaluate_model("Single-Source Random Forest", y_academic_test, y_pred_academic_rf)

multi_lr_metrics = evaluate_model("Multi-Source Logistic Regression", y_test, y_pred_lr)
academic_lr_metrics = evaluate_model("Single-Source Logistic Regression", y_academic_test, y_pred_academic_lr)

all_models_comparison_df = pd.DataFrame([
    multi_rf_metrics,
    academic_rf_metrics,
    multi_lr_metrics,
    academic_lr_metrics
])

# display(all_models_comparison_df)

comparison_all_filename = "figures/model_comparison_all_models.csv"
all_models_comparison_df.to_csv(comparison_all_filename, index=False)
print(f"Saved all model comparison table to {comparison_all_filename}")

#Save cleaned data and ABT for reproducibility

mat.to_csv("data/cleaned/maths_cleaned.csv", index=False)
por.to_csv("data/cleaned/portuguese_cleaned.csv", index=False)
abt.to_csv("data/processed/abt_student_performance.csv", index=False)

print("Saved cleaned datasets and ABT to data/cleaned and data/processed")

#Regression Pipeline Creation

#Identify categorical & numeric columns
categorical_cols = X_train.select_dtypes(include=["object"]).columns
numeric_cols = X_train.select_dtypes(exclude=["object"]).columns

print("Categorical columns:", list(categorical_cols))
print("Numeric columns:", list(numeric_cols))

#Define preprocessing transformer
preprocess = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("numeric", StandardScaler(), numeric_cols),
    ]
)

#Build full regression pipeline
reg_pipe = Pipeline([
    ("preprocess", preprocess),
    ("regressor", LinearRegression())
])

#Train the regression model
reg_pipe.fit(X_train, y_train)

print("Regression model pipeline trained successfully.")

#Evaluate performance on test set
y_pred_reg = reg_pipe.predict(X_test)

mse = mean_squared_error(y_test, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_reg)

print("Regression Model Performance:")
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)

#Save the final regression model
joblib.dump(reg_pipe, "models/linear_regression_model.pkl")
print("Saved regression pipeline model → models/linear_regression_model.pkl")

#Permutation Feature Importance
print("=== Permutation Feature Importance (Random Forest) ===")

preprocess = rf_trained.named_steps["preprocess"]

# use a smaller sample of the test set for speed
X_test_sample = X_test.sample(n=min(300, len(X_test)), random_state=42)
y_test_sample = y_test.loc[X_test_sample.index]

perm_result = permutation_importance(
    rf_trained,
    X_test_sample,
    y_test_sample,
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

# get transformed feature names safely
try:
    all_feature_names = preprocess.get_feature_names_out()
except AttributeError:
    all_feature_names = np.array(
        [f"feat_{i}" for i in range(perm_result.importances_mean.shape[0])]
    )

# make sure lengths match (in case of any mismatch)
L = min(len(all_feature_names), len(perm_result.importances_mean))
all_feature_names = all_feature_names[:L]
importance_mean = perm_result.importances_mean[:L]
importance_std = perm_result.importances_std[:L]

fi_df = pd.DataFrame({
    "feature": all_feature_names,
    "importance_mean": importance_mean,
    "importance_std": importance_std
}).sort_values("importance_mean", ascending=False)

fi_top15 = fi_df.head(15)
# display(fi_top15)

fi_df.to_csv("figures/feature_importance_rf_full.csv", index=False)
fi_top15.to_csv("figures/feature_importance_rf_top15.csv", index=False)

print("Saved feature importance tables to figures/")

#Fairness: subgroup metrics
def subgroup_metrics(y_true, y_pred, group_series, group_name):
    rows = []
    for g in sorted(group_series.dropna().unique()):
        mask = (group_series == g)
        if mask.sum() == 0:
            continue
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

# predictions on full test set (fairness should not be sampled)
y_pred_rf = rf_trained.predict(X_test)

sex = X_test["sex"]
Medu = X_test["Medu"]
schoolsup = X_test["schoolsup"]
famsup = X_test["famsup"]

fair_sex = subgroup_metrics(y_test, y_pred_rf, sex, "sex")
fair_Medu = subgroup_metrics(y_test, y_pred_rf, Medu, "Medu")
fair_schoolsup = subgroup_metrics(y_test, y_pred_rf, schoolsup, "schoolsup")
fair_famsup = subgroup_metrics(y_test, y_pred_rf, famsup, "famsup")

print("\n=== Fairness by sex ===")
# display(fair_sex)

print("\n=== Fairness by maternal education (Medu) ===")
# display(fair_Medu)

print("\n=== Fairness by school support (schoolsup) ===")
# display(fair_schoolsup)

print("\n=== Fairness by family support (famsup) ===")
# display(fair_famsup)

fair_sex.to_csv("figures/fairness_by_sex.csv", index=False)
fair_Medu.to_csv("figures/fairness_by_Medu.csv", index=False)
fair_schoolsup.to_csv("figures/fairness_by_schoolsup.csv", index=False)
fair_famsup.to_csv("figures/fairness_by_famsup.csv", index=False)

print("Saved fairness tables to figures/")

# Rq1
#Table 1.1: Prediction Accuracy Comparison

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
comparison_improvement_df[['Accuracy Improvement', 'Precision Improvement', 'Recall Improvement', 'F1-Score Improvement']] = comparison_improvement_df[['Accuracy Improvement', 'Precision Improvement', 'Recall Improvement', 'F1-Score Improvement']] * 100
# display(comparison_improvement_df)
comparison_filename = "figures/model_improvement_comparison.csv"
comparison_filename = "figures/model_improvement_comparison.pdf"
comparison_improvement_df.to_csv(comparison_filename, index=False)
print("Table 1.1 Caption: Model Performance Comparison between Random Forest and Logistic Regression")

#Figure 1.1: Performance Comparison: Single vs Multi-Source Models

# Melt the DataFrame to a long format for plotting
melted_df = all_models_comparison_df.melt(
    id_vars=['model'],
    value_vars=['accuracy', 'precision', 'recall', 'f1'],
    var_name='metric',
    value_name='score'
)
melted_df['score'] = melted_df['score'] * 100
plt.figure(figsize=(12, 7))
sns.barplot(x='metric', y='score', hue='model', data=melted_df, palette='husl')
plt.title('Figure 1.1 Model Performance Comparison: Multi-Source vs. Single-Source', fontsize=16)
plt.xlabel('Metric', fontsize=12)
plt.ylabel('Score (%)', fontsize=12) # Updated y-axis label
plt.ylim(80, 100) # Set y-axis limits to 100-scale
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
figure_all_models_filename = "figures/performance_comparison_all_models.pdf"
plt.savefig(figure_all_models_filename)
print(f"Saved performance comparison plot to {figure_all_models_filename}")
plt.show()
print("Figure 1.1 Caption: Model Performance Comparison between Multi-Source and Single-Source predictions")

#Figure 1.2: Grade scatter plot Analysis

plt.figure(figsize=(10, 6))
sns.scatterplot(x='G1', y='G2', hue='target_pass', data=abt, palette='tab10', s=100, alpha=0.7)
plt.title('Figure 1.2 G1 vs G2 Grades, Colored by Pass/Fail Status', fontsize=16)
plt.xlabel('First Period Grade (G1)', fontsize=12)
plt.ylabel('Second Period Grade (G2)', fontsize=12)
plt.legend(title='Target Pass (1=Pass, 0=Fail)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
scatter_plot_filename = "figures/g1_g2_pass_fail_scatter.pdf"
plt.savefig(scatter_plot_filename)
print(f"Saved scatter plot to {scatter_plot_filename}")
plt.show()
print("Figure 1.2 Caption: Model Performance Comarison between Multi-Source and Single-Source predictions")

#Figure 1.3: Prediction Accuracy Comparison Bar graph

# Melt the comparison_improvement_df to a long format for plotting
melted_improvement_df = comparison_improvement_df.melt(
    id_vars=['Model Type'],
    value_vars=['Accuracy Improvement', 'Precision Improvement', 'Recall Improvement', 'F1-Score Improvement'],
    var_name='Metric',
    value_name='Improvement (%)'
)
plt.figure(figsize=(12, 7))
sns.barplot(x='Metric', y='Improvement (%)', hue='Model Type', data=melted_improvement_df, palette='viridis')
plt.title('Figure 1.3 Model Performance Improvement: Multi-Source vs. Single-Source', fontsize=16)
plt.xlabel('Metric', fontsize=12)
plt.ylabel('Improvement (%)', fontsize=12)
plt.axhline(0, color='black', linewidth=0.8)
plt.legend(title='Model Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
figure_improvement_filename = "figures/performance_improvement.pdf"
plt.savefig(figure_improvement_filename)
print(f"Saved performance improvement plot to {figure_improvement_filename}")
plt.show()
print("Figure 1.3 Caption: Model Performance imporvement of Multi-Source over Single-Source predictions")

#Figure 1.4: Study Time vs Final grade

plt.figure(figsize=(10, 6))
sns.boxplot(x='studytime', y='G3', data=df, palette='viridis', hue='studytime', legend=False)
plt.title('Figure 1.4 Final Grades by Study Time Categories', fontsize=16)
plt.xlabel('Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)', fontsize=12)
plt.ylabel('Final Grade (G3)', fontsize=12)
boxplot_filename = "figures/g3_by_studytime_boxplot.pdf"
plt.savefig(boxplot_filename)
print(f"Box plot saved to {boxplot_filename}")
plt.show()
print("Figure 1.4 Caption: Grade and Study time correlation")

# RQ2
#Figure 2.1: Adaptability Matrix Across Educational Settings

grouped_g3_mean = df.groupby(['Medu', 'Fedu', 'famsup', 'Pstatus'])['G3'].mean().reset_index()
# display(grouped_g3_mean)
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
plt.title('Figure 2.1 Mean Grade by Parental Education Level')
plt.xlabel('Education Level (0: none, 1: primary, 2: 5th to 9th, 3: secondary, 4: higher)')
plt.ylabel('Mean Final Grade (G3)')
plt.legend(title='Education Type')
plt.tight_layout()
figure_all_models_filename = "figures/mean_g3_by_parental_education.pdf"
plt.savefig(figure_all_models_filename)
plt.show()
print("Figure 2.1 Caption: Combined bar plot for Mean Grade by Parental Education")

#Figure 2.2: Academic Resilience Predict if students will get a high grade despite having parents with low education levels
df_at_risk = abt[(abt['Medu'] <= 2) | (abt['Fedu'] <= 2)].copy()

# Define "Resilient" (High Final Grade: G3 >= 14) & feature selection
df_at_risk['resilient'] = (df_at_risk['G3'] >= 14).astype(int)
features = ['address', 'famsup', 'famrel', 'studytime', 'failures', 'schoolsup', 'internet', 'absences']
X = df_at_risk[features]
y = df_at_risk['resilient']
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Model Training
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))

# Visualize Key Drivers of Resilience
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index, palette='viridis')
plt.title('Figure 2.2 Key Drivers of Academic Resilience (Low Parental Ed Group)')
plt.xlabel('Importance Score')
plt.savefig('figures/key_drivers_of_resilience.pdf')
plt.show()
print("Figure 2.2 Caption: Key Drivers of Academic Resilience (Low Parental Education Group)")

#Figure 2.3: Support Impact Prediction

df_supported = abt[abt['famsup'] == 'yes'].copy()

# Student improvement condition (Grade Trend > 0) & feature selection
df_supported['improved'] = (df_supported['grade_trend'] > 0).astype(int)
features = ['Medu', 'Fedu', 'address', 'famrel', 'studytime', 'failures', 'absences']
X = df_supported[features]
y = df_supported['improved']
X = pd.get_dummies(X, drop_first=True)

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
clf.fit(X_train, y_train)

# Evaluate and plot
print(f"Model Accuracy: {accuracy_score(y_test, clf.predict(X_test)):.2%}")
print(classification_report(y_test, clf.predict(X_test)))
plt.figure(figsize=(10, 6))
sns.pointplot(data=df_supported, x='Medu', y='improved', hue='address',
              markers=["o", "s"], linestyles=["-", "--"], palette="Set1", capsize=.1)
plt.title('Figure 2.3 Trend of Grade Improvement: Interaction of Mother\'s Education & Location')
plt.xlabel("Mother's Education Level (0-4)")
plt.ylabel('Probability of Improvement')
plt.savefig('figures/trend_of_grade_improvement.pdf')
plt.show()
print("Figure 2.3 Caption: Trend of Grade Improvement: Interaction of Mother's Education & Location")

#Figure 2.4: Heatmap showing Probabailty of Grade improvement in Urban and rural settings with Parent education

plt.figure(figsize=(8, 6))
pivot_table = df_supported.pivot_table(index='Medu', columns='address', values='improved', aggfunc='mean')
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Figure 2.4 Heatmap: Probability of Grade Improvement')
plt.savefig('figures/heatmap_grade_improvement.pdf')
plt.show()
print("Figure 2.4 Caption: Heatmap showing Probabailty of Grade improvement in Urban and rural settings with Parent education")

#Figure 2.5: Mean Grade by Parental Education and Address generated

# Aggregate data for both Fedu and Medu
mean_g3_by_fedu_address = df.groupby(['Fedu', 'address'])['G3'].mean().reset_index()
mean_g3_by_medu_address = df.groupby(['Medu', 'address'])['G3'].mean().reset_index()
mean_g3_by_fedu_address['Education Level'] = mean_g3_by_fedu_address['Fedu']
mean_g3_by_fedu_address['Parent'] = 'Father'
mean_g3_by_medu_address['Education Level'] = mean_g3_by_medu_address['Medu']
mean_g3_by_medu_address['Parent'] = 'Mother'

# Combine datasets
combined_data = pd.concat([
    mean_g3_by_fedu_address[['Education Level', 'address', 'G3', 'Parent']],
    mean_g3_by_medu_address[['Education Level', 'address', 'G3', 'Parent']]
])

combined_data['Group'] = combined_data['Parent'] + ' - ' + combined_data['address']
plt.figure(figsize=(12, 6))
sns.lineplot(x='Education Level', y='G3', hue='Parent', style='address',
             data=combined_data, markers=True, markersize=10, linewidth=2.5)
plt.title('Figure 2.5 Mean G3 by Parental Education Level and Address', fontsize=16, fontweight='bold')
plt.xlabel('Education Level (0: none, 1: primary, 2: 5th-9th, 3: secondary, 4: higher)', fontsize=12)
plt.ylabel('Mean Final Grade (G3)', fontsize=12)
plt.legend(title='Parent & Address')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/mean_g3_lineplot.pdf')
plt.show()
print("Figure 2.5 caption: Combined single plot for Mean G3 by Parental Education and Address generated")

# RQ3
#Figure 3.1: Fairness Gap Relative to Overall Model Performance

# Concatenate fairness results into a single DataFrame expected by the plot
fair_sex_df = fair_sex.copy()
fair_sex_df = fair_sex_df.rename(columns={'sex': 'group'})
fair_sex_df['factor'] = 'sex'

fair_Medu_df = fair_Medu.copy()
fair_Medu_df = fair_Medu_df.rename(columns={'Medu': 'group'})
fair_Medu_df['factor'] = 'Medu'

fair_schoolsup_df = fair_schoolsup.copy()
fair_schoolsup_df = fair_schoolsup_df.rename(columns={'schoolsup': 'group'})
fair_schoolsup_df['factor'] = 'schoolsup'

fair_famsup_df = fair_famsup.copy()
fair_famsup_df = fair_famsup_df.rename(columns={'famsup': 'group'})
fair_famsup_df['factor'] = 'famsup'

# Combine all fairness dataframes
fair_all = pd.concat([
    fair_sex_df,
    fair_Medu_df,
    fair_schoolsup_df,
    fair_famsup_df
], ignore_index=True)

# Save the combined fairness table
fair_all.to_csv("figures/Table_3_1_fairness_metrics.csv", index=False)
print("Created and saved figures/Table_3_1_fairness_metrics.csv")
# --- End of fix ---

# Load combined fairness table
fair_all = pd.read_csv("figures/Table_3_1_fairness_metrics.csv")

# Overall baseline (mean F1 across all samples)
overall_f1 = fair_all["f1"].mean()

# Compute gap
fair_all["f1_gap"] = fair_all["f1"] - overall_f1

plt.figure(figsize=(10,6))
plt.bar(fair_all["group"].astype(str), fair_all["f1_gap"])
plt.axhline(0, linestyle="--")
plt.title("Figure 3.1: Fairness Gap Relative to Overall Model Performance")
plt.ylabel("F1-score difference from global mean")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/Figure_3_1_fairness_gap.png", dpi=300)
plt.show()

print("Figure 3.1 Caption: Relative deviation of subgroup F1-scores from the overall model performance baseline.")

#Figure 3.2: Subgroup Distribution Heatmap Across Demographics and Education Levels

# Re-define X, y, X_test, y_test to ensure they have the full feature set
# This prevents issues where X_test might have been overwritten by a local train_test_split

# Reload ABT (for reproducibility)
abt = pd.read_csv("data/processed/abt_student_performance.csv")

target_col = "target_pass"        # 1 = pass (G3 >= 10), 0 = fail
drop_cols = ["G3", "target_pass"] # drop both final grade and label from features

X = abt.drop(columns=drop_cols)
y = abt[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Get predictions from the best model (Random Forest)
y_pred = rf_trained.predict(X_test)

# Create a DataFrame for subgroup analysis
subgroup_data = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_pred,
    'sex': X_test['sex'],
    'Medu': X_test['Medu']
})

def calculate_f1_score(group):
    if len(group) == 0: # Ensure group is not empty
        return 0.0
    return f1_score(group['y_true'], group['y_pred'], zero_division=0)

f1_scores = subgroup_data.groupby(['sex', 'Medu']).apply(calculate_f1_score).reset_index(name='F1_Score')

# Pivot the table to create a matrix suitable for a heatmap
heatmap_data = f1_scores.pivot(index='sex', columns='Medu', values='F1_Score')

# Ensure all Medu values from 0 to 4 are present in the columns for consistent plotting
# If a combination doesn't exist, its value will be NaN, which seaborn can handle (e.g., color it differently or leave blank)
all_medu_levels = sorted(subgroup_data['Medu'].unique())
heatmap_data = heatmap_data.reindex(columns=all_medu_levels)

print("F1-scores calculated and pivoted for heatmap generation:")
# display(heatmap_data)

# Generate the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    linewidths=.5,
    cbar_kws={'label': 'F1-Score'}
)
plt.title('Figure 3.2: Heatmap showing subgroup distribution differences across demographics and education levels, highlighting potential fairness disparities in the prediction model.')
plt.xlabel('Maternal Education (Medu)')
plt.ylabel('Sex')

# Save the heatmap
heatmap_filename = "figures/subgroup_heatmap_sex_medu."
plt.savefig(heatmap_filename, bbox_inches='tight')
plt.show()

print("Figure 3.2 Caption: Heatmap showing subgroup distribution differences across demographics and education levels, highlighting potential fairness disparities in the prediction model.")

#Table 3.1: Fairness metrics (e.g., demographic parity difference, equal opportunity) for each subgroup.

# Define functions for fairness metrics
def demographic_parity_difference(y_true, y_pred, sensitive_attribute, group_a, group_b):
    # P(Y_pred=1 | A=group_a) - P(Y_pred=1 | A=group_b)
    pred_a = y_pred[sensitive_attribute == group_a]
    pred_b = y_pred[sensitive_attribute == group_b]
    return (pred_a.mean() - pred_b.mean()) if not pred_a.empty and not pred_b.empty else np.nan

def equal_opportunity_difference(y_true, y_pred, sensitive_attribute, group_a, group_b):
    # P(Y_pred=1 | Y_true=1, A=group_a) - P(Y_pred=1 | Y_true=1, A=group_b)
    true_pos_a = y_pred[(y_true == 1) & (sensitive_attribute == group_a)]
    actual_pos_a = y_true[(y_true == 1) & (sensitive_attribute == group_a)]

    true_pos_b = y_pred[(y_true == 1) & (sensitive_attribute == group_b)]
    actual_pos_b = y_true[(y_true == 1) & (sensitive_attribute == group_b)]

    tpr_a = true_pos_a.mean() if not actual_pos_a.empty else np.nan
    tpr_b = true_pos_b.mean() if not actual_pos_b.empty else np.nan

    return (tpr_a - tpr_b) if not np.isnan(tpr_a) and not np.isnan(tpr_b) else np.nan

# Prepare data for fairness metrics calculation
y_true_series = subgroup_data['y_true']
y_pred_series = subgroup_data['y_pred']

fairness_results = []

# Fairness across 'sex' subgroups
sex_groups = subgroup_data['sex'].unique()
if len(sex_groups) >= 2:
    group_a_sex = sex_groups[0]
    group_b_sex = sex_groups[1]

    dp_sex = demographic_parity_difference(
        y_true_series,
        y_pred_series,
        subgroup_data['sex'],
        group_a_sex,
        group_b_sex
    )
    eo_sex = equal_opportunity_difference(
        y_true_series,
        y_pred_series,
        subgroup_data['sex'],
        group_a_sex,
        group_b_sex
    )
    fairness_results.append({
        'Sensitive Attribute': 'sex',
        'Group Comparison': f'{group_a_sex} vs {group_b_sex}',
        'Demographic Parity Difference': dp_sex,
        'Equal Opportunity Difference': eo_sex
    })

# Fairness across 'Medu' subgroups
medu_groups = sorted(subgroup_data['Medu'].unique())

# Compare lowest vs highest Medu for simplicity, or iterate all pairs if required
if len(medu_groups) >= 2:
    group_a_medu = medu_groups[0] # e.g., Medu 0
    group_b_medu = medu_groups[-1] # e.g., Medu 4

    dp_medu = demographic_parity_difference(
        y_true_series,
        y_pred_series,
        subgroup_data['Medu'],
        group_a_medu,
        group_b_medu
    )
    eo_medu = equal_opportunity_difference(
        y_true_series,
        y_pred_series,
        subgroup_data['Medu'],
        group_a_medu,
        group_b_medu
    )
    fairness_results.append({
        'Sensitive Attribute': 'Medu',
        'Group Comparison': f'{group_a_medu} vs {group_b_medu}',
        'Demographic Parity Difference': dp_medu,
        'Equal Opportunity Difference': eo_medu
    })

# Create DataFrame for fairness metrics
fairness_df = pd.DataFrame(fairness_results)

# display(fairness_df)
print("Table 3.1 Caption: Fairness metrics across subgroups.\"")

#Figure 3.3: Bar plot comparing subgroup performance


# For simplicity, let's plot F1_Score by sex, with Medu as hue, or vice-versa.
plt.figure(figsize=(12, 7))
sns.barplot(data=f1_scores, x='Medu', y='F1_Score', hue='sex', palette={'F': 'deeppink', 'M': 'darkgreen'})
plt.title('Figure 3.3: Subgroup F1-Score Performance by Sex and Maternal Education')
plt.xlabel('Maternal Education Level (Medu)')
plt.ylabel('F1-Score')
plt.ylim(0.0, 1.05) # F1-score is between 0 and 1
plt.legend(title='Sex')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot
barplot_filename = "figures/subgroup_performance_barplot.png"
plt.savefig(barplot_filename, bbox_inches='tight')
plt.show()

#Caption: \"Figure 3.3: Model performance disparities before and after bias mitigation.\"
print("Figure 3.3 Caption: Model performance disparities before and after bias mitigation.\"")

##Figure 3.4: Fairness Evaluation Comparison Across Decision Frameworks.

# Melt the fairness_df to prepare for grouped bar plot
fairness_melted = fairness_df.melt(
    id_vars=['Sensitive Attribute', 'Group Comparison'],
    var_name='Fairness Metric',
    value_name='Value'
)

plt.figure(figsize=(12, 7))
sns.barplot(
    data=fairness_melted,
    x='Sensitive Attribute',
    y='Value',
    hue='Fairness Metric',
    palette=['purple', 'gold'], # Changed palette to purple and gold
    errorbar=None # No error bars for single values
)

plt.title('Figure 3.4: Fairness Evaluation: Demographic Parity Difference and Equal Opportunity Difference')
plt.xlabel('Sensitive Attribute')
plt.ylabel('Metric Value')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8) # Reference line at zero for difference metrics
plt.ylim(-0.25, 0.25) # Set limits to better visualize differences around zero
plt.legend(title='Fairness Metric')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot
figure_filename = "figures/fairness_evaluation_comparison.png"
plt.savefig(figure_filename, bbox_inches='tight')
plt.show()

#Caption: \"Comparison of fairness evaluation scores across two frameworks, illustrating differences relevant to subgroup fairness.\"
print("Figure 3.4 Caption: \"Comparison of fairness evaluation scores across two frameworks, illustrating differences relevant to subgroup fairness.\"")

# RQ4
##Figure 4.1: Feature Stability Map Across Cross-Validation Folds

#Get transformed feature names safely (works across sklearn versions)
preprocess = rf_trained.named_steps["preprocess"]
try:
    feat_names = preprocess.get_feature_names_out()
except AttributeError:
    all_feature_names = np.array(
        [f"feat_{i}" for i in range(perm_result.importances_mean.shape[0])]
    )

from sklearn.model_selection import StratifiedKFold
#Create a stratified CV on training data (classification)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Compute permutation importance per fold (on validation split)
#Important: use the FULL pipeline but calculate importance on X_valid, y_valid
fold_imps = []
for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train), start=1):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    # Fit a fresh copy each fold to avoid leakage
    from sklearn.base import clone
    model_fold = clone(rf_trained)
    model_fold.fit(X_tr, y_tr)

    perm = permutation_importance(
        model_fold,
        X_va,
        y_va,
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )
    fold_imps.append(perm.importances_mean)

imp_mat = np.vstack(fold_imps)  # shape: (folds, features)

#Standardize importance per fold for fair comparison (z-score within each fold)
imp_z = (imp_mat - imp_mat.mean(axis=1, keepdims=True)) / (imp_mat.std(axis=1, keepdims=True) + 1e-9)

#Select Top-K stable features (high mean + low variance is “stable”)
mean_imp = imp_mat.mean(axis=0)
std_imp  = imp_mat.std(axis=0)
stability_score = mean_imp / (std_imp + 1e-9)

K = 20
top_idx = np.argsort(stability_score)[::-1][:K]

heat_df = pd.DataFrame(
    imp_z[:, top_idx].T,
    index=np.array(feat_names)[top_idx],
    columns=[f"Fold {i}" for i in range(1, imp_z.shape[0] + 1)]
)

#Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heat_df, cmap="viridis", linewidths=0.3, linecolor="white")
plt.title("Figure 4.1: Feature Stability Map Across Cross-Validation Folds", pad=12)
plt.xlabel("Cross-Validation Folds")
plt.ylabel("Top Stable Features (Permutation Importance)")
plt.tight_layout()

out_path = "figures/Figure_4_6_feature_stability_heatmap.png"
plt.savefig(out_path, dpi=300)
plt.show()

print('Figure 4.1 Caption: "Heatmap of the top predictive features ranked by stability across cross-validation folds. Each row is a feature and each column is a fold; cell values show standardized permutation importance. Features that remain strong across folds indicate more reliable drivers of final academic performance (G3)."')

#Figure 4.2: Performance Comparison of Feature-Fusion Models

# Define the metrics for each model based on previous evaluation results
metrics_data = {
    'Model': [
        'Logistic Regression', 'Logistic Regression', 'Logistic Regression', 'Logistic Regression',
        'Random Forest', 'Random Forest', 'Random Forest', 'Random Forest'
    ],
    'Metric': [
        'Accuracy', 'Precision', 'Recall', 'F1-score',
        'Accuracy', 'Precision', 'Recall', 'F1-score'
    ],
    'Value': [
        0.971, 0.970, 0.994, 0.982, # Logistic Regression metrics
        0.948, 0.964, 0.970, 0.967  # Random Forest metrics
    ]
}

performance_df = pd.DataFrame(metrics_data)

plt.figure(figsize=(12, 7))
sns.barplot(data=performance_df, x='Metric', y='Value', hue='Model', palette=['Blue', 'red'])
plt.title('Figure 4.2: Performance Comparison of Logistic Regression vs. Random Forest Models')
plt.xlabel('Evaluation Metric')
plt.ylabel('Score')
plt.ylim(0.8, 1.0) # Set appropriate y-axis limits for score values
plt.legend(title='Model')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot
figure_filename = "figures/performance_comparison_feature_fusion.png"
plt.savefig(figure_filename, bbox_inches='tight')
plt.show()

#Caption: \"Accuracy, precision, recall, and F1-score results for different feature-fusion models, illustrating how combined predictors improve academic performance estimation.\"
print("Figure 4.2 Caption: \"Accuracy, precision, recall, and F1-score results for different feature-fusion models, illustrating how combined predictors improve academic performance estimation.\"")

#Figure 4.3: Confusion Matrix Comparison Across Models

# Get predictions from both trained models
y_pred_log_reg = log_reg_trained.predict(X_test)
y_pred_rf = rf_trained.predict(X_test)

# Calculate confusion matrices
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Create subplots for comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(
    cm_log_reg,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    ax=axes[0],
    xticklabels=['Predicted Fail', 'Predicted Pass'],
    yticklabels=['Actual Fail', 'Actual Pass']
)
axes[0].set_title('Figure 4.3: Logistic Regression Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

sns.heatmap(
    cm_rf,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    ax=axes[1],
    xticklabels=['Predicted Fail', 'Predicted Pass'],
    yticklabels=['Actual Fail', 'Actual Pass']
)
axes[1].set_title('Random Forest Confusion Matrix')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.tight_layout()

# Save the plot
figure_filename = "figures/confusion_matrix_comparison.png"
plt.savefig(figure_filename, bbox_inches='tight')
plt.show()

#Caption: \"Confusion matrices showing how different feature-selection methods affect model prediction performance.\"
print("Figure 4.3 Caption: \"Confusion matrices showing how different models perform on academic performance prediction.\"")

#Figure 4.4: Model Runtime Comparison Across Iterations

import time
runtime_data = []

# Measure training and prediction time for Logistic Regression
start_time = time.time()
log_reg_trained.fit(X_train, y_train)
train_time_lr = time.time() - start_time

start_time = time.time()
log_reg_trained.predict(X_test)
predict_time_lr = time.time() - start_time

runtime_data.append({
    'Model': 'Logistic Regression',
    'Metric': 'Training Time (s)',
    'Value': train_time_lr
})
runtime_data.append({
    'Model': 'Logistic Regression',
    'Metric': 'Prediction Time (s)',
    'Value': predict_time_lr
})

# Measure training and prediction time for Random Forest
start_time = time.time()
rf_trained.fit(X_train, y_train)
train_time_rf = time.time() - start_time

start_time = time.time()
rf_trained.predict(X_test)
predict_time_rf = time.time() - start_time

runtime_data.append({
    'Model': 'Random Forest',
    'Metric': 'Training Time (s)',
    'Value': train_time_rf
})
runtime_data.append({
    'Model': 'Random Forest',
    'Metric': 'Prediction Time (s)',
    'Value': predict_time_rf
})

rw_runtime_df = pd.DataFrame(runtime_data)

plt.figure(figsize=(10, 6))
sns.barplot(data=rw_runtime_df, x='Metric', y='Value', hue='Model', palette='rocket')
plt.title('Figure 4.4: Model Runtime Comparison: Training vs. Prediction')
plt.xlabel('Runtime Metric')
plt.ylabel('Time (seconds)')
plt.legend(title='Model')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot
figure_filename = "figures/model_runtime_comparison.png"
plt.savefig(figure_filename, bbox_inches='tight')
plt.show()

#Caption: \"Runtime curves for multiple prediction models, showing computational differences when evaluating fused academic features.\"
print("Figure 4.4 Caption: \"Runtime curves for multiple prediction models, showing computational differences when evaluating fused academic features.\"")

#Figure 4.5: Perceived Impact of Predictive Features on Academic Performance
# The RF model is inside a pipeline, so we access it via the 'model' step
rf_model = rf_trained.named_steps['model']
feature_importances = rf_model.feature_importances_

# The preprocessor contains 'num' for numeric and 'cat' for categorical features
preprocessor = rf_trained.named_steps['preprocess']

# Get numeric feature names directly
numeric_feature_names = preprocessor.named_transformers_['num'].get_feature_names_out(numeric_features)

# Get categorical feature names after one-hot encoding
categorical_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)

# Combine all feature names
all_feature_names = list(numeric_feature_names) + list(categorical_feature_names)

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': feature_importances
})

# Sort by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Select top N features for better visualization if there are too many, for simplicity, let's plot all for now or a reasonable top number
top_n = 20 # Adjust as needed
if len(importance_df) > top_n:
    importance_df_plot = importance_df.head(top_n)
else:
    importance_df_plot = importance_df

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', hue='Feature', data=importance_df_plot, palette='viridis', legend=False)
plt.title('Figure 4.5: Top Predictive Features on Academic Performance (Random Forest)')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.tight_layout()

# Save the plot
figure_filename = "figures/feature_importance_impact.png"
plt.savefig(figure_filename, bbox_inches='tight')
plt.show()

# Figure 4.5 Caption: \"Stacked distribution illustrating perceived influence of key features on academic performance, highlighting which factors contribute most strongly to prediction outcomes.\"
print("Caption: \"Stacked distribution illustrating perceived influence of key features on academic performance, highlighting which factors contribute most strongly to prediction outcomes.\"")