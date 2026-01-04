"""
RANDOM FOREST MODEL BUILDING AND ANALYSIS
Business Analytics Project: Predicting Corporate Environmental Risk
====================================================================

This script builds the Random Forest classifier, evaluates performance,
and generates all tables and figures for Chapter 4.

Author: Adebayo
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             roc_curve, accuracy_score, precision_score, recall_score, 
                             f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("RANDOM FOREST MODEL BUILDING AND ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

print("\n1. Loading data...")
df = pd.read_csv('/home/hp/matched_dataset.csv')

print(f"   ✓ Loaded {len(df)} observations")
print(f"   Columns: {len(df.columns)}")
print(f"   Time period: {df['year'].min()}-{df['year'].max()}")

# Define features
print("\n2. Defining features...")

textual_features = [
    'climate_keywords',
    'pollution_keywords', 
    'compliance_keywords',
    'negative_sentiment',
    'positive_sentiment',
    'flesch_reading_ease',
    'avg_sentence_length'
]

control_variables = [
    'log_total_assets',
    'roa',
    'leverage'
]

all_features = textual_features + control_variables

print(f"   Textual features: {len(textual_features)}")
print(f"   Control variables: {len(control_variables)}")
print(f"   Total features: {len(all_features)}")

# Create feature matrix and target
X = df[all_features]
y = df['high_risk']

print(f"\n   Feature matrix shape: {X.shape}")
print(f"   Target distribution:")
print(f"   - Low Risk (0): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
print(f"   - High Risk (1): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")

# ============================================================================
# 2. TEMPORAL TRAIN-VALIDATION-TEST SPLIT
# ============================================================================

print("\n3. Creating temporal train-validation-test split...")

# 2015-2018: Training (67%)
# 2019: Validation (17%)
# 2020: Testing (16%)

train_df = df[df['year'].isin([2015, 2016, 2017, 2018])].copy()
val_df = df[df['year'] == 2019].copy()
test_df = df[df['year'] == 2020].copy()

X_train = train_df[all_features]
y_train = train_df['high_risk']

X_val = val_df[all_features]
y_val = val_df['high_risk']

X_test = test_df[all_features]
y_test = test_df['high_risk']

print(f"\n   Training set (2015-2018): {len(train_df)} obs, {y_train.sum()} high-risk ({y_train.mean()*100:.1f}%)")
print(f"   Validation set (2019):    {len(val_df)} obs, {y_val.sum()} high-risk ({y_val.mean()*100:.1f}%)")
print(f"   Test set (2020):          {len(test_df)} obs, {y_test.sum()} high-risk ({y_test.mean()*100:.1f}%)")

# ============================================================================
# 3. HYPERPARAMETER OPTIMIZATION
# ============================================================================

print("\n4. Performing hyperparameter optimization...")
print("   Using GridSearchCV with 3-fold cross-validation on training set")

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}

print(f"\n   Parameter grid size: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['class_weight'])} combinations")

# Create base model
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

# Grid search
grid_search = GridSearchCV(
    rf_base,
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)

print("\n   Running grid search (this may take a few minutes)...")
grid_search.fit(X_train, y_train)

print(f"\n   ✓ Grid search complete!")
print(f"   Best F1-score (CV): {grid_search.best_score_:.4f}")
print(f"   Best parameters:")
for param, value in grid_search.best_params_.items():
    print(f"     - {param}: {value}")

# Get best model
best_rf = grid_search.best_estimator_

# Save hyperparameter results
grid_results = pd.DataFrame(grid_search.cv_results_)
grid_results_top = grid_results.nlargest(10, 'mean_test_score')[
    ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
].copy()
grid_results_top.to_csv('/home/hp/hyperparameter_results.csv', index=False)
print(f"\n   ✓ Saved top 10 hyperparameter configurations")

# ============================================================================
# 4. MODEL TRAINING AND VALIDATION
# ============================================================================

print("\n5. Training final model with best parameters...")

# Validation performance
val_pred = best_rf.predict(X_val)
val_pred_proba = best_rf.predict_proba(X_val)[:, 1]

val_f1 = f1_score(y_val, val_pred)
val_precision = precision_score(y_val, val_pred)
val_recall = recall_score(y_val, val_pred)

print(f"\n   Validation set (2019) performance:")
print(f"   - F1-Score:  {val_f1:.4f}")
print(f"   - Precision: {val_precision:.4f}")
print(f"   - Recall:    {val_recall:.4f}")

# ============================================================================
# 5. TEST SET EVALUATION
# ============================================================================

print("\n6. Evaluating on test set (2020)...")

# Predictions
test_pred = best_rf.predict(X_test)
test_pred_proba = best_rf.predict_proba(X_test)[:, 1]

# Calculate metrics
test_accuracy = accuracy_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred, zero_division=0)
test_recall = recall_score(y_test, test_pred, zero_division=0)
test_f1 = f1_score(y_test, test_pred, zero_division=0)
test_roc_auc = roc_auc_score(y_test, test_pred_proba)

# Confusion matrix
cm = confusion_matrix(y_test, test_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\n   TEST SET PERFORMANCE:")
print(f"   {'=' * 50}")
print(f"   Accuracy:   {test_accuracy:.4f}")
print(f"   Precision:  {test_precision:.4f}")
print(f"   Recall:     {test_recall:.4f}")
print(f"   F1-Score:   {test_f1:.4f}")
print(f"   ROC-AUC:    {test_roc_auc:.4f}")
print(f"   Specificity:{specificity:.4f}")
print(f"   {'=' * 50}")

print(f"\n   Confusion Matrix:")
print(f"                 Predicted")
print(f"                 Low  High")
print(f"   Actual Low    {tn:3d}  {fp:3d}")
print(f"   Actual High   {fn:3d}  {tp:3d}")

# ============================================================================
# 6. BASELINE COMPARISONS
# ============================================================================

print("\n7. Comparing with baseline models...")

# Naive baseline (always predict majority class)
naive_classifier = DummyClassifier(strategy='most_frequent', random_state=42)
naive_classifier.fit(X_train, y_train)
naive_pred = naive_classifier.predict(X_test)

naive_accuracy = accuracy_score(y_test, naive_pred)
naive_precision = precision_score(y_test, naive_pred, zero_division=0)
naive_recall = recall_score(y_test, naive_pred, zero_division=0)
naive_f1 = f1_score(y_test, naive_pred, zero_division=0)

print(f"\n   Naive Baseline (Always predict majority):")
print(f"   - Accuracy: {naive_accuracy:.4f}")
print(f"   - F1-Score: {naive_f1:.4f}")

# Industry heuristic (predict high-risk for mining/chemicals, low for utilities)
industry_pred = test_df['industry'].apply(
    lambda x: 1 if x in ['Mining', 'Chemicals'] else 0
).values

industry_accuracy = accuracy_score(y_test, industry_pred)
industry_precision = precision_score(y_test, industry_pred, zero_division=0)
industry_recall = recall_score(y_test, industry_pred, zero_division=0)
industry_f1 = f1_score(y_test, industry_pred, zero_division=0)

print(f"\n   Industry Heuristic Baseline:")
print(f"   - Accuracy: {industry_accuracy:.4f}")
print(f"   - F1-Score: {industry_f1:.4f}")

# Compare improvements
f1_improvement_naive = ((test_f1 - naive_f1) / naive_f1 * 100) if naive_f1 > 0 else float('inf')
f1_improvement_industry = ((test_f1 - industry_f1) / industry_f1 * 100) if industry_f1 > 0 else float('inf')

print(f"\n   IMPROVEMENTS OVER BASELINES:")
print(f"   - vs Naive: {f1_improvement_naive:.1f}% improvement")
print(f"   - vs Industry Heuristic: {f1_improvement_industry:.1f}% improvement")

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': ['Naive Baseline', 'Industry Heuristic', 'Random Forest'],
    'Accuracy': [naive_accuracy, industry_accuracy, test_accuracy],
    'Precision': [naive_precision, industry_precision, test_precision],
    'Recall': [naive_recall, industry_recall, test_recall],
    'F1-Score': [naive_f1, industry_f1, test_f1],
    'ROC-AUC': [0.5, '-', test_roc_auc]
})

comparison_df.to_csv('/home/hp/model_comparison.csv', index=False)
print(f"\n   ✓ Saved model comparison table")

# ============================================================================
# 7. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print("\n8. Analyzing feature importance...")

# Get feature importances
feature_importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': best_rf.feature_importances_
}).sort_values('Importance', ascending=False)

feature_importance['Importance_Percent'] = feature_importance['Importance'] * 100

print(f"\n   TOP 10 MOST IMPORTANT FEATURES:")
print(f"   {'=' * 60}")
for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['Feature']:25s} {row['Importance_Percent']:6.2f}%")
print(f"   {'=' * 60}")

feature_importance.to_csv('/home/hp/feature_importance.csv', index=False)
print(f"\n   ✓ Saved complete feature importance rankings")

# ============================================================================
# 8. ROBUSTNESS CHECKS
# ============================================================================

print("\n9. Performing robustness checks...")

# Performance by industry
print(f"\n   Performance by Industry:")
print(f"   {'=' * 60}")

industry_results = []
for industry in test_df['industry'].unique():
    mask = test_df['industry'] == industry
    if mask.sum() > 0:
        y_ind = y_test[mask]
        pred_ind = test_pred[mask]
        
        if len(y_ind) > 0 and y_ind.sum() > 0:  # Only if we have positive cases
            ind_f1 = f1_score(y_ind, pred_ind, zero_division=0)
            ind_acc = accuracy_score(y_ind, pred_ind)
            
            industry_results.append({
                'Industry': industry,
                'N': mask.sum(),
                'High_Risk_Count': y_ind.sum(),
                'Accuracy': ind_acc,
                'F1_Score': ind_f1
            })
            
            print(f"   {industry:15s} N={mask.sum():3d}, Acc={ind_acc:.3f}, F1={ind_f1:.3f}")

industry_results_df = pd.DataFrame(industry_results)
industry_results_df.to_csv('/home/hp/performance_by_industry.csv', index=False)

print(f"   {'=' * 60}")
print(f"   ✓ Saved industry-specific performance")

# ============================================================================
# 9. SAVE PREDICTIONS AND RESULTS
# ============================================================================

print("\n10. Saving predictions and results...")

# Save test set with predictions
test_results = test_df.copy()
test_results['predicted_risk'] = test_pred
test_results['predicted_probability'] = test_pred_proba
test_results['correct_prediction'] = (test_pred == y_test).astype(int)

test_results.to_csv('/home/hp/test_predictions.csv', index=False)
print(f"   ✓ Saved test set predictions")

# Create comprehensive results summary
results_summary = {
    'Dataset': {
        'Total_Observations': len(df),
        'Training_Observations': len(train_df),
        'Validation_Observations': len(val_df),
        'Test_Observations': len(test_df),
        'High_Risk_Percentage': f"{y.mean()*100:.1f}%"
    },
    'Best_Hyperparameters': grid_search.best_params_,
    'Test_Performance': {
        'Accuracy': round(test_accuracy, 4),
        'Precision': round(test_precision, 4),
        'Recall': round(test_recall, 4),
        'F1_Score': round(test_f1, 4),
        'ROC_AUC': round(test_roc_auc, 4),
        'Specificity': round(specificity, 4)
    },
    'Baseline_Comparison': {
        'Naive_F1': round(naive_f1, 4),
        'Industry_F1': round(industry_f1, 4),
        'RF_F1': round(test_f1, 4),
        'Improvement_vs_Naive_pct': round(f1_improvement_naive, 1),
        'Improvement_vs_Industry_pct': round(f1_improvement_industry, 1)
    },
    'Confusion_Matrix': {
        'True_Negatives': int(tn),
        'False_Positives': int(fp),
        'False_Negatives': int(fn),
        'True_Positives': int(tp)
    }
}

import json
with open('/home/hp/results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"   ✓ Saved comprehensive results summary")

print("\n" + "=" * 80)
print("✓ MODEL BUILDING AND ANALYSIS COMPLETE!")
print("=" * 80)

print("\nFiles created:")
print("1. hyperparameter_results.csv - Top 10 parameter combinations")
print("2. model_comparison.csv - Performance vs baselines")
print("3. feature_importance.csv - Feature importance rankings")
print("4. performance_by_industry.csv - Industry-specific performance")
print("5. test_predictions.csv - Predictions on test set")
print("6. results_summary.json - Comprehensive results")

print("\nNext: Run visualization script to create all figures...")
