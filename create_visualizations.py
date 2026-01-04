"""
VISUALIZATION SCRIPT FOR CHAPTER 4
====================================
Creates all figures and tables for the Results and Analysis chapter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import json

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("CREATING VISUALIZATIONS FOR CHAPTER 4")
print("=" * 80)

# Load data
print("\n1. Loading data and results...")
df = pd.read_csv('/home/claude/realistic_matched_dataset.csv')
feature_importance = pd.read_csv('/home/claude/feature_importance.csv')
comparison = pd.read_csv('/home/claude/model_comparison.csv')
test_predictions = pd.read_csv('/home/claude/test_predictions.csv')

with open('/home/claude/results_summary.json', 'r') as f:
    results = json.load(f)

print("   ✓ Data loaded")

# ============================================================================
# FIGURE 1: SAMPLE DISTRIBUTION
# ============================================================================

print("\n2. Creating Figure 1: Sample Distribution...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: Distribution by year
year_counts = df['year'].value_counts().sort_index()
axes[0].bar(year_counts.index, year_counts.values, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Year', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Number of Observations', fontsize=12, fontweight='bold')
axes[0].set_title('(A) Sample Distribution by Year', fontsize=13, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Panel B: Distribution by industry
industry_counts = df['industry'].value_counts()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
axes[1].bar(industry_counts.index, industry_counts.values, color=colors, alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Industry', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Number of Observations', fontsize=12, fontweight='bold')
axes[1].set_title('(B) Sample Distribution by Industry', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
axes[1].tick_params(axis='x', rotation=15)

# Panel C: High-risk vs Low-risk
risk_counts = df['high_risk'].value_counts()
risk_labels = ['Low Risk (0)', 'High Risk (1)']
risk_colors = ['#2ECC71', '#E74C3C']
axes[2].bar(risk_labels, [risk_counts[0], risk_counts[1]], color=risk_colors, alpha=0.7, edgecolor='black')
axes[2].set_ylabel('Number of Observations', fontsize=12, fontweight='bold')
axes[2].set_title('(C) Dependent Variable Distribution', fontsize=13, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

# Add percentages on bars
for i, v in enumerate([risk_counts[0], risk_counts[1]]):
    pct = v / len(df) * 100
    axes[2].text(i, v + 5, f'{v}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/figure1_sample_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved figure1_sample_distribution.png")

# ============================================================================
# FIGURE 2: DESCRIPTIVE STATISTICS - BOX PLOTS
# ============================================================================

print("\n3. Creating Figure 2: Textual Features Distribution...")

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()

textual_features = ['climate_keywords', 'pollution_keywords', 'compliance_keywords',
                   'negative_sentiment', 'positive_sentiment', 'flesch_reading_ease',
                   'avg_sentence_length']

feature_labels = {
    'climate_keywords': 'Climate Keywords\n(per 10k words)',
    'pollution_keywords': 'Pollution Keywords\n(per 10k words)',
    'compliance_keywords': 'Compliance Keywords\n(per 10k words)',
    'negative_sentiment': 'Negative Sentiment\n(%)',
    'positive_sentiment': 'Positive Sentiment\n(%)',
    'flesch_reading_ease': 'Flesch Reading Ease\n(Score)',
    'avg_sentence_length': 'Avg Sentence Length\n(words)'
}

for idx, feature in enumerate(textual_features):
    data_to_plot = [df[df['high_risk']==0][feature].dropna(), 
                    df[df['high_risk']==1][feature].dropna()]
    
    bp = axes[idx].boxplot(data_to_plot, labels=['Low Risk', 'High Risk'],
                           patch_artist=True, widths=0.6)
    
    # Color boxes
    bp['boxes'][0].set_facecolor('#2ECC71')
    bp['boxes'][1].set_facecolor('#E74C3C')
    
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    
    axes[idx].set_title(feature_labels[feature], fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Value', fontsize=10)
    axes[idx].grid(axis='y', alpha=0.3)

# Remove extra subplot
axes[7].axis('off')

plt.suptitle('Distribution of Textual Features by Risk Category', 
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('/home/claude/figure2_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved figure2_feature_distributions.png")

# ============================================================================
# FIGURE 3: CONFUSION MATRIX
# ============================================================================

print("\n4. Creating Figure 3: Confusion Matrix...")

cm = results['Confusion_Matrix']
cm_array = np.array([[cm['True_Negatives'], cm['False_Positives']],
                     [cm['False_Negatives'], cm['True_Positives']]])

fig, ax = plt.subplots(figsize=(8, 6))

# Create heatmap
sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', cbar=True,
            square=True, linewidths=2, linecolor='black',
            xticklabels=['Predicted Low', 'Predicted High'],
            yticklabels=['Actual Low', 'Actual High'],
            annot_kws={'size': 16, 'weight': 'bold'})

plt.title('Confusion Matrix - Test Set (2020)', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Actual Risk', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Risk', fontsize=12, fontweight='bold')

# Add accuracy annotation
accuracy = results['Test_Performance']['Accuracy']
plt.text(1, -0.3, f'Overall Accuracy: {accuracy:.2%}', 
         ha='center', fontsize=12, fontweight='bold', transform=ax.transData)

plt.tight_layout()
plt.savefig('/home/claude/figure3_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved figure3_confusion_matrix.png")

# ============================================================================
# FIGURE 4: ROC CURVE
# ============================================================================

print("\n5. Creating Figure 4: ROC Curve...")

# Load test data to recreate ROC curve
test_df = df[df['year'] == 2020]
y_test = test_df['high_risk']
y_pred_proba = test_predictions['predicted_probability']

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = results['Test_Performance']['ROC_AUC']

fig, ax = plt.subplots(figsize=(8, 8))

# Plot ROC curve
ax.plot(fpr, tpr, color='#E74C3C', linewidth=3, 
        label=f'Random Forest (AUC = {roc_auc:.3f})')

# Plot diagonal (random classifier)
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.500)')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve - Random Forest Model', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/figure4_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved figure4_roc_curve.png")

# ============================================================================
# FIGURE 5: FEATURE IMPORTANCE
# ============================================================================

print("\n6. Creating Figure 5: Feature Importance...")

# Top 10 features
top_features = feature_importance.head(10).copy()

fig, ax = plt.subplots(figsize=(10, 7))

colors_gradient = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(top_features)))

bars = ax.barh(range(len(top_features)), top_features['Importance_Percent'], 
               color=colors_gradient, edgecolor='black', linewidth=1.5)

ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'])
ax.invert_yaxis()
ax.set_xlabel('Feature Importance (%)', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)

# Add percentage labels on bars
for i, (idx, row) in enumerate(top_features.iterrows()):
    ax.text(row['Importance_Percent'] + 0.3, i, f"{row['Importance_Percent']:.2f}%",
            va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('/home/claude/figure5_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved figure5_feature_importance.png")

# ============================================================================
# FIGURE 6: MODEL COMPARISON
# ============================================================================

print("\n7. Creating Figure 6: Model Performance Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Bar chart comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.25

rf_values = [comparison[comparison['Model']=='Random Forest'][m].values[0] for m in metrics]
naive_values = [comparison[comparison['Model']=='Naive Baseline'][m].values[0] for m in metrics]
industry_values = [comparison[comparison['Model']=='Industry Heuristic'][m].values[0] for m in metrics]

axes[0].bar(x - width, naive_values, width, label='Naive Baseline', 
           color='#95A5A6', alpha=0.8, edgecolor='black')
axes[0].bar(x, industry_values, width, label='Industry Heuristic', 
           color='#3498DB', alpha=0.8, edgecolor='black')
axes[0].bar(x + width, rf_values, width, label='Random Forest', 
           color='#E74C3C', alpha=0.8, edgecolor='black')

axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
axes[0].set_title('(A) Performance Metrics Comparison', fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics)
axes[0].legend(fontsize=10, frameon=True, shadow=True)
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim([0, 1.0])

# Panel B: F1-Score focus
models = ['Naive\nBaseline', 'Industry\nHeuristic', 'Random\nForest']
f1_scores = [naive_values[3], industry_values[3], rf_values[3]]
colors_f1 = ['#95A5A6', '#3498DB', '#E74C3C']

bars = axes[1].bar(models, f1_scores, color=colors_f1, alpha=0.8, 
                   edgecolor='black', linewidth=2)

axes[1].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
axes[1].set_title('(B) F1-Score Comparison', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_ylim([0, 0.6])

# Add value labels on bars
for bar, value in zip(bars, f1_scores):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('/home/claude/figure6_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved figure6_model_comparison.png")

# ============================================================================
# FIGURE 7: PERFORMANCE BY INDUSTRY
# ============================================================================

print("\n8. Creating Figure 7: Performance by Industry...")

industry_perf = pd.read_csv('/home/claude/performance_by_industry.csv')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

industries = industry_perf['Industry']
colors_ind = ['#FF6B6B', '#45B7D1', '#4ECDC4']

# Panel A: Accuracy by industry
axes[0].bar(industries, industry_perf['Accuracy'], color=colors_ind, 
           alpha=0.8, edgecolor='black', linewidth=2)
axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title('(A) Accuracy by Industry', fontsize=13, fontweight='bold')
axes[0].set_ylim([0, 1.0])
axes[0].grid(axis='y', alpha=0.3)

for i, (ind, acc) in enumerate(zip(industries, industry_perf['Accuracy'])):
    axes[0].text(i, acc + 0.03, f'{acc:.3f}', ha='center', 
                fontweight='bold', fontsize=11)

# Panel B: F1-Score by industry
axes[1].bar(industries, industry_perf['F1_Score'], color=colors_ind, 
           alpha=0.8, edgecolor='black', linewidth=2)
axes[1].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
axes[1].set_title('(B) F1-Score by Industry', fontsize=13, fontweight='bold')
axes[1].set_ylim([0, 0.6])
axes[1].grid(axis='y', alpha=0.3)

for i, (ind, f1) in enumerate(zip(industries, industry_perf['F1_Score'])):
    axes[1].text(i, f1 + 0.02, f'{f1:.3f}', ha='center', 
                fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('/home/claude/figure7_performance_by_industry.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved figure7_performance_by_industry.png")

# ============================================================================
# FIGURE 8: CORRELATION HEATMAP
# ============================================================================

print("\n9. Creating Figure 8: Feature Correlation Matrix...")

features_for_corr = ['climate_keywords', 'pollution_keywords', 'compliance_keywords',
                     'negative_sentiment', 'positive_sentiment', 'flesch_reading_ease',
                     'avg_sentence_length', 'log_total_assets', 'roa', 'leverage']

corr_matrix = df[features_for_corr].corr()

fig, ax = plt.subplots(figsize=(12, 10))

# Create heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, linecolor='black',
            cbar_kws={'label': 'Correlation Coefficient'},
            annot_kws={'size': 9, 'weight': 'bold'})

plt.title('Correlation Matrix of Features', fontsize=14, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('/home/claude/figure8_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved figure8_correlation_matrix.png")

print("\n" + "=" * 80)
print("✓ ALL VISUALIZATIONS CREATED!")
print("=" * 80)

print("\nFigures created:")
print("1. figure1_sample_distribution.png - Sample breakdown by year/industry/risk")
print("2. figure2_feature_distributions.png - Box plots of textual features")
print("3. figure3_confusion_matrix.png - Model confusion matrix")
print("4. figure4_roc_curve.png - ROC curve with AUC")
print("5. figure5_feature_importance.png - Top 10 feature importance")
print("6. figure6_model_comparison.png - Performance vs baselines")
print("7. figure7_performance_by_industry.png - Industry-specific performance")
print("8. figure8_correlation_matrix.png - Feature correlation heatmap")

print("\nAll figures are publication-ready at 300 DPI!")
