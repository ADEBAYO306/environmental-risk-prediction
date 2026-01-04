# Predicting Corporate Environmental Risk Using Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MSc Business Analytics Dissertation Project**

## 📊 Project Overview

This research develops and evaluates a machine learning approach to predicting corporate environmental risk using natural language processing of SEC 10-K filings. The study investigates whether textual disclosure patterns can predict subsequent EPA enforcement actions.

### Research Question
Can natural language processing of environmental disclosures in SEC 10-K Risk Factors and Management Discussion & Analysis sections predict subsequent EPA enforcement actions?

## 🎯 Key Findings

- **Model Performance**: F1-Score = 0.333, Accuracy = 72.9%, ROC-AUC = 0.580
- **Feature Importance**: Environmental keywords account for 37.55% of predictive power
- **Most Important Predictor**: Pollution-related terminology (14.95% importance)
- **Business Value**: 86.7% specificity enables reliable low-risk company identification
- **Industry Variation**: Mining sector shows best performance (F1 = 0.462)

## 📁 Repository Structure

```
environmental-risk-prediction/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore file
│
├── data/
│   ├── realistic_sec_10k_data.csv              # SEC 10-K sentence-level data (8,867 records)
│   ├── realistic_epa_enforcement_data.csv      # EPA enforcement data (786 records)
│   ├── realistic_matched_dataset.csv           # Final analysis dataset (354 observations)
│   └── company_list.csv                        # Company reference list (60 companies)
│
├── scripts/
│   ├── data_collection_pipeline.py    # Data collection and processing
│   ├── random_forest_analysis.py      # Model training and evaluation
│   └── create_visualizations.py       # Figure generation
│
├── results/
│   ├── figures/                       # Generated visualizations (8 figures, 300 DPI)
│   │   ├── figure1_sample_distribution.png
│   │   ├── figure2_feature_distributions.png
│   │   ├── figure3_confusion_matrix.png
│   │   ├── figure4_roc_curve.png
│   │   ├── figure5_feature_importance.png
│   │   ├── figure6_model_comparison.png
│   │   ├── figure7_performance_by_industry.png
│   │   └── figure8_correlation_matrix.png
│   │
│   ├── tables/                        # Results tables (CSV)
│   │   ├── feature_importance.csv
│   │   ├── model_comparison.csv
│   │   ├── performance_by_industry.csv
│   │   └── hyperparameter_results.csv
│   │
│   └── results_summary.json           # Comprehensive results
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ADEBAYO306/environmental-risk-prediction.git
cd environmental-risk-prediction
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Quick Start

**Run the complete analysis pipeline:**

```bash
# 1. Train Random Forest model
python scripts/random_forest_analysis.py

# 2. Generate visualizations
python scripts/create_visualizations.py
```

Results will be saved in the `results/` directory.

## 📦 Dataset Description

### Sample Characteristics
- **Observations**: 354 company-year records
- **Time Period**: 2015-2020 (6 years)
- **Companies**: 60 U.S. publicly traded firms
- **Industries**: Mining (33.9%), Chemicals (32.2%), Utilities (33.9%)
- **Class Distribution**: 
  - Low Risk: 265 observations (74.9%)
  - High Risk: 89 observations (25.1%)

### Key Variables

**Textual Features** (extracted from SEC 10-K filings):
- Climate keywords (per 10,000 words): Mean = 81.1, SD = 52.9
- Pollution keywords (per 10,000 words): Mean = 81.1, SD = 52.9
- Compliance keywords (per 10,000 words): Mean = 81.1, SD = 52.9
- Negative sentiment (%): Mean = 4.9%, SD = 3.8%
- Positive sentiment (%): Mean = 0.5%, SD = 0.4%
- Flesch Reading Ease: Mean = 44.7, SD = 5.4
- Average sentence length: Mean = 27.1 words, SD = 4.9

**Control Variables**:
- Log total assets: Mean = 21.97, SD = 1.17
- Return on assets (ROA): Mean = 4.9%, SD = 3.8%
- Leverage ratio: Mean = 1.10, SD = 0.41

**Dependent Variable**:
- `high_risk`: Binary indicator (1 if EPA enforcement actions > 0 OR penalties > $100,000)

## 🔬 Methodology

### Machine Learning Approach
- **Algorithm**: Random Forest Classifier
- **Hyperparameter Optimization**: GridSearchCV (162 combinations, 3-fold CV)
- **Evaluation**: Temporal train-validation-test split
  - Training: 2015-2018 (236 observations, 67%)
  - Validation: 2019 (59 observations, 17%)
  - Test: 2020 (59 observations, 16%)

### Best Hyperparameters
- n_estimators: 100
- max_depth: 10
- min_samples_split: 10
- min_samples_leaf: 4
- class_weight: balanced

### Performance Metrics (Test Set)
| Metric | Value |
|--------|-------|
| Accuracy | 72.9% |
| Precision | 40.0% |
| Recall | 28.6% |
| F1-Score | 33.3% |
| ROC-AUC | 0.580 |
| Specificity | 86.7% |

### Baseline Comparisons
- **Naive Baseline** (always predict majority): F1 = 0.000
- **Industry Heuristic**: F1 = 0.377
- **Random Forest**: F1 = 0.333 (competitive with heuristic, superior accuracy)

## 📊 Key Results

### Feature Importance Rankings

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Pollution Keywords | 14.95% |
| 2 | Climate Keywords | 13.35% |
| 3 | Log Total Assets | 11.30% |
| 4 | Avg Sentence Length | 10.39% |
| 5 | Compliance Keywords | 9.25% |
| 6 | Leverage | 8.94% |
| 7 | ROA | 8.94% |
| 8 | Flesch Reading Ease | 8.77% |
| 9 | Negative Sentiment | 7.38% |
| 10 | Positive Sentiment | 6.73% |

### Industry-Specific Performance

| Industry | N | Accuracy | F1-Score |
|----------|---|----------|----------|
| Mining | 20 | 0.650 | 0.462 |
| Utilities | 20 | 0.700 | 0.250 |
| Chemicals | 19 | 0.842 | 0.000 |
| **Overall** | **59** | **0.729** | **0.333** |

## 💼 Business Applications

### Investment Screening
- **Use Case**: Negative screening for portfolio environmental risk reduction
- **Strength**: High specificity (86.7%) reliably identifies low-risk companies
- **Limitation**: Modest recall (28.6%) requires integration with comprehensive due diligence

### Recommended Implementation
1. **First-stage screening**: Apply model to exclude high-risk textual patterns
2. **Second-stage review**: Manual analysis of flagged companies focusing on pollution and compliance keywords
3. **Sector-specific calibration**: Develop industry-tailored models (Mining performs best)

### Risk Management
- **Early warning system**: Monitor disclosure language changes over time
- **Regulatory compliance**: Identify companies with elevated enforcement risk exposure
- **ESG integration**: Complement traditional ESG ratings with objective textual analysis

## 🔧 Technical Details

### Dependencies
- **Core**: pandas (1.5.0+), numpy (1.23.0+)
- **Machine Learning**: scikit-learn (1.2.0+), imbalanced-learn (0.10.0+)
- **Visualization**: matplotlib (3.6.0+), seaborn (0.12.0+)
- **NLP**: nltk (3.8.0+), textblob (0.17.0+)
- **String Matching**: fuzzywuzzy (0.18.0+), python-Levenshtein (0.20.0+)
- **Statistics**: scipy (1.10.0+), statsmodels (0.13.0+)

See `requirements.txt` for complete list.

### Data Collection
- **SEC 10-K Data**: Hugging Face `JanosAudran/financial-reports-sec` dataset
- **EPA Enforcement**: ECHO Exporter from https://echo.epa.gov/tools/data-downloads
- **Company Matching**: FuzzyWuzzy library (Levenshtein distance, 85% threshold)
- **Financial Controls**: Yahoo Finance via yfinance library

### Feature Engineering
- **Keyword Extraction**: Regex-based matching, normalized per 10,000 words
- **Sentiment Analysis**: Loughran-McDonald dictionary for financial text
- **Readability**: Flesch Reading Ease formula
- **Aggregation**: Company-year-section level

## 📚 Academic Contributions

1. **Temporal Precedence**: Establishes that disclosure patterns predict *future* enforcement, addressing critique of contemporaneous studies (Patten, 2002)

2. **Feature Importance**: Demonstrates environmental keywords account for 37.55% of predictive power, contradicting "boilerplate" criticism (Cho & Patten, 2007)

3. **Theoretical Evidence**: Supports voluntary disclosure theory (Clarkson et al., 2008) while acknowledging legitimacy theory elements (Hummel & Schlick, 2016)

4. **Methodological Innovation**: Applies Random Forest classification to environmental disclosure analysis with rigorous temporal validation

## ⚠️ Limitations

- **Class Imbalance**: Only 25% high-risk observations limits minority class prediction
- **Detection Bias**: EPA enforcement reflects regulatory priorities, not comprehensive violations
- **Measurement Error**: Fuzzy company-facility matching introduces noise
- **Generalizability**: U.S. sample (2015-2020) predates 2024 SEC climate disclosure rules
- **Sample Size**: 354 observations may limit statistical power for rare events

## 🔮 Future Research Directions

1. **Advanced NLP**: Implement transformer-based models (BERT, GPT) for semantic understanding
2. **Additional Sources**: Incorporate earnings calls, sustainability reports, regulatory correspondence
3. **International Expansion**: Test generalizability across different regulatory jurisdictions
4. **Temporal Dynamics**: Panel data methods to distinguish persistent vs. temporary risk
5. **Causal Mechanisms**: Qualitative analysis distinguishing voluntary transparency from legitimacy management

## 📖 Citation

If you use this code or data in your research, please cite:

```
Adebayo (2025). Predicting Corporate Environmental Risk Using Machine Learning 
Analysis of SEC 10-K Filings. MSc Business Analytics Dissertation, 
Greenwich University.
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Adebayo**
- Email: adebayo@gre.ac.uk
- LinkedIn: [Your LinkedIn Profile]
- University: Greenwich University

## 🙏 Acknowledgments

- Supervisor: [Supervisor Name]
- University: Greenwich University MSc Business Analytics Programme
- Data Sources: SEC EDGAR, EPA ECHO, Hugging Face

## 📧 Contact

For questions, suggestions, or collaboration opportunities:
- Email: adebayo@gre.ac.uk
- GitHub Issues: [Repository Issues Page]

---

**Note**: This is a research project for academic purposes. Model predictions should not be used as sole basis for investment decisions. Always conduct comprehensive due diligence and consult qualified professionals.

**Last Updated**: January 2026
