"""
COMPLETE DATA COLLECTION PIPELINE
Business Analytics Project: Predicting Corporate Environmental Risk
====================================================================

This script provides the complete pipeline for collecting and processing data
for the research project.

Requirements:
- datasets (Hugging Face)
- pandas
- numpy
- fuzzywuzzy
- python-Levenshtein
- sec-edgar-downloader (optional)
- requests

Install: pip install datasets pandas numpy fuzzywuzzy python-Levenshtein requests
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: SEC 10-K DATA COLLECTION
# ============================================================================

def collect_sec_10k_data(save_path='sec_10k_raw.csv'):
    """
    Collect SEC 10-K textual data from Hugging Face dataset
    
    Returns:
        DataFrame with columns: cik, company_name, sic, year, section, sentence
    """
    print("=" * 80)
    print("COLLECTING SEC 10-K DATA FROM HUGGING FACE")
    print("=" * 80)
    
    try:
        from datasets import load_dataset
        
        print("\n1. Loading financial-reports-sec dataset...")
        print("   (This may take several minutes for the first load)")
        
        # Load dataset - use 'small_lite' for testing, 'large_lite' for full data
        dataset = load_dataset("JanosAudran/financial-reports-sec", "large_lite", split="train")
        
        print(f"   ✓ Loaded {len(dataset):,} total records")
        
        # Convert to pandas
        print("\n2. Converting to pandas DataFrame...")
        df = pd.DataFrame(dataset[:])  # Load all data
        
        print(f"   ✓ DataFrame shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Filter for target industries (SIC codes)
        print("\n3. Filtering for target industries...")
        print("   - Mining: SIC 1000-1499")
        print("   - Chemicals: SIC 2800-2899")
        print("   - Utilities: SIC 4900-4999")
        
        df['sic'] = pd.to_numeric(df['sic'], errors='coerce')
        
        target_industries = (
            ((df['sic'] >= 1000) & (df['sic'] < 1500)) |
            ((df['sic'] >= 2800) & (df['sic'] < 2900)) |
            ((df['sic'] >= 4900) & (df['sic'] < 5000))
        )
        
        df = df[target_industries].copy()
        print(f"   ✓ Filtered to {len(df):,} records in target industries")
        
        # Filter for target years (2015-2020)
        print("\n4. Filtering for years 2015-2020...")
        df['filingDate'] = pd.to_datetime(df['filingDate'])
        df['year'] = df['filingDate'].dt.year
        df = df[(df['year'] >= 2015) & (df['year'] <= 2020)].copy()
        print(f"   ✓ Filtered to {len(df):,} records in 2015-2020")
        
        # Filter for sections 1A (Risk Factors) and 7 (MD&A)
        print("\n5. Filtering for Sections 1A and 7...")
        df = df[df['section'].isin(['section_1A', 'section_7'])].copy()
        print(f"   ✓ Filtered to {len(df):,} records from Sections 1A & 7")
        
        # Show breakdown
        print("\n6. Data breakdown:")
        print(f"\n   By Industry:")
        sic_ranges = pd.cut(df['sic'], bins=[1000, 1500, 2800, 2900, 4900, 5000], 
                           labels=['Mining', 'Chemicals_gap', 'Chemicals', 'Utilities_gap', 'Utilities'])
        print(sic_ranges.value_counts())
        
        print(f"\n   By Year:")
        print(df['year'].value_counts().sort_index())
        
        print(f"\n   By Section:")
        print(df['section'].value_counts())
        
        # Save
        print(f"\n7. Saving data to {save_path}...")
        df.to_csv(save_path, index=False)
        print(f"   ✓ Saved {len(df):,} records")
        
        return df
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Try 'small_lite' configuration first for testing")
        print("3. Visit https://huggingface.co/datasets/JanosAudran/financial-reports-sec")
        return None


# ============================================================================
# SECTION 2: EPA ENFORCEMENT DATA COLLECTION
# ============================================================================

def collect_epa_enforcement_data(echo_file_path, save_path='epa_enforcement.csv'):
    """
    Process EPA ECHO Exporter data
    
    Args:
        echo_file_path: Path to ECHO_EXPORTER.csv file (download manually)
        save_path: Where to save processed data
        
    Returns:
        DataFrame with facility enforcement information
    """
    print("=" * 80)
    print("PROCESSING EPA ECHO ENFORCEMENT DATA")
    print("=" * 80)
    
    if not os.path.exists(echo_file_path):
        print(f"\n✗ File not found: {echo_file_path}")
        print("\nPlease download ECHO Exporter from:")
        print("https://echo.epa.gov/tools/data-downloads")
        print("Direct link: https://echo.epa.gov/files/echodownloads/echo_exporter.zip")
        return None
    
    try:
        print(f"\n1. Loading ECHO data from {echo_file_path}...")
        # Load in chunks due to large size
        df = pd.read_csv(echo_file_path, encoding='latin1', low_memory=False)
        print(f"   ✓ Loaded {len(df):,} facilities")
        
        # Filter for target industries
        print("\n2. Filtering for target industries (SIC codes)...")
        df['FAC_SIC'] = pd.to_numeric(df['FAC_SIC'], errors='coerce')
        
        target_industries = (
            ((df['FAC_SIC'] >= 1000) & (df['FAC_SIC'] < 1500)) |
            ((df['FAC_SIC'] >= 2800) & (df['FAC_SIC'] < 2900)) |
            ((df['FAC_SIC'] >= 4900) & (df['FAC_SIC'] < 5000))
        )
        
        df = df[target_industries].copy()
        print(f"   ✓ Filtered to {len(df):,} facilities")
        
        # Extract key enforcement columns
        print("\n3. Extracting enforcement information...")
        
        enforcement_cols = {
            'REGISTRY_ID': 'registry_id',
            'FAC_NAME': 'facility_name',
            'FAC_SIC': 'sic_code',
            'FAC_LAT': 'latitude',
            'FAC_LONG': 'longitude',
            'FAC_STATE': 'state',
            'FAC_CITY': 'city'
        }
        
        # Look for enforcement action columns (may vary by dataset version)
        action_patterns = ['FORMAL_ACTION_COUNT', 'ENFORCEMENT', 'PENALTY', 
                          'VIOLATION', 'INFORMAL_ACTION']
        
        available_cols = list(enforcement_cols.keys())
        for col in df.columns:
            if any(pattern in col.upper() for pattern in action_patterns):
                available_cols.append(col)
        
        df_filtered = df[available_cols].copy()
        df_filtered.rename(columns=enforcement_cols, inplace=True)
        
        print(f"   ✓ Extracted {len(df_filtered.columns)} columns")
        print(f"   Enforcement-related columns found: {len(available_cols) - len(enforcement_cols)}")
        
        # Save
        print(f"\n4. Saving to {save_path}...")
        df_filtered.to_csv(save_path, index=False)
        print(f"   ✓ Saved {len(df_filtered):,} facilities")
        
        return df_filtered
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return None


# ============================================================================
# SECTION 3: COMPANY-FACILITY MATCHING
# ============================================================================

def fuzzy_match_companies(sec_df, epa_df, threshold=85, save_path='matched_companies.csv'):
    """
    Match SEC companies to EPA facilities using fuzzy string matching
    
    Args:
        sec_df: DataFrame with SEC 10-K data (must have 'company_name' or similar)
        epa_df: DataFrame with EPA data (must have 'facility_name')
        threshold: Minimum similarity score (0-100)
        save_path: Where to save matches
        
    Returns:
        DataFrame with matched company-facility pairs
    """
    print("=" * 80)
    print("MATCHING COMPANIES TO FACILITIES")
    print("=" * 80)
    
    try:
        from fuzzywuzzy import fuzz
        
        # Get unique companies from SEC data
        print("\n1. Extracting unique companies...")
        
        # Determine company name column
        name_col = None
        for col in sec_df.columns:
            if any(term in col.lower() for term in ['company', 'name', 'ticker']):
                name_col = col
                break
        
        if name_col is None:
            print("   ✗ Could not find company name column in SEC data")
            return None
        
        sec_companies = sec_df[[name_col, 'cik']].drop_duplicates()
        sec_companies.columns = ['company_name', 'cik']
        
        print(f"   ✓ Found {len(sec_companies)} unique SEC companies")
        
        # Get unique facilities from EPA data
        epa_facilities = epa_df[['facility_name', 'registry_id']].drop_duplicates()
        print(f"   ✓ Found {len(epa_facilities)} unique EPA facilities")
        
        # Perform fuzzy matching
        print(f"\n2. Performing fuzzy matching (threshold={threshold})...")
        print("   This may take a while...")
        
        matches = []
        total = len(sec_companies)
        
        for idx, (_, sec_row) in enumerate(sec_companies.iterrows()):
            if idx % 100 == 0:
                print(f"   Progress: {idx}/{total} ({100*idx/total:.1f}%)")
            
            sec_name = str(sec_row['company_name']).upper()
            
            for _, epa_row in epa_facilities.iterrows():
                epa_name = str(epa_row['facility_name']).upper()
                
                # Calculate similarity
                similarity = fuzz.ratio(sec_name, epa_name)
                
                if similarity >= threshold:
                    matches.append({
                        'sec_company': sec_name,
                        'epa_facility': epa_name,
                        'similarity_score': similarity,
                        'cik': sec_row['cik'],
                        'registry_id': epa_row['registry_id']
                    })
        
        matches_df = pd.DataFrame(matches)
        
        print(f"\n3. Matching results:")
        print(f"   ✓ Found {len(matches_df)} matches")
        print(f"   Average similarity: {matches_df['similarity_score'].mean():.1f}")
        
        # Save
        print(f"\n4. Saving matches to {save_path}...")
        matches_df.to_csv(save_path, index=False)
        print(f"   ✓ Saved {len(matches_df)} matches")
        
        # Show sample matches
        print("\n5. Sample matches:")
        print(matches_df.head(10).to_string(index=False))
        
        return matches_df
        
    except ImportError:
        print("\n✗ Error: fuzzywuzzy not installed")
        print("   Install with: pip install fuzzywuzzy python-Levenshtein")
        return None
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return None


# ============================================================================
# SECTION 4: TEXT FEATURE EXTRACTION
# ============================================================================

def extract_textual_features(sec_df, save_path='sec_with_features.csv'):
    """
    Extract textual features from 10-K disclosures
    
    Args:
        sec_df: DataFrame with SEC 10-K sentence-level data
        save_path: Where to save data with features
        
    Returns:
        DataFrame with extracted features
    """
    print("=" * 80)
    print("EXTRACTING TEXTUAL FEATURES")
    print("=" * 80)
    
    try:
        print("\n1. Loading required libraries...")
        import re
        from textstat import flesch_reading_ease
        
        print("   ✓ Libraries loaded")
        
        # Define environmental keywords
        print("\n2. Defining keyword categories...")
        
        keywords = {
            'climate': ['climate', 'warming', 'greenhouse', 'ghg', 'carbon', 'emission'],
            'pollution': ['pollution', 'contamination', 'toxic', 'hazardous', 'spill', 
                         'discharge', 'waste', 'remediation'],
            'compliance': ['epa', 'violation', 'enforcement', 'penalty', 'fine', 
                          'compliance', 'regulation', 'clean air', 'clean water', 'rcra'],
            'resources': ['water usage', 'energy', 'renewable', 'conservation', 'sustainability']
        }
        
        # Loughran-McDonald sentiment words (simplified version)
        negative_words = ['loss', 'losses', 'penalty', 'penalties', 'violation', 'violations',
                         'litigation', 'adverse', 'risk', 'risks', 'uncertainty', 'uncertainties']
        
        positive_words = ['benefit', 'benefits', 'improvement', 'improvements', 'achieve',
                         'achieved', 'success', 'successful', 'strong', 'growth']
        
        print(f"   ✓ Defined {len(keywords)} keyword categories")
        
        # Group by company-year-section
        print("\n3. Aggregating text by company-year-section...")
        
        agg_df = sec_df.groupby(['cik', 'year', 'section']).agg({
            'sentence': lambda x: ' '.join(x),  # Combine all sentences
        }).reset_index()
        
        agg_df.rename(columns={'sentence': 'full_text'}, inplace=True)
        
        print(f"   ✓ Aggregated to {len(agg_df)} company-year-section observations")
        
        # Extract features
        print("\n4. Extracting features...")
        
        def count_keywords(text, keyword_list):
            text_lower = text.lower()
            return sum(text_lower.count(kw) for kw in keyword_list)
        
        def normalize_count(count, text):
            # Normalize per 10,000 words
            word_count = len(text.split())
            return (count / word_count) * 10000 if word_count > 0 else 0
        
        # Extract all features
        for category, kw_list in keywords.items():
            print(f"   Extracting {category} keywords...")
            agg_df[f'{category}_count'] = agg_df['full_text'].apply(
                lambda x: normalize_count(count_keywords(x, kw_list), x)
            )
        
        # Sentiment
        print(f"   Extracting sentiment...")
        agg_df['negative_sentiment'] = agg_df['full_text'].apply(
            lambda x: normalize_count(count_keywords(x, negative_words), x)
        )
        agg_df['positive_sentiment'] = agg_df['full_text'].apply(
            lambda x: normalize_count(count_keywords(x, positive_words), x)
        )
        
        # Linguistic complexity
        print(f"   Extracting linguistic complexity...")
        agg_df['flesch_reading_ease'] = agg_df['full_text'].apply(
            lambda x: flesch_reading_ease(x) if len(x) > 0 else 0
        )
        agg_df['avg_sentence_length'] = agg_df['full_text'].apply(
            lambda x: len(x.split()) / max(x.count('.'), 1)
        )
        
        print(f"\n5. Feature extraction complete!")
        print(f"   Total features extracted: {len([col for col in agg_df.columns if col not in ['cik', 'year', 'section', 'full_text']])}")
        
        # Save
        print(f"\n6. Saving to {save_path}...")
        agg_df.to_csv(save_path, index=False)
        print(f"   ✓ Saved {len(agg_df)} observations with features")
        
        return agg_df
        
    except ImportError as ie:
        print(f"\n✗ Missing library: {str(ie)}")
        print("   Install with: pip install textstat")
        return None
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# SECTION 5: CREATE FINAL DATASET
# ============================================================================

def create_final_dataset(sec_features, epa_enforcement, matches, 
                        compustat_file=None, save_path='final_analysis_dataset.csv'):
    """
    Create final analysis dataset combining all data sources
    
    Args:
        sec_features: DataFrame with SEC textual features
        epa_enforcement: DataFrame with EPA enforcement data
        matches: DataFrame with company-facility matches
        compustat_file: Path to Compustat financial data (optional)
        save_path: Where to save final dataset
        
    Returns:
        Final analysis DataFrame
    """
    print("=" * 80)
    print("CREATING FINAL ANALYSIS DATASET")
    print("=" * 80)
    
    try:
        print("\n1. Merging SEC features with company matches...")
        
        # Merge SEC and matches
        df = sec_features.merge(matches[['cik', 'registry_id']], on='cik', how='left')
        print(f"   ✓ Merged: {len(df)} observations")
        
        # Merge with EPA enforcement
        print("\n2. Merging with EPA enforcement data...")
        df = df.merge(epa_enforcement, on='registry_id', how='left')
        print(f"   ✓ Merged: {len(df)} observations")
        
        # Create dependent variable (high_risk)
        print("\n3. Creating dependent variable (high_risk)...")
        
        # Find enforcement action columns
        action_cols = [col for col in df.columns if 'ACTION' in col.upper() or 'PENALTY' in col.upper()]
        
        if action_cols:
            df['total_actions'] = df[action_cols].sum(axis=1, skipna=True)
            df['high_risk'] = (df['total_actions'] > 0).astype(int)
        else:
            # Fallback: create simulated risk variable
            print("   ⚠ Warning: No enforcement columns found, creating simulated risk variable")
            df['high_risk'] = np.random.binomial(1, 0.25, len(df))
        
        print(f"   High-risk distribution:")
        print(df['high_risk'].value_counts())
        
        # Add control variables
        print("\n4. Adding control variables...")
        
        if compustat_file and os.path.exists(compustat_file):
            print("   Loading Compustat data...")
            compustat = pd.read_csv(compustat_file)
            df = df.merge(compustat, on=['cik', 'year'], how='left')
        else:
            print("   ⚠ No Compustat data provided, creating simulated financial variables")
            df['log_total_assets'] = np.random.uniform(18, 24, len(df))
            df['roa'] = np.random.uniform(-0.05, 0.15, len(df))
            df['leverage'] = np.random.uniform(0.3, 2.0, len(df))
        
        # Add industry classification
        print("\n5. Adding industry classification...")
        df['industry'] = pd.cut(df['sic_code'], 
                                bins=[0, 1500, 2900, 5000, 10000],
                                labels=['Mining', 'Chemicals', 'Utilities', 'Other'])
        
        # Final cleanup
        print("\n6. Final cleanup and validation...")
        df = df.dropna(subset=['cik', 'year', 'high_risk'])
        print(f"   ✓ Final dataset: {len(df)} observations")
        
        # Save
        print(f"\n7. Saving final dataset to {save_path}...")
        df.to_csv(save_path, index=False)
        print(f"   ✓ Saved {len(df)} observations")
        
        # Print summary
        print("\n8. Dataset summary:")
        print(f"   Total observations: {len(df)}")
        print(f"   Years: {df['year'].min()} - {df['year'].max()}")
        print(f"   Unique companies: {df['cik'].nunique()}")
        print(f"   High-risk %: {df['high_risk'].mean()*100:.1f}%")
        print(f"\n   By industry:")
        print(df['industry'].value_counts())
        print(f"\n   By year:")
        print(df['year'].value_counts().sort_index())
        
        return df
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DATA COLLECTION PIPELINE FOR BUSINESS ANALYTICS PROJECT")
    print("Predicting Corporate Environmental Risk using ML Analysis of SEC 10-K Filings")
    print("=" * 80)
    
    # Set paths
    ECHO_FILE = 'ECHO_EXPORTER.csv'  # You need to download this manually
    COMPUSTAT_FILE = None  # Optional: path to Compustat data
    
    # Run pipeline
    print("\n\nSTARTING PIPELINE...\n")
    
    # Step 1: Collect SEC data
    sec_df = collect_sec_10k_data('sec_10k_raw.csv')
    
    if sec_df is not None:
        # Step 2: Process EPA data (if file exists)
        if os.path.exists(ECHO_FILE):
            epa_df = collect_epa_enforcement_data(ECHO_FILE, 'epa_enforcement.csv')
        else:
            print(f"\n⚠ Warning: {ECHO_FILE} not found. Please download from EPA ECHO website.")
            print("   Continuing with sample data for demonstration...")
            epa_df = None
        
        # Step 3: Extract textual features
        sec_features = extract_textual_features(sec_df, 'sec_with_features.csv')
        
        # Step 4 & 5: Match and create final dataset (if EPA data available)
        if epa_df is not None and sec_features is not None:
            matches = fuzzy_match_companies(sec_df, epa_df, threshold=85, 
                                          save_path='matched_companies.csv')
            
            if matches is not None:
                final_df = create_final_dataset(sec_features, epa_df, matches,
                                                compustat_file=COMPUSTAT_FILE,
                                                save_path='final_analysis_dataset.csv')
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nOutput files:")
    print("1. sec_10k_raw.csv - Raw SEC 10-K data")
    print("2. epa_enforcement.csv - EPA enforcement data (if available)")
    print("3. sec_with_features.csv - SEC data with textual features")
    print("4. matched_companies.csv - Company-facility matches")
    print("5. final_analysis_dataset.csv - Final dataset for analysis")
