import os
import sys
import logging
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

# ===============================================
#  CONFIG – Edit these
# ===============================================
INPUT_FILE = 'datadb/student_placement_data_clean.csv'
OUTPUT_CSV = 'datadb/student_data_with_features.csv'
DB_PATH = 'datadb/college_placement.db'
LOG_LEVEL = logging.INFO

# ===============================================
#  LOGGING
# ===============================================
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===============================================
#  FEATURE ENGINEERING
# ===============================================
def engineer_features(df):
    """Add all derived features to the DataFrame."""
    
    logger.info("Starting feature engineering...")
    initial_cols = len(df.columns)
    
    # 1. Academic Score (weighted average)
    df['academic_score'] = (df['ssc_percentage'] * 0.2 + 
                           df['hsc_percentage'] * 0.3 + 
                           (df['degree_cgpa'] / 10 * 100) * 0.5)
    logger.info("✅ Created academic_score")
    
    # 2. Experience Score
    df['experience_score'] = (df['internships'] * 3) + (df['projects'] * 2)
    logger.info("✅ Created experience_score")
    
    # 3. Overall Strength (composite)
    df['overall_strength'] = (df['academic_score'] * 0.4 + 
                             df['coding_score'] * 0.3 + 
                             df['aptitude_score'] * 0.2 + 
                             df['experience_score'] * 0.1)
    logger.info("✅ Created overall_strength")
    
    # 4. Risk Category (based on attendance)
    df['risk_category'] = pd.cut(df['attendance'], 
                                 bins=[0, 70, 90, 101], 
                                 labels=['High Risk', 'Medium Risk', 'Low Risk'])
    logger.info("✅ Created risk_category")
    
    # 5. Placement Probability (logistic score)
    prob_score = (df['academic_score'] * 0.3 + 
                 df['coding_score'] * 0.25 + 
                 df['aptitude_score'] * 0.2 + 
                 df['experience_score'] * 0.15 + 
                 df['attendance'] * 0.1)
    df['placement_probability'] = 1 / (1 + np.exp(-(prob_score - 55) / 12))
    logger.info("✅ Created placement_probability")
    
    # 6. Skill Gap (vs class average)
    class_avg_coding = df['coding_score'].mean()
    df['skill_gap'] = df['coding_score'] - class_avg_coding
    logger.info(f"✅ Created skill_gap (class average coding: {class_avg_coding:.2f})")
    
    # 7. Add department column if not exists (for analysis)
    if 'department' not in df.columns:
        np.random.seed(42)
        depts = ['CSE', 'IT', 'ECE', 'ME']
        dept_probs = {'CSE': 0.50, 'IT': 0.25, 'ECE': 0.15, 'ME': 0.10}
        df['department'] = np.random.choice(
            depts, 
            size=len(df), 
            p=[dept_probs[d] for d in depts]
        )
        logger.info("✅ Added dummy department column for analysis")
    
    # 8. Interaction feature: Coding × Aptitude (synergy)
    df['coding_aptitude_interaction'] = (df['coding_score'] * df['aptitude_score']) / 100
    logger.info("✅ Created coding_aptitude_interaction")
    
    new_cols = len(df.columns) - initial_cols
    logger.info(f"✅ Added {new_cols} new features. Total columns: {len(df.columns)}")
    
    return df

# ===============================================
#  SAVE TO DATABASE
# ===============================================
def save_to_database(df):
    """Save feature-engineered data to a new table in database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DROP TABLE IF EXISTS student_features")
        df.to_sql('student_features', conn, if_exists='replace', index=False)
        logger.info(f"✅ Saved {len(df)} rows to 'student_features' table in database")
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to save to database: {e}")
        return False

# ===============================================
#  GENERATE SUMMARY REPORT
# ===============================================
def generate_summary(df, output_dir='reports'):
    """Create a summary report of feature distributions."""
    os.makedirs(output_dir, exist_ok=True)
    
    numeric_cols = ['academic_score', 'experience_score', 'overall_strength', 
                   'placement_probability', 'skill_gap']
    
    summary = df[numeric_cols].describe().round(2)
    summary.to_csv(os.path.join(output_dir, 'feature_summary.csv'))
    
    logger.info(f"✅ Feature summary saved to {output_dir}/feature_summary.csv")
    return summary

# ===============================================
#  MAIN PIPELINE
# ===============================================
def main():
    logger.info("="*50)
    logger.info("🚀 FEATURE ENGINEERING PIPELINE")
    logger.info("="*50)
    
    # 1. Load data
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        return
    
    logger.info(f"Loading data from {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Save original column count
    original_cols = len(df.columns)
    
    # 2. Engineer features
    df = engineer_features(df)
    
    # 3. Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV) or '.', exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"💾 Data saved to {OUTPUT_CSV}")
    
    # 4. Save to database
    save_to_database(df)
    
    # 5. Generate summary
    generate_summary(df)
    
    # 6. Print sample
    logger.info("\n📊 Sample of new features (first 5 students):")
    sample_cols = ['student_id', 'academic_score', 'experience_score', 
                   'overall_strength', 'risk_category', 'placement_probability']
    print(df[sample_cols].head())
    
    logger.info("="*50)
    logger.info("🎉 FEATURE ENGINEERING COMPLETE!")
    logger.info(f"   Added {len(df.columns) - original_cols} new features")
    logger.info(f"   Total columns: {len(df.columns)}")
    logger.info("="*50)

# ===============================================
#  RUN THE SCRIPT
# ===============================================
if __name__ == '__main__':
    main()