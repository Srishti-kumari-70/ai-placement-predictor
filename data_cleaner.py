import os, json, logging
import pandas as pd
import numpy as np
from datetime import datetime

# ===== CONFIG =====
INPUT_FILE = 'datadb/student_placement_data.csv'
OUTPUT_FILE = 'datadb/student_placement_data_clean.csv'
MISSING_THRESHOLD = 0.5
CAP_OUTLIERS = True
IQR_MULTIPLIER = 1.5
IMPUTE_CONFIG = {
    'ssc_percentage': 'mean',
    'hsc_percentage': 'mean',
    'degree_cgpa': 'median',
    'coding_score': 'median',
    'aptitude_score': 'median',
    'attendance': 'median',
    'internships': 'mode',
    'projects': 'mode',
    'age': 'median'
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def drop_high_missing(df):
    threshold = MISSING_THRESHOLD
    missing_frac = df.isnull().sum() / len(df)
    cols_to_drop = missing_frac[missing_frac > threshold].index.tolist()
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        logger.info(f"❌ Dropped columns: {cols_to_drop}")
    return df, cols_to_drop

def remove_duplicates(df):
    if 'student_id' in df.columns:
        before = len(df)
        df.drop_duplicates(subset=['student_id'], keep='first', inplace=True)
        removed = before - len(df)
        if removed:
            logger.info(f"🗑️ Removed {removed} duplicates.")
        return df, removed
    return df, 0

def impute_missing(df):
    for col in df.columns:
        if df[col].isnull().any():
            strategy = IMPUTE_CONFIG.get(col, 'median')
            if strategy == 'mean':
                val = df[col].mean()
            elif strategy == 'median':
                val = df[col].median()
            elif strategy == 'mode':
                mode_vals = df[col].mode()
                val = mode_vals[0] if not mode_vals.empty else 0
            else:
                val = df[col].median()
            df[col] = df[col].fillna(val)
            logger.info(f"🔄 Imputed '{col}' with {strategy}: {val:.2f}")
    return df

def cap_outliers(df):
    if not CAP_OUTLIERS:
        return df, {}
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    outlier_summary = {}
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) == 0:
            continue
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - IQR_MULTIPLIER * iqr
        upper = q3 + IQR_MULTIPLIER * iqr
        df[f'{col}_outlier'] = ((df[col] < lower) | (df[col] > upper)).astype(int)
        count = df[f'{col}_outlier'].sum()
        if count > 0:
            df[col] = df[col].clip(lower, upper)
            logger.info(f"📏 Capped {count} outliers in '{col}'")
            outlier_summary[col] = {'count': int(count), 'lower': float(lower), 'upper': float(upper)}
    return df, outlier_summary

def fix_dtypes(df):
    if 'age' in df.columns:
        df['age'] = df['age'].astype(int)
    for col in ['gender', 'placement_status']:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

def handle_infinite(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def main():
    logger.info("🚀 Starting smart cleaning...")
    df = pd.read_csv(INPUT_FILE)
    report = {'initial_shape': df.shape}
    df = handle_infinite(df)
    df, dropped = drop_high_missing(df)
    report['dropped_columns'] = dropped
    df, dup_removed = remove_duplicates(df)
    report['duplicates_removed'] = dup_removed
    df, outliers = cap_outliers(df)
    report['outliers_capped'] = outliers
    df = impute_missing(df)
    report['imputation_config'] = IMPUTE_CONFIG
    df = fix_dtypes(df)
    report['final_shape'] = df.shape
    report['remaining_missing'] = int(df.isnull().sum().sum())
    os.makedirs(os.path.dirname(OUTPUT_FILE) or '.', exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"💾 Saved to {OUTPUT_FILE}")
    report_path = OUTPUT_FILE.replace('.csv', '_cleaning_report.json')
    report['timestamp'] = datetime.now().isoformat()
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"📄 Report saved to {report_path}")
    logger.info("✅ Cleaning complete.")

if __name__ == '__main__':
    main()