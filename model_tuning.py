import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from datetime import datetime

# ===============================================
#  CONFIG
# ===============================================
INPUT_FILE = 'datadb/student_data_with_features.csv'
MODEL_DIR = 'models'
REPORT_DIR = 'reports/ml'
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Features to use
FEATURES = ['academic_score', 'coding_score', 'aptitude_score', 
            'experience_score', 'overall_strength', 'attendance', 
            'internships', 'projects']

# Parameter grid for tuning
PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],           # Added
    'colsample_bytree': [0.8, 1.0],    # Added
    'gamma': [0, 0.1, 0.2]             # Added
}

# ===============================================
#  LOGGING
# ===============================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===============================================
#  LOAD DATA
# ===============================================
def load_data():
    logger.info(f"Loading data from {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        logger.error(f"File not found: {INPUT_FILE}")
        return None
    
    df = pd.read_csv(INPUT_FILE)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

# ===============================================
#  PREPARE DATA
# ===============================================
def prepare_data(df):
    X = df[FEATURES]
    y = df['placement_status'].map({'Placed': 1, 'Not_Placed': 0})
    logger.info(f"X: {X.shape}, y: {y.shape}")
    logger.info(f"Placed: {sum(y)}, Not Placed: {len(y)-sum(y)}")
    return X, y

# ===============================================
#  LOAD BASELINE MODEL
# ===============================================
def load_baseline_model():
    """Load baseline model for comparison."""
    baseline_path = os.path.join(MODEL_DIR, 'placement_predictor_model.pkl')
    if os.path.exists(baseline_path):
        try:
            model = joblib.load(baseline_path)
            logger.info(f"Loaded baseline model from {baseline_path}")
            return model
        except Exception as e:
            logger.warning(f"Could not load baseline model: {e}")
            return None
    logger.warning("No baseline model found")
    return None

# ===============================================
#  HYPERPARAMETER TUNING
# ===============================================
def tune_model(X_train, y_train):
    """Run GridSearchCV for hyperparameter tuning."""
    logger.info("Starting hyperparameter tuning...")
    logger.info(f"Parameter grid: {PARAM_GRID}")
    
    total_combinations = 1
    for v in PARAM_GRID.values():
        total_combinations *= len(v)
    logger.info(f"Total combinations: {total_combinations}")
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    # Create model
    xgb = XGBClassifier(random_state=RANDOM_SEED, n_jobs=-1)
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=PARAM_GRID,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    logger.info("Running GridSearchCV... (this may take 2-3 minutes)")
    grid_search.fit(X_train, y_train)
    
    logger.info("✅ Tuning complete!")
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search

# ===============================================
#  EVALUATE MODEL
# ===============================================
def evaluate_model(model, X_test, y_test):
    """Evaluate model on test data."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {acc:.4f}")
    return y_pred, y_proba, acc

# ===============================================
#  SAVE RESULTS
# ===============================================
def save_results(grid_search, X_test, y_test, baseline_acc, tuned_acc, out_dir):
    """Save tuned model and results."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Save tuned model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'xgboost_tuned_model.pkl'))
    logger.info(f"💾 Tuned model saved to {MODEL_DIR}/xgboost_tuned_model.pkl")
    
    # 2. Save CV results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv(os.path.join(out_dir, 'cv_results.csv'), index=False)
    logger.info(f"💾 CV results saved to {out_dir}/cv_results.csv")
    
    # 3. Save summary report
    with open(os.path.join(out_dir, 'tuning_summary.txt'), 'w') as f:
        f.write("="*50 + "\n")
        f.write("HYPERPARAMETER TUNING SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Best parameters: {grid_search.best_params_}\n")
        f.write(f"Best CV score: {grid_search.best_score_:.4f}\n\n")
        
        f.write("-"*30 + "\n")
        f.write("COMPARISON\n")
        f.write("-"*30 + "\n")
        f.write(f"Baseline model accuracy: {baseline_acc:.2%}\n")
        f.write(f"Tuned model accuracy: {tuned_acc:.2%}\n")
        f.write(f"Improvement: {(tuned_acc - baseline_acc)*100:.2f}%\n")
    
    # 4. Save classification report
    y_pred = grid_search.predict(X_test)
    report = classification_report(y_test, y_pred, 
                                   target_names=['Not Placed', 'Placed'], 
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(out_dir, 'classification_report_tuned.csv'))
    
    # 5. Create final summary DataFrame
    comparison_df = pd.DataFrame({
        'model': ['Baseline', 'Tuned'],
        'accuracy': [baseline_acc, tuned_acc],
        'test_set_size': [len(X_test), len(X_test)]
    })
    comparison_df.to_csv(os.path.join(out_dir, 'model_comparison.csv'), index=False)
    
    logger.info(f"💾 All reports saved to {out_dir}")

# ===============================================
#  MAIN
# ===============================================
def main():
    logger.info("="*50)
    logger.info("🚀 HYPERPARAMETER TUNING PIPELINE")
    logger.info("="*50)
    
    # 1. Load data
    df = load_data()
    if df is None:
        return
    
    # 2. Prepare data
    X, y = prepare_data(df)
    
    # 3. Split data (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 4. Load baseline model (if exists)
    baseline_model = load_baseline_model()
    baseline_acc = None
    if baseline_model:
        _, _, baseline_acc = evaluate_model(baseline_model, X_test, y_test)
    else:
        # Train a simple model as baseline
        logger.info("No baseline model found. Training a simple XGBoost as baseline...")
        xgb_base = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_SEED)
        xgb_base.fit(X_train, y_train)
        _, _, baseline_acc = evaluate_model(xgb_base, X_test, y_test)
    
    # 5. Hyperparameter tuning
    grid_search = tune_model(X_train, y_train)
    
    # 6. Evaluate tuned model
    _, _, tuned_acc = evaluate_model(grid_search.best_estimator_, X_test, y_test)
    
    # 7. Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(REPORT_DIR, f'tuning_{timestamp}')
    
    # 8. Save results
    save_results(grid_search, X_test, y_test, baseline_acc, tuned_acc, out_dir)
    
    # 9. Final summary
    print("\n" + "="*50)
    print("📊 TUNING SUMMARY")
    print("="*50)
    print(f"🎯 Best Parameters: {grid_search.best_params_}")
    print(f"📈 Best CV Score: {grid_search.best_score_:.2%}")
    print(f"🎯 Baseline Accuracy: {baseline_acc:.2%}")
    print(f"🚀 Tuned Accuracy: {tuned_acc:.2%}")
    print(f"📊 Improvement: {(tuned_acc - baseline_acc)*100:.2f}%")
    print("="*50)
    print(f"💾 Tuned model saved to: {MODEL_DIR}/xgboost_tuned_model.pkl")
    print(f"📁 Reports saved to: {out_dir}/")
    print("🎉 TUNING COMPLETE!")

if __name__ == '__main__':
    main()