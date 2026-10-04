import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier
import joblib
from datetime import datetime

# ===============================================
#  CONFIG – Edit these
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

# ===============================================
#  LOGGING
# ===============================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===============================================
#  1. LOAD DATA
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
#  2. PREPARE DATA
# ===============================================
def prepare_data(df):
    X = df[FEATURES]
    y = df['placement_status'].map({'Placed': 1, 'Not_Placed': 0})
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"Class distribution: Placed={sum(y)}, Not_Placed={len(y)-sum(y)}")
    return X, y

# ===============================================
#  3. TRAIN MODELS
# ===============================================
def train_models(X, y):
    """Train models with cross-validation (no tuning)."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, 
                                                        random_state=RANDOM_SEED, stratify=y)
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    results = {}
    
    # ---------- Random Forest ----------
    logger.info("\n" + "="*50)
    logger.info("Training Random Forest...")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    cv_score_rf = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy').mean()
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    
    results['rf'] = {
        'model': rf,
        'cv_score': cv_score_rf,
        'test_accuracy': accuracy_score(y_test, y_pred_rf),
        'roc_auc': roc_auc_score(y_test, y_proba_rf),
        'y_pred': y_pred_rf,
        'y_proba': y_proba_rf
    }
    
    logger.info(f"✅ RF CV Score: {cv_score_rf:.4f}")
    logger.info(f"✅ RF Test Accuracy: {results['rf']['test_accuracy']:.4f}")
    logger.info(f"✅ RF ROC-AUC: {results['rf']['roc_auc']:.4f}")
    
    # ---------- XGBoost ----------
    logger.info("\n" + "="*50)
    logger.info("Training XGBoost...")
    
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_SEED, n_jobs=-1)
    xgb.fit(X_train, y_train)
    
    cv_score_xgb = cross_val_score(xgb, X_train, y_train, cv=cv, scoring='accuracy').mean()
    y_pred_xgb = xgb.predict(X_test)
    y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    
    results['xgb'] = {
        'model': xgb,
        'cv_score': cv_score_xgb,
        'test_accuracy': accuracy_score(y_test, y_pred_xgb),
        'roc_auc': roc_auc_score(y_test, y_proba_xgb),
        'y_pred': y_pred_xgb,
        'y_proba': y_proba_xgb
    }
    
    logger.info(f"✅ XGB CV Score: {cv_score_xgb:.4f}")
    logger.info(f"✅ XGB Test Accuracy: {results['xgb']['test_accuracy']:.4f}")
    logger.info(f"✅ XGB ROC-AUC: {results['xgb']['roc_auc']:.4f}")
    
    return results, X_train, X_test, y_train, y_test

# ===============================================
#  4. PLOT FEATURE IMPORTANCE
# ===============================================
def plot_feature_importance(model, feature_names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance (XGBoost)")
        plt.bar(range(len(indices)), importances[indices], color='indigo', alpha=0.7)
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'feature_importance.png'), dpi=150)
        plt.close()
        logger.info(f"✅ Feature importance plot saved")
        
        # Print top 5 features
        logger.info("\n📊 Top 5 Features:")
        for i in range(min(5, len(indices))):
            logger.info(f"   {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.3f}")

# ===============================================
#  5. PLOT CONFUSION MATRIX
# ===============================================
def plot_confusion_matrix(y_test, y_pred, out_dir, model_name):
    os.makedirs(out_dir, exist_ok=True)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Placed', 'Placed'],
                yticklabels=['Not Placed', 'Placed'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'confusion_matrix_{model_name}.png'), dpi=150)
    plt.close()
    logger.info(f"✅ Confusion matrix saved for {model_name}")

# ===============================================
#  6. PLOT ROC CURVE
# ===============================================
def plot_roc_curve(y_test, y_proba, out_dir, model_name):
    os.makedirs(out_dir, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'roc_curve_{model_name}.png'), dpi=150)
    plt.close()
    logger.info(f"✅ ROC curve saved for {model_name}")

# ===============================================
#  7. SAVE MODEL & METRICS (FIXED: y_test passed)
# ===============================================
def save_results(results, y_test, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save XGBoost model (better performer)
    joblib.dump(results['xgb']['model'], os.path.join(MODEL_DIR, 'placement_predictor_model.pkl'))
    logger.info(f"💾 Model saved to {MODEL_DIR}/placement_predictor_model.pkl")
    
    # Save metrics to CSV
    metrics = []
    for name in ['rf', 'xgb']:
        metrics.append({
            'model': name,
            'cv_accuracy': round(results[name]['cv_score'], 4),
            'test_accuracy': round(results[name]['test_accuracy'], 4),
            'roc_auc': round(results[name]['roc_auc'], 4)
        })
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(out_dir, 'model_metrics.csv'), index=False)
    logger.info(f"💾 Metrics saved to {out_dir}/model_metrics.csv")
    
    # Save classification report (using XGBoost predictions)
    report = classification_report(y_test, results['xgb']['y_pred'], 
                                   target_names=['Not Placed', 'Placed'], 
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(out_dir, 'classification_report.csv'))
    logger.info(f"💾 Classification report saved")

# ===============================================
#  8. MAIN
# ===============================================
def main():
    logger.info("="*50)
    logger.info("🚀 ML TRAINING (No Tuning)")
    logger.info("="*50)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Prepare data
    X, y = prepare_data(df)
    
    # Train models
    results, X_train, X_test, y_train, y_test = train_models(X, y)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(REPORT_DIR, timestamp)
    
    # Generate plots
    plot_feature_importance(results['xgb']['model'], FEATURES, out_dir)
    plot_confusion_matrix(y_test, results['rf']['y_pred'], out_dir, 'RandomForest')
    plot_confusion_matrix(y_test, results['xgb']['y_pred'], out_dir, 'XGBoost')
    plot_roc_curve(y_test, results['rf']['y_proba'], out_dir, 'RandomForest')
    plot_roc_curve(y_test, results['xgb']['y_proba'], out_dir, 'XGBoost')
    
    # Save results (pass y_test)
    save_results(results, y_test, out_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("📊 FINAL RESULTS")
    print("="*50)
    print(f"🎯 Random Forest CV Accuracy : {results['rf']['cv_score']:.2%}")
    print(f"🎯 Random Forest Test Acc    : {results['rf']['test_accuracy']:.2%}")
    print(f"🚀 XGBoost CV Accuracy       : {results['xgb']['cv_score']:.2%}")
    print(f"🚀 XGBoost Test Acc          : {results['xgb']['test_accuracy']:.2%}")
    print(f"📈 XGBoost ROC-AUC           : {results['xgb']['roc_auc']:.3f}")
    print("="*50)
    print(f"💾 Model saved to: {MODEL_DIR}/placement_predictor_model.pkl")
    print(f"📁 Reports saved to: {out_dir}/")
    print("🎉 TRAINING COMPLETE!")

if __name__ == '__main__':
    main()