
import os
import logging
import pandas as pd
import numpy as np
import joblib
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime

# ===== CONFIG =====
INPUT_FILE = 'datadb/student_data_with_features.csv'
MODEL_FILE = 'models/placement_predictor_model.pkl'
REPORT_DIR = 'reports/explainability'
RANDOM_SEED = 42
TEST_SIZE = 0.2
FEATURES = ['academic_score', 'coding_score', 'aptitude_score', 
            'experience_score', 'overall_strength', 'attendance', 
            'internships', 'projects']

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data():
    df = pd.read_csv(INPUT_FILE)
    logger.info(f"Loaded {len(df)} rows")
    return df

def load_model():
    if not os.path.exists(MODEL_FILE):
        # Fallback to tuned model
        fallback = 'models/xgboost_tuned_model.pkl'
        if os.path.exists(fallback):
            model = joblib.load(fallback)
            logger.info(f"Loaded tuned model from {fallback}")
            return model
        else:
            raise FileNotFoundError("No model found")
    model = joblib.load(MODEL_FILE)
    logger.info("Model loaded")
    return model

def main():
    logger.info("🚀 LIME Explainability")
    df = load_data()
    X = df[FEATURES]
    y = df['placement_status'].map({'Placed': 1, 'Not_Placed': 0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
    model = load_model()
    
    # LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=FEATURES,
        class_names=['Not Placed', 'Placed'],
        mode='classification'
    )
    
    # Explain first test instance
    exp = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba)
    
    # Save explanation as HTML
    out_dir = os.path.join(REPORT_DIR, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)
    exp.save_to_file(os.path.join(out_dir, 'lime_explanation.html'))
    logger.info(f"✅ LIME explanation saved to {out_dir}")
    
    # Also plot feature importance from model (built‑in)
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        plt.figure(figsize=(10,6))
        plt.barh(FEATURES, imp, color='indigo')
        plt.xlabel('Importance')
        plt.title('Model Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'feature_importance.png'))
        logger.info("✅ Feature importance plot saved")

if __name__ == '__main__':
    main()