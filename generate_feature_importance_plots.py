import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

GBRF_WINNER_MODEL_PATH = os.path.join(MODELS_DIR, 'winner_model.joblib')
XGB_WINNER_MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_winner_model.joblib')

def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile = f"feature_importance_plotter_{ts}.log"
    logpath = os.path.join(LOG_DIR, logfile)
    
    logger = logging.getLogger("FeatureImportancePlotter")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(logpath)
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
    return logger

LOGGER = setup_logger()

def plot_feature_importance(model, feature_names, title, save_path):
    if not hasattr(model, 'feature_importances_'):
        LOGGER.error(f"Model of type {type(model).__name__} does not have feature_importances_ attribute.")
        return

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plt.savefig(save_path)
    LOGGER.info(f"Saved feature importance plot to {save_path}")
    plt.close()

def main():
    LOGGER.info("Starting feature importance plotting.")

    try:
        LOGGER.info(f"Loading Gradient Boosting model from {GBRF_WINNER_MODEL_PATH}")
        gbrf_payload = joblib.load(GBRF_WINNER_MODEL_PATH)
        gbrf_model = gbrf_payload['model']
        gbrf_features = gbrf_payload['feature_columns']
        LOGGER.info("Gradient Boosting model loaded successfully.")
        
        gbrf_plot_path = os.path.join(LOG_DIR, 'gbrf_winner_model_feature_importance.png')
        plot_feature_importance(gbrf_model, gbrf_features, 'Gradient Boosting Winner Model - Feature Importance', gbrf_plot_path)

    except FileNotFoundError:
        LOGGER.error(f"Model file not found at {GBRF_WINNER_MODEL_PATH}. Please train the model first.")
    except Exception as e:
        LOGGER.error(f"An error occurred while processing the Gradient Boosting model: {e}")

    try:
        LOGGER.info(f"Loading XGBoost model from {XGB_WINNER_MODEL_PATH}")
        xgb_payload = joblib.load(XGB_WINNER_MODEL_PATH)
        xgb_model = xgb_payload['model']
        xgb_features = xgb_payload['feature_columns']
        LOGGER.info("XGBoost model loaded successfully.")
        
        xgb_plot_path = os.path.join(LOG_DIR, 'xgb_winner_model_feature_importance.png')
        plot_feature_importance(xgb_model, xgb_features, 'XGBoost Winner Model - Feature Importance', xgb_plot_path)

    except FileNotFoundError:
        LOGGER.error(f"Model file not found at {XGB_WINNER_MODEL_PATH}. Please train the model first.")
    except Exception as e:
        LOGGER.error(f"An error occurred while processing the XGBoost model: {e}")

    LOGGER.info("Feature importance plotting finished.")

if __name__ == "__main__":
    main()
