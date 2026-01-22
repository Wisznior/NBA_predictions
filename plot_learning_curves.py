import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import os
import logging
from datetime import datetime
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
WINNER_FEATURES_PATH = os.path.join(DATA_DIR, 'ml_winner_features.csv')

def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile = f"learning_curve_plotter_{ts}.log"
    logpath = os.path.join(LOG_DIR, logfile)
    
    logger = logging.getLogger("LearningCurvePlotter")
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

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    plt.legend(loc="best")
    return plt

def main():
    LOGGER.info("Starting learning curve plotting.")

    if not os.path.exists(WINNER_FEATURES_PATH):
        LOGGER.error(f"Winner features file not found at {WINNER_FEATURES_PATH}")
        return

    try:
        df = pd.read_csv(WINNER_FEATURES_PATH)
        LOGGER.info(f"Loaded {len(df)} samples from {WINNER_FEATURES_PATH}")
    except Exception as e:
        LOGGER.error(f"Error loading winner features: {e}")
        return

    X = df.drop(columns=['game_id', 'game_date', 'outcome'])
    y = df['outcome'].apply(lambda x: 1 if x == 'H' else 0)

    LOGGER.info("Generating learning curve for Gradient Boosting Classifier.")
    gbrt_title = "Learning Curve (Gradient Boosting)"
    gbrt_estimator = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=67)
    gbrt_plot = plot_learning_curve(gbrt_estimator, gbrt_title, X, y, cv=5)
    gbrt_save_path = os.path.join(LOG_DIR, 'gbrf_winner_model_learning_curve.png')
    gbrt_plot.savefig(gbrt_save_path)
    LOGGER.info(f"Saved Gradient Boosting learning curve to {gbrt_save_path}")
    plt.close()

    LOGGER.info("Generating learning curve for XGBoost Classifier.")
    xgb_title = "Learning Curve (XGBoost)"
    xgb_estimator = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, 
                                      n_estimators=100, learning_rate=0.1, max_depth=3, random_state=67)
    xgb_plot = plot_learning_curve(xgb_estimator, xgb_title, X, y, cv=5)
    xgb_save_path = os.path.join(LOG_DIR, 'xgb_winner_model_learning_curve.png')
    xgb_plot.savefig(xgb_save_path)
    LOGGER.info(f"Saved XGBoost learning curve to {xgb_save_path}")
    plt.close()

    LOGGER.info("Learning curve plotting finished.")

if __name__ == "__main__":
    main()
