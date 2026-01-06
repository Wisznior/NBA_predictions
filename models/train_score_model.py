import pandas as pd
import joblib
import os
import sys
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import logging

# configuration
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
LOG_DIR = os.path.join(BASE, "logs")

INPUT_FEATURES_FILE = os.path.join(DATA_DIR, 'ml_score_features.csv')
HOME_SCORE_MODEL_FILE = os.path.join(MODELS_DIR, 'home_score_model.joblib')
AWAY_SCORE_MODEL_FILE = os.path.join(MODELS_DIR, 'away_score_model.joblib')

def setup_logger():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile = f"score_model_training_{ts}.log"
    logpath = os.path.join(LOG_DIR, logfile)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(logpath), logging.StreamHandler(sys.stdout)])
    return logging.getLogger(__name__)

def train_model(logger):
    logger.info("Starting training of score prediction models")

    if not os.path.exists(INPUT_FEATURES_FILE):
        logger.error(f"Feature file not found {INPUT_FEATURES_FILE} Run the feature generation script first.")
        return

    try:
        df = pd.read_csv(INPUT_FEATURES_FILE)
        logger.info(f"Loaded {len(df)} samples from {INPUT_FEATURES_FILE}")
    except Exception as e:
        logger.error(f"Error loading feature file: {e}")
        return
    
    X = df.drop(columns=['game_date', 'home_team_id', 'away_team_id', 'home_pts', 'away_pts'])
    y_home = df['home_pts']
    y_away = df['away_pts']

    feature_columns = X.columns.tolist()
    logger.info(f"Prepared {len(feature_columns)} features for training.")

    X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
        X, y_home, y_away, test_size=0.2, random_state=42
    )
    logger.info(f"Training set size: {len(X_train)}, test set size: {len(X_test)}")

    logger.info("Training Home Score Model")
    home_score_model = RandomForestRegressor(n_estimators=100, random_state=42)
    home_score_model.fit(X_train, y_home_train)
    home_predictions = home_score_model.predict(X_test)
    logger.info(f"Home Score Model MAE: {mean_absolute_error(y_home_test, home_predictions):.2f}")
    logger.info(f"Home Score Model R2: {r2_score(y_home_test, home_predictions):.2f}")

    logger.info("Training Away Score Model")
    away_score_model = RandomForestRegressor(n_estimators=100, random_state=42)
    away_score_model.fit(X_train, y_away_train)
    away_predictions = away_score_model.predict(X_test)
    logger.info(f"Away Score Model MAE: {mean_absolute_error(y_away_test, away_predictions):.2f}")
    logger.info(f"Away Score Model R2: {r2_score(y_away_test, away_predictions):.2f}")

    # Save models
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    joblib.dump({'model': home_score_model, 'feature_columns': feature_columns}, HOME_SCORE_MODEL_FILE)
    joblib.dump({'model': away_score_model, 'feature_columns': feature_columns}, AWAY_SCORE_MODEL_FILE)
    logger.info("Score prediction models trained and saved successfully.")

if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Starting score model training script.")
    train_model(logger)
    logger.info("Score model training script finished.")
