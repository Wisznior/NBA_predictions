import pandas as pd
import os
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
LOG_DIR = os.path.join(BASE, "logs")

def setup_logger():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile = f"model_training_{ts}.log"
    logpath = os.path.join(LOG_DIR, logfile)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(logpath), logging.StreamHandler()])
    return logging.getLogger(__name__)

def train_model(features_path, model_output_path, logger):
    
    if not os.path.exists(features_path):
        logger.error(f"Feature file not found at {features_path}")
        return

    try:
        df = pd.read_csv(features_path)
        logger.info(f"Loaded {len(df)} samples")
    except Exception as e:
        logger.error(f"Error loading feature file: {e}")
        return

    X = df.drop(columns=['game_id', 'game_date', 'outcome'])
    # 1 for home win H, 0 for away win A
    y = df['outcome'].apply(lambda x: 1 if x == 'H' else 0)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67, stratify=y)
    logger.info(f"Training set: {len(X_train)} samples. Test set: {len(X_test)} samples.")

    logger.info("Starting model training")
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=67)
    model.fit(X_train, y_train)
    logger.info("Traning finished")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Model accuracy: {accuracy:.4f}")

    model_payload = {
        'model': model,
        'feature_columns': list(X.columns)
    }
    joblib.dump(model_payload, model_output_path)
    logger.info(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Starting model training")
    
    features_path = os.path.join(DATA_DIR, 'ml_game_features.csv')
    model_output_path = os.path.join(MODELS_DIR, 'winner_model.joblib')
    
    train_model(features_path, model_output_path, logger)
    
    logger.info("Model training finished.")

