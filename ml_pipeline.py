import pandas as pd
import sqlite3
import os
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import joblib
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

DB_PATH = os.path.join(DATA_DIR, 'nba_history.db')
WINNER_FEATURES_PATH = os.path.join(DATA_DIR, 'ml_winner_features.csv')
SCORE_FEATURES_PATH = os.path.join(DATA_DIR, 'ml_score_features.csv')

WINNER_MODEL_PATH = os.path.join(MODELS_DIR, 'winner_model.joblib')
HOME_SCORE_MODEL_PATH = os.path.join(MODELS_DIR, 'home_score_model.joblib')
AWAY_SCORE_MODEL_PATH = os.path.join(MODELS_DIR, 'away_score_model.joblib')

STATS_COLS = ['pts', 'fg_pct', 'fg3_pct', 'ft_pct', 'ast', 'reb', 'tov', 'stl', 'blk', 'pf', 'plus_minus']
GAMES_TO_LOOK_BACK = 10

def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile = f"ml_pipeline_{ts}.log"
    logpath = os.path.join(LOG_DIR, logfile)
    
    logger = logging.getLogger("MLPipeline")
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

def generate_winner_features(window_size=10):
    LOGGER.info("Starting winner feature generation.")
    if not os.path.exists(DB_PATH):
        LOGGER.error(f"Database not found at {DB_PATH}")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        games_df = pd.read_sql("SELECT * FROM games", conn, parse_dates=['game_date'])
        conn.close()
        LOGGER.info(f"Loaded {len(games_df)} games from database for winner features.")
    except Exception as e:
        LOGGER.error(f"Error loading data for winner features: {e}")
        return

    games_df = games_df.sort_values('game_date').reset_index(drop=True)

    home_df = games_df[['game_id', 'game_date', 'home_team_id'] + [f'home_{col}' for col in STATS_COLS]].copy()
    home_df.rename(columns={'home_team_id': 'team_id'}, inplace=True)
    home_df.columns = ['game_id', 'game_date', 'team_id'] + STATS_COLS

    away_df = games_df[['game_id', 'game_date', 'away_team_id'] + [f'away_{col}' for col in STATS_COLS]].copy()
    away_df.rename(columns={'away_team_id': 'team_id'}, inplace=True)
    away_df.columns = ['game_id', 'game_date', 'team_id'] + STATS_COLS
    
    team_stats_df = pd.concat([home_df, away_df]).sort_values(['team_id', 'game_date'])

    rolling_stats = team_stats_df.groupby('team_id')[STATS_COLS].rolling(window=window_size, min_periods=1).mean().shift(1)
    rolling_stats.columns = [f'{col}_roll_avg' for col in STATS_COLS]
    
    team_stats_with_rolling = pd.concat([team_stats_df.reset_index(drop=True), rolling_stats.reset_index(drop=True)], axis=1)
    team_stats_with_rolling.dropna(subset=[f'{col}_roll_avg' for col in STATS_COLS], inplace=True)

    home_rolling = team_stats_with_rolling.rename(columns={'team_id': 'home_team_id'})
    away_rolling = team_stats_with_rolling.rename(columns={'team_id': 'away_team_id'})

    final_df = games_df.merge(home_rolling[['game_id', 'home_team_id'] + [f'{col}_roll_avg' for col in STATS_COLS]], on=['game_id', 'home_team_id'])
    final_df.rename(columns={f'{col}_roll_avg': f'home_{col}_roll_avg' for col in STATS_COLS}, inplace=True)
    
    final_df = final_df.merge(away_rolling[['game_id', 'away_team_id'] + [f'{col}_roll_avg' for col in STATS_COLS]], on=['game_id', 'away_team_id'])
    final_df.rename(columns={f'{col}_roll_avg': f'away_{col}_roll_avg' for col in STATS_COLS}, inplace=True)

    feature_cols = [f'home_{col}_roll_avg' for col in STATS_COLS] + [f'away_{col}_roll_avg' for col in STATS_COLS]
    final_df = final_df[['game_id', 'game_date', 'outcome'] + feature_cols].copy()
    final_df.dropna(inplace=True)

    final_df.to_csv(WINNER_FEATURES_PATH, index=False)
    LOGGER.info(f"Winner features created with {len(final_df)} samples. Saved to {WINNER_FEATURES_PATH}")


def _get_stats_for_one_team_for_training(team_id, game_date, all_games_dataframe):

    all_games_for_team = all_games_dataframe[
        ((all_games_dataframe['home_team_id'] == team_id) | (all_games_dataframe['away_team_id'] == team_id)) &
        (all_games_dataframe['game_date'] < game_date)
    ]
    all_games_for_team = all_games_for_team.sort_values('game_date')

    if len(all_games_for_team) < GAMES_TO_LOOK_BACK:
        return None

    last_n_games = all_games_for_team.tail(GAMES_TO_LOOK_BACK)
    calculated_averages = {}

    for stat in STATS_COLS:
        home_stats = last_n_games[last_n_games['home_team_id'] == team_id][f'home_{stat}']
        away_stats = last_n_games[last_n_games['away_team_id'] == team_id][f'away_{stat}']
        
        combined_stats = pd.concat([home_stats, away_stats])
        calculated_averages[f'{stat}_roll_avg'] = combined_stats.mean()
        
    return calculated_averages

def generate_score_features():
    LOGGER.info("Starting score feature generation.")
    if not os.path.exists(DB_PATH):
        LOGGER.error(f"Database not found at {DB_PATH}")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        all_games_df = pd.read_sql("SELECT * FROM games", conn, parse_dates=['game_date'])
        all_games_df.sort_values('game_date', inplace=True)
        conn.close()
        LOGGER.info(f"Loaded {len(all_games_df)} games from database for score features.")
    except Exception as e:
        LOGGER.error(f"Error loading data for score features: {e}")
        return

    features_data = []
    for index, game in all_games_df.iterrows():
        home_features = _get_stats_for_one_team_for_training(game['home_team_id'], game['game_date'], all_games_df)
        away_features = _get_stats_for_one_team_for_training(game['away_team_id'], game['game_date'], all_games_df)

        if home_features and away_features:
            sample = {
                'home_pts': game['home_pts'],
                'away_pts': game['away_pts']
            }
            for key, value in home_features.items():
                sample[f'home_{key}'] = value
            for key, value in away_features.items():
                sample[f'away_{key}'] = value
            features_data.append(sample)

    features_df = pd.DataFrame(features_data)
    features_df.to_csv(SCORE_FEATURES_PATH, index=False)
    LOGGER.info(f"Score features created with {len(features_df)} samples. Saved to {SCORE_FEATURES_PATH}")


def train_winner_model():
    LOGGER.info("Starting winner model training.")
    if not os.path.exists(WINNER_FEATURES_PATH):
        LOGGER.error(f"Winner features file not found at {WINNER_FEATURES_PATH}")
        return

    try:
        df = pd.read_csv(WINNER_FEATURES_PATH)
        LOGGER.info(f"Loaded {len(df)} samples for winner model training.")
    except Exception as e:
        LOGGER.error(f"Error loading winner features: {e}")
        return

    X = df.drop(columns=['game_id', 'game_date', 'outcome'])
    y = df['outcome'].apply(lambda x: 1 if x == 'H' else 0) # 1 for home win, 0 for away win

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67, stratify=y)
    LOGGER.info(f"Training set: {len(X_train)} samples. Test set: {len(X_test)} samples.")

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=67)
    model.fit(X_train, y_train)
    LOGGER.info("Winner model training finished.")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    LOGGER.info(f"Winner model accuracy: {accuracy:.4f}")

    model_payload = {
        'model': model,
        'feature_columns': list(X.columns)
    }
    joblib.dump(model_payload, WINNER_MODEL_PATH)
    LOGGER.info(f"Winner model saved to {WINNER_MODEL_PATH}")


def train_score_models():
    LOGGER.info("Starting score models training.")
    if not os.path.exists(SCORE_FEATURES_PATH):
        LOGGER.error(f"Score features file not found at {SCORE_FEATURES_PATH}")
        return

    try:
        df = pd.read_csv(SCORE_FEATURES_PATH)
        LOGGER.info(f"Loaded {len(df)} samples for score model training.")
    except Exception as e:
        LOGGER.error(f"Error loading score features: {e}")
        return
    
    X = df.drop(columns=['home_pts', 'away_pts'])
    y_home = df['home_pts']
    y_away = df['away_pts']

    feature_columns = X.columns.tolist()
    
    X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
        X, y_home, y_away, test_size=0.2, random_state=42
    )
    LOGGER.info(f"Training set size: {len(X_train)}, test set size: {len(X_test)}")

    LOGGER.info("Training Home Score Model")
    home_model = RandomForestRegressor(n_estimators=100, random_state=42)
    home_model.fit(X_train, y_home_train)
    home_predictions = home_model.predict(X_test)
    LOGGER.info(f"Home Score Model MAE: {mean_absolute_error(y_home_test, home_predictions):.2f}, R2: {r2_score(y_home_test, home_predictions):.2f}")

    LOGGER.info("Training Away Score Model")
    away_model = RandomForestRegressor(n_estimators=100, random_state=42)
    away_model.fit(X_train, y_away_train)
    away_predictions = away_model.predict(X_test)
    LOGGER.info(f"Away Score Model MAE: {mean_absolute_error(y_away_test, away_predictions):.2f}, R2: {r2_score(y_away_test, away_predictions):.2f}")

    joblib.dump({'model': home_model, 'feature_columns': feature_columns}, HOME_SCORE_MODEL_PATH)
    joblib.dump({'model': away_model, 'feature_columns': feature_columns}, AWAY_SCORE_MODEL_PATH)
    LOGGER.info(f"Score models saved to {HOME_SCORE_MODEL_PATH} and {AWAY_SCORE_MODEL_PATH}")


if __name__ == "__main__":
    LOGGER.info("Starting ML Pipeline.")
    generate_winner_features()
    generate_score_features()
    train_winner_model()
    train_score_models()
    LOGGER.info("ML Pipeline finished.")
