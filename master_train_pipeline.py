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
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

DB_PATH = os.path.join(DATA_DIR, 'nba_history.db')
WINNER_FEATURES_PATH = os.path.join(DATA_DIR, 'ml_winner_features.csv')
SCORE_FEATURES_PATH = os.path.join(DATA_DIR, 'ml_score_features.csv')

GBRF_WINNER_MODEL_PATH = os.path.join(MODELS_DIR, 'winner_model.joblib')
GBRF_HOME_SCORE_MODEL_PATH = os.path.join(MODELS_DIR, 'home_score_model.joblib')
GBRF_AWAY_SCORE_MODEL_PATH = os.path.join(MODELS_DIR, 'away_score_model.joblib')

XGB_WINNER_MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_winner_model.joblib')
XGB_HOME_SCORE_MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_home_score_model.joblib')
XGB_AWAY_SCORE_MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_away_score_model.joblib')

STATS_COLS = ['pts', 'fg_pct', 'fg3_pct', 'ft_pct', 'ast', 'reb', 'tov', 'stl', 'blk', 'pf', 'plus_minus']
GAMES_TO_LOOK_BACK = 10

def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile = f"master_train_pipeline_{ts}.log"
    logpath = os.path.join(LOG_DIR, logfile)
    
    logger = logging.getLogger("MasterTrainPipeline")
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

def generate_winner_features(window_size=GAMES_TO_LOOK_BACK):
    LOGGER.info("Starting winner feature generation.")
    if not os.path.exists(DB_PATH):
        LOGGER.error(f"Database not found at {DB_PATH}")
        return False

    try:
        conn = sqlite3.connect(DB_PATH)
        games_df = pd.read_sql("SELECT * FROM games", conn, parse_dates=['game_date'])
        conn.close()
        LOGGER.info(f"Loaded {len(games_df)} games from database for winner features.")
    except Exception as e:
        LOGGER.error(f"Error loading data for winner features: {e}")
        return False

    games_df = games_df.sort_values('game_date').reset_index(drop=True)

    all_team_appearances = []
    for index, row in games_df.iterrows():
        all_team_appearances.append({'game_id': row['game_id'], 'game_date': row['game_date'], 'team_id': row['home_team_id'], 'is_home': 1})
        all_team_appearances.append({'game_id': row['game_id'], 'game_date': row['game_date'], 'team_id': row['away_team_id'], 'is_home': 0})
    team_appearances_df = pd.DataFrame(all_team_appearances)
    team_appearances_df = team_appearances_df.sort_values(by=['team_id', 'game_date']).reset_index(drop=True)

    team_appearances_df['prev_game_date'] = team_appearances_df.groupby('team_id')['game_date'].shift(1)
    team_appearances_df['days_rest'] = (team_appearances_df['game_date'] - team_appearances_df['prev_game_date']).dt.days

    home_rest = team_appearances_df[team_appearances_df['is_home'] == 1][['game_id', 'days_rest']].rename(columns={'days_rest': 'home_team_days_rest'})
    away_rest = team_appearances_df[team_appearances_df['is_home'] == 0][['game_id', 'days_rest']].rename(columns={'days_rest': 'away_team_days_rest'})

    games_df = games_df.merge(home_rest, on='game_id', how='left')
    games_df = games_df.merge(away_rest, on='game_id', how='left')

    home_df = games_df[['game_id', 'game_date', 'home_team_id'] + [f'home_{col}' for col in STATS_COLS] + ['away_pts']].copy()
    home_df.rename(columns={'home_team_id': 'team_id'}, inplace=True)
    home_df.columns = ['game_id', 'game_date', 'team_id'] + STATS_COLS + ['pts_allowed']

    away_df = games_df[['game_id', 'game_date', 'away_team_id'] + [f'away_{col}' for col in STATS_COLS] + ['home_pts']].copy()
    away_df.rename(columns={'away_team_id': 'team_id'}, inplace=True)
    away_df.columns = ['game_id', 'game_date', 'team_id'] + STATS_COLS + ['pts_allowed']
    
    team_stats_df = pd.concat([home_df, away_df]).sort_values(['team_id', 'game_date'])
    team_stats_df['pts_differential'] = team_stats_df['pts'] - team_stats_df['pts_allowed']

    ROLLING_STATS_TO_CALCULATE = STATS_COLS + ['pts_allowed', 'pts_differential']

    rolling_stats = team_stats_df.groupby('team_id')[ROLLING_STATS_TO_CALCULATE].rolling(window=window_size, min_periods=1).mean().shift(1)
    rolling_stats.columns = [f'{col}_roll_avg' for col in ROLLING_STATS_TO_CALCULATE]
    
    team_stats_with_rolling = pd.concat([team_stats_df.reset_index(drop=True), rolling_stats.reset_index(drop=True)], axis=1)
    team_stats_with_rolling.dropna(subset=[f'{col}_roll_avg' for col in ROLLING_STATS_TO_CALCULATE], inplace=True)

    home_rolling = team_stats_with_rolling.rename(columns={'team_id': 'home_team_id'})
    away_rolling = team_stats_with_rolling.rename(columns={'team_id': 'away_team_id'})

    final_df = games_df.merge(home_rolling[['game_id', 'home_team_id'] + [f'{col}_roll_avg' for col in ROLLING_STATS_TO_CALCULATE]], on=['game_id', 'home_team_id'])
    final_df.rename(columns={f'{col}_roll_avg': f'home_{col}_roll_avg' for col in ROLLING_STATS_TO_CALCULATE}, inplace=True)
    
    final_df = final_df.merge(away_rolling[['game_id', 'away_team_id'] + [f'{col}_roll_avg' for col in ROLLING_STATS_TO_CALCULATE]], on=['game_id', 'away_team_id'])
    final_df.rename(columns={f'{col}_roll_avg': f'away_{col}_roll_avg' for col in ROLLING_STATS_TO_CALCULATE}, inplace=True)

    feature_cols = ([f'home_{col}_roll_avg' for col in STATS_COLS] +
                    [f'away_{col}_roll_avg' for col in STATS_COLS] +
                    ['home_pts_allowed_roll_avg', 'away_pts_allowed_roll_avg'] +
                    ['home_team_days_rest', 'away_team_days_rest'] +
                    ['home_pts_differential_roll_avg', 'away_pts_differential_roll_avg'])
    final_df = final_df[['game_id', 'game_date', 'outcome'] + feature_cols].copy()
    final_df.dropna(inplace=True)

    final_df.to_csv(WINNER_FEATURES_PATH, index=False)
    LOGGER.info(f"Winner features created with {len(final_df)} samples. Saved to {WINNER_FEATURES_PATH}")
    return True

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
        
    home_pts_allowed = last_n_games[last_n_games['home_team_id'] == team_id]['away_pts']
    away_pts_allowed = last_n_games[last_n_games['away_team_id'] == team_id]['home_pts']
    combined_pts_allowed = pd.concat([home_pts_allowed, away_pts_allowed])
    calculated_averages['pts_allowed_roll_avg'] = combined_pts_allowed.mean()
        
    home_pts = last_n_games[last_n_games['home_team_id'] == team_id]['home_pts']
    away_pts = last_n_games[last_n_games['away_team_id'] == team_id]['away_pts']

    home_diff = home_pts - home_pts_allowed
    away_diff = away_pts - away_pts_allowed
    combined_pts_diff = pd.concat([home_diff, away_diff])
    calculated_averages['pts_differential_roll_avg'] = combined_pts_diff.mean()
        
    return calculated_averages

def generate_score_features():
    LOGGER.info("Starting score feature generation.")
    if not os.path.exists(DB_PATH):
        LOGGER.error(f"Database not found at {DB_PATH}")
        return False

    try:
        conn = sqlite3.connect(DB_PATH)
        all_games_df = pd.read_sql("SELECT * FROM games", conn, parse_dates=['game_date'])
        all_games_df.sort_values('game_date', inplace=True)

        all_team_appearances = []
        for index, row in all_games_df.iterrows():
            all_team_appearances.append({'game_id': row['game_id'], 'game_date': row['game_date'], 'team_id': row['home_team_id'], 'is_home': 1})
            all_team_appearances.append({'game_id': row['game_id'], 'game_date': row['game_date'], 'team_id': row['away_team_id'], 'is_home': 0})
        team_appearances_df = pd.DataFrame(all_team_appearances)
        team_appearances_df = team_appearances_df.sort_values(by=['team_id', 'game_date']).reset_index(drop=True)

        team_appearances_df['prev_game_date'] = team_appearances_df.groupby('team_id')['game_date'].shift(1)
        team_appearances_df['days_rest'] = (team_appearances_df['game_date'] - team_appearances_df['prev_game_date']).dt.days

        home_rest = team_appearances_df[team_appearances_df['is_home'] == 1][['game_id', 'days_rest']].rename(columns={'days_rest': 'home_team_days_rest'})
        away_rest = team_appearances_df[team_appearances_df['is_home'] == 0][['game_id', 'days_rest']].rename(columns={'days_rest': 'away_team_days_rest'})

        all_games_df = all_games_df.merge(home_rest, on='game_id', how='left')
        all_games_df = all_games_df.merge(away_rest, on='game_id', how='left')
        conn.close()
        LOGGER.info(f"Loaded {len(all_games_df)} games from database for score features.")
    except Exception as e:
        LOGGER.error(f"Error loading data for score features: {e}")
        return False

    features_data = []
    for index, game in all_games_df.iterrows():
        home_features = _get_stats_for_one_team_for_training(game['home_team_id'], game['game_date'], all_games_df)
        away_features = _get_stats_for_one_team_for_training(game['away_team_id'], game['game_date'], all_games_df)

        if home_features and away_features:
            sample = {
                'home_pts': game['home_pts'],
                'away_pts': game['away_pts'],
                'home_team_days_rest': game['home_team_days_rest'],
                'away_team_days_rest': game['away_team_days_rest']
            }
            for key, value in home_features.items():
                sample[f'home_{key}'] = value
            for key, value in away_features.items():
                sample[f'away_{key}'] = value
            features_data.append(sample)

    features_df = pd.DataFrame(features_data)
    features_df.to_csv(SCORE_FEATURES_PATH, index=False)
    LOGGER.info(f"Score features created with {len(features_df)} samples. Saved to {SCORE_FEATURES_PATH}")
    return True

def train_gbrf_winner_model():
    LOGGER.info("Starting Gradient Boosting winner model training.")
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
    LOGGER.info("Gradient Boosting winner model training finished.")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    LOGGER.info(f"Gradient Boosting winner model accuracy: {accuracy:.4f}")

    model_payload = {
        'model': model,
        'feature_columns': list(X.columns)
    }
    joblib.dump(model_payload, GBRF_WINNER_MODEL_PATH)
    LOGGER.info(f"Gradient Boosting winner model saved to {GBRF_WINNER_MODEL_PATH}")


def train_gbrf_score_models():
    LOGGER.info("Starting Random Forest score models training.")
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

    LOGGER.info("Training Random Forest Home Score Model")
    home_model = RandomForestRegressor(n_estimators=100, random_state=42)
    home_model.fit(X_train, y_home_train)
    home_predictions = home_model.predict(X_test)
    LOGGER.info(f"Random Forest Home Score Model MAE: {mean_absolute_error(y_home_test, home_predictions):.2f}, R2: {r2_score(y_home_test, home_predictions):.2f}")

    LOGGER.info("Training Random Forest Away Score Model")
    away_model = RandomForestRegressor(n_estimators=100, random_state=42)
    away_model.fit(X_train, y_away_train)
    away_predictions = away_model.predict(X_test)
    LOGGER.info(f"Random Forest Away Score Model MAE: {mean_absolute_error(y_away_test, away_predictions):.2f}, R2: {r2_score(y_away_test, away_predictions):.2f}")

    joblib.dump({'model': home_model, 'feature_columns': feature_columns}, GBRF_HOME_SCORE_MODEL_PATH)
    joblib.dump({'model': away_model, 'feature_columns': feature_columns}, GBRF_AWAY_SCORE_MODEL_PATH)
    LOGGER.info(f"Random Forest score models saved to {GBRF_HOME_SCORE_MODEL_PATH} and {GBRF_AWAY_SCORE_MODEL_PATH}")

def train_xgboost_winner_model():
    LOGGER.info("Starting XGBoost winner model training.")
    if not os.path.exists(WINNER_FEATURES_PATH):
        LOGGER.error(f"Winner features file not found at {WINNER_FEATURES_PATH}. Please run feature generation.")
        return

    try:
        df = pd.read_csv(WINNER_FEATURES_PATH)
        LOGGER.info(f"Loaded {len(df)} samples for XGBoost winner model training.")
    except Exception as e:
        LOGGER.error(f"Error loading winner features: {e}")
        return

    X = df.drop(columns=['game_id', 'game_date', 'outcome'])
    y = df['outcome'].apply(lambda x: 1 if x == 'H' else 0) # 1 for home win, 0 for away win

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67, stratify=y)
    LOGGER.info(f"Training set: {len(X_train)} samples. Test set: {len(X_test)} samples.")

    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, 
                              n_estimators=100, learning_rate=0.1, max_depth=3, random_state=67)
    model.fit(X_train, y_train)
    LOGGER.info("XGBoost winner model training finished.")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    LOGGER.info(f"XGBoost winner model accuracy: {accuracy:.4f}")

    model_payload = {
        'model': model,
        'feature_columns': list(X.columns)
    }
    joblib.dump(model_payload, XGB_WINNER_MODEL_PATH)
    LOGGER.info(f"XGBoost winner model saved to {XGB_WINNER_MODEL_PATH}")


def train_xgboost_score_models():
    LOGGER.info("Starting XGBoost score models training.")
    if not os.path.exists(SCORE_FEATURES_PATH):
        LOGGER.error(f"Score features file not found at {SCORE_FEATURES_PATH}. Please run feature generation.")
        return

    try:
        df = pd.read_csv(SCORE_FEATURES_PATH)
        LOGGER.info(f"Loaded {len(df)} samples for XGBoost score model training.")
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

    LOGGER.info("Training XGBoost Home Score Model")
    home_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    home_model.fit(X_train, y_home_train)
    home_predictions = home_model.predict(X_test)
    LOGGER.info(f"XGBoost Home Score Model MAE: {mean_absolute_error(y_home_test, home_predictions):.2f}, R2: {r2_score(y_home_test, home_predictions):.2f}")

    LOGGER.info("Training XGBoost Away Score Model")
    away_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    away_model.fit(X_train, y_away_train)
    away_predictions = away_model.predict(X_test)
    LOGGER.info(f"XGBoost Away Score Model MAE: {mean_absolute_error(y_away_test, away_predictions):.2f}, R2: {r2_score(y_away_test, away_predictions):.2f}")

    joblib.dump({'model': home_model, 'feature_columns': feature_columns}, XGB_HOME_SCORE_MODEL_PATH)
    joblib.dump({'model': away_model, 'feature_columns': feature_columns}, XGB_AWAY_SCORE_MODEL_PATH)
    LOGGER.info(f"XGBoost score models saved to {XGB_HOME_SCORE_MODEL_PATH} and {XGB_AWAY_SCORE_MODEL_PATH}")


if __name__ == "__main__":
    LOGGER.info("Starting Master Training Pipeline.")
    
    winner_features_generated = generate_winner_features()
    score_features_generated = generate_score_features()

    if winner_features_generated:
        train_gbrf_winner_model()
        train_xgboost_winner_model()
    else:
        LOGGER.error("Skipping winner model training due to feature generation failure.")

    if score_features_generated:
        train_gbrf_score_models()
        train_xgboost_score_models()
    else:
        LOGGER.error("Skipping score model training due to feature generation failure.")
        
    LOGGER.info("Master Training Pipeline finished.")