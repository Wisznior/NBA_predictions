import pandas as pd
import sqlite3
import os
import sys
from datetime import datetime
import logging

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
LOG_DIR = os.path.join(BASE, "logs")

DATABASE_FILE = os.path.join(DATA_DIR, 'nba_history.db')
OUTPUT_FEATURES_FILE = os.path.join(DATA_DIR, 'ml_score_features.csv')

GAMES_TO_LOOK_BACK = 10
STATS_LIST = ['pts', 'fg_pct', 'fg3_pct', 'ft_pct', 'ast', 'reb', 'tov', 'stl', 'blk', 'pf', 'plus_minus']

def setup_logger():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile = f"generate_score_features_{ts}.log"
    logpath = os.path.join(LOG_DIR, logfile)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(logpath), logging.StreamHandler(sys.stdout)])
    return logging.getLogger(__name__)

def get_team_id(team_name, teams_df):
    exact_match = teams_df[teams_df['team_name'].str.lower() == team_name.lower()]
    if not exact_match.empty:
        return exact_match.iloc[0]['team_id'], exact_match.iloc[0]['team_name']

    partial_match = teams_df[teams_df['team_name'].str.lower().str.contains(team_name.lower())]
    if not partial_match.empty:
        return partial_match.iloc[0]['team_id'], partial_match.iloc[0]['team_name']
    
    return None, None

def get_stats_for_one_team_for_training(team_id, game_date, all_games_dataframe):
    all_games_for_team = all_games_dataframe[
        ((all_games_dataframe['home_team_id'] == team_id) | (all_games_dataframe['away_team_id'] == team_id)) &
        (all_games_dataframe['game_date'] < game_date)
    ]
    all_games_for_team = all_games_for_team.sort_values('game_date').reset_index(drop=True)

    if len(all_games_for_team) < GAMES_TO_LOOK_BACK:
        return None

    last_n_games = all_games_for_team.tail(GAMES_TO_LOOK_BACK)
    calculated_averages = {}

    for stat in STATS_LIST:
        home_stats = last_n_games[last_n_games['home_team_id'] == team_id][f'home_{stat}']
        away_stats = last_n_games[last_n_games['away_team_id'] == team_id][f'away_{stat}']
        
        combined_stats = pd.concat([home_stats, away_stats])
        calculated_averages[f'{stat}_roll_avg'] = combined_stats.mean()
        
    return calculated_averages

def generate_features(logger):
    logger.info("Starting feature generation for score prediction models")

    if not os.path.exists(DATABASE_FILE):
        logger.error(f"Database not found at {DATABASE_FILE} Run the data collection scripts first.")
        return

    conn = sqlite3.connect(DATABASE_FILE)
    try:
        teams_df = pd.read_sql("SELECT * FROM teams", conn) # Needed for get_team_id
        all_games_df = pd.read_sql("SELECT * FROM games", conn, parse_dates=['game_date'])
        all_games_df.sort_values('game_date', inplace=True)
        logger.info(f"Loaded {len(all_games_df)} historical games from database.")
    finally:
        conn.close()

    features_data = []

    for index, game in all_games_df.iterrows():
        game_date = game['game_date']
        home_team_id = game['home_team_id']
        away_team_id = game['away_team_id']
        home_score = game['home_pts']
        away_score = game['away_pts']

        home_features = get_stats_for_one_team_for_training(home_team_id, game_date, all_games_df)
        away_features = get_stats_for_one_team_for_training(away_team_id, game_date, all_games_df)

        if home_features and away_features:
            sample_features = {
                'game_date': game_date,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'home_pts': home_score,
                'away_pts': away_score
            }
            for key, value in home_features.items():
                sample_features[f'home_{key}'] = value
            for key, value in away_features.items():
                sample_features[f'away_{key}'] = value
            
            features_data.append(sample_features)
        else:
            logger.debug(f"Skipping game {index} due to insufficient historical data for one or both teams.")

    if not features_data:
        logger.error("No features generated")
        return

    features_df = pd.DataFrame(features_data)
    features_df.to_csv(OUTPUT_FEATURES_FILE, index=False)
    logger.info(f"Generated {len(features_df)} feature samples and saved to {OUTPUT_FEATURES_FILE}")

if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Starting feature generation script.")
    generate_features(logger)
    logger.info("Feature generation script finished.")
