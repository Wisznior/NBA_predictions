import pandas as pd
import sqlite3
import os
import logging
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
LOG_DIR = os.path.join(BASE, "logs")

os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile = f"feature_engineering_{ts}.log"
    logpath = os.path.join(LOG_DIR, logfile)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(logpath), logging.StreamHandler()])
    return logging.getLogger(__name__)

#Loads data from the database and calculates rolling averages for teams. Saves them to CSV file
def create_features(db_path, output_csv_path, logger, window_size=10):

    if not os.path.exists(db_path):
        logger.error(f"Database not found - {db_path}")
        return

    try:
        conn = sqlite3.connect(db_path)
        games_df = pd.read_sql("SELECT * FROM games", conn, parse_dates=['game_date'])
        conn.close()
        logger.info(f"Loaded {len(games_df)} games from database")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    games_df = games_df.sort_values('game_date').reset_index(drop=True)

    # Statistics we want use for rolling averages
    stats_cols = ['pts', 'fg_pct', 'fg3_pct', 'ft_pct', 'ast', 'reb', 'tov', 'stl', 'blk', 'pf', 'plus_minus']
    
    # Separate data for home and away teams
    home_df = games_df[['game_id', 'game_date', 'home_team_id'] + [f'home_{col}' for col in stats_cols]].copy()
    home_df.rename(columns={'home_team_id': 'team_id'}, inplace=True)
    home_df.columns = ['game_id', 'game_date', 'team_id'] + stats_cols

    away_df = games_df[['game_id', 'game_date', 'away_team_id'] + [f'away_{col}' for col in stats_cols]].copy()
    away_df.rename(columns={'away_team_id': 'team_id'}, inplace=True)
    away_df.columns = ['game_id', 'game_date', 'team_id'] + stats_cols
    
    # Single datadrame with all team stats
    team_stats_df = pd.concat([home_df, away_df]).sort_values(['team_id', 'game_date'])

    rolling_stats = team_stats_df.groupby('team_id')[stats_cols].rolling(window=window_size, min_periods=1).mean().shift(1)
    rolling_stats.columns = [f'{col}_roll_avg' for col in stats_cols]
    
    # We merge statistics with original data
    team_stats_with_rolling = pd.concat([team_stats_df.reset_index(drop=True), rolling_stats.reset_index(drop=True)], axis=1)
    team_stats_with_rolling.dropna(subset=[f'{col}_roll_avg' for col in stats_cols], inplace=True)

    home_rolling = team_stats_with_rolling.rename(columns={'team_id': 'home_team_id'})
    away_rolling = team_stats_with_rolling.rename(columns={'team_id': 'away_team_id'})

    # Merge with the main games table
    final_df = games_df.merge(home_rolling[['game_id', 'home_team_id'] + [f'{col}_roll_avg' for col in stats_cols]], on=['game_id', 'home_team_id'])
    final_df.rename(columns={f'{col}_roll_avg': f'home_{col}_roll_avg' for col in stats_cols}, inplace=True)
    
    final_df = final_df.merge(away_rolling[['game_id', 'away_team_id'] + [f'{col}_roll_avg' for col in stats_cols]], on=['game_id', 'away_team_id'])
    final_df.rename(columns={f'{col}_roll_avg': f'away_{col}_roll_avg' for col in stats_cols}, inplace=True)

    # Selection of final columns
    feature_cols = [f'home_{col}_roll_avg' for col in stats_cols] + [f'away_{col}_roll_avg' for col in stats_cols]
    final_df = final_df[['game_id', 'game_date', 'outcome'] + feature_cols].copy()
    final_df.dropna(inplace=True)

    logger.info(f"Created {len(final_df)} samples for model training")
    
    final_df.to_csv(output_csv_path, index=False)
    logger.info(f"Saved features to {output_csv_path}")


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Starting feature creation process")
    
    db_path = os.path.join(DATA_DIR, 'nba_history.db')
    output_csv_path = os.path.join(DATA_DIR, 'ml_game_features.csv')
    
    create_features(db_path, output_csv_path, logger)
    
    logger.info("Feature creation finished.")