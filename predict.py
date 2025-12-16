import argparse
import pandas as pd
import sqlite3
import joblib
from datetime import datetime
import os
import sys

PROJECT_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(PROJECT_FOLDER_PATH, "data")
MODELS_FOLDER = os.path.join(PROJECT_FOLDER_PATH, "models")
DATABASE_FILE = os.path.join(DATA_FOLDER, 'nba_history.db')
MODEL_FILE = os.path.join(MODELS_FOLDER, 'winner_model.joblib')

GAMES_TO_LOOK_BACK = 10

STATS_LIST = ['pts', 'fg_pct', 'fg3_pct', 'ft_pct', 'ast', 'reb', 'tov', 'stl', 'blk', 'pf', 'plus_minus']

def get_team_id(team_name, teams_df):

    # Finds a team's ID
    exact_match = teams_df[teams_df['team_name'].str.lower() == team_name.lower()]
    if not exact_match.empty:
        return exact_match.iloc[0]['team_id'], exact_match.iloc[0]['team_name']

    partial_match = teams_df[teams_df['team_name'].str.lower().str.contains(team_name.lower())]
    if not partial_match.empty:
        return partial_match.iloc[0]['team_id'], partial_match.iloc[0]['team_name']
    
    return None, None

def get_stats_for_one_team(team_id, game_date, all_games_dataframe):

    all_games_for_team = all_games_dataframe[
        ((all_games_dataframe['home_team_id'] == team_id) | (all_games_dataframe['away_team_id'] == team_id)) &
        (all_games_dataframe['game_date'] < game_date)
    ]

    last_n_games = all_games_for_team.tail(GAMES_TO_LOOK_BACK)
    if len(last_n_games) < GAMES_TO_LOOK_BACK:
        return None

    calculated_averages = {}

    for stat in STATS_LIST:
        home_stats = last_n_games[last_n_games['home_team_id'] == team_id][f'home_{stat}']
        away_stats = last_n_games[last_n_games['away_team_id'] == team_id][f'away_{stat}']
        
        combined_stats = pd.concat([home_stats, away_stats])
        calculated_averages[f'{stat}_roll_avg'] = combined_stats.mean()
        
    return calculated_averages

def predict_winner(home_team_name, away_team_name):

    if not os.path.exists(DATABASE_FILE):
        print(f"ERROR: Database not found at {DATABASE_FILE}. Run the data collection scripts first.")
        sys.exit(1)
    if not os.path.exists(MODEL_FILE):
        print(f"ERROR: Model not found at {MODEL_FILE}. Run the training script first.")
        sys.exit(1)

    conn = sqlite3.connect(DATABASE_FILE)

    try:
        teams_df = pd.read_sql("SELECT * FROM teams", conn)
        all_games_df = pd.read_sql("SELECT * FROM games", conn, parse_dates=['game_date'])
        all_games_df.sort_values('game_date', inplace=True)
    finally:
        conn.close()

    model_payload = joblib.load(MODEL_FILE)
    model = model_payload['model']
    feature_columns = model_payload['feature_columns']

    home_id, full_home_name = get_team_id(home_team_name, teams_df)
    away_id, full_away_name = get_team_id(away_team_name, teams_df)

    if not home_id:
        print(f"ERROR: Could not find home team: '{home_team_name}'")
        sys.exit(1)
    if not away_id:
        print(f"ERROR: Could not find away team: '{away_team_name}'")
        sys.exit(1)

    print(f"\nPredicting for: {full_home_name} (Home) vs. {full_away_name} (Away)")

    today = datetime.now()
    home_features = get_stats_for_one_team(home_id, today, all_games_df)
    away_features = get_stats_for_one_team(away_id, today, all_games_df)

    if home_features is None:
        print(f"WARNING: Not enough historical data for {full_home_name} to make a prediction")
        sys.exit(1)
    if away_features is None:
        print(f"WARNING: Not enough historical data for {full_away_name} to make a prediction")
        sys.exit(1)

    model_input = {}
    for key, value in home_features.items():
        model_input[f'home_{key}'] = value
    for key, value in away_features.items():
        model_input[f'away_{key}'] = value
    
    input_df = pd.DataFrame([model_input], columns=feature_columns)

    probabilities = model.predict_proba(input_df)[0]
    away_win_prob = probabilities[0]
    home_win_prob = probabilities[1]

    if home_win_prob > away_win_prob:
        winner = full_home_name
        confidence = home_win_prob
    else:
        winner = full_away_name
        confidence = away_win_prob

    print(f"Predicted Winner: {winner}")
    print(f"Confidence: {confidence:.1%}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Predict the winner of an upcoming NBA game.")
    parser.add_argument("--home", required=True, type=str, help="Name of the home team.")
    parser.add_argument("--away", required=True, type=str, help="Name of the away team.")
    
    args = parser.parse_args()
    
    predict_winner(args.home, args.away)
