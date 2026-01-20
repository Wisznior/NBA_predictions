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

GBRF_WINNER_MODEL_PATH = os.path.join(MODELS_FOLDER, 'winner_model.joblib')
GBRF_HOME_SCORE_MODEL_PATH = os.path.join(MODELS_FOLDER, 'home_score_model.joblib')
GBRF_AWAY_SCORE_MODEL_PATH = os.path.join(MODELS_FOLDER, 'away_score_model.joblib')

XGB_WINNER_MODEL_PATH = os.path.join(MODELS_FOLDER, 'xgb_winner_model.joblib')
XGB_HOME_SCORE_MODEL_PATH = os.path.join(MODELS_FOLDER, 'xgb_home_score_model.joblib')
XGB_AWAY_SCORE_MODEL_PATH = os.path.join(MODELS_FOLDER, 'xgb_away_score_model.joblib')

GAMES_TO_LOOK_BACK = 10
STATS_LIST = ['pts', 'fg_pct', 'fg3_pct', 'ft_pct', 'ast', 'reb', 'tov', 'stl', 'blk', 'pf', 'plus_minus']

def get_team_id(team_name, teams_df):
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
    all_games_for_team = all_games_for_team.sort_values('game_date')

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

def master_predict_all(home_team_name, away_team_name, model_type):
    if not os.path.exists(DATABASE_FILE):
        print(f"ERROR: Database not found at {DATABASE_FILE}. Run the data collection scripts first.")
        sys.exit(1)
    
    winner_model_path = None
    home_score_model_path = None
    away_score_model_path = None

    if model_type == 'gb_rf':
        winner_model_path = GBRF_WINNER_MODEL_PATH
        home_score_model_path = GBRF_HOME_SCORE_MODEL_PATH
        away_score_model_path = GBRF_AWAY_SCORE_MODEL_PATH
    elif model_type == 'xgboost':
        winner_model_path = XGB_WINNER_MODEL_PATH
        home_score_model_path = XGB_HOME_SCORE_MODEL_PATH
        away_score_model_path = XGB_AWAY_SCORE_MODEL_PATH
    else:
        print(f"ERROR: Invalid model_type '{model_type}'. Choose 'gb_rf' or 'xgboost'.")
        sys.exit(1)

    if not (os.path.exists(winner_model_path) and 
            os.path.exists(home_score_model_path) and 
            os.path.exists(away_score_model_path)):
        print(f"ERROR: One or more '{model_type}' models not found in {MODELS_FOLDER}. Ensure master_train_pipeline.py has been run.")
        sys.exit(1)

    conn = sqlite3.connect(DATABASE_FILE)
    try:
        teams_df = pd.read_sql("SELECT * FROM teams", conn)
        all_games_df = pd.read_sql("SELECT * FROM games", conn, parse_dates=['game_date'])
        all_games_df.sort_values('game_date', inplace=True)
    finally:
        conn.close()

    winner_model_payload = joblib.load(winner_model_path)
    winner_model = winner_model_payload['model']
    winner_feature_columns = winner_model_payload['feature_columns']

    home_score_model_payload = joblib.load(home_score_model_path)
    home_score_model = home_score_model_payload['model']
    home_score_feature_columns = home_score_model_payload['feature_columns']

    away_score_model_payload = joblib.load(away_score_model_path)
    away_score_model = away_score_model_payload['model']
    away_score_feature_columns = away_score_model_payload['feature_columns']

    home_id, full_home_name = get_team_id(home_team_name, teams_df)
    away_id, full_away_name = get_team_id(away_team_name, teams_df)

    if not home_id:
        print(f"ERROR: Could not find home team: '{home_team_name}'")
        sys.exit(1)
    if not away_id:
        print(f"ERROR: Could not find away team: '{away_team_name}'")
        sys.exit(1)

    print(f"\n--- NBA Game Prediction ({model_type.upper()} Models) ---")
    print(f"Matchup: {full_home_name} (Home) vs. {full_away_name} (Away)")

    today = datetime.now()
    home_features = get_stats_for_one_team(home_id, today, all_games_df)
    away_features = get_stats_for_one_team(away_id, today, all_games_df)

    if home_features is None:
        print(f"WARNING: Not enough historical data for {full_home_name} to make a prediction")
        sys.exit(1)
    if away_features is None:
        print(f"WARNING: Not enough historical data for {full_away_name} to make a prediction")
        sys.exit(1)

    model_input_dict = {}
    for key, value in home_features.items():
        model_input_dict[f'home_{key}'] = value
    for key, value in away_features.items():
        model_input_dict[f'away_{key}'] = value
    
    input_df = pd.DataFrame([model_input_dict])

    winner_input_df = input_df.reindex(columns=winner_feature_columns, fill_value=0)
    probabilities = winner_model.predict_proba(winner_input_df)[0]
    away_win_prob = probabilities[0]
    home_win_prob = probabilities[1]

    if home_win_prob > away_win_prob:
        predicted_winner = full_home_name
        winner_confidence = home_win_prob
    else:
        predicted_winner = full_away_name
        winner_confidence = away_win_prob

    print(f"\nPredicted Winner: {predicted_winner}")
    print(f"Confidence: {winner_confidence:.1%}")

    home_score_input_df = input_df.reindex(columns=home_score_feature_columns, fill_value=0)
    away_score_input_df = input_df.reindex(columns=away_score_feature_columns, fill_value=0)

    predicted_home_score = home_score_model.predict(home_score_input_df)[0]
    predicted_away_score = away_score_model.predict(away_score_input_df)[0]

    print(f"\nPredicted Scores:")
    print(f"  {full_home_name}: {predicted_home_score:.0f}")
    print(f"  {full_away_name}: {predicted_away_score:.0f}")
    print(f"Predicted Match Result: {full_home_name} {predicted_home_score:.0f} - {predicted_away_score:.0f} {full_away_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the winner and scores of an NBA game using specified model type.")
    parser.add_argument("--home", required=True, type=str, help="Name of the home team.")
    parser.add_argument("--away", required=True, type=str, help="Name of the away team.")
    parser.add_argument("--model_type", required=True, choices=['gb_rf', 'xgboost'], 
                        help="Type of model to use for prediction: 'gb_rf' (Gradient Boosting/Random Forest) or 'xgboost'.")
    
    args = parser.parse_args()
    
    master_predict_all(args.home, args.away, args.model_type)
