from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from datetime import date, datetime, timedelta
import json
import os
import sys
import sqlite3
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
sys.path.insert(0, BASE_DIR)

from bets.services import get_nba_schedule
from bets.constants import TEAM_LOGOS
from predict import get_team_id, get_stats_for_one_team


def get_score_prediction(home_team, away_team):
    db_path = os.path.join(DATA_DIR, 'nba_history.db')
    
    try:
        conn = sqlite3.connect(db_path)
        teams_df = pd.read_sql("SELECT * FROM teams", conn)
        all_games_df = pd.read_sql("SELECT * FROM games", conn, parse_dates=['game_date'])
        all_games_df.sort_values('game_date', inplace=True)
        conn.close()
        
        home_model_path = os.path.join(MODELS_DIR, "home_score_model.joblib")
        away_model_path = os.path.join(MODELS_DIR, "away_score_model.joblib")
        
        if not os.path.exists(home_model_path) or not os.path.exists(away_model_path):
            return None
        
        home_model_payload = joblib.load(home_model_path)
        away_model_payload = joblib.load(away_model_path)
        
        home_score_model = home_model_payload['model']
        away_score_model = away_model_payload['model']
        home_feature_columns = home_model_payload['feature_columns']
        away_feature_columns = away_model_payload['feature_columns']
        
        home_id, _ = get_team_id(home_team, teams_df)
        away_id, _ = get_team_id(away_team, teams_df)
        
        if not home_id or not away_id:
            return None
            
        today = datetime.now()
        home_features = get_stats_for_one_team(home_id, today, all_games_df)
        away_features = get_stats_for_one_team(away_id, today, all_games_df)
        
        if home_features is None or away_features is None:
            return None
        
        model_input = {}
        for key, value in home_features.items():
            model_input[f'home_{key}'] = value
        for key, value in away_features.items():
            model_input[f'away_{key}'] = value
        
        input_df = pd.DataFrame([model_input])
        
        home_input_df = input_df.reindex(columns=home_feature_columns, fill_value=0)
        away_input_df = input_df.reindex(columns=away_feature_columns, fill_value=0)
        
        predicted_home_score = home_score_model.predict(home_input_df)[0]
        predicted_away_score = away_score_model.predict(away_input_df)[0]
        
        return {
            'predicted_home_score': int(predicted_home_score),
            'predicted_away_score': int(predicted_away_score)
        }
    except Exception as e:
        print(f"error in score prediction: {e}")
        return None


def future(request):
    nba_schedule = get_nba_schedule()
    api_games = nba_schedule.get("data", [])
    
    upcoming_games = [
        g for g in api_games 
        if g.get("status") in ["", None] or "Final" not in str(g.get("status", ""))
    ]
    
    future_games = []
    
    for game in upcoming_games[:10]:
        try:
            home_team = game["home_team"]["full_name"]
            away_team = game["visitor_team"]["full_name"]
            
            score_info = get_score_prediction(home_team, away_team)
            
            if score_info:
                future_games.append({
                    "home_team": home_team,
                    "away_team": away_team,
                    "match_date": game["date"][:10],
                    "predicted_home": score_info['predicted_home_score'],
                    "predicted_away": score_info['predicted_away_score'],
                    "home_logo": TEAM_LOGOS.get(home_team, "placeholder.png"),
                    "away_logo": TEAM_LOGOS.get(away_team, "placeholder.png"),
                })
        except Exception as e:
            print(f"error generating prediction: {e}")
            continue
    
    return render(request, "future.html", {
        "future_games": future_games,
        "active": "future",
    })


def history(request):    
    db_path = os.path.join(DATA_DIR, 'nba_history.db')
    conn = sqlite3.connect(db_path)
    
    query = """
        SELECT 
            g.game_date,
            g.home_pts,
            g.away_pts,
            ht.team_name as home_team,
            at.team_name as away_team,
            g.home_team_id,
            g.away_team_id
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        WHERE g.home_pts IS NOT NULL 
          AND g.away_pts IS NOT NULL
        ORDER BY g.game_date DESC
        LIMIT 50
    """
    
    games_df = pd.read_sql(query, conn)
    
    all_games_df = pd.read_sql("SELECT * FROM games", conn, parse_dates=['game_date'])
    all_games_df.sort_values('game_date', inplace=True)
    
    teams_df = pd.read_sql("SELECT * FROM teams", conn)
    conn.close()
    
    try:
        home_model_path = os.path.join(MODELS_DIR, "home_score_model.joblib")
        away_model_path = os.path.join(MODELS_DIR, "away_score_model.joblib")
        
        home_model_payload = joblib.load(home_model_path)
        away_model_payload = joblib.load(away_model_path)
        
        home_score_model = home_model_payload['model']
        away_score_model = away_model_payload['model']
        home_feature_columns = home_model_payload['feature_columns']
        away_feature_columns = away_model_payload['feature_columns']
        
        models_loaded = True
    except Exception as e:
        print(f"error loading models: {e}")
        models_loaded = False
    
    history_games = []
    
    for _, game in games_df.iterrows():
        game_dict = {
            "home": game['home_team'],
            "away": game['away_team'],
            "home_pts": int(game['home_pts']),
            "away_pts": int(game['away_pts']),
            "date": game['game_date'][:10],
            "home_logo": TEAM_LOGOS.get(game['home_team'], "placeholder.png"),
            "away_logo": TEAM_LOGOS.get(game['away_team'], "placeholder.png"),
        }
        
        if models_loaded:
            try:
                home_id = game['home_team_id']
                away_id = game['away_team_id']
                
                game_date = datetime.strptime(game['game_date'][:10], '%Y-%m-%d')
                
                home_features = get_stats_for_one_team(home_id, game_date, all_games_df)
                away_features = get_stats_for_one_team(away_id, game_date, all_games_df)
                
                if home_features and away_features:
                    model_input = {}
                    for key, value in home_features.items():
                        model_input[f'home_{key}'] = value
                    for key, value in away_features.items():
                        model_input[f'away_{key}'] = value
                    
                    input_df = pd.DataFrame([model_input])
                    
                    home_input_df = input_df.reindex(columns=home_feature_columns, fill_value=0)
                    away_input_df = input_df.reindex(columns=away_feature_columns, fill_value=0)
                    
                    predicted_home = home_score_model.predict(home_input_df)[0]
                    predicted_away = away_score_model.predict(away_input_df)[0]
                    
                    game_dict["predicted_home"] = int(predicted_home)
                    game_dict["predicted_away"] = int(predicted_away)
                else:
                    game_dict["predicted_home"] = None
                    game_dict["predicted_away"] = None
            except Exception as e:
                print(f"error predicting game: {e}")
                game_dict["predicted_home"] = None
                game_dict["predicted_away"] = None
        else:
            game_dict["predicted_home"] = None
            game_dict["predicted_away"] = None
        
        history_games.append(game_dict)
    
    return render(request, "history.html", {
        "history_games": history_games,
        "active": "history",
    })