import pandas as pd
import json
import os
import sys
import joblib
from datetime import datetime
from predict import get_team_id, get_stats_for_one_team
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BETTING_DIR = os.path.join(BASE_DIR, "betting_data")

DB_PATH = os.path.join(DATA_DIR, 'nba_history.db')
BETTING_DATA_PATH = os.path.join(BETTING_DIR, 'parsed_data_new.json')
WINNER_MODEL_PATH = os.path.join(MODELS_DIR, 'winner_model.joblib')
COMPARISON_RESULTS_PATH = os.path.join(DATA_DIR, 'betting_comparison_results.json')


def odds_to_probability(decimal_odds):
    if decimal_odds <= 1.0:
        return 0.0
    return 1.0 / decimal_odds


def get_average_odds(bookmakers_dict):
    if not bookmakers_dict:
        return None
    
    odds_list = list(bookmakers_dict.values())
    return sum(odds_list) / len(odds_list)


def load_betting_data():
    if not os.path.exists(BETTING_DATA_PATH):
        print(f"Nie znaleziono pliku z kursami: {BETTING_DATA_PATH}")
        return []
    
    try:
        with open(BETTING_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Wczytano {len(data)} meczów z kursami")
        return data
    except Exception as e:
        print(f"Błąd podczas wczytywania kursów: {e}")
        return []


def predict_game_probabilities(home_team_name, away_team_name):
    if not os.path.exists(DB_PATH):
        print(f"brak bazy danych: {DB_PATH}")
        return None
    
    if not os.path.exists(WINNER_MODEL_PATH):
        print(f"brak modelu: {WINNER_MODEL_PATH}")
        return None
    
    try:
        conn = sqlite3.connect(DB_PATH)
        teams_df = pd.read_sql("SELECT * FROM teams", conn)
        all_games_df = pd.read_sql("SELECT * FROM games", conn, parse_dates=['game_date'])
        all_games_df.sort_values('game_date', inplace=True)
        conn.close()
        
        model_payload = joblib.load(WINNER_MODEL_PATH)
        model = model_payload['model']
        feature_columns = model_payload['feature_columns']
        
        home_id, full_home_name = get_team_id(home_team_name, teams_df)
        away_id, full_away_name = get_team_id(away_team_name, teams_df)
        
        if not home_id or not away_id:
            print(f"Nie znaleziono drużyn: {home_team_name} lub {away_team_name}")
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
        
        input_df = pd.DataFrame([model_input], columns=feature_columns)
        
        probabilities = model.predict_proba(input_df)[0]
        
        return {
            'home_team': full_home_name,
            'away_team': full_away_name,
            'home_prob': probabilities[1],
            'away_prob': probabilities[0]
        }
        
    except Exception as e:
        print(f"Błąd podczas predykcji: {e}")
        return None


def calculate_value_bet(ml_probability, bookmaker_probability, threshold=0.05):
    value_difference = ml_probability - bookmaker_probability
    is_value_bet = value_difference >= threshold
    
    if bookmaker_probability > 0:
        implied_odds = 1.0 / bookmaker_probability
        expected_value = (ml_probability * (implied_odds - 1) * 100) - ((1 - ml_probability) * 100)
    else:
        expected_value = 0
    
    return {
        'is_value_bet': is_value_bet,
        'value': value_difference,
        'expected_value': expected_value
    }

def convert_types(obj):
    import numpy as np
    if hasattr(np, 'bool_') and isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_types(item) for item in obj]
    return obj

def compare_predictions_with_odds():
    print("porównanie ML - kursy")
    
    betting_data = load_betting_data()
    if not betting_data:
        print("Brak danych bukmacherskich do porównania")
        return []
    
    results = []
    
    for i, game in enumerate(betting_data, 1):
        print(f"\n[{i}/{len(betting_data)}] {game['home_team']} vs {game['away_team']}")
        
        ml_predictions = predict_game_probabilities(game['home_team'], game['away_team'])
        
        if not ml_predictions:
            print(f"brak predykcji ML")
            continue

        if 'h2h' not in game or 'home_price' not in game['h2h'] or 'away_price' not in game['h2h']:
            print(f"brak kompletnych danych o kursach")
            continue
        
        avg_home_odds = get_average_odds(game['h2h']['home_price'])
        avg_away_odds = get_average_odds(game['h2h']['away_price'])
        
        if not avg_home_odds or not avg_away_odds:
            print(f"brak kursów")
            continue
        
        bookie_home_prob = odds_to_probability(avg_home_odds)
        bookie_away_prob = odds_to_probability(avg_away_odds)
        
        home_value = calculate_value_bet(ml_predictions['home_prob'], bookie_home_prob)
        away_value = calculate_value_bet(ml_predictions['away_prob'], bookie_away_prob)
        
        game_result = {
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'game_date': game.get('time', 'Unknown'),
            'ml_home_prob': round(ml_predictions['home_prob'], 4),
            'ml_away_prob': round(ml_predictions['away_prob'], 4),
            'bookie_home_prob': round(bookie_home_prob, 4),
            'bookie_away_prob': round(bookie_away_prob, 4),
            'avg_home_odds': round(avg_home_odds, 2),
            'avg_away_odds': round(avg_away_odds, 2),
            'home_value_bet': home_value['is_value_bet'],
            'home_value_difference': round(home_value['value'], 4),
            'home_expected_value': round(home_value['expected_value'], 2),
            'away_value_bet': away_value['is_value_bet'],
            'away_value_difference': round(away_value['value'], 4),
            'away_expected_value': round(away_value['expected_value'], 2),
        }
        
        results.append(game_result)
        
        print(f"Model ML:")
        print(f"- {game['home_team']}: {ml_predictions['home_prob']*100:.1f}%")
        print(f"- {game['away_team']}: {ml_predictions['away_prob']*100:.1f}%")
        print(f"Bukmacherzy:")
        print(f"- {game['home_team']}: {avg_home_odds:.2f} (implikuje {bookie_home_prob*100:.1f}%)")
        print(f"- {game['away_team']}: {avg_away_odds:.2f} (implikuje {bookie_away_prob*100:.1f}%)")
        
        if home_value['is_value_bet']:
            print(f"VALUE BET: {game['home_team']} (różnica: {home_value['value']*100:.1f}%, EV: {home_value['expected_value']:.2f} PLN)")
        if away_value['is_value_bet']:
            print(f"VALUE BET: {game['away_team']} (różnica: {away_value['value']*100:.1f}%, EV: {away_value['expected_value']:.2f} PLN)")
    
    try:
        results_converted = [convert_types(r) for r in results]
        with open(COMPARISON_RESULTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"\nbłąd podczas zapisywania wyników: {e}")
    
    return results


def generate_comparison_report(results):
    print("\nraport")
    
    if not results:
        print("brak wyników do analizy")
        return
    
    total_games = len(results)
    home_value_bets = sum(1 for r in results if r['home_value_bet'])
    away_value_bets = sum(1 for r in results if r['away_value_bet'])
    total_value_bets = home_value_bets + away_value_bets
    
    print(f"Przeanalizowane mecze: {total_games}")
    print(f"Value bets: {total_value_bets}")
    print(f"-gospodarze: {home_value_bets}")
    print(f"-goście: {away_value_bets}")
    
    print("\n5 najlepszych VB")
    
    all_value_bets = []
    for r in results:
        if r['home_value_bet']:
            all_value_bets.append({
                'team': r['home_team'],
                'opponent': r['away_team'],
                'value': r['home_value_difference'],
                'ev': r['home_expected_value'],
                'ml_prob': r['ml_home_prob'],
                'odds': r['avg_home_odds']
            })
        if r['away_value_bet']:
            all_value_bets.append({
                'team': r['away_team'],
                'opponent': r['home_team'],
                'value': r['away_value_difference'],
                'ev': r['away_expected_value'],
                'ml_prob': r['ml_away_prob'],
                'odds': r['avg_away_odds']
            })
    
    all_value_bets.sort(key=lambda x: x['ev'], reverse=True)
    
    for i, bet in enumerate(all_value_bets[:5], 1):
        print(f"\n{i}. {bet['team']} vs {bet['opponent']}")
        print(f"Prawdopodobieństwo ML: {bet['ml_prob']*100:.1f}%")
        print(f"Kurs bukmachera: {bet['odds']:.2f}")
        print(f"Przewaga: {bet['value']*100:.1f}%")
        print(f"Oczekiwana wartość: {bet['ev']:.2f} PLN (na 100 PLN stawki)")
    

def main():
    results = compare_predictions_with_odds()
    generate_comparison_report(results)


if __name__ == "__main__":
    main()
