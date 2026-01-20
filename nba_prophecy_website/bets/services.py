import requests
import json
import os
import sqlite3
from datetime import timedelta, datetime
from django.conf import settings
from .models import Team, Game, Bet
from django.utils import timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ML_DB_PATH = os.path.join(BASE_DIR, "data", "nba_history.db")


def update_nba_schedule() -> dict:
    start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d')
    
    url = f"https://api.balldontlie.io/v1/games?start_date={start_date}&end_date={end_date}"
    headers = {"Authorization": settings.NBA_SCHEDULE_API_KEY}

    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        data = res.json()

        path = settings.BASE_DIR / "bets" / "data" / "nba_schedule.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return data
    except Exception as e:
        print(f"błąd API: {e}")
        return {"data": []}


def get_nba_schedule() -> dict:
    nba_schedule_path = settings.BASE_DIR / "bets" / "data" / "nba_schedule.json"
    
    should_update = False
    
    if not os.path.exists(nba_schedule_path):
        should_update = True
    else:
        file_mod_time = os.path.getmtime(nba_schedule_path)
        last_update = datetime.fromtimestamp(file_mod_time)
        
        if datetime.now() - last_update > timedelta(hours=12):
            should_update = True

    if should_update:
        return update_nba_schedule()
        
    with open(nba_schedule_path, encoding="utf-8", mode="r") as f:
        return json.load(f)


def get_or_create_team(team_data):
    team, _ = Team.objects.get_or_create(
        team_api_id=team_data["id"],
        defaults={
            "city": team_data["city"],
            "name": team_data["name"],
            "full_name": team_data["full_name"],
            "abbreviation": team_data["abbreviation"],
            "conference": team_data["conference"],
            "division": team_data["division"],
        }
    )
    return team


def get_game_result_from_ml_db(home_team_name, away_team_name, game_date):
    if not os.path.exists(ML_DB_PATH):
        return None
    
    try:
        if isinstance(game_date, datetime):
            game_date_str = game_date.strftime('%Y-%m-%d')
        else:
            game_date_str = str(game_date)
        
        conn = sqlite3.connect(ML_DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT team_id, team_name FROM teams")
        ml_teams = {row[1].lower(): row[0] for row in cursor.fetchall()}
        
        home_team_id = None
        away_team_id = None
        
        home_lower = home_team_name.lower()
        away_lower = away_team_name.lower()
        
        for team_name, team_id in ml_teams.items():
            if home_lower in team_name or team_name in home_lower:
                home_team_id = team_id
            if away_lower in team_name or team_name in away_lower:
                away_team_id = team_id
        
        if not home_team_id or not away_team_id:
            conn.close()
            return None
        
        query = """
            SELECT home_pts, away_pts, game_date
            FROM games
            WHERE home_team_id = ? 
              AND away_team_id = ?
              AND date(game_date) BETWEEN date(?, '-1 day') AND date(?, '+1 day')
            LIMIT 1
        """
        
        cursor.execute(query, (home_team_id, away_team_id, game_date_str, game_date_str))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            home_score, away_score, found_date = result
            
            return {
                'home_score': int(home_score) if home_score else None,
                'away_score': int(away_score) if away_score else None
            }
        else:
            return None
            
    except Exception as e:
        return None


def update_finished_games_from_ml_db():    
    if not os.path.exists(ML_DB_PATH):
        return 0
    
    updated_count = 0

    games_without_scores = Game.objects.filter(
        home_team_score__isnull=True,
        visitor_team_score__isnull=True
    )
        
    for game in games_without_scores:
        home_name = game.home_team.full_name
        away_name = game.visitor_team.full_name
        game_date = game.game_date
            
        result = get_game_result_from_ml_db(home_name, away_name, game_date)
        
        if result and result['home_score'] is not None and result['away_score'] is not None:
            game.home_team_score = result['home_score']
            game.visitor_team_score = result['away_score']
            game.save()
            
            updated_count += 1
        
    return updated_count


def update_finished_games_from_api():    
    schedule_data = get_nba_schedule()
    api_games = schedule_data.get("data", [])
    
    if not api_games:
        return update_finished_games_from_ml_db()
    
    updated_count = 0

    for api_game in api_games:
        status = api_game.get("status", "")
        if "Final" not in str(status):
            continue
            
        game_api_id = api_game.get("id")
        home_score = api_game.get("home_team_score")
        visitor_score = api_game.get("visitor_team_score")

        if home_score is None or visitor_score is None:
            continue

        try:
            game = Game.objects.get(game_api_id=game_api_id)
            
            if game.home_team_score is not None and game.visitor_team_score is not None:
                continue
            
            game.home_team_score = home_score
            game.visitor_team_score = visitor_score
            game.save()
            
            updated_count += 1
            
        except Game.DoesNotExist:
            continue
        except Exception as e:
            continue
    
    return updated_count


def update_bets_status():    
    finished_games = Game.objects.filter(
        home_team_score__isnull=False,
        visitor_team_score__isnull=False
    )

    updated_count = 0

    for game in finished_games:
        pending_bets = Bet.objects.filter(game=game, status='pending')
        
        for bet in pending_bets:
            bet.update_status()
            updated_count += 1
            
            status_emoji = "🎉" if bet.status == 'won' else "😞"
            print(f"  {status_emoji} Zakład {bet.user.username}: {bet.status}")
    return updated_count


def update_game_results_and_bets():
    games_updated = update_finished_games_from_ml_db()
    
    bets_updated = update_bets_status()
        
    return {
        'games_updated': games_updated,
        'bets_updated': bets_updated
    }