import requests
import json
import os
from datetime import timedelta, datetime
from django.conf import settings
from .models import Team
from django.utils import timezone
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
        print(f"Błąd API: {e}")
        return {"data": []}

def get_nba_schedule() -> dict:
    """Zwraca dane z pliku. Jeśli plik nie istnieje lub jest stary - aktualizuje go."""
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
        print("Dane NBA są nieaktualne. Pobieram nowe...")
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

def update_game_results_and_bets():
    from .models import Game, Bet
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
    return updated_count
