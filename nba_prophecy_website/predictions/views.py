from django.shortcuts import render
from datetime import date

def future(request):

    #future games and predictions for testing
    future_games =[
        {"home_team": "Los Angeles Lakers", "away_team": "Golden State Warriors", "match_date": date.today().isoformat(), "predicted_home": 112, "predicted_away": 108,},
        {"home_team": "Raptors", "away_team": "Trail Blazers", "match_date": date.today().isoformat(), "predicted_home": 98, "predicted_away": 110,},
        {"home_team": "Celtics", "away_team": "Spurs", "match_date": date.today().isoformat(), "predicted_home": 107, "predicted_away": 88,},
        {"home_team": "Thunder", "away_team": "Warriors", "match_date": date.today().isoformat(), "predicted_home": 122, "predicted_away": 101,},
        {"home_team": "Nuggets", "away_team": "Pacers", "match_date": date.today().isoformat(), "predicted_home": 114, "predicted_away": 104,},
        {"home_team": "Wizards", "away_team": "Pelicans", "match_date": date.today().isoformat(), "predicted_home": 117, "predicted_away": 108,},
        {"home_team": "Grizzlies", "away_team": "Hawks", "match_date": date.today().isoformat(), "predicted_home": 117, "predicted_away": 108,},
    ]

    return render(request, "future.html", {
        "future_games": future_games,
        "active": "future",
    })

def history(request):
    # past games and predictions for testing
    history_games = [
        {"home": "Lakers", "away": "Clippers", "home_pts": 120, "away_pts": 110, "date": "2025-01-31", "predicted_home": 112, "predicted_away": 108,},
        {"home": "Los Angeles Lakers", "away": "Nuggets", "home_pts": 99, "away_pts": 103, "date": "2025-02-01", "predicted_home": 98, "predicted_away": 110,},
        {"home": "Suns", "away": "Pacers", "home_pts": 89, "away_pts": 107, "date": "2025-02-01", "predicted_home": 107, "predicted_away": 88,},
        {"home": "Pelicans", "away": "Trail Blazers", "home_pts": 92, "away_pts": 94, "date": "2025-02-01", "predicted_home": 122, "predicted_away": 101,},
        {"home": "Grizzlies", "away": "Nuggets", "home_pts": 110, "away_pts": 103, "date": "2025-02-01", "predicted_home": 114, "predicted_away": 104,},
        {"home": "Raptors", "away": "Celtics", "home_pts": 99, "away_pts": 112, "date": "2025-02-01", "predicted_home": 117, "predicted_away": 108,},
        {"home": "Hawks", "away": "Nuggets", "home_pts": 113, "away_pts": 111, "date": "2025-02-01", "predicted_home": 117, "predicted_away": 108,},
        {"home": "Golden State Warriors", "away": "Nuggets", "home_pts": 99, "away_pts": 115, "date": "2025-02-01", "predicted_home": 117, "predicted_away": 108,},
    ]
    return render(request, "history.html", {
        "history_games": history_games,
        "active": "history",
    })

