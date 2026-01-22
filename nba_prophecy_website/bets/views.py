from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.contrib import messages
from .services import get_nba_schedule, get_or_create_team, update_game_results_and_bets
from .constants import TEAM_LOGOS
from .forms import BetForm
from .models import Bet, Game
import json
import os
import hashlib
from datetime import datetime
from django.utils import timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
COMPARISON_PATH = os.path.join(DATA_DIR, 'betting_comparison_results.json')


def generate_game_id(home_team, away_team, game_date):
    game_string = f"{home_team}_{away_team}_{game_date}"
    return int(hashlib.md5(game_string.encode()).hexdigest()[:8], 16)


def normalize_team_name(name):
    replacements = {
        'LA Lakers': 'Los Angeles Lakers',
        'LA Clippers': 'Los Angeles Clippers',
        'LA': 'Los Angeles'
    }
    normalized = name.strip()
    for old, new in replacements.items():
        if normalized.startswith(old):
            normalized = normalized.replace(old, new, 1)
    return normalized.lower()


def parse_game_datetime(date_string):
    if not date_string or date_string == "Unknown":
        return timezone.now()
    
    try:
        if 'T' in date_string:
            clean_date = date_string.replace('Z', '').replace('+00:00', '')
            dt = datetime.fromisoformat(clean_date)
            if timezone.is_naive(dt):
                dt = timezone.make_aware(dt)
            return dt
        
        if len(date_string) >= 10:
            date_part = date_string[:10]
            dt = datetime.strptime(date_part, '%Y-%m-%d')
            dt = dt.replace(hour=12, minute=0, second=0)
            return timezone.make_aware(dt)
        
        return timezone.now()
    except Exception as e:
        print(f"error parsing date '{date_string}': {e}")
        return timezone.now()


def find_api_match(home_team, away_team, api_games):
    h_normalized = normalize_team_name(home_team)
    a_normalized = normalize_team_name(away_team)
    
    for api_game in api_games:
        h_api = normalize_team_name(api_game["home_team"]["full_name"])
        a_api = normalize_team_name(api_game["visitor_team"]["full_name"])
        
        if h_normalized == h_api and a_normalized == a_api:
            return api_game
        if (h_normalized in h_api or h_api in h_normalized) and \
           (a_normalized in a_api or a_api in a_normalized):
            return api_game
    return None


@login_required
def bets_view(request):
    update_game_results_and_bets()
    
    if request.method == "POST":
        game_id = request.POST.get("game_id")
        home_score = request.POST.get("home_score")
        visitor_score = request.POST.get("visitor_score")
        home_team_name = request.POST.get("home_team_name")
        away_team_name = request.POST.get("away_team_name")
        game_date = request.POST.get("game_date")
        
        if game_id and home_score and visitor_score and home_team_name and away_team_name:
            try:
                game_datetime = parse_game_datetime(game_date)
                
                with transaction.atomic():
                    nba_schedule = get_nba_schedule()
                    api_games = nba_schedule.get("data", [])

                    api_match = find_api_match(home_team_name, away_team_name, api_games)
                    
                    if api_match:
                        home_team_data = api_match["home_team"]
                        visitor_team_data = api_match["visitor_team"]
                        game_datetime_final = parse_game_datetime(api_match.get("date", game_date))
                        season = api_match.get("season", 2024)
                    else:
                        home_team_data = {
                            "id": generate_game_id(home_team_name, "home", str(game_datetime.date())),
                            "full_name": home_team_name,
                            "name": home_team_name.split()[-1] if home_team_name else "Unknown",
                            "city": " ".join(home_team_name.split()[:-1]) if home_team_name else "Unknown",
                            "abbreviation": "".join([w[0] for w in home_team_name.split()]).upper()[:3] if home_team_name else "UNK",
                            "conference": "Unknown",
                            "division": "Unknown"
                        }
                        visitor_team_data = {
                            "id": generate_game_id(away_team_name, "away", str(game_datetime.date())),
                            "full_name": away_team_name,
                            "name": away_team_name.split()[-1] if away_team_name else "Unknown",
                            "city": " ".join(away_team_name.split()[:-1]) if away_team_name else "Unknown",
                            "abbreviation": "".join([w[0] for w in away_team_name.split()]).upper()[:3] if away_team_name else "UNK",
                            "conference": "Unknown",
                            "division": "Unknown"
                        }
                        game_datetime_final = game_datetime
                        season = 2024
                    
                    home_team = get_or_create_team(home_team_data)
                    visitor_team = get_or_create_team(visitor_team_data)
                    
                    game_defaults = {
                        "home_team": home_team,
                        "visitor_team": visitor_team,
                        "season": season,
                        "game_date": game_datetime_final,
                    }
                    
                    from django.db import connection
                    with connection.cursor() as cursor:
                        cursor.execute("PRAGMA table_info(bets_game)")
                        columns = cursor.fetchall()
                        if any(col[1] == 'datetime' for col in columns):
                            game_defaults["datetime"] = game_datetime_final
                    
                    game, created = Game.objects.get_or_create(
                        game_api_id=game_id,
                        defaults=game_defaults
                    )
                    
                    existing_bet = Bet.objects.filter(user=request.user, game=game).first()
                    if existing_bet:
                        messages.warning(request, "You have already bet on this game!")
                    else:
                        Bet.objects.create(
                            user=request.user,
                            game=game,
                            home_score=int(home_score),
                            visitor_score=int(visitor_score),
                            status='pending'
                        )
                        messages.success(request, f"Bet placed successfully!")
                    
                return redirect("bet")
            except Exception as e:
                print(f"error creating bet: {e}")
                import traceback
                traceback.print_exc()
                messages.error(request, f"error: {str(e)}")
                return redirect("bet")
        else:
            messages.error(request, "Missing required fields")
            return redirect("bet")
    
    nba_schedule = get_nba_schedule()
    api_games = nba_schedule.get("data", [])
    
    upcoming_api_games = [
        g for g in api_games 
        if g.get("status") in ["", None] or "Final" not in str(g.get("status", ""))
    ]
    
    comparison_data = []
    if os.path.exists(COMPARISON_PATH):
        try:
            with open(COMPARISON_PATH, 'r', encoding='utf-8') as f:
                comparison_data = json.load(f)
        except Exception as e:
            print(f"Error loading comparison data: {e}")
    
    display_games = []
    
    if comparison_data:
        for ml_game in comparison_data:
            home_team = ml_game["home_team"]
            away_team = ml_game["away_team"]
            game_date = ml_game.get("game_date", "Unknown")
            
            api_match = find_api_match(home_team, away_team, upcoming_api_games)
            
            if api_match:
                game_id = api_match["id"]
            else:
                game_id = generate_game_id(home_team, away_team, game_date[:10] if len(game_date) >= 10 else game_date)
            
            home_odds = ml_game.get('avg_home_odds', 2.0)
            away_odds = ml_game.get('avg_away_odds', 2.0)
            
            bookie_home_prob = (1 / home_odds) * 100 if home_odds > 0 else 50
            bookie_away_prob = (1 / away_odds) * 100 if away_odds > 0 else 50
            
            total = bookie_home_prob + bookie_away_prob
            if total > 0:
                bookie_home_prob = (bookie_home_prob / total) * 100
                bookie_away_prob = (bookie_away_prob / total) * 100
            
            is_home_value = ml_game.get('home_value_bet', False)
            is_away_value = ml_game.get('away_value_bet', False)
            
            predicted_home = ml_game.get('predicted_home_score', 110)
            predicted_away = ml_game.get('predicted_away_score', 105)
            
            display_games.append({
                "id": game_id,
                "date": game_date,
                "is_value_bet": is_home_value or is_away_value,
                "value_side": 'home' if is_home_value else ('away' if is_away_value else None),
                "home_team": {
                    "full_name": home_team,
                    "logo": TEAM_LOGOS.get(home_team, "placeholder.png"),
                    "ml_prob": round(ml_game.get('ml_home_prob', 0.5) * 100, 1),
                    "bookie_prob": round(bookie_home_prob, 1),
                    "odds": home_odds,
                    "is_value": is_home_value,
                    "predicted_score": predicted_home
                },
                "visitor_team": {
                    "full_name": away_team,
                    "logo": TEAM_LOGOS.get(away_team, "placeholder.png"),
                    "ml_prob": round(ml_game.get('ml_away_prob', 0.5) * 100, 1),
                    "bookie_prob": round(bookie_away_prob, 1),
                    "odds": away_odds,
                    "is_value": is_away_value,
                    "predicted_score": predicted_away
                },
                "has_bet": Bet.objects.filter(
                    user=request.user, 
                    game__game_api_id=game_id
                ).exists(),
            })
    else:
        print("No comparison data found")
        for game in upcoming_api_games[:15]:
            display_games.append({
                "id": game["id"],
                "date": game.get("date", ""),
                "is_value_bet": False,
                "value_side": None,
                "home_team": {
                    "full_name": game["home_team"]["full_name"],
                    "logo": TEAM_LOGOS.get(game["home_team"]["full_name"], "placeholder.png"),
                    "ml_prob": 50.0,
                    "bookie_prob": 50.0,
                    "odds": 2.0,
                    "is_value": False,
                    "predicted_score": 110
                },
                "visitor_team": {
                    "full_name": game["visitor_team"]["full_name"],
                    "logo": TEAM_LOGOS.get(game["visitor_team"]["full_name"], "placeholder.png"),
                    "ml_prob": 50.0,
                    "bookie_prob": 50.0,
                    "odds": 2.0,
                    "is_value": False,
                    "predicted_score": 105
                },
                "has_bet": Bet.objects.filter(user=request.user, game__game_api_id=game["id"]).exists(),
            })

    return render(request, "bet.html", {
        "display_games": display_games, 
        "form": BetForm(),
        "active": "bet"
    })


@login_required
def bets_history_view(request):
    update_game_results_and_bets()
    
    user_bets = Bet.objects.filter(user=request.user).select_related(
        'game', 'game__home_team', 'game__visitor_team'
    ).order_by('-created_at')
    
    all_bets_list = list(user_bets)
    for bet in all_bets_list:
        bet.home_logo = TEAM_LOGOS.get(bet.game.home_team.full_name, 'placeholder.png')
        bet.visitor_logo = TEAM_LOGOS.get(bet.game.visitor_team.full_name, 'placeholder.png')
    
    pending_bets = [b for b in all_bets_list if b.status == 'pending']
    won_bets = [b for b in all_bets_list if b.status == 'won']
    lost_bets = [b for b in all_bets_list if b.status == 'lost']
    
    finished_bets = len(won_bets) + len(lost_bets)
    win_percentage = (len(won_bets) / finished_bets * 100) if finished_bets > 0 else 0
    
    context = {
        'active': 'bets_history',
        'all_bets': all_bets_list,
        'pending_bets': pending_bets,
        'won_bets': won_bets,
        'lost_bets': lost_bets,
        'total_bets': len(all_bets_list),
        'won_count': len(won_bets),
        'lost_count': len(lost_bets),
        'pending_count': len(pending_bets),
        'win_percentage': round(win_percentage, 1),
    }
    return render(request, 'bets_history.html', context)