from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import Http404
from .services import get_nba_schedule, get_or_create_team
from .constants import TEAM_LOGOS
from .forms import BetForm
from .models import Bet, Game

@login_required
def bets_view(request):
    nba_schedule: dict = get_nba_schedule()
    games_by_id = {g["id"]: g for g in nba_schedule["data"]}
    user_bets = (Bet.objects.filter(user=request.user).values_list("game__game_api_id", flat=True))
    for game in nba_schedule["data"]:
        game["home_team"]["logo"] = TEAM_LOGOS[game["home_team"]["full_name"]]
        game["visitor_team"]["logo"] = TEAM_LOGOS[game["visitor_team"]["full_name"]]
        game["has_bet"] = game["id"] in user_bets

    if request.method == "POST":
        form = BetForm(request.POST)
        if form.is_valid():
            with transaction.atomic():
                game_data = games_by_id[int(form.cleaned_data["game_id"])]
                if not game_data:
                    raise Http404("Game not found")
                home_team = get_or_create_team(game_data["home_team"])
                visitor_team = get_or_create_team(game_data["visitor_team"])
                game, _ = Game.objects.get_or_create(
                    game_api_id=form.cleaned_data["game_id"],
                    defaults={
                        "game_date": game_data["date"],
                        "season": game_data["season"],
                        "datetime": game_data["datetime"],
                        "home_team": home_team,
                        "visitor_team": visitor_team,
                    },
                )

                Bet.objects.create(
                    user=request.user,
                    game=game,
                    home_score=form.cleaned_data["home_score"],
                    visitor_score=form.cleaned_data["visitor_score"],
                )
            return redirect("bet")
        else:
            print(form.errors)

    else:
        form = BetForm()

    return render(request, "bet.html", {"active": "bet", "form": form, "nba_schedule": nba_schedule})

@login_required
def bets_history_view(request):
    user_bets = Bet.objects.filter(user=request.user).select_related(
        'game',
        'game__home_team',
        'game__visitor_team'
    )
    
    all_bets_list = list(user_bets)
    for bet in all_bets_list:
        bet.home_logo = TEAM_LOGOS.get(bet.game.home_team.abbreviation, 'placeholder.png')
        bet.visitor_logo = TEAM_LOGOS.get(bet.game.visitor_team.abbreviation, 'placeholder.png')
    
    pending_bets = [b for b in all_bets_list if b.status == 'pending']
    won_bets = [b for b in all_bets_list if b.status == 'won']
    lost_bets = [b for b in all_bets_list if b.status == 'lost']
    
    total_bets = len(all_bets_list)
    won_count = len(won_bets)
    lost_count = len(lost_bets)
    pending_count = len(pending_bets)
    
    finished_bets = won_count + lost_count
    win_percentage = (won_count / finished_bets * 100) if finished_bets > 0 else 0
    
    context = {
        'active': 'bets_history',
        'all_bets': all_bets_list,
        'pending_bets': pending_bets,
        'won_bets': won_bets,
        'lost_bets': lost_bets,
        'total_bets': total_bets,
        'won_count': won_count,
        'lost_count': lost_count,
        'pending_count': pending_count,
        'win_percentage': round(win_percentage, 1),
    }
    return render(request, 'bets_history.html', context)