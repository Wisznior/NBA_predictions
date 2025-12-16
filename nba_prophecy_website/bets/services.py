import requests
import json
from datetime import timedelta, datetime
from django.conf import settings
from .models import Team


def update_nba_schedule() -> None:

    tomorrow = datetime.now() + timedelta(days=1)
    next_week = tomorrow + timedelta(days=7)
    url: str = f"https://api.balldontlie.io/v1/games?start_date={tomorrow.year}-{tomorrow.month}-{tomorrow.day}&end_date={next_week.year}-{next_week.month}-{next_week.day}"
    headers: dict = {
        "Authorization": settings.NBA_SCHEDULE_API_KEY
    }

    res: requests.Response = requests.get(url, headers=headers, timeout=5)

    data = res.json()

    nba_schedule_path = settings.BASE_DIR / "bets" / "data" / "nba_schedule.json"
    nba_schedule_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nba_schedule_path, encoding="utf-8", mode="w") as f:
        json.dump(data, f, indent=4)


def get_nba_schedule() -> dict:
    nba_schedule_path = settings.BASE_DIR / "bets" / "data" / "nba_schedule.json"
    nba_schedule_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nba_schedule_path, encoding="utf-8", mode="r") as f:
        file_content: str = f.read()
        file_content: dict = json.loads(file_content)
        return file_content


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
