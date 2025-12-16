from django.db import models
from django.conf import settings

class Team(models.Model):
    team_api_id = models.PositiveIntegerField(unique=True)
    city = models.CharField(max_length=50)
    name = models.CharField(max_length=50)
    full_name = models.CharField(max_length=100)
    abbreviation = models.CharField(max_length=5)
    conference = models.CharField(max_length=10)
    division = models.CharField(max_length=20)


class Game(models.Model):
    home_team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="home_games")
    visitor_team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="visitor_games")

    game_api_id = models.PositiveIntegerField(unique=True)
    game_date = models.DateField()
    season = models.PositiveSmallIntegerField()
    datetime = models.DateTimeField()

    home_team_score = models.PositiveSmallIntegerField(null=True)
    visitor_team_score = models.PositiveSmallIntegerField(null=True)


class Bet(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="bets")
    game = models.ForeignKey(Game, on_delete=models.CASCADE, related_name="bets")

    home_score = models.PositiveSmallIntegerField()
    visitor_score = models.PositiveSmallIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "game")

