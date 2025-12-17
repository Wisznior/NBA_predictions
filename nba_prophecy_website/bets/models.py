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

    
    BET_STATUS_CHOICES = [
        ('pending', 'Pending'),      
        ('won', 'Won'),              
        ('lost', 'Lost'),            
    ]
    status = models.CharField(
        max_length=10, 
        choices=BET_STATUS_CHOICES, 
        default='pending'
    )
    
    class Meta:
        unique_together = ("user", "game")
        ordering = ['-created_at'] 
    
    def __str__(self):
        return f"{self.user.username} - {self.game.home_team.abbreviation} vs {self.game.visitor_team.abbreviation}"
    
    def check_if_won(self):
        if self.game.home_team_score is None or self.game.visitor_team_score is None:
            return None
        
        actual_home = self.game.home_team_score
        actual_visitor = self.game.visitor_team_score

        predicted_home = self.home_score
        predicted_visitor = self.visitor_score
        
        if actual_home > actual_visitor:
            return predicted_home > predicted_visitor
        elif actual_visitor > actual_home:
            return predicted_visitor > predicted_home
        else:
            return predicted_home == predicted_visitor
    
    def update_status(self):
        if self.game.home_team_score is not None and self.game.visitor_team_score is not None:
            won = self.check_if_won()
            if won is not None:
                self.status = 'won' if won else 'lost'
                self.save()
