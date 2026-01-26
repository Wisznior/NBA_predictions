from django.core.management.base import BaseCommand
from bets.services import update_game_results_and_bets

class Command(BaseCommand):

    def handle(self, *args, **options):        
        try:
            stats = update_game_results_and_bets()
            
            self.stdout.write(
                self.style.SUCCESS(
                    f"Zaktualizowano gier: {stats['games_updated']}, zaktualizowano zakładów: {stats['bets_updated']}"
                )
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Wystąpił błąd podczas aktualizacji: {str(e)}")
            )