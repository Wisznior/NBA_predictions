from data_collection.nba_collector import NBADataCollector
from data_collection.players_collector import NBAPlayerCollector

if __name__ == "__main__":    
    games = NBADataCollector()
    players  = NBAPlayerCollector()
    
    games.update_all()
    players.update_players_list()
    players.calculate_recent_form()
    
    games.close()
    players.close()
