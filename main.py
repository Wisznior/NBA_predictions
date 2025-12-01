from data_collection.nba_collector import NBADataCollector

if __name__ == "__main__":    
    collector = NBADataCollector()
    collector.update_all()
    collector.close()
