import pandas as pd
import sqlite3
import logging
import time
import os
from datetime import datetime
from nba_api.stats.endpoints import leaguegamelog

class NBADataCollector:
    def __init__(self, path_to_db: str ='nba_history.db'):
        self.path_to_db = path_to_db
        self.conn = None
        self.logger = None
        self._create_logs()
        self._db_connection()
        self._create_tables()

    def _create_logs(self) -> None:
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        logfile = f"nba_data_collection_{ts}.log"
        logpath = os.path.join('logs', logfile)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logpath, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("NBA Data Collector initialized")

    def _db_connection(self) -> None:
        try:
            self.conn = sqlite3.connect(self.path_to_db)
            self.logger.info(f"Connected to db: {self.path_to_db}")
        except sqlite3.Error as e:
            self.logger.error(f"Db connection error: {e}")
            raise

    def _create_tables(self) -> None:
        sql_create = """
        CREATE TABLE IF NOT EXISTS historical_games (
            season_id TEXT NOT NULL,
            team_id INTEGER NOT NULL,
            team_abbreviation TEXT,
            team_name TEXT,
            game_id TEXT NOT NULL,
            game_date DATE,
            matchup TEXT,
            is_home_game INTEGER,
            outcome TEXT,
            points_scored INTEGER,
            points_allowed INTEGER,
            field_goal_percentage REAL,
            three_point_percentage REAL,
            free_throw_percentage REAL,
            assists INTEGER,
            rebounds INTEGER,
            offensive_rebounds INTEGER,
            defensive_rebounds INTEGER,
            turnovers INTEGER,
            steals INTEGER,
            blocks INTEGER,
            fouls INTEGER,
            plus_minus INTEGER,
            PRIMARY KEY (game_id, team_id)
        );
        """
    
        try:
            curs = self.conn.cursor()
            curs.execute(sql_create)
            self.conn.commit()
            self.logger.info("Table created")
        except sqlite3.Error as e:
            self.logger.error(f"Cant create a table: {e}")
            raise

    def existing_ids(self, season: str) -> set:
        q = "SELECT DISTINCT game_id FROM historical_games WHERE season_id = ?"
        try:
            temp = pd.read_sql(q, self.conn, params=(season,))
            existing_ids = set(temp['game_id'].tolist())
            self.logger.debug(f"Found {len(existing_ids)} existing for {season}")
            return existing_ids
        except Exception as e:
            self.logger.warning(f"Couldn't get existing IDs: {e}")
            return set()

    def collect_season_games(self, season: str) -> None:
        self.logger.info(f"Collecting season: {season}")
        
        try:
            gamelog = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star='Regular Season',
                player_or_team_abbreviation='T'
            )
            raw_data_from_api = gamelog.get_data_frames()[0]

            if raw_data_from_api.empty:
                self.logger.warning(f"API returned empty")
                return
            
            self.logger.info(f"Got {len(raw_data_from_api)} records from API")
            
        except Exception as e:
            self.logger.error(f"API failed: {e}")
            return

        raw_data_from_api['season_id'] = season

        if 'GAME_DATE' not in raw_data_from_api.columns:
            self.logger.error("Game date missing")
            return
        
        transformed_data = self._transform_data(raw_data_from_api)

        existing = self.existing_ids(season)
        only_new_data = transformed_data[~transformed_data['game_id'].isin(existing)]
        
        if only_new_data.empty:
            self.logger.info(f"No new games for {season}")
            return
        
        self._save_games(only_new_data)
        time.sleep(1.0)

    def _transform_data(self, raw_data : pd.DataFrame) -> pd.DataFrame:
        result = raw_data.copy()
        
        cols = {
            'SEASON_ID': 'season_id',
            'TEAM_ID': 'team_id',
            'TEAM_ABBREVIATION': 'team_abbreviation',
            'TEAM_NAME': 'team_name',
            'GAME_ID': 'game_id',
            'GAME_DATE': 'game_date',
            'MATCHUP': 'matchup',
            'WL': 'outcome',
            'PTS': 'points_scored',
            'FG_PCT': 'field_goal_percentage',
            'FG3_PCT': 'three_point_percentage',
            'FT_PCT': 'free_throw_percentage',
            'AST': 'assists',
            'REB': 'rebounds',
            'OREB': 'offensive_rebounds',
            'DREB': 'defensive_rebounds',
            'TOV': 'turnovers',
            'STL': 'steals',
            'BLK': 'blocks',
            'PF': 'fouls',
            'PLUS_MINUS': 'plus_minus'
        }
        
        avail = {k: v for k, v in cols.items() if k in result.columns}
        result = result.rename(columns=avail)
        
        result['game_date'] = pd.to_datetime(result['game_date']).dt.strftime('%Y-%m-%d')
        
        if 'matchup' in result.columns:
            result['is_home_game'] = result['matchup'].apply(
                lambda x: 1 if 'vs.' in str(x) else 0
            )
        
        return result

    def _save_games(self, data: pd.DataFrame) -> None:
        req_cols = [
            'season_id', 'team_id', 'team_abbreviation', 'team_name',
            'game_id', 'game_date', 'matchup', 'outcome', 'points_scored',
            'field_goal_percentage', 'three_point_percentage', 'assists',
            'rebounds', 'turnovers'
        ]
        
        cols_to_store = [c for c in req_cols if c in data.columns]
        opt_cols = ['is_home_game', 'offensive_rebounds', 'defensive_rebounds', 
                    'steals', 'blocks', 'fouls', 'plus_minus', 'free_throw_percentage']
        
        for c in opt_cols:
            if c in data.columns:
                cols_to_store.append(c)
        
        count = len(data)
        
        try:
            data[cols_to_store].to_sql(
                'historical_games',
                self.conn,
                if_exists='append',
                index=False

            )
            self.logger.info(f"Saved {count} records")
        except Exception as e:
            self.logger.error(f"Save failed: {e}")

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.logger.info("Connection closed")


if __name__ == "__main__":
    seasons = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
    
    collector = NBADataCollector()
    
    for s in seasons:
        collector.collect_season_games(s)
    
    collector.close()