import pandas as pd
import sqlite3
import logging
import time
import os
from datetime import datetime
from nba_api.stats.endpoints import leaguegamelog

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(BASE, "logs")
DATA = os.path.join(BASE, "data")

os.makedirs(LOG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

class NBADataCollector:
    def __init__(self, path_to_db: str ='nba_history.db'):
        self.path_to_db = os.path.join(DATA, path_to_db)
        self.conn = None
        self.logger = None
        self._create_logs()
        self._db_connection()
        self._create_tables()

    def _create_logs(self) -> None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        logfile = f"nba_data_collection_{ts}.log"
        logpath = os.path.join(LOG, logfile)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logpath, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("NBA Data initialized")

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

    def latest_stored_date(self):
        q = "SELECT MAX(game_date) FROM historical_games"
        df = pd.read_sql(q, self.conn)
        if df.iloc[0, 0] is None:
            return None
        return df.iloc[0, 0]
    
    def fetch_games(self, seasone):
        try:
            gamelog = leaguegamelog.LeagueGameLog(
                season = seasone,
                season_type_all_star = 'Regular Season',
                player_or_team_abbreviation = 'T'
            )
            df = gamelog.get_data_frames()[0]
            if df.empty:
                self.logger.info(f"No more data for {seasone}")
                return pd.DataFrame()
            df['SEASON_ID'] = seasone
            return df
        except Exception as e:
            self.logger.info("Failed to collect data {e}")
            return pd.DataFrame()
    
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
            'rebounds', 'turnovers']
        
        cols_to_store = [c for c in req_cols if c in data.columns]
        
        opt_cols = [
            'is_home_game', 'offensive_rebounds', 'defensive_rebounds',
            'steals', 'blocks','fouls','plus_minus', 'free_throw_percentage']

        for c in opt_cols:
            if c in data.columns:
                cols_to_store.append(c)

        data = data[cols_to_store].copy()

        try:
            data.to_sql(
                'historical_games',
                self.conn,
                if_exists = 'append',
                index = False
            )
            self.logger.info(f"Saved {len(data)} rows")
        except Exception as e:
            self.logger.info(f"Save failed: {e}")
    
    def update_all(self):
        latest = self.latest_stored_date()

        if latest is None:
            self.logger.info("Collecting all seasons")

            seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(2020, 2026)]

            for s in seasons:
                df = self.fetch_games(s)
                if df.empty:
                    continue
                trans = self._transform_data(df)
                self._save_games(trans)
                time.sleep(1)
            return
        
        self.logger.info(f"Latest game in db: {latest}")

        latest = datetime.strptime(latest, "%Y-%m-%d")
        current =  datetime.today().year

        to_check = [f"{current-1}-{str(current)[-2:]}",
                    f"{current}-{str(current+1)[-2:]}"]

        for s in to_check:
            df = self.fetch_games(s)
            if df.empty:
                continue

            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df[df['GAME_DATE'] > latest]

            if df.empty:
                continue

            transformed = self._transform_data(df)
            self._save_games(transformed)
        self.logger.info("Updated")

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.logger.info("Connection closed")
