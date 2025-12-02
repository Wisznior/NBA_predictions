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
    def __init__(self, path_to_db: str = 'nba_history.db'):
        self.path_to_db = os.path.join(DATA, path_to_db)
        self.conn = None
        self.logger = None
        self._create_logs()
        self._db_connection()
        self._create_tables()

    def _create_logs(self) -> None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        logfile = f"nba_collector_{ts}.log"
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
        self.logger.info("NBA Game Collector initialized")

    def _db_connection(self) -> None:
        try:
            self.conn = sqlite3.connect(self.path_to_db)
            self.logger.info(f"Connected to db: {self.path_to_db}")
        except sqlite3.Error as e:
            self.logger.error(f"Db connection error: {e}")
            raise

    def _create_tables(self) -> None:
        sql_teams = """
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            team_abbreviation TEXT,
            team_name TEXT
        );
        """

        sql_games = """
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            season_id TEXT NOT NULL,
            game_date DATE,
            
            home_team_id INTEGER,
            away_team_id INTEGER,
            outcome TEXT,
            
            home_pts INTEGER,
            home_fg_pct REAL,
            home_fg3_pct REAL,
            home_ft_pct REAL,
            home_ast INTEGER,
            home_reb INTEGER,
            home_tov INTEGER,
            home_stl INTEGER,
            home_blk INTEGER,
            home_pf INTEGER,
            home_plus_minus INTEGER,

            away_pts INTEGER,
            away_fg_pct REAL,
            away_fg3_pct REAL,
            away_ft_pct REAL,
            away_ast INTEGER,
            away_reb INTEGER,
            away_tov INTEGER,
            away_stl INTEGER,
            away_blk INTEGER,
            away_pf INTEGER,
            away_plus_minus INTEGER,
            
            FOREIGN KEY(home_team_id) REFERENCES teams(team_id),
            FOREIGN KEY(away_team_id) REFERENCES teams(team_id)
        );
        """
    
        try:
            curs = self.conn.cursor()
            curs.execute(sql_teams)
            curs.execute(sql_games)
            self.conn.commit()
            self.logger.info("Tables created")
        except sqlite3.Error as e:
            self.logger.error(f"Cant create tables: {e}")
            raise

    def latest_stored_date(self):
        try:
            q = "SELECT MAX(game_date) FROM games"
            df = pd.read_sql(q, self.conn)
            if df.empty or df.iloc[0, 0] is None:
                return None
            return df.iloc[0, 0]
        except Exception:
            return None
    
    def fetch_games(self, seasone):
        try:
            gamelog = leaguegamelog.LeagueGameLog(
                season=seasone,
                season_type_all_star='Regular Season',
                player_or_team_abbreviation='T'
            )
            df = gamelog.get_data_frames()[0]

            if df.empty:
                self.logger.info(f"No data for {seasone}")
                return pd.DataFrame()
            df['SEASON_ID'] = seasone
            return df
        except Exception as e:
            self.logger.error(f"Failed to collect data: {e}")
            return pd.DataFrame()
    
    def _update_teams(self, data: pd.DataFrame) -> None:
        if data.empty:
            return

        teams = data[['TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME']].drop_duplicates()
        teams = teams.rename(columns={
            'TEAM_ID': 'team_id',
            'TEAM_ABBREVIATION': 'team_abbreviation',
            'TEAM_NAME': 'team_name'
        })

        existing_id = pd.read_sql("SELECT team_id FROM teams", self.conn)['team_id'].tolist()
        new_teams = teams[~teams['team_id'].isin(existing_id)]

        if not new_teams.empty:
            new_teams.to_sql('teams', self.conn, if_exists='append', index=False)
            self.logger.info(f"Added {len(new_teams)} new teams to db")

    def _transform_games(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame()

        home_games = data[data['MATCHUP'].str.contains('vs.')].copy()
        away_games = data[data['MATCHUP'].str.contains('@')].copy()

        stats_map = {
            'PTS': 'pts',
            'FG_PCT': 'fg_pct',
            'FG3_PCT': 'fg3_pct',
            'FT_PCT': 'ft_pct',
            'AST': 'ast',
            'REB': 'reb',
            'TOV': 'tov',
            'STL': 'stl',
            'BLK': 'blk',
            'PF': 'pf',
            'PLUS_MINUS': 'plus_minus'
        }

        cols_home = ['GAME_ID', 'SEASON_ID', 'GAME_DATE', 'TEAM_ID', 'WL'] + list(stats_map.keys())
        home_df = home_games[cols_home].copy()
        
        home_col = {k: f"home_{v}" for k, v in stats_map.items()}
        home_col['TEAM_ID'] = 'home_team_id'
        home_col['GAME_ID'] = 'game_id'
        home_col['SEASON_ID'] = 'season_id'
        home_col['GAME_DATE'] = 'game_date'

        cols_away = ['GAME_ID', 'TEAM_ID'] + list(stats_map.keys())
        away_df = away_games[cols_away].copy()

        away_col = {k: f"away_{v}" for k, v in stats_map.items()}
        away_col['TEAM_ID'] = 'away_team_id'
        away_col['GAME_ID'] = 'game_id'
        
        home_df = home_df.rename(columns=home_col)
        away_df = away_df.rename(columns=away_col)

        final_df = pd.merge(home_df, away_df, on='game_id', how='inner')

        final_df['outcome'] = final_df['WL'].apply(lambda x: 'H' if x == 'W' else 'A')
        
        final_df['game_date'] = pd.to_datetime(final_df['game_date']).dt.strftime('%Y-%m-%d')

        target_cols = [
            'game_id', 'season_id', 'game_date', 
            'home_team_id', 'away_team_id', 'outcome',            
            'home_pts', 'home_fg_pct', 'home_fg3_pct', 'home_ft_pct', 
            'home_ast', 'home_reb', 'home_tov', 'home_stl', 'home_blk', 'home_pf', 'home_plus_minus',
            'away_pts', 'away_fg_pct', 'away_fg3_pct', 'away_ft_pct', 
            'away_ast', 'away_reb', 'away_tov', 'away_stl', 'away_blk', 'away_pf', 'away_plus_minus'
        ]
        
        return final_df[target_cols]

    def _save_games(self, data: pd.DataFrame) -> None:
        if data.empty:
            return        
        try:
            existing_ids = pd.read_sql("SELECT game_id FROM games", self.conn)
            
            merged = data.merge(existing_ids, on='game_id', how='left', indicator=True)
            to_insert = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

            if not to_insert.empty:
                to_insert.to_sql('games', self.conn, if_exists='append', index=False)
                self.logger.info(f"Saved {len(to_insert)} new games")
            else:
                self.logger.info("No new games")

        except Exception as e:
            self.logger.error(f"Save failed: {e}")
    
    def update_all(self):
        latest = self.latest_stored_date()
        current_year = datetime.today().year
        
        if latest is None:
            self.logger.info("Collecting historical data")
            seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(2020, 2026)]
        else:
            self.logger.info(f"Latest game in db: {latest}")
            seasons = [f"{current_year-1}-{str(current_year)[-2:]}",
                       f"{current_year}-{str(current_year+1)[-2:]}"]

        for s in seasons:
            df = self.fetch_games(s)
            if df.empty:
                continue
            
            self._update_teams(df)
            
            transformed = self._transform_games(df)
            
            if latest and not transformed.empty:
                transformed['game_date'] = pd.to_datetime(transformed['game_date'])
                latest_dt = pd.to_datetime(latest)
                transformed = transformed[transformed['game_date'] > latest_dt]
                transformed['game_date'] = transformed['game_date'].dt.strftime('%Y-%m-%d')

            self._save_games(transformed)
            time.sleep(1)

        self.logger.info("Update complete")

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.logger.info("Connection closed")
