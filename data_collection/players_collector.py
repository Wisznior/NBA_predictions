import pandas as pd
import sqlite3
import logging
import time
import os
from datetime import datetime, timedelta
from nba_api.stats.endpoints import leaguegamelog, commonallplayers

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(BASE, "logs")
DATA = os.path.join(BASE, "data")

os.makedirs(LOG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

class NBAPlayerCollector:
    def __init__(self, path_to_db: str = 'nba_history.db'):
        self.path_to_db = os.path.join(DATA, path_to_db)
        self.conn = None
        self.logger = None
        self._create_logs()
        self._db_connection()
        self._create_tables()

    def _create_logs(self) -> None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        logfile = f"nba_players_stats_{ts}.log"
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
        self.logger.info("NBA Player Collector initialized")

    def _db_connection(self) -> None:
        try:
            self.conn = sqlite3.connect(self.path_to_db)
            self.logger.info(f"Connected to db: {self.path_to_db}")
        except sqlite3.Error as e:
            self.logger.error(f"Db connection error: {e}")
            raise

    def _create_tables(self) -> None:
        sql_players = """
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY,
            player_name TEXT,
            team_id INTEGER
        );
        """
        sql_recent_stats = """
        CREATE TABLE IF NOT EXISTS player_recent_stats (
            player_id INTEGER PRIMARY KEY,
            last_game_date DATE,
            days_since_last_game INTEGER,
            avg_pts REAL,
            avg_reb REAL,
            avg_ast REAL,
            avg_stl REAL,
            avg_blk REAL,
            avg_tov REAL,
            avg_fg_pct REAL,
            avg_plus_minus REAL,
            games_played_count INTEGER,
            FOREIGN KEY(player_id) REFERENCES players(player_id)
        );
        """
    
        try:
            curs = self.conn.cursor()
            curs.execute(sql_players)
            curs.execute(sql_recent_stats)
            self.conn.commit()
            self.logger.info("Tables created")
        except sqlite3.Error as e:
            self.logger.error(f"Cant create tables: {e}")
            raise

    def get_current_season_string(self):
        now = datetime.now()
        if now.month >= 10:
            y = now.year
        else:
            y = now.year - 1
        return f"{y}-{str(y+1)[-2:]}"

    def update_players_list(self):
        self.logger.info("Update player list")
        try:
            players = commonallplayers.CommonAllPlayers(is_only_current_season=1)
            df = players.get_data_frames()[0]
            
            df_db = df[['PERSON_ID', 'DISPLAY_FIRST_LAST', 'TEAM_ID']].copy()
            df_db.columns = ['player_id', 'player_name', 'team_id']

            existing = pd.read_sql("SELECT player_id FROM players", self.conn)
            new_players = df_db[~df_db['player_id'].isin(existing['player_id'])]
            
            if not new_players.empty:
                new_players.to_sql('players', self.conn, if_exists='append', index=False)
                self.logger.info(f"Added {len(new_players)} new players")
            
            self.logger.info("Players updated")
            return df_db
            
        except Exception as e:
            self.logger.error(f"Cant update player list: {e}")
            return pd.DataFrame()

    def calculate_recent_form(self):
        season = self.get_current_season_string()        
        try:
            logs = leaguegamelog.LeagueGameLog(
                season=season,
                player_or_team_abbreviation='P',
                season_type_all_star='Regular Season'
            )
            df = logs.get_data_frames()[0]
            
            if df.empty:
                self.logger.info("No game logs")
                return

            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            
            def get_last_5(x):
                last_5 = x.sort_values('GAME_DATE', ascending=False).head(5)
                
                stats = {
                    'last_game_date': last_5['GAME_DATE'].max(),
                    'days_since_last_game': (datetime.now() - last_5['GAME_DATE'].max()).days,
                    'avg_pts': last_5['PTS'].mean(),
                    'avg_reb': last_5['REB'].mean(),
                    'avg_ast': last_5['AST'].mean(),
                    'avg_stl': last_5['STL'].mean(),
                    'avg_blk': last_5['BLK'].mean(),
                    'avg_tov': last_5['TOV'].mean(),
                    'avg_fg_pct': last_5['FG_PCT'].mean(),
                    'avg_plus_minus': last_5['PLUS_MINUS'].mean(),
                    'games_played_count': len(last_5)
                }
                return pd.Series(stats)

            recent_stats = df.groupby('PLAYER_ID').apply(get_last_5, include_groups=False)
            recent_stats = recent_stats.reset_index().rename(columns={'PLAYER_ID': 'player_id'})
            
            recent_stats['last_game_date'] = recent_stats['last_game_date'].dt.strftime('%Y-%m-%d')

            recent_stats.to_sql('player_recent_stats', self.conn, if_exists='replace', index=False)
            self.logger.info(f"Updated {len(recent_stats)} players")

        except Exception as e:
            self.logger.error(f"Failed to calculate: {e}")

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.logger.info("Connection closed")
