import json
import logging
from datetime import date, timedelta

# returns a dict with game info if found
def find_game(home_team, away_team, file_from = "parsed_data.json"):
    with open(file_from, "r") as f:
        data = json.load(f)

    for record in data:
        if (record["home_team"] == home_team and record["away_team"] == away_team) or record["id"] == id:
            logging.info(f"game found of {home_team} and {away_team}") 
            return record      
                            
    logging.info("game not found") 

def count_records(file_from):
    with open(file_from, "r") as f:
        data = json.load(f)
    
    count = 0
    for i in data:
        count += 1

    logging.info(f"{count} records in file {file_from}")
    return count

# probability calculated from bookmakers odds - returns a tuple of floats in range [0, 1]
def implied_probability(game):
    try:
        if game:
            home_team = game["home_team"]
            away_team = game["away_team"]
            home = [1/i for i in game["h2h"]["home_price"].values()]
            away = [1/i for i in game["h2h"]["away_price"].values()]
            length = len(home)      
            home_avg = sum(home) / length
            away_avg = sum(away) / length
            home_avg_norm = round(home_avg / (home_avg + away_avg), 2)
            away_avg_norm = round(away_avg / (home_avg + away_avg), 2)
            logging.info(f"succesfully calculated for {home_team} and {away_team}")
            return (home_avg_norm, away_avg_norm)
        logging.warning(f"couldnt calculate for {home_team} and {away_team}")
    
    except Exception as e:
        logging.warning(e)

# calculate and write probabilities to each record
def write_probability(file_from = "parsed_data.json"):
    with open(file_from, "r") as f:
        data = json.load(f)
        logging.info(f"loading of file {file_from} succesful")

    for g in data:
        g["home_prob"], g["away_prob"] = implied_probability(g)
    with open(file_from, "w") as f:
        data = json.dump(data, f, indent=4)
        logging.info(f"saving to file {file_from} succesful")

# returns cutoff date
def expiration_date(d):
    return date.today() - timedelta(days = d)

# returns date of game as object of date
def game_to_date(game):
    d = game["time"]
    return date(int(d[0:4]), int(d[5:7]), int(d[8:10]))

# checks if game is within relevant time bound (3 days)
def is_expired(game):
    game_date = game_to_date(game)
    exp = expiration_date(3)
    if exp < game_date:
        return False
    return True

# for each record in parsed data file checks if within time bound
def write_relevant_records(file_from, file_to):
    try:
        with open(file_from, "r") as f:
            data = json.load(f)
            data_new = []
            logging.info(f"loading of file {file_from} succesful")

        for g in data:
            if not is_expired(g):
                data_new.append(g)

        with open(file_to, "w") as f:
            json.dump(data_new, f, indent=4)
            logging.info(f"saving to file {file_to} succesful")

    except Exception as e:
        logging.warning(e)
        raise

# process raw data to desired format
def parse(file_from, file_to):
    try:
        with open(file_from, "r") as f:
            data = json.load(f)
            logging.info(f"loading of file {file_from} succesful")
        
        games = []
        for i in data:
            id = i["id"]
            time = i["commence_time"]
            home_team = i["home_team"]
            away_team = i["away_team"]
            h2h = {"home_price":{}, "away_price":{}}
            spread = {"home_price":{}, "away_price":{}, "home_point":{}, "away_point":{}}
            total = {"over_price":{}, "under_price":{}, "point":{}}

            for j in i["bookmakers"]:
                for q in j["markets"]:
                    buk = j["title"]
                    if q["key"] == "h2h":
                        for gg in q["outcomes"]:
                            if gg["name"] == home_team:
                                h2h["home_price"][buk] = gg["price"]
                            if gg["name"] == away_team:
                                h2h["away_price"][buk] = gg["price"]
                    elif q["key"] == "spreads":
                        for gg in q["outcomes"]:
                            if gg["name"] == home_team:
                                spread["home_price"][buk] = gg["price"]
                                spread["home_point"][buk] = gg["point"]
                            if gg["name"] == away_team:
                                spread["away_price"][buk] = gg["price"]
                                spread["away_point"][buk] = gg["point"]
                    elif q["key"] == "totals":
                        total["over_price"][buk] = q["outcomes"][0]["price"]
                        total["under_price"][buk] = q["outcomes"][1]["price"]
                        total["point"][buk] = q["outcomes"][0]["point"]

            games.append({
                        "id": id,
                        "time": time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_prob": None,
                        "away_prob": None,
                        "h2h": h2h,
                        "spread": spread,
                        "total": total
                        })
            
        with open(file_to, 'w') as f:
            json.dump(games, f, indent=4) 
            logging.info(f"saving to file {file_to} succesful")

    except Exception as e:
        logging.warning(e)
        raise

# data from 2 input files is saved in order to output file
def merge(file_1, file_2, file_out):
    try:
        data_out = []
        with open(file_1, 'r') as f:
            data_1 = json.load(f)
            logging.info(f"loading of file {file_1} succesful")

        with open(file_2, 'r') as f:
            data_2 = json.load(f)
            logging.info(f"loading of file {file_2} succesful")

        if isinstance(data_1, list) and isinstance(data_2, list):
            for r in data_1:
                if not find_game(r["home_team"], r["away_team"], file_2):
                    data_out.append(r)
            data_out += data_2
        else:
            raise Exception("wrong data format")
            
        with open(file_out, 'w') as f:
            json.dump(data_out, f, indent=4)
            logging.info(f"saving to file {file_out} succesful")

    except Exception as e:
        logging.warning(e)
        raise

def calculate_expected_points(home_team, away_team, file_from = "parsed_data.json"):
    logging.info("calc")

    game = find_game(home_team, away_team, file_from)
    home = 0
    away = 0

    total_dict = game["total"]["point"].values()
    total = sum(total_dict) / len(total_dict) / 2.0

    spread_dict = game["spread"]["home_point"].values()
    spread = sum(abs(x) for x in spread_dict) / len(spread_dict) / 2.0

    home = round(total + spread, 0)
    away = round(total - spread, 0)

    if (game["home_prob"] < game["away_prob"]):
        home, away = away, home

    logging.info(f"home: {home}, away: {away}")
    return (home, away)

if __name__ == "__main__":
    pass    