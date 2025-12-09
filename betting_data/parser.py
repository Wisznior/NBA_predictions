import json
import logging

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

def parse(file_from, file_to):
    with open(file_from, "r") as f:
        data = json.load(f)
        logging.info(f"loading of file {file_from} succesful")
        games = []

        for i in data:
            id = i["id"]
            home_team = i["home_team"]
            away_team = i["away_team"]
            h2h = {"home_price":[], "away_price":[]}
            spread = {"home_price":[], "away_price":[], "home_point":[], "away_point":[]}
            total = {"over_price":[], "under_price":[], "point":[]}

            for j in i["bookmakers"]:
                for q in j["markets"]:
                    if q["key"] == "h2h":
                        h2h["home_price"].append(q["outcomes"][0]["price"])
                        h2h["away_price"].append(q["outcomes"][1]["price"])
                    elif q["key"] == "spreads":
                        spread["home_price"].append(q["outcomes"][0]["price"])
                        spread["home_point"].append(q["outcomes"][0]["point"])
                        spread["away_price"].append(q["outcomes"][1]["price"])
                        spread["away_point"].append(q["outcomes"][1]["point"])
                    elif q["key"] == "totals":
                        total["over_price"].append(q["outcomes"][0]["price"])
                        total["under_price"].append(q["outcomes"][1]["price"])
                        total["point"].append(q["outcomes"][0]["point"])

            games.append({"id": i["id"], "home_team":home_team, "away_team": away_team, "h2h": h2h, "spread": spread, "total": total})
        
    with open(file_to, 'w') as f:
        json.dump(games, f, indent=4) 
        logging.info(f"saving to file {file_to} succesful")

def merge(file_1, file_2, file_out):
    try:
        with open(file_1, 'r') as f:
            data_1 = json.load(f)
            logging.info(f"loading file {file_1} succesful")

        with open(file_2, 'r') as f:
            data_2 = json.load(f)
            logging.info(f"loading file {file_2} succesful")

        if isinstance(data_1, list) and isinstance(data_2, list):
            data_out = data_1 + data_2
        else:
            raise Exception("wrong data format")
            
        with open(file_out, 'w') as f:
            json.dump(data_out, f, indent=4)
            logging.info(f"saving to file {file_out} succesful")
    except Exception as e:
        logging.warning(e)



if __name__ == "__main__":
    logging.info("parser")
    
