import json


with open("full_data.json", "r") as f:
    data = json.load(f)
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
    
with open('parsed_data.json', 'w') as f:
    json.dump(games, f, indent=4) 
    
