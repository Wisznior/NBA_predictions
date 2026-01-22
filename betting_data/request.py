import requests
import json
import logging
import os
from dotenv import load_dotenv

def request_odds_api(file_out = "full_data_new.json"):
    try:
        load_dotenv()

        with open("parameters.json", "r") as f:
            params = json.load(f)

        API_KEY = os.getenv("ODDS_API_KEY")

        if not API_KEY:
            raise ValueError("Nie znaleziono API KEY w pliku .env")

        SPORT = params["request"]["sport"]
        REGION = params["request"]["region"]
        MARKET = params["request"]["market"]
        FORMAT = params["request"]["format"]  

        url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds/?apiKey={API_KEY}&regions={REGION}&markets={MARKET}&oddsFormat={FORMAT}'

        response = requests.get(url)

        used = int(response.headers.get('x-requests-used', 0))

        logging.info(f"quota: {used}/500")

        if response.status_code != 200:
            logging.warning(f"{response.status_code} - {response.text}")
        else:
            data = response.json()
            logging.info(f"pulled {len(data)} games")

            with open(file_out, 'w') as f:
                json.dump(data, f, indent=4) 
                logging.info("save succesful")

    except Exception as e:
        logging.warning(e)
        raise

if __name__ == "__main__":
    request_odds_api()

