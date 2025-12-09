import requests
import json
import config
import logging

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

def request_odds_api(file_out):
    try:
        with open("parameters.json", "r") as f:
            params = json.load(f)
    except Exception as e:
        logging.warning(e.msg)

    API_KEY = config.API_KEY
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

if __name__ == "__main__":
    request_odds_api()

