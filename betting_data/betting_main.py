import parser
import request
import logging

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

def main():
    logging.info("=== main ===")
    request.request_odds_api("full_data_new.json")
    parser.merge("full_data.json", "full_data_new.json", "full_data.json")
    parser.parse("full_data_new.json", "parsed_data_new.json")
    parser.merge("parsed_data.json", "parsed_data_new.json", "parsed_data.json")

    logging.info("============")




if __name__ == "__main__":
    main()
