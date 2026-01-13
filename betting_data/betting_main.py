import parser
import request
import logging

logging.basicConfig(
    level=logging.INFO, 
    style = '{',
    format = '{asctime} [{levelname:^8.8}] {funcName:^16.16} | {message}',
    datefmt='%H:%M:%S'
)


def main():
    try:
        request.request_odds_api()
        parser.parse("full_data_new.json", "parsed_data_new.json")
        parser.write_probability("parsed_data_new.json")
        parser.merge("parsed_data.json", "parsed_data_new.json", "parsed_data.json")
        parser.write_relevant_records("parsed_data.json", "relevant_data.json")

        parser.count_records("parsed_data.json")
        parser.count_records("relevant_data.json")

        # parser.merge("full_data.json", "full_data_new.json", "full_data.json")
        # parser.parse("full_data_new.json", "parsed_data_new.json")
        # parser.merge("parsed_data.json", "parsed_data_new.json", "parsed_data.json")

        # parser.count_records("full_data.json")
        # parser.count_records("full_data_new.json")
        # parser.count_records("parsed_data_new.json")

    except Exception as e:
        logging.warning(f"program stopped | {e}")


        
if __name__ == "__main__":
    main()
