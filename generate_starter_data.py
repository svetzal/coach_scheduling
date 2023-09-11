import json
import pandas as pd

def generate_starter_data():

    #
    # Rejig to use the calendar created in services.py
    #

    with open("starter_data.json", "w") as f:
        json.dump({}, f, indent=4)


if __name__ == "__main__":
    generate_starter_data()
