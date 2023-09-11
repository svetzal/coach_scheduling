import json
import pandas as pd

def generate_starter_data():
    coaches = ["Coach 1", "Coach 2", "Coach 3", "Coach 4"]
    areas = ["Area 1", "Area 2", "Area 3", "Area 4"]

    start_date = pd.to_datetime("2023-11-01")
    simultaneous_assignments = 2

    # Generate starter data, assigning a pair of coaches to each area for 6-week intervals, with two assignments active at any given time

    # Target assignment data structure:
    # {
    #     "coaches": {
    #         "prime": "Coach 1",
    #         "second": "Coach 2"
    #     },
    #     "area": "Area 1",
    #     "start_date": "2023-11-01",
    #     "end_date": "2023-12-12"
    # }

    # assignment_start_date = start_date + pd.DateOffset(weeks=k)
    # assignment_end_date = assignment_start_date + pd.DateOffset(weeks=6)

    starter_data = []
    for k in range(0, len(areas)):
        assignment_start_date = start_date + pd.DateOffset(weeks=k)
        assignment_end_date = assignment_start_date + pd.DateOffset(weeks=6)
        starter_data.append({
            "coaches": {
                "prime": coaches[k*2 % len(coaches)],
                "second": coaches[(k*2 + 1) % len(coaches)]
            },
            "area": areas[k],
            "start_date": assignment_start_date.strftime("%Y-%m-%d"),
            "end_date": assignment_end_date.strftime("%Y-%m-%d")
        })

    file_structure = {
        "coaches": coaches,
        "areas": areas,
        "simultaneous_assignments": simultaneous_assignments,
        "assignments": starter_data
    }

    with open("starter_data.json", "w") as f:
        json.dump(file_structure, f, indent=4)


if __name__ == "__main__":
    generate_starter_data()
