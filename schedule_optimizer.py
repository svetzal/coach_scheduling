import pandas as pd
from pandas._libs import NaTType

from models import Schedule

schedule = Schedule(
    start_date=pd.to_datetime("2023-11-01"),
    end_date=pd.to_datetime("2024-10-31"),
    assignments=[]
)

date_range = pd.bdate_range(start=schedule.start_date, end=schedule.end_date, freq="C")
df = pd.DataFrame(date_range, columns=["date"])
df["year"] = df["date"].dt.year
df["day"] = df["date"].dt.dayofweek
df["week"] = df["date"].dt.strftime("%U")

grid_df = df.pivot(index=["year", "week"], columns="day", values="date")

doc = ""

doc += "<table>"
for index,row in grid_df.iterrows():
    doc += "<tr>"
    for day in row:
        if pd.notna(day):
            doc += f"<td>{day.strftime('%m/%d/%Y')}</td>"
        else:
            doc += "<td></td>"
    doc += "</tr>"
doc += "</table>"

with open("schedule.html", "w") as f:
    f.write(doc)

