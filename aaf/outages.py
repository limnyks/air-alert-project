"""D36 fix 2: days that look like an outage of the official feed.

A UTC day is flagged if the official file has no alert at all (any oblast, any level) while the
volunteer file has alerts that day - independent evidence, so a truly calm day is not removed."""
import pandas as pd

from . import config


def daily_counts(path, start, end):
    d = pd.read_csv(path, usecols=["started_at"]).drop_duplicates()
    day = pd.to_datetime(d["started_at"], utc=True).dt.floor("D")
    day = day[(day >= start.floor("D")) & (day <= end.floor("D"))]
    return day.value_counts().sort_index()


def detect():
    days = pd.date_range(config.OFFICIAL_START.floor("D"), config.OFFICIAL_END.floor("D"), freq="D")
    off = daily_counts(config.RAW_DIR / "official_data_en.csv", config.OFFICIAL_START, config.OFFICIAL_END).reindex(days, fill_value=0)
    vol = daily_counts(config.RAW_DIR / "volunteer_data_en.csv", config.OFFICIAL_START, config.OFFICIAL_END).reindex(days, fill_value=0)
    t = pd.DataFrame({"day": days.date, "official_alerts": off.values, "volunteer_alerts": vol.values})
    t["outage"] = (t["official_alerts"] == 0) & (t["volunteer_alerts"] > 0)
    return t
