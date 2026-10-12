"""Paths and the frozen protocol dates (NOTES.md section 12, D31; D34 amendment). All times UTC."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EPISODES = ROOT / "data" / "processed" / "raion_episodes.csv"
UNIT_CHECKS = ROOT / "logs" / "step10b_raion_checks.csv"
UNITS_CLEAN = ROOT / "config" / "units_clean.csv"          # D36 fix 1 (written by step12a)
OUTAGE_DAYS = ROOT / "config" / "outage_days.csv"          # D36 fix 2 (written by step12a)
OUTAGE_PERIODS = ROOT / "config" / "outage_periods.csv"    # D37: periods excluded from fitting and scoring
RAW_DIR = ROOT / "data" / "raw" / "7ccd436"

TZ = "Europe/Kyiv"
HOUR = pd.Timedelta(hours=1)
HOUR_NS = 3_600 * 10**9
MIN_NS = 60 * 10**9
HORIZONS = tuple(range(1, 25))                              # +1 h ... +24 h

OFFICIAL_START = pd.Timestamp("2022-03-15 16:10:34", tz="UTC")
OFFICIAL_END = pd.Timestamp("2026-09-07 05:28:22", tz="UTC")
VAL_START = pd.Timestamp("2026-05-18", tz="UTC")            # also the end of exploration data
FOLDS = tuple((pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")) for a, b in
              [("2026-05-18", "2026-06-01"), ("2026-06-01", "2026-06-15"),
               ("2026-06-15", "2026-06-29"), ("2026-06-29", "2026-07-13")])
PURGE = pd.Timedelta(hours=24)
PRE_TEST = (pd.Timestamp("2026-07-13", tz="UTC"), OFFICIAL_END)
SNAPSHOT = pd.Timestamp("2026-10-11 08:22:01", tz="UTC")
ADAPT = (pd.Timestamp("2026-09-11 08:22:01", tz="UTC"), pd.Timestamp("2026-09-18 08:22:01", tz="UTC"))
POST_TEST = (ADAPT[1], SNAPSHOT)

# Label times inside these windows may only be read in final-test mode (protocol 12.1: touched once).
PROTECTED = (PRE_TEST, POST_TEST)
