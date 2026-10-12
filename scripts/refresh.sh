#!/usr/bin/env bash
# One command to refresh the forecast and the map:
#   fresh alerts.in.ua snapshot (~14 min) -> final-model forecast -> token check -> commit forecast
#   -> rebuild the map (map/index.html, docs/map.html) -> token check -> commit map.
# Needs YOUR OWN alerts.in.ua API token in .env (ALERTS_IN_UA_TOKEN=...; request one at https://alerts.in.ua/api-request).
# .env is git-ignored and the token is never printed or written anywhere else.
# Usage (from anywhere):  bash scripts/refresh.sh      then publish with:  git push
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

$PY -u scripts/step07a_snapshot_alerts_in_ua.py --live | tee logs/refresh_snapshot_output.txt
$PY scripts/step22_final_live.py | tee logs/step22_output.txt       # stops if the snapshot is incomplete
SNAP=$(ls -1 data/raw/alerts_in_ua_live | sort | tail -1)

$PY scripts/token_check.py
git add "forecasts/final_${SNAP}.csv" "forecasts/final_${SNAP}.json" logs/step22_output.txt
git commit -q -m "Live forecast (final model D47), snapshot ${SNAP}"   # commit time = proof it precedes the forecast hours

$PY scripts/step15_build_map.py
$PY scripts/token_check.py
git add map/index.html docs/map.html
git commit -q -m "Map: forecast from snapshot ${SNAP}"
echo "done: forecast and map committed for snapshot ${SNAP}. Publish with: git push"
