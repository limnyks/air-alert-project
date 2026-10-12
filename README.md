# Raion Alert Outlook

KSE Agentic AI School, Stage 2 mini project. An hour-by-hour forecast of **announced air-raid alerts** for
110 Ukrainian raions: the chance that a raion is under alert at each of the next 24 hours, shown on a map.
The whole process (decisions D1–D49, mistakes M1–M13, every result) is logged in [NOTES.md](NOTES.md).

> **Research project, not a warning system.** Always follow official air-raid alerts. The forecasts are about
> *announced alerts*, not attacks or damage, and a low number never means safe.

**Map:** [`map/index.html`](map/index.html) (self-contained, no external requests) – tabs *Forecast* (+1…+24 h
slider), *Past days: forecast vs real*, and *Can I go?* (one-hour trip check, with a prominent disclaimer).

## The final model (D47)

```
logit p(raion, t+h) = logit(smoothed level of the raion at t) + hour shift(Kyiv hour of t+h) + w_h · alert_now
```

- **Level:** exponentially weighted share of time under alert (half-life 7 days), smoothed toward the national
  level so quiet raions never get 0 %.
- **Hour shift** (24 values) and **w_h** (24 horizons): one weighted logistic regression with the level as an
  offset; recent data weighted most (half-life 30 days); refit every day on labels before that day.
- **alert_now:** an alert still running at the data end counts as ON. Kyiv time with DST handled.
- Half-lives chosen on 2025 from {7, 14, 30, 60, 90} days, confirmed on 2026.

## Results

Primary metric: mean macro Brier over +1…+24 h (lower is better), skill = 1 − Brier(model)/Brier(flat B4),
95 % joint day-block bootstrap intervals. Each day learns only from earlier days.

| Period | Final | Flat B4 | Skill [95 %] |
|---|---|---|---|
| **2026, confirmation (276 days)** | 0.0937 | 0.0946 | **+0.9 % [+0.6 %, +1.3 %]** |
| alerts.in.ua weeks, 12 Sep – 11 Oct (30 days) | 0.1231 | 0.1236 | +0.4 % [−0.2 %, +1.1 %] – no evidence of a difference |

- The gain is at the first hours (+1 h +9.9 %, +2 h +3.7 %, +6 h +1.2 %); from about +13 h the model equals B4.
- What each part adds (2026): hour shift +0.4 % [+0.1, +0.7], current alert +1.2 % [+1.0, +1.4].
- Calibration: ECE 0.0073 vs B4 0.0039, difference +0.0033 [−0.0029, +0.0054] – worse as a point estimate,
  not significantly; the rule was read as "not significantly worse", like every other criterion (D47).
- 2025 (+9.7 %) is in-sample for the half-life choice and is **not** a result.
- Past days on the map (89 forecasts, each refit only on earlier data): Brier 0.1227; mean forecast 48 % where an
  alert happened vs 16 % where not.
- Earlier one-time test (post-reform, 22 days): the 7-day share B3 was best (0.1262); the M0 stack was worse.

## Data

| Source | Use |
|---|---|
| [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset), official file (MIT) | history 2022-03-15 → 2026-09-07; outage 29 Aug – 3 Sep masked |
| [alerts.in.ua API](https://devs.alerts.in.ua) (token required) | last 30 days and the live forecast; raw responses kept local only |
| [HDX COD-AB Ukraine](https://data.humdata.org/dataset/cod-ab-ukr) (CC BY-IGO) | raion and oblast shapes |
| [McDonald's restaurant locator](https://restaurants.mcdonalds.ua/) (fetched 2026-10-12) | city list in *Can I go?* (only derived facts saved) |

A raion is under alert when its own alert **or** an oblast-wide alert is on (air raid only). 9 raions with
near-permanent alerts are shown hatched, not forecast. The two sources agree on adjacent weeks (national share
0.21 vs 0.22, raion correlation 0.94). Committed files hold only aggregated forecasts and alert flags.

## Leakage protection

`History.observable(t)` (a model sees only what was known at t), a guard on test-window labels, automated
leakage tests (output must not change when the future is cut off or faked), daily refits on earlier labels
only (asserted), and forecasts committed before the hours they predict.

## How to run

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python -m unittest discover -s tests
```

Data preparation: `scripts/step05` … `step10b`. Model: `step21a_final_data.py`, `step21b_final_model.py`,
`step23_past_days.py`, map `step15_build_map.py`. **Refresh the live forecast and map:** put your own token in
`.env` (`ALERTS_IN_UA_TOKEN=...`, git-ignored), then `bash scripts/refresh.sh` (~14 min download, runs the
token check, commits the forecast and the map). Outputs of every step are in `logs/`.

**GitHub Pages:** Settings → Pages → Deploy from a branch → `main`, folder `/ (root)`; the map is at
`https://<user>.github.io/<repo>/map/`.

## Limitations

- Small gain over a flat level, only in the first hours; no evidence of a gain in the most recent weeks.
- Mass-attack days cannot be predicted from alert history alone (day-to-day correlation +0.02).
- Short post-reform history; the regime shift around September 2026 has no identified cause.
- The live forecast expires 24 h after its data time unless refreshed.
- Duration intervals were too narrow (72–74 % coverage instead of 80 %).
