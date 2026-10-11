# Project notes: Air raid alert forecasting (KSE Agentic AI School, Stage 2)

Single source of truth for decisions, verified facts, hypotheses and mistakes.
Updated at the end of every step, after the user approves the changes.

---

## 1. Task & rules

### Organizers' requirements (as given by the user)
- Build a mini pet project in Python: time series analysis and forecasting of air raid alerts in Ukraine.
- Must be a genuine model that reflects the true state of affairs and, based on that, predicts an alert and its duration.
- Use the most recent data available and evaluate on the latest period.
- No data leakage: nothing the model is trained on or uses as input may come from after the time the forecast is made.
- Optional: text data (news, official statements, public Telegram channels), same rule: only content published before the forecast time.
- Deliverables: a GitHub repo with working Python code, the conversation log, and a ~100-word reflection.
- They evaluate the process: how the user iterates, catches mistakes and guides the architecture.

### Working rules (user's rules for the assistant)
1. Explain the plan before writing code, wait for the user's decision, and write code only after approval.
2. Small steps. After each step the user runs the code and shows the output.
3. Do not invent datasets, APIs, columns or results. If unsure something exists, say so and give a link to check.
4. Every claim about the data must come from output we actually saw.
5. The user makes the key decisions. The assistant gives options with pros and cons.
6. If more info is needed, ask. Details about the task are known only if the user states them.

---

## 2. Decisions log

| # | Date | Decision | Options considered | Reason |
|---|------|----------|--------------------|--------|
| D1 | 2026-10-10 | Keep notes in one file `NOTES.md` inside the repo | Repo markdown / assistant's internal memory / Notion or Google Doc | Visible to organizers, versioned in git, readable by both of us |
| D2 | 2026-10-10 | Project lives in new subfolder `air-alerts-forecast/` with its own git repo | New subfolder / use parent folder as is | Clean repo for submission |
| D3 | 2026-10-10 | Conversation log: final export via the app at project end, converted to `CONVERSATION_LOG.md`; raw `.jsonl` also committed | Raw export only / readable markdown only / both | Readable for organizers, raw file proves nothing was edited |
| D4 | 2026-10-10 | Scope: big country map by oblast → select an oblast → breakdown by hromada. Kyiv map later. Build order: data → oblast model → country map → drill-down → Kyiv | One region / all oblasts / few targets + all as inputs | User wants several levels of detail; core model first so there is always a working, honest result |
| D5 | 2026-10-10 | Alert start: one hourly hazard model gives both hourly probabilities and predicted time of next alert | Hourly grid only / survival only / combined | User wants both views; one model keeps them consistent |
| D6 | 2026-10-10 | Duration: one model with "elapsed minutes" input gives both duration at start and remaining duration | Duration at start / remaining / both | User wants both |
| D7 | 2026-10-11 | Text data included (news, official statements, Telegram). Added after the alert-history model; each source stays only if it beats "without it" on the same test period | No text / text later as experiment / text from start | User wants maximum accuracy from all sources that meet quality standards (see below) |
| D8 | 2026-10-11 | Forecast horizon: 24 hours | shorter / 24h / longer | User wants a 24-hour view per region |
| D9 | 2026-10-11 | Interface: big map + time slider (now … +24h), red = alert, gray = normal; region view with 24h hourly status strip | — | User's design |
| D10 | 2026-10-11 | The model is grounded in mathematics. Each region is an on/off process X_r(t). Alert starts: point process with conditional intensity λ_r(t \| history before t), candidate form = multivariate Hawkes (seasonal baseline μ_r(t) via Fourier terms + self-excitation α_rr + cross-region excitation α_rs, decay kernel g; intensity active only in "off" state). Durations: survival analysis S(d) with censoring (log-normal / Weibull / mixture); remaining duration = S(e+x)/S(e). Map outputs (P(active at +h), next alert time, duration) all come from Monte Carlo simulation of the fitted model. Interface is minimal and only renders the model's probabilities | Pipeline/ML-centered plan with UI focus (rejected) | User: mathematical calculations must underpin every action; real impact and a model matter, not a beautiful interface |
| D11 | 2026-10-11 | Option C: mathematical model is the core; gradient boosting is a challenger on the same inputs. Model ladder M0 inhomogeneous Poisson → M1 + self-excitation → M2 + cross-region excitation → M3 + regime adaptation → M4 + text as exogenous input to λ; ML challenger. Each level must beat the previous on the same walk-forward test windows by proper scores | A math only / B ML only / C math core + ML challenger | Interpretable core plus an honest check whether ML adds anything |
| D12 | 2026-10-11 | Python 3.12.15 installed via Homebrew; project virtual environment in `.venv`; libraries added step by step and pinned in `requirements.txt`; `.gitignore` excludes `.venv/`, `data/`, secrets | Keep system Python 3.9.6 / install newer | 3.9 is old; scipy/statsmodels needed for likelihood and survival models |
| D13 | 2026-10-11 | Luhansk oblast and Crimea excluded from the model; shown on the map as "no data". To re-check against our own data in Step 4 | Exclude / try to model | Other projects report no usable data (permanent alert / absent); user approved |
| D14 | 2026-10-11 | Option b: generative model (λ + S + simulation) is the core; direct ML challenger compared per horizon. Hybrid (use whichever wins per horizon) decided only later, on evidence | a generative only / b core + challenger / c hybrid | Keeps consistency and interpretability; challenger reveals structural weaknesses; hybrid would bring back inconsistency, so only if the gain is clearly worth it |
| D15 | 2026-10-11 | Adopt changes 1–24 from related work, each with its mitigation (section 10) | Adopt all / subset | Research showed concrete ways to improve data handling, model structure and evaluation |
| D16 | 2026-10-11 | Map boundaries: HDX cod-ab-ukr (Kartographia, reference 2025, updated 2026-01-26, CC BY-IGO) | HDX / geoBoundaries | geoBoundaries ADM2 is from 2006 (495 old raions) and won't match the data's post-2020 raions |
| D17 | 2026-10-11 | Data source strategy decided by evidence: Step 5 measures agreement between official and volunteer files on their overlap (2022-03-15 → 2026-09-07). Then choose: (1) splice official→volunteer, (2) volunteer as single label source + official for repair/sub-oblast/cross-check + alerts.in.ua for test-period check/live, (3) re-collect from Telegram | A–E single/combined options | Combining sources risks an artificial break at the test period; the overlap lets us measure source disagreement instead of guessing |

**Evaluation (D10/D11, to be frozen before modeling):** proper scoring rules (log-likelihood, Brier) with walk-forward windows; time-rescaling theorem goodness-of-fit (rescaled inter-arrival times ~ Exp(1), KS test); calibration; significance of score differences across windows; durations: log-score and CRPS with censoring.
**Implementation caveat:** unverified whether a ready Hawkes library works on our Python/Mac; fallback is our own likelihood with scipy, or an hourly logistic model whose inputs are the Hawkes kernel sums (discretized version of the same process).

**Source quality standards (D7):** reliable publication timestamp; archive covers the whole train + test period without big gaps; still updated; accessible legally and repeatably. Telegram API credentials (api_id/api_hash) are created and stored by the user, read from a local file that is never committed.

---

## 3. Verified data facts
_Only facts seen in actual output, tagged with the step they came from._

**Step 4 (read-only: GitHub API, file heads/tails via byte ranges, docs pages, Telegram previews; dataset commit `7ccd4367b7057a4541ce109ca09cf1f0220f6a75`, 2026-10-10 05:01 UTC):**
- Vadimkin repo is updated daily (commits every day up to 2026-10-10). Files: `official_data_en.csv` (31.0 MB), `official_data_uk.csv`, `volunteer_data_en.csv` (8.9 MB), `volunteer_data_uk.csv`. License MIT.
- Official columns: `oblast,raion,hromada,level,started_at,finished_at,source`; `level` values seen: `oblast`, `hromada`, `raion`. Times UTC with `+00:00`, second precision. First line 2022-03-15; **last lines 2026-09-07** (max date to confirm on full file in Step 5).
- Volunteer columns: `region,started_at,finished_at,naive`; oblast level only. First line 2022-02-25; last lines 2026-10-10.
- Dataset README: all times UTC; since December 2025 mostly raion-level alerts in the official data; volunteer data from eTryvoga; **if no end message: `naive=True` and `finished_at = started_at + 30 min`** (also seen in parser code) → these durations are invented and must be treated as censored.
- README: permanent sirens in Luhansk (since 2022-04-04) and Crimea (since 2022-12-10) are **not listed** in the datasets (supports D13).
- Parsers: official data from Telegram `@air_alert_ua`, volunteer from `@UkraineAlarmSignal` (eTryvoga), both via Telethon `iter_messages`.
- Open issue #7 (2026-09-25): official dataset stopped 2026-09-07 05:28 UTC; likely cause: message format change after the alert reform. `@air_alert_ua` still posts (new format, emoji 🔴/🟠/🟢).
- `@UkraineAlarmSignal` posts "Жовтий рівень тривоги" / "Червоний рівень тривоги" / "Відбій тривоги" with threat type (e.g. БПЛА).
- Telegram public web preview (`t.me/s/<channel>`) works without login and paginates back to early message IDs; dates not visible in extracted text (only times).
- alerts.in.ua API: token required; history endpoint `/v1/regions/{uid}/alerts/month_ago.json` = **1 month only**; 2 history requests/min; has `alert_level`, `location_type` (oblast/raion/city/hromada); not for critical infrastructure.
- geoBoundaries UKR: ADM1 27 units (2017), ADM2 495 units (2006, pre-reform), ADM3 10375 units. HDX cod-ab-ukr: Kartographia, dataset date 2025-09-01→2025-12-18, last modified 2026-01-26, CC BY-IGO, GeoJSON/SHP/GDB/XLSX.
**Step 5 (script `scripts/step05_inspect_data.py`, output `logs/step05_output.txt`; python 3.12.15, pandas 3.0.6, numpy 2.5.4; sha256 official `b396f876…`, volunteer `e441ecaa…`):**
- Official: 299,275 rows; **113,845 exact duplicate rows (38%)**; not sorted; started_at 2022-03-15 16:10:34 → **2026-09-07 05:28:22 UTC** (whole file).
- Official levels: raion 132,251 / oblast 130,235 / hromada 36,789. **Raion-level rows start rising in 2025Q1 (814) and dominate from 2025Q3** (13,002 raion vs 7,176 oblast); from 2025Q4 oblast-level rows almost vanish (1,806; then ~139–193 per quarter). README's "since December 2025" understates this.
- Official durations: 4 rows ≤ 0 min, 779 < 1 min, **1,318 > 24 h**; median 48.4 min.
- Volunteer: 115,793 rows, 0 duplicates, not sorted; 2022-02-25 16:36 → **2026-10-10 04:47 UTC**.
- Volunteer naive rows: 5,791 (5.0%), **all exactly 30 min** (rule confirmed). Naive share by month: 2026-08 0.045, 2026-09 0.063, 2026-10 0.091 (partial); weekly up to 0.084–0.092 in Oct.
- Volunteer real (non-naive) durations: 6 ≤ 0 min, 1,348 < 1 min, 14 > 24 h; median 31.2 min overall, but 18.3 / 17.0 min in the 28 days before / after 2026-09-06.
- Region names: 25 in each file, identical sets (no mapping needed). Luhanska oblast: 2 official rows, 3 volunteer rows; no Crimea/Sevastopol (D13 confirmed).
- Reform window (volunteer): 136.8 → 145.8 alerts/day (+6.6%); weekly counts rise from the week ending 2026-08-30, i.e. before the reform. No sharp break in counts.
- Agreement (official rolled up as union of all levels per oblast vs volunteer, starts within ±5 min): pooled 0.519 (volunteer found in official) / 0.710 (official found in volunteer). **Declines over time: 2022Q2–Q3 ≈ 0.93–0.95 / 0.91–0.95 → 2024Q1 0.636 / 0.758 → 2025Q1 0.410 / 0.590 → 2026Q3 0.225 / 0.392.**
- Matched alerts have nearly identical times: median |start diff| 0.25–0.6 min, median end diff 0.2–0.7 min.
- By region: western oblasts agree well (Zakarpatska 0.967, Chernivetska 0.942, Ternopilska 0.913); frontline oblasts badly (Dnipropetrovska 0.144 with 12,953 volunteer alerts vs 3,417 official episodes; Kharkivska 0.277; Sumska 0.356).
- Only 10.5% of naive volunteer alerts have an official match within 5 min (repairable end).
- **Interpretation (hypothesis, not proven):** low agreement is mostly a definition difference (volunteer may log a new region alert per local alert; the union roll-up merges them, and long official alerts swallow many volunteer alerts), not timing errors. → **Splicing sources (D17 option 1) is not safe.**

- External (news, multiple sources): **Cabinet resolution №1092 of 4 Sep 2026** introduced yellow (drones) and red (missiles/ballistic/massive) alert levels; rollout Kyiv 6 Sep, most regions ~8 Sep, Lviv 12 Oct; amended by №1128 (15 Sep).

---

## 4. Hypotheses
_Status: untested / confirmed / rejected. These come from background knowledge, not from our data._

| # | Hypothesis | Mathematical test | Status | Evidence |
|---|-----------|-------------------|--------|----------|
| H1 | Alerts cluster at night (drone/missile attacks) | Shape of seasonal baseline μ_r(t); start rate by hour of week; Fourier terms significant (likelihood-ratio test) | untested | none |
| H2 | Short alerts tend to be ballistic threats, long ones drone waves | Two-component mixture fits durations better than a single distribution (likelihood / information criterion) | untested | none (threat type itself may not be in the data) |
| H3 | Alerts in neighboring/frontline regions before t raise the probability of an alert in the target region soon after | Cross-excitation α_rs > 0 with confidence interval; lead-lag cross-correlation of hourly counts; M2 beats M1 out of sample | untested | none |
| H4 | Recent activity (alerts in last 24h / 7d, time since last alert) is predictive | Overdispersion of counts (variance/mean > 1), non-exponential inter-arrival times; self-excitation α_rr > 0; M1 beats M0 out of sample. **Warning (IDS Hawkes paper, arXiv:2601.14299): excitation can look strong at short scales but fade on long data; a too-rigid baseline makes slow rate changes look like self-excitation. M0 must include a causal (past-only) slowly varying trend so any M1 gain is real** | untested | none |
| H5 | Monitoring Telegram channels post leading signals (e.g. "drones launched") before alerts | Text intensity as exogenous input to λ improves out-of-sample score (M4 beats M3) | untested | none |
| H6 | An oblast-wide alert covers all its hromadas, so a hromada's alert status combines oblast, raion and hromada alerts | Check on raw records: oblast-level rows vs hromada-level rows overlapping in time | partly checked | Step 5: sub-oblast rows exist (raion 132,251, hromada 36,789); since 2025Q3 alerts are mostly raion-level. How to roll up to "oblast under alert" is still open (Step 6) |
| H8 | The September 2026 yellow/red reform (res. №1092) changed the alert process (rate, duration, geography), so the test period is a new regime | Compare start rate, duration distribution and naive share before vs after 2026-09-06; change-point term in μ(t) or separate post-reform parameters must improve out-of-sample score | partial evidence | Reform confirmed in news. Step 5 (volunteer): no sharp break in counts (136.8 → 145.8/day, rise started before the reform); median real duration 18.3 → 17.0 min; naive share rising 0.045 → 0.063 (Sep) → 0.091 (Oct, partial) — possibly a parser problem with the new message format (unverified) |
| H7 | Weather affects attacks (e.g. drones less active in strong wind / heavy rain) | Weather known at time t (observations or forecasts issued before t) as exogenous input to λ improves out-of-sample score. Stretch goal | untested | none (idea from Nujabesuuu/AlarmForecast, which uses weather) |

---

## 5. Leakage checklist
- [ ] Censoring: an alert ongoing at time t has no known end time; never use its real end time as input.
- [ ] Rolling/lag features use only data strictly before t.
- [ ] Other regions' alerts are used only if they started before t.
- [ ] Timezones & DST handled consistently (Kyiv time vs UTC).
- [ ] Text data uses the original publication time, not edit or scrape time.
- [ ] Time-ordered split only: train on the past, test on the latest period. No random shuffling.
- [ ] Hyperparameters tuned on a validation period, never on the test period.
- [ ] Purge: no training label window (up to +24h) may reach into the validation/test period.
- [ ] Lag guard in code: any feature lag ≤ 0 raises an error.
- [ ] Automated leakage test: λ(t) computed from data cut at t equals λ(t) from full data and from data with fake future alerts; deliberately leaky features must make the test fail.
- [ ] Trend in μ(t) is causal (computed from the past only; no two-sided smoothing).
- [ ] Weather (if used): only values known at t (observations or forecasts issued before t), never actual future weather.
- [ ] Known limitation, not fixable in code: the source data may contain later corrections (retroactive edits).

---

## 6. Mistakes & fixes
| # | Step | What went wrong | How it was caught | Fix |
|---|------|-----------------|-------------------|-----|
| M1 | Step 1 | Assistant stated H1/H2 from background knowledge, labeled only softly as "expectation" | User asked the assistant to restate the rules; self-check | Moved them to an explicit hypotheses list; they count as facts only once confirmed by output |
| M2 | Plan review | Plan was pipeline- and UI-centered with ad hoc features; no mathematical model of the alert process | User caught it: "mathematical calculations have to underpin every action" | Rebuilt the core around an on/off point process (intensity λ, survival S, simulation); each hypothesis got a mathematical test (D10, D11) |
| M3 | Step 5 | Part 6 divided official "after" rows by 28 days although the official file covers only ~1.2 days after 2026-09-06 → misleading "14.5 rows/day" (real rate ≈ 330/day, same as before) | Assistant while reading the output | Ignore that number; future scripts divide by the days actually covered by the data |

---

## 7. Open questions
_Waiting for the user's answer or for data._

Answered: A → D4, B → D5 + D8, C → D6, D → D7, Q1 → deadline not relevant (user: focus on the project), Q3 → Python 3.9.6, Q10 → option b (simulation from the fitted model, D10), Q11 → newer Python (D12).

- Q2. Tokens: alerts.in.ua token / Telegram api_id needed? (Decide after Step 4 shows which sources require them.)
- Q4. Data source freshness: which candidate source has the most recent data? (Unverified candidates: Vadimkin dataset https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset ; alerts.in.ua API https://devs.alerts.in.ua ; Telegram https://t.me/air_alert_ua ; Kaggle)
  **⚠ Freshness warning (reported by other projects, NOT verified by us):** the official file in the Vadimkin dataset reportedly ends 2026-09-07 (vladysalvkrutii, maranferij-lang); volunteer (eTryvoga) file reportedly reaches 2026-10-09 (vladysalvkrutii, Ddrakkx) but was called unreliable after mid-2025 (maranferij-lang); feed outage reported Aug 29 – Sep 3, 2026 (maranferij-lang); ~10% of volunteer alerts have no end message (vladysalvkrutii). lbgos/AirAlerts got fresh data by scraping the Telegram alert channel @UkraineAlarmSignal.
- Q5. Does an alert dataset with raion/hromada granularity exist, and from what date?
- Q6. Boundary (map shapes) source: HDX cod-ab-ukr https://data.humdata.org/dataset/cod-ab-ukr (to verify levels and freshness).
- Q7. Interface technology: Streamlit / static HTML (Folium) / Dash?
- Q9. Map coloring: (a) threshold red/gray, (b) gray→red gradient, (c) threshold + exact % on hover/panel?
- **Q13. ⚠ TELEGRAM IS AN UNSOLVED ISSUE (keep in mind).** Possible needs: (1) fresher alert data if public datasets are stale, (2) text signals for H5. Problems: access needs the user's personal API credentials; organizers cannot re-collect without their own; terms of use not checked; history length unknown; edited/deleted posts; parsing errors. Options: A no Telegram / B Telegram API + commit only derived frozen snapshot / C public web preview t.me/s/ (unverified) / D text datasets collected by others (unverified). Decide after Step 4 facts.
  **Step 4 update:** public preview works without login and paginates far back, but dates not visible in extracted text; full history = thousands of page requests; terms not checked. New option E: run the dataset's own MIT collection script (Telethon, needs user's credentials) with a parser fixed for the post-reform format → fresh official raion-level data with yellow/red levels.
- Q12. Plan review (3 cycles of drawbacks): which fixes are accepted? (skeleton first, freshness measurement, frozen evaluation protocol, live + replay mode, conditional hromada/text, reproducibility, walk-forward, label definitions, UTC handling, known-limitation of retroactive data edits, disclaimer, derived text features only, Parquet, must-have vs stretch lists)

---

## 8. Step log
| Step | Goal | Script | Key result |
|------|------|--------|------------|
| 1 | Understand the task, what is realistically predictable, decisions to make | none | Framing: probabilistic short-horizon forecast + duration; open decisions A–D |
| 1b | Restate the user's rules | none | Rules confirmed by the user |
| 2 | Set up project folder, git and notes | none (no code) | `air-alerts-forecast/` created, `git init`, `NOTES.md` created |
| 3 | Test export of the conversation transcript | none (app export + inspection) | Format: JSONL; user and assistant text plus tool calls kept, reasoning blocks empty, export is a snapshot up to the export moment |
| 3b | Requirements gathering | none | Decisions D4–D9 recorded; map coloring and "active at +h" approach still open (Q9, Q10) |
| 3c | Plan review: 3 cycles of drawbacks, then mathematical redesign | none | User rejected the UI-centered framing (M2); model rebuilt as on/off point process + survival + simulation (D10); option C chosen (D11) |
| 3d | Environment setup | shell commands (no project code) | Homebrew Python 3.12.15; venv at `.venv/` confirmed (`which python` → `.venv/bin/python`, `--version` → 3.12.15); `.gitignore` created |
| 3e | Research of similar projects (web, EN + UA), then analysis: useful ideas, negative impacts, challenges | none | 12+ similar projects found (3 KSE applications); no point-process / Hawkes model found; careful projects report modest gains; data-freshness warnings; changes 1–24 adopted with mitigations (D15); D13, D14 decided; Telegram flagged as unsolved (Q13) |
| 4 | Check data sources (read-only) | none (API calls, byte-range reads, docs) | Official file stops 2026-09-07 (confirmed, issue #7); volunteer fresh to 2026-10-10 but oblast-only with invented +30 min ends; alert reform №1092 (Sep 2026) = structural break before test period (H8); alerts.in.ua history only 1 month; HDX chosen for boundaries (D16); source strategy to be decided by agreement measurement (D17) |
| 5 | Download pinned data, print facts, measure official vs volunteer agreement | `scripts/step05_inspect_data.py` → `logs/step05_output.txt` | Official ends 2026-09-07 (confirmed), 38% duplicate rows, raion-level from 2025Q3; volunteer fresh to 2026-10-10, naive 5.0% and rising after reform; agreement falls from ~0.95 (2022) to 0.225/0.392 (2026Q3) → splicing not safe; cause probably a definition difference. User chose a diagnostic step next (Step 6) |

---

## 9. Related work
_Found 2026-10-11. All numbers are the authors' own claims from their pages; nobody re-ran their code. We take ideas with credit, not code._

| Project | What it does | Method | Reported result | Ideas we adopt |
|---|---|---|---|---|
| [lbgos/AirAlerts](https://github.com/lbgos/AirAlerts) | P(siren start) and P(all-clear) per oblast, 15m–6h; live map | LightGBM per horizon, 60-day recency half-life, Platt + cap calibration; Telegram scraped from 2025-12-01 | Test 2026-09-26→10-10, AUC 0.88–0.92, Brier below climatology at all horizons | Rings, recency weighting, Telegram alert channel, confident-but-wrong metric, live logging |
| [vladysalvkrutii/kse-air-alerts](https://github.com/vladysalvkrutii/kse-air-alerts) (KSE) | Alert start within 3h, 24 oblasts | Logistic regression, 16-week window, walk-forward | Test 2026-09-11→10-09, Brier skill +7.4%, within-region AUC 0.570 | Leakage test with deliberately leaky features, pinned commit, window selection |
| [Ddrakkx/alerts-forecast](https://github.com/Ddrakkx/alerts-forecast) (KSE) | Alert start 15m–6h, Poltava (+Kyiv, Kharkiv, Lviv) | Logistic regression, gradient boosting | No robust gain for Poltava; gains for Kyiv/Kharkiv at short horizons | Week-block bootstrap CIs, per-region reporting, off-state-only scoring |
| [andriiboichuk13/ua_alerts_analysis](https://github.com/andriiboichuk13/ua_alerts_analysis) (KSE) | Time series analysis | not described | — | — |
| [Dopplerr24/air-raid-alert-analysis](https://github.com/Dopplerr24/air-raid-alert-analysis) | Alert start within 1h, 24 oblasts | One LightGBM, 5 walk-forward folds, 24h embargo | PR-AUC 0.324 vs 0.152 baseline; adjacency 0.337 | Lag guard, purge, PR-AUC |
| [maranferij-lang/Ukraine-air-alerts](https://github.com/maranferij-lang/Ukraine-air-alerts) | Alert start / active within 1/3/6h, 24 units | LightGBM per H | Test 2026-08-07→09-05, within-oblast AUC 0.64–0.69; GDELT news didn't help | Outage masking, official vs volunteer comparison, documented negative results |
| [koalla05/air-alerts](https://github.com/koalla05/air-alerts) | Remaining alert duration, quiet time | Survival framing, Kaplan-Meier, LightGBM quantiles | No aggregate results | Episode merging, KM baseline, 10/50/90% quantiles |
| [Nujabesuuu/AlarmForecast](https://github.com/Nujabesuuu/AlarmForecast) | Hourly probability, 25 regions, 24h; map | LightGBM; alerts + weather + Telegram (385k msgs) + ISW | ROC-AUC 0.911 (test period not stated) | Weather idea (H7) |
| [3olena3/air-raid-alerts-forecast](https://github.com/3olena3/air-raid-alerts-forecast) | Daily alert count + total duration, next day | CatBoost | Duration MAE ~35–55 min | Expected alert minutes in 24h |
| [mykoladomanskyi25-dotcom/Air-raid-alerts](https://github.com/mykoladomanskyi25-dotcom/Air-raid-alerts) | Daily counts, baselines | Naive, rolling mean | Best: 7-day rolling mean; misses mass-strike spikes | 7-day rolling mean baseline |
| [mak74ik/ukraine-alert-forecast](https://github.com/mak74ik/ukraine-alert-forecast), [Kamenqq/air-raid-alerts-forecast](https://github.com/Kamenqq/air-raid-alerts-forecast) | Not described | — | — | — |

**Papers and other sources**
- [Pavlyshenko & Pavlyshenko, arXiv:2411.14625](https://arxiv.org/abs/2411.14625): random forests, 5–15 min ahead, 4 oblasts. [Review](https://pith.science/paper/2411.14625) flags leakage (unlagged own-region feature), no baselines, single split. Used as a warning example.
- [IDS Hawkes paper, arXiv:2601.14299](https://arxiv.org/pdf/2601.14299): excitation fades on long data → H4 warning.
- [Euromaidan Hawkes, arXiv:2405.12253](https://arxiv.org/pdf/2405.12253): discretized Hawkes on counts → supports our hourly fallback.
- [BFI working paper](https://bfi.uchicago.edu/wp-content/uploads/2022/10/BFI_WP_2022-148.pdf): public response to alerts (not a forecast).
- [VoxUkraine](https://voxukraine.org/en/lost-lessons-due-to-air-raid-alerts-how-many-and-what-can-be-done): median 33 min for weekday daytime alerts, Sep 2024 – Feb 2025.
- Statistics sites: [tryvogy.com](https://tryvogy.com/en), [air-alarms.in.ua](https://air-alarms.in.ua/en/region/zakarpatiia).

**Uniqueness:** no project found combines a two-state point process, ring-structured Hawkes excitation, survival mixture for durations, simulation-based outputs and a prospective test. Not finding it does not prove it doesn't exist, nor that it works better.

---

## 10. Adopted changes (D15) with mitigations

| # | Change | From | Possible negative impact → mitigation |
|---|---|---|---|
| 1 | Cross-region influence by rings (neighbor / 2nd ring / national) | lbgos | Symmetric rings miss directional attack routes → compare with directional pairs on validation |
| 2 | Exponential recency weighting of the likelihood | lbgos | Less effective data, forgets yearly campaigns, extra parameter → use either this or #8, small grid |
| 3 | Check Telegram alert channel as fresher source with threat type | lbgos | Credentials, reproducibility, parsing errors, time → see Q13 |
| 4 | "Confident but wrong" share | lbgos | Not a proper score; selecting by it pushes to 50% → report only |
| 5 | Live forecast logging, scored later (prospective test) | lbgos | Needs live data and days of running; small sample → accept the cost |
| 6 | Automated leakage test (cut vs full vs fake-future; leaky features must fail) | vladysalvkrutii | None (cannot catch retroactive source edits) |
| 7 | Pin dataset commit, drop last 24h | vladysalvkrutii | Conflicts with "most recent data" → re-pin to latest commit right before final run |
| 8 | Choose training window on validation | vladysalvkrutii | Overlaps #2, overfitting risk → one mechanism or very small grid |
| 9 | Week-block bootstrap CIs | Ddrakkx | None (intervals wide with few weeks; that is honest) |
| 10 | Per-region results | Ddrakkx | None (beware chance results across 24 regions) |
| 11 | Lag guard (lag ≤ 0 → error) | Dopplerr24 | None |
| 12 | Purge between train and test | Dopplerr24 | Extra embargo throws away the most recent data → purge only, no extra embargo |
| 13 | PR-AUC alongside Brier and log-likelihood; no pooled ROC-AUC | Dopplerr24 | Depends on base rate, ignores calibration → report only; select by log-likelihood/Brier |
| 14 | Detect outages, mask from likelihood | maranferij-lang | Real lulls mistaken for outages → only with independent evidence |
| 15 | Compare official vs volunteer files | maranferij-lang | None |
| 16 | Document negative experiments | maranferij-lang | None |
| 17 | Merge overlapping records into episodes | koalla05 | Merge rule changes counts/durations; loses hromada detail → keep raw + merged, test gap rules |
| 18 | Kaplan-Meier duration baseline | koalla05 | None (assumes non-informative censoring, same as our model) |
| 19 | Duration as median + 10–90% range | koalla05 | None |
| 20 | Slowly varying trend in μ(t) | IDS paper | Two-sided smoothing leaks; too flexible trend hides real excitation → causal trend, smoothness chosen on validation |
| 21 | 7-day rolling mean baseline | mykoladomanskyi25 | None |
| 22 | Weather as candidate input (H7), stretch | AlarmForecast | Leakage if actual future weather used → only weather known at t |
| 23 | Expected alert minutes in next 24h, stretch | 3olena3 | None (inherits errors of both models) |
| 24 | Compare our statistics with tryvogy / air-alarms / VoxUkraine | sites | Their methods differ/changed → rough sanity check only |

**Not adopted:** deleting long alerts as outliers (biases durations); ISW reports as text (leakage-prone timestamps); anything from projects without described methods; unlagged own-region features (Pavlyshenko).

**Cross-cutting risk:** many tunable choices (half-life, window, trend smoothness, merge gap) = many ways to overfit validation → freeze the evaluation protocol before modeling, keep grids small, touch the test period once.

---

## 11. Known limitations (no full solution)
1. Adversary decisions (start of a mass attack) cannot be predicted from alert history; Hawkes can only react once a wave starts.
2. Retroactive corrections in source data (leakage hidden in the history itself); documented, not fixable.
3. Regions without data (Luhansk, Crimea, occupied areas) → "no data" on the map (D13).
4. Freshness depends on dataset maintainers → backup source only reduces the risk.
5. Predictability ceiling: careful projects all report modest gains; our goal is honest measurement, not impressive numbers.
6. Separating self-excitation from a slowly changing rate is mathematically hard; only the out-of-sample test (M1 vs M0) decides.
7. Telegram (Q13): unsolved until Step 4 facts are in.
