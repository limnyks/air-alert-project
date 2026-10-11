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
7. (added 2026-10-11) Every option lists its external dependencies and what happens if each fails (after mistake M4).
8. (added 2026-10-11) No bare risk lists: for each risk, write a solution right away, critique it three times, and present only the finalized result.

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
| D18 | 2026-10-11 | Independent reference labels for the test period (2026-09-07 → 2026-10-10) from **both** (A) official Telegram channel @air_alert_ua via public web preview (no login) and (B) alerts.in.ua history `month_ago` (token requested by the user, stored in `.env`, never committed). Every metric reported twice: vs volunteer labels and vs reference labels; the gap = label uncertainty | A only / B only / both | Volunteer file is the only fresh source; its accuracy after the reform is unknown (Q14). Two independent checks are strongest |
| D19 | 2026-10-11 | **Data-quality gate:** the model is adapted to a change in the data (change-point, recency weighting, etc.) only if the reference labels confirm the change is real; a change seen only in the volunteer file is treated as a data artifact (fix labels, not the model) | Adapt to any change / gate | Prevents "fitting the bug" (e.g. parser missing post-reform message types) |
| D20 | 2026-10-11 | Model uses **on/off states**, not raw records: overlapping alerts per region merged into "on" periods; a start = off→on switch; very short gaps joined (gap size tested). Unknown endings use **interval censoring** (end between start and next start in the region); invented 30-min ends never used. Sources are converted into one standard episodes table (region, start, end, end-type, source) | Raw records / states | Removes recording-style bursts that would fake Hawkes excitation; correct math for unknown endings; sources become replaceable |
| D21 | 2026-10-11 | **SUPERSEDED by D23.** **Provisional:** "oblast under alert" = the volunteer (eTryvoga) definition for the first model, because it is technically easiest (one file, already clean on/off with 0 overlaps, fresh to 2026-10-10, no merging or collection needed). **Revisit after the first test** (first baseline evaluation); likely switch target is option c (volunteer for oblasts + official per-raion/hromada series for drill-down). Risks: depends on volunteer accuracy (Q14), unexplained extra alert time in Volyn/Lviv (Q15). Switching must not require rewriting the model (standard episodes table, D20) | a volunteer / b official "any part" (U) / c two levels | User: start with the technically easiest, change after the first test |
| D22 | 2026-10-11 | Step 7a: one-time `month_ago` snapshot from alerts.in.ua for 24 units (23 oblasts + Kyiv City; no Luhansk/Crimea/Sevastopol per D13; corrected from "25" — Ukraine has 24 oblasts), ~35 s between requests (under 2/min), token from `.env` sent in the request header (never printed, never in URLs). Raw responses saved unchanged **locally only** (`data/`, gitignored) because terms §9.2 forbid reproducing materials without consent; only our derived results may be published (small legal uncertainty → Q17). No bulk fetch of 1,622 raion/hromada UIDs (~13.5 h; risk of token block for anomalous volume) | Oblast-level snapshot / all UIDs | Time-sensitive (1-month window); respects terms and limits |
| D23 | 2026-10-11 | **Data combination (supersedes D21):** train on the **official file up to 2026-09-07**; test on **alerts.in.ua 2026-09-11 → 2026-10-10**; target = **"is the oblast capital under air-raid alert"** (option C: status = oblast-wide alert OR the capital's raion OR the capital's own hromada/city alert). Costs accepted: 4-day gap 7–11 Sep skipped; test labels cannot be published (only metrics); Telegram grey zone documented in README; volunteer file kept only as a comparison | A volunteer (D21) / B official "any part" / C oblast capital | Step 7b: after the reform the volunteer file shows Kyiv City under alert 9.8% of time vs 32.0% in alerts.in.ua, missing endings up to 47% by region; "any part" saturates at the front (Kharkivska 100%, Dnipropetrovska 99.5%). Official file and alerts.in.ua come from the official alert stream → consistent definition for train and test. User: "lets do it" |
| D24 | 2026-10-11 | Target details for D23: **Donetska → Kramatorsk** (temporary administrative centre; check for saturation, exclude if always under alert); **Kyivska → Bila Tserkva** (largest city; the oblast administration sits in Kyiv City, a separate unit); **test labels = air_raid only** (measure how often other alert types occur at capital locations, risk H9). Capital names in each source to be taken from the data (Step 8a), not typed from memory | Kramatorsk / exclude; Bila Tserkva / exclude; air_raid only / all types | User decision |
| D25 | 2026-10-11 | **Text data stays in the plan** (user: "telegram or news is important"). Text is still added only after the alert-history model works and kept only if it beats "without text" on the same test period (D7). Source to be chosen then: news websites (lower legal risk, slower), GDELT (open, but did not help in another project), Telegram (fastest signals, but **Telegram terms forbid using its data for ML**; needs the user's API credentials; risk of account limits; raw messages not publishable). Suggested order: news first, Telegram if needed and the user accepts the terms risk | Drop Telegram/text / keep text | User decision; risks disclosed before deciding |
| D26 | 2026-10-11 | **Capital labels (Step 8 plan, finalized after 3 self-critiques):** (A) rule R1 for all capitals: capital under alert = oblast-wide alert in its oblast OR its raion OR its own city/hromada; other hromadas in the raion excluded; mapping in `config/capitals.csv` (every name verified verbatim against `logs/step08a_output.txt`). (B) official labels changed structure (oblast-wide alerts stop ~Oct–Nov 2025): if a capital's time-under-alert share after its switch differs from the 12 months before by more than ×1.5 either way, train it only after the switch, else full history; parameters shared across capitals. (C) consistency official→alerts.in.ua: adaptation week 2026-09-11 → 09-17 (re-estimate each capital's baseline level), final test 2026-09-18 → 10-10; evidence = volunteer bridge in healthy regions (Kharkivska, Kyivska, Dnipropetrovska, Mykolaivska) + before/after statistics; the 56-alert overlap check dropped (only 4 usable cases). (D) a capital is excluded if under alert > 80% of time in the last 3 training months or in the adaptation week (rule fixed before seeing the numbers) | R1 / R2 (raion only) | Step 8a: no non-air-raid alerts at capitals in alerts.in.ua; Kharkiv/Zaporizhzhia have city-level alerts a raion-only rule would miss |
| D27 | 2026-10-11 | **Step 8b outcomes (finalized after 3 self-critiques):** (1) gap rule: join gaps ≤ 2 min in both sources (0/5 min reported as sensitivity); (2) capital rule R1 confirmed (city/hromada add < 1.1% of alert time); (3) training start per capital from Check B: after-switch only for Vinnytsia, Zhytomyr, Zaporizhzhia, Bila Tserkva, Lviv, Odesa, Ternopil, Kherson, Khmelnytskyi, Cherkasy; full history for the rest; parameters shared across capitals; (4) adaptation week estimates one shared post-reform level multiplier + per-capital multipliers shrunk toward it; the "no adaptation" variant is also reported; (5) **excluded by the 80% rule: Kramatorsk, Zaporizhzhia, Sumy** → 21 capitals modeled; the 3 shown on the map as "near-permanent alert (>80% of time)", not gray; (6) **no design decision may use data after 2026-09-18 08:22 UTC; the final test window is touched once** | — | Step 8b output |

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
**Step 6 (script `scripts/step06_diagnose_sources.py`, output `logs/step06_output.txt`):**
- Official duplicates: every duplicated row appears **exactly twice**; until 2025Q4 exactly half of each quarter's rows are duplicates (e.g. 2022Q1 3,312 rows / 1,656 duplicates); 2026Q1 partly (6,954 of 33,091); 2026Q2–Q3 none → the official file wrote every row twice until early 2026, then fixed. **Correction to Step 5: all official raw counts before 2026 are doubled** (the raion shift itself still holds).
- Volunteer self-overlap: **0.0 in every quarter and region** → the volunteer file is a clean on/off series; the hypothesis "it logs each district alert as a new oblast alert" is **rejected**.
- Volunteer starts vs official U (any part of oblast): "inside an already running official alert" grows 0.040 (2022Q2) → 0.661 (2026Q3); "outside" stays 0.06–0.11 after 2023; inside episodes > 24 h: 0.15–0.26 of starts since 2024Q3 (Dnipropetrovska 0.601, Kharkivska 0.365).
- Official O (oblast-level rows only) collapses: from 2025Q4, 0.87–0.99 of volunteer starts are "outside" O.
- Time under alert (pooled): volunteer stable 0.086 (2022Q2) → 0.109 (2026Q3); official U 0.092 → **0.306**. P(U | volunteer) 0.90–0.98 until 2025Q3, then 0.855 / 0.903 / 0.820 / **0.772**; P(volunteer | U) 0.912 → **0.276**. → **Volunteer alert time is almost a subset of official U; U covers ~3× more time by 2026.**
- Dnipropetrovska: share of time under alert U 0.762 vs volunteer 0.197. Under U, frontline oblasts are almost always "red".
- Official alerts > 24 h (deduplicated): 897; by year 2022: 3, 2024: 107, 2025: 272, 2026: 515; mostly Donetska (392) and Dnipropetrovska (386, Nikopolskyi raion hromadas). Longest: Lypetska hromada 14,497.3 h (from 2024-05-12), Vovchanska hromada 10,525.4 h (from 2024-05-20).
- **West exception:** Volynska and Lvivska — volunteer marks *more* alert time than official (Volyn 0.058 vs 0.026), P(U | volunteer) only 0.384 / 0.543; milder in Chernivetska 0.746, Ternopilska 0.795, Zakarpatska 0.796. Cause unknown (Q15).
**Step 7 preparation (read-only: devs.alerts.in.ua, alerts.in.ua/terms-of-usage, token request form, UID sheet):**
- alerts.in.ua oblast UIDs (docs): Khmelnytska 3, Vinnytska 4, Rivnenska 5, Volynska 8, Dnipropetrovska 9, Zhytomyrska 10, Zakarpatska 11, Zaporizka 12, Ivano-Frankivska 13, Kyivska 14, Kirovohradska 15, Luhanska 16, Mykolaivska 17, Odeska 18, Poltavska 19, Sumska 20, Ternopilska 21, Kharkivska 22, Khersonska 23, Cherkaska 24, Chernihivska 25, Chernivetska 26, Lvivska 27, Donetska 28, Crimea 29, Sevastopol 30, Kyiv City 31.
- UID sheet: 1,622 UIDs = 25 oblasts + 2 special-status cities + 126 raions + 1,469 hromadas.
- History endpoint fields (docs example): id, location_title, location_title_en, location_type, started_at, finished_at, updated_at, alert_type, location_uid, location_oblast, location_oblast_uid, location_raion, notes. `alert_level` appears only in the active-alerts model, not in the history example. Whether oblast history includes raion/hromada alerts: **not stated**. Only period documented: `month_ago`; separate limit 2 requests/min.
- Docs: 401 = token missing/wrong/revoked/expired; 403 = IP blocked or not available in country; 429 = too many requests; systematic violations or anomalous daily volume → IP/token blocked; don't use for critical infrastructure; data "at your own risk", delays possible.
- **Terms of use v1.4 (last updated and effective 2025-07-19)**: scraping / private endpoints forbidden (§8.1–8.2); **reproducing or modifying materials without electronic consent prohibited (§9.2)** → raw API data must not be published; mass caching/aggregation that emulates the service needs consent (§7.4); attribution required for map/embeds/screenshots (§7.1); data "as is" without accuracy guarantees (§10); research use not mentioned; new versions apply automatically (§14).

**Step 7a (script `scripts/step07a_snapshot_alerts_in_ua.py`, output `logs/step07a_output.txt`; raw snapshot local only: `data/raw/alerts_in_ua/20261011T082201Z/`):**
- 24/24 units HTTP 200, **16,527 records**, 13.5 min; response shape `{"alerts": [...]}`.
- Window: most units 2026-09-11 → 2026-10-11 ~08:00 UTC (= one month before the request at 08:22 UTC). Exceptions: Kharkivska earliest start **2024-05-20 09:31:37** (4 s from the official file's Vovchanska hromada long alert start 09:31:41 — likely the same alert, unverified), Dnipropetrovska earliest 2026-08-29 → `month_ago` apparently includes alerts that started earlier but overlap the window.
- Fields: alert_level, alert_type, country, finished_at, id, location_oblast, location_oblast_uid, (location_raion for Dnipropetrovska), location_title, location_title_en, location_type, location_uid, notes, started_at, updated_at. **`alert_level` exists in history** (the docs example omits it); its values not yet inspected.
- location_type: almost all `raion`; frontline oblasts also `hromada` and `city` (Dnipropetrovska 397/393, Zaporizka 167/167, Kirovohradska 218/218, Kharkivska 293/289); **Kyiv City = `oblast`** (217 records, 1:1 comparable with volunteer); Mykolaivska 1 `oblast` record.
- alert_type: mostly `air_raid`; Dnipropetrovska 280 `artillery_shelling`; Kharkivska 2 `artillery_shelling` + 1 `urban_fights`.
- finished_at missing (ongoing at snapshot): 0–11 per unit (censored).
- **Zakarpatska: only 18 records, all 2026-09-15 → 2026-09-17** (no alerts 17 Sep – 11 Oct: real or gap? → check vs volunteer).
- Test-period gap: 2026-09-07 → 2026-09-11 not covered by this source (needs source A).

**Step 7b (script `scripts/step07b_compare_reference.py`, output `logs/step07b_output.txt`; window 2026-09-11 08:22 → 2026-10-10 04:47 UTC; volunteer 4,380 alerts; alerts.in.ua 16,244 air_raid records, 283 other types excluded):**
- alert_level: yellow 10,246 / red 6,281. Weekly red share (air_raid): 0.487 → 0.464 → 0.344 → 0.333 → 0.299. Red dominates at the front (Dnipropetrovska 0.848, Donetska 0.723, Zaporizka 0.597), yellow in the rear (Rivnenska 0.949, Kyiv City 0.926 yellow). Duration (finished air_raid): red median 55.4 / 90% 343.8 min; yellow median 42.0 / 90% 161.2 min.
- **Kyiv City 1:1 (both oblast-level):** volunteer 134 alerts vs alerts.in.ua 217; **time under alert 0.098 vs 0.320**; Jaccard 0.212; P(aic | vol) 0.743, P(vol | aic) 0.229; matched within 5 min: 55/134; matched timing identical (start diff median 0.30 min, end diff median 0.28 min); 79 volunteer alerts unmatched (28 naive).
- All units, alerts.in.ua rolled up as "any part": mean time_vol 0.128 vs time_aic 0.431; P(aic | vol) 0.820, P(vol | aic) 0.311. Saturation: Kharkivska 1.000, Dnipropetrovska 0.995, Donetska 0.971, Zaporizka 0.913.
- **Western excess confirmed independently (Q15):** share of volunteer alert time confirmed by alerts.in.ua: Lvivska 0.098 (113 volunteer alerts vs 17 episodes; 0.885 of volunteer starts where alerts.in.ua shows no alert), Ternopilska 0.150, Volynska 0.214.
- Zakarpatska: volunteer 0 alerts in window, alerts.in.ua 3 episodes (15–17 Sep) → genuinely quiet, not a gap.
- **Naive share in window 0.075 overall but 0.002 (Kharkivska) → 0.471 (Ternopilska)**; Donetska 0.400, Khersonska 0.370, Khmelnytska 0.368, Kyiv City 0.284 → missing endings are not random (R4).
- Normal volunteer alerts: own duration median 16.5 min vs time to end of the matching alerts.in.ua episode 53.9 min; |diff| ≤ 2 min for only 0.358.
- Old alerts.in.ua starts (56 before the window): Vovchanska hromada 2024-05-20 09:31:37 / 09:31:41 = `artillery_shelling` / `urban_fights`, still running (~20,975 h) — **not air raid**; Kharkivska "any part" saturation comes from an `air_raid` alert in Lypetska hromada running since 2026-03-10 (~5,155 h).

**Tie-breaker attempt (2026-10-11):** Слово і діло article (2026-10-01) on September yellow/red alerts **uses alerts.in.ua data** → not independent; gives alert counts (Kyiv City: 130 yellow, 32 red; all Ukraine 6,728 yellow, 5,998 red, 12,726 total; Lviv 11 yellow) but no hours in text and no method. Usable only as a processing sanity check of our snapshot, not to decide which source is true.

**Telegram terms (read 2026-10-11):** Content Licensing terms prohibit "the scraping, indexing, harvesting, aggregation or use of data" from Telegram "to train, fine-tune, validate or otherwise engage in the development, enhancement, benchmarking" of "artificial intelligence, machine learning models and similar technologies"; no research or public-channel exception; API terms §1.5 point to the same rule; general ToS do not mention the t.me/s/ preview. Open Terms Archive dates the change 2026-02-03 (not verified on Telegram's page). Both Vadimkin files were built by reading Telegram via the API → grey zone (not a lawyer; unclear whether statistical models count).

**Step 8a (script `scripts/step08a_find_capitals.py`, output `logs/step08a_output.txt`; official deduplicated rows: 185,430):**
- All 24 capitals found in both sources. Naming: official `<Name>skyi raion` and `m. <City> ta <Name>ska terytorialna hromada`; alerts.in.ua `<City> raion`, plus `<City> city` and `<Name>ska terytorialna hromada` for Dnipro (3+3), Zaporizhzhia (167+167), Kharkiv (289+289) (city and hromada counts identical → likely the same alerts listed twice).
- **alerts.in.ua: 0 non-air-raid alerts at any capital location** (all 24).
- Official capital hromada rows: Kharkiv 3,580 (1 over 24 h), Zaporizhzhia 2,317 (2 over 24 h); most other capital hromada rows only in 2022.
- **Official oblast-wide alerts stop around Oct–Nov 2025** (last starts 2025-10-19 … 2025-11-04; Khmelnytska 2025-08-20). Capital-raion rows start in 2022 for many capitals, but only in 2025 for Dnipro (05-01), Bila Tserkva (07-24), Khmelnytskyi (07-26), Poltava (08-01), Rivne (08-20), Kherson (08-30), Lviv (09-03), and Lutsk / Uzhhorod / Ivano-Frankivsk / Ternopil (11-06).
- Kramatorskyi raion: 1,119 official rows, **45 over 24 h**; alerts.in.ua Kramatorsk raion red share 0.722.
- Mykolaivska: 1 oblast-level alerts.in.ua record ("Mykolaiv Oblast", 2026-10-02). Kyiv City: 2,267 official oblast-wide rows (last 2026-09-05), 217 alerts.in.ua oblast records.
- 56 alerts.in.ua alerts that started before the window: only 4 started before the official file ended; 2 matched (artillery_shelling, urban_fights) and the official file shows them ended while alerts.in.ua shows them still running; the long air-raid one is not in the official file.

**Step 8b (script `scripts/step08b_build_capital_episodes.py`, output `logs/step08b_output.txt`, monthly shares `logs/step08b_monthly_share_official.csv`; episodes table local only: `data/processed/capital_episodes.csv`):**
- 24 capitals, episodes in both sources, no overlaps after merging; **90,066 episodes**; censored at snapshot 0–1 per capital. Merging matters for Kharkiv (12,905 official rows → 9,279 episodes; 905 alerts.in.ua records → 311) and Zaporizhzhia (10,510 → 8,194; 509 → 163).
- Gaps between episodes: official ≤1 min 0.004, ≤2 min 0.008, ≤5 min 0.024 (median 178.5 min); alerts.in.ua ≤1 min 0.011, ≤2 min 0.026, ≤5 min 0.055 (median 57.3 min).
- Check A: alert time only from city/hromada records ≤ 0.0068 (official, Mykolaiv) and ≤ 0.0100 (alerts.in.ua, Kharkiv); official capital-hromada alerts > 24 h: Zaporizhzhia 2, Kharkiv 1.
- Check B: switch dates 2025-10-19 … 2025-11-04 (Khmelnytskyi 2025-08-20); ratios after/before from 0.343 (Bila Tserkva) to 2.480 (Zaporizhzhia).
- Check C: volunteer-bridge Jaccard before → after: Kharkiv 0.256 → 0.284, Bila Tserkva 0.226 → 0.206, Dnipro 0.261 → 0.247, Mykolaiv 0.459 → 0.423 (stable → supports same capital definition in both sources). Time under alert, official last 28 days → alerts.in.ua window: Kyiv 0.147 → 0.322, Zhytomyr 0.034 → 0.233, Bila Tserkva 0.083 → 0.348, Rivne 0.009 → 0.120, Chernihiv 0.460 → 0.785; front stable (Kramatorsk 0.969 → 0.971, Zaporizhzhia 0.916 → 0.912). Official Kyiv monthly share already 0.107 (Aug) → 0.210 (1–7 Sep) before the source change → the jump is at least partly real. Kyiv 7.233 episodes/day after vs Слово і діло 130+32 alerts to 30 Sep (processing sanity check, not independent).
- Check D (share last 92 training days / adaptation week): Kramatorsk 0.958 / 0.945, Zaporizhzhia 0.905 / 0.881, Sumy 0.816 / 0.755 → excluded; Kharkiv 0.770 / 0.657 kept. Share = minutes under alert / minutes in window on a 1-minute grid (minutes rounded outward: overstatement ≤ 2 min per episode, i.e. ≤ ~1–1.4 percentage points for Kyiv/Kharkiv; no decision flips).

- **Conclusion:** the disagreement is mainly a **definition difference** ("any part of the oblast", inflated by long local frontline alerts and the raion shift) rather than volunteer timing errors — except the unexplained western excess.

- **Interpretation (Step 5, superseded by Step 6 above):** low agreement is mostly a definition difference (volunteer may log a new region alert per local alert; the union roll-up merges them, and long official alerts swallow many volunteer alerts), not timing errors. → **Splicing sources (D17 option 1) is not safe.**

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
| H9 | Some long official "alerts" in frontline hromadas are non-air-raid threats (artillery shelling, urban fights), which inflate "any part" alert time | Compare official long alerts with alerts.in.ua `alert_type` for the same location and start | partial evidence | Step 7b: Vovchanska hromada (official: hromada alert from 2024-05-20 09:31:41, 10,525 h) is `artillery_shelling` / `urban_fights` in alerts.in.ua. The official file has no type column, so training labels may include non-air-raid alerts (risk for D23 consistency) |
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
| M4 | Step 6 / D18 | Proposed alerts.in.ua (source B) as "a free token you request" without saying the token could be refused or delayed, and missed that its 1-month history makes every day of delay lose test-period data | User: "You have never mentioned this risk and it is not okay" | Full risk table with fallbacks given; permanent rule: every option lists its dependencies and what happens if each fails |
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
- **Q14. Is the volunteer file accurate after 2026-09-07?** Fresh ≠ accurate. Known: precise timing when matched; naive share rising after the reform; its recording rules unknown; no independent check yet for the test period. Checks: (1) Step 6 diagnostics (pre-Sep), (2) spot-check vs @air_alert_ua public preview, (3) alerts.in.ua `month_ago` history, (4) rough comparison with public monthly statistics. → D18.
  **Step 7b answer:** after the reform the volunteer file disagrees strongly with alerts.in.ua even 1:1 (Kyiv City 0.098 vs 0.320 time under alert) and has region-dependent missing endings → not used for test labels (D23). Independent tie-breaker not available (article uses alerts.in.ua; Telegram ruled out by its terms).
  **To do before Step 7:** check Telegram's terms of use for reading the public web preview; user requests an alerts.in.ua token at https://alerts.in.ua/api-request.
- **Q15. Why does the volunteer file mark more alert time than the official one in Volynska and Lvivska** (and mildly in Chernivetska, Ternopilska, Zakarpatska)? Extra alerts or errors? Check against reference labels (D18).
  **Step 7b answer:** confirmed by alerts.in.ua (Lvivska 0.098, Ternopilska 0.150, Volynska 0.214 of volunteer alert time confirmed). Cause still unknown; volunteer file no longer used for labels (D23).
- **Q16. alerts.in.ua token: dependencies and fallbacks (M4).** Received 2026-10-11; activation requires the user's reply (sent 2026-10-11); usage implies accepting their terms (to be read). ⚠ **The token appeared in this conversation (pasted + screenshot), and the conversation log is a deliverable** → redact it from every published file (log, raw .jsonl) and request a replacement token before publishing if live mode needs one. Token stored only in `.env` (gitignored). Every day of delay loses one day of test-period history (1-month window) → snapshot immediately when active. If refused/never active: source A only; if both A and B fail: manual spot-check, rough monthly statistics, Telegram API (Q13), or report R1 as unmitigated.
- **Q17. Consent email to alerts.in.ua** asking permission to publish derived research results (terms §9.2). User: **later**. Depends on their answer; if no/none: publish code + own analysis + volunteer-based results, describe the alerts.in.ua comparison in words without their numbers. Also re-check: a future live mode showing their live status may count as "emulating the service" (§7.4).
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
| 6 | Diagnose why official and volunteer disagree | `scripts/step06_diagnose_sources.py` → `logs/step06_output.txt` | Official duplicates = every row written twice until early 2026; volunteer is clean on/off (0 overlaps); volunteer alert time ≈ subset of official "any part" (U), which covers ~3× more time by 2026 due to long local frontline alerts + raion shift → definition difference, not timing errors; unexplained western excess (Q15). alerts.in.ua token received (Q16, M4). Provisional target: volunteer definition (D21) |
| 7a | One-time alerts.in.ua `month_ago` snapshot (reference labels, D18/D22) | `scripts/step07a_snapshot_alerts_in_ua.py` → `logs/step07a_output.txt` | 24/24 OK, 16,527 records, window ~2026-09-11 → 10-11; mostly raion-level; Kyiv City oblast-level (clean 1:1 test); `alert_level` present; Zakarpatska suspiciously sparse; 7–11 Sep not covered |
| 7b | Compare volunteer vs alerts.in.ua in the test period | `scripts/step07b_compare_reference.py` → `logs/step07b_output.txt` | yellow/red levels recorded, red share falling weekly; Kyiv City 1:1: volunteer 0.098 vs alerts.in.ua 0.320 time under alert; western excess confirmed; naive share non-random; "any part" saturated at the front; Vovchansk = artillery/urban fights (H9) |
| 7c | Tie-breaker attempt + Telegram terms check | none (read-only web) | Article not independent (uses alerts.in.ua); Telegram terms forbid using its data for ML → decision D23: train official (to 7 Sep), test alerts.in.ua (11 Sep → 10 Oct), target = oblast capital |
| 8a | Find each capital's names in both sources | `scripts/step08a_find_capitals.py` → `logs/step08a_output.txt` | All 24 capitals found; no non-air-raid alerts at capitals; official oblast-wide alerts stop Oct–Nov 2025; mapping `config/capitals.csv`; solutions A–D finalized (D26) |
| 8b | Build capital episodes and run checks A–D | `scripts/step08b_build_capital_episodes.py` → `logs/step08b_output.txt` | 90,066 episodes; R1 confirmed; 10 capitals train after switch; post-reform level jump in rear capitals (partly real); Kramatorsk, Zaporizhzhia, Sumy excluded → 21 capitals; D27 |

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
8. **Telegram AI/ML terms (2026-10-11):** using Telegram data to train/validate/benchmark ML models is prohibited; our long history (Vadimkin files) was built from Telegram → grey zone, documented in README. **User decision D25: text (Telegram or news) stays in the plan; Telegram risk accepted knowingly if Telegram is chosen later.**
9. **Train/test label consistency (D23) cannot be checked directly:** the official file (to 2026-09-07) and the alerts.in.ua snapshot (from 2026-09-11) do not overlap in time, except alerts that started before the window; the official file has no alert-type column (H9).

### Risks from uncertain volunteer-data accuracy (Q14) and how they are handled

| # | Risk | Severity | Solution | Remaining risk |
|---|---|---|---|---|
| R1 | Fake evaluation: test labels may not reflect reality | High | Reference labels from 2 independent sources; report metrics vs both; gap = label uncertainty (D18) | Low (only if both sources share an error) |
| R2 | Fitting the bug: adapting the model to a data artifact (e.g. post-reform parser gaps) | High | Data-quality gate (D19); weekly quality metrics; inspect which messages lose their ending | Low |
| R3 | Wrong math conclusions: recording-style bursts fake Hawkes excitation (H4) and distort cross-region effects (H3). **Step 6: overlapping-record bursts ruled out (0 overlaps); sequential re-alerts still to check via the gap rule** | High → Medium | On/off states with merged periods and tested gap rule (D20); fit on volunteer vs official states on the overlap and compare α; per-region baseline μ_r; refit without weak-agreement regions; time-rescaling test | Medium (robustness is evidence, not proof) |
| R4 | Biased durations from invented / missing endings. **Step 7b: confirmed non-random (naive share 0.002–0.471 by region)** | Medium | Interval censoring (D20); repair from official matches where possible; test whether missing endings are random; sensitivity analysis (exclude / interval / fake-30) | Low–medium |
| R5 | Single source dependency | Medium | Pinned snapshot (I1); standard episodes table so sources are replaceable (D20); reference collection doubles as backup; alerts.in.ua live feed | Low |
| R6 | Credibility with organizers | Low if documented | README "Data quality" section with numbers; label-uncertainty band; committed logs; mistakes log | None if honest |
| R7 | Misleading map users | Low | Disclaimer; "data as of" timestamp; data-quality marker per region; probabilities, not certainties | Low |

**Planned order:** Step 6 run diagnostics (R3 evidence) → Step 7 reference labels for the test period (R1, R2, R4) → Step 8 standard episodes table: states, gap rule, interval censoring, repairs (R3–R5) → exploration & hypotheses → models with robustness checks → README data quality, disclaimer, map markers (R6, R7).
