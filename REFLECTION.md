# Reflection

I built an hour-by-hour, leakage-safe forecast of announced air-raid alerts for 110 Ukrainian raions, tested day by day, with a map and a one-hour "can I go?" check.

I worked with Claude, deciding every step myself. The stack is Python (pandas, scipy, statsmodels) on the official alert record, the alerts.in.ua API and HDX boundaries. The model is a logistic regression on a raion's recent alert level, the hour of day, and the alert running now. Every result passed a bootstrap and an automated leakage test. The map is a plain SVG page.

I changed the model on the last day, and the reason was a flaw I noticed. The 12-hour and 24-hour maps were identical, which made no sense. Together we traced it: a flat model cannot see the time of day, and our first fix, multiplying by an hour factor, broke calibration. The rule for adopting a new model was written before each run. The final model passed, but the calibration rule was ambiguous, so I had that written down before reading the result. Because the one-time test was already spent, the final model is supporting evidence, not an untouched test.

The September 2026 yellow and red alert levels made the data harder. I decided to forecast any alert and show the levels only as observed. Alert records also had gaps and an outage, which we masked. I noticed that a 7-day window made Odesa look safe, and I caught an unsupported claim that the reform caused the shift. The gain is small, and I reported it that way.
