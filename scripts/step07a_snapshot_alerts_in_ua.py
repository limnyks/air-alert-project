"""Step 7a: one-time snapshot of the alerts.in.ua "month_ago" history (reference labels, D18/D22).

- Reads the API token from .env (ALERTS_IN_UA_TOKEN=...). The token is sent only in the
  Authorization header and is never printed or written anywhere.
- Requests 24 units: 23 oblasts + Kyiv City (no Luhansk, Crimea, Sevastopol; D13).
- Waits WAIT_SECONDS between requests (API limit for history: 2 requests/minute).
- Saves every raw response unchanged to data/raw/alerts_in_ua/<UTC time>/ (local only:
  terms of use section 9.2 forbid reproducing their materials), plus manifest.json.
- Prints only derived facts: record counts, date ranges, location types, field names.

Stops immediately on 401/403 (token not active / blocked). On 429 waits and retries once.

Run from the project root:
    python -u scripts/step07a_snapshot_alerts_in_ua.py | tee logs/step07a_output.txt
"""
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
OUT_ROOT = ROOT / "data" / "raw" / "alerts_in_ua"
if "--live" in sys.argv:   # D42: fresh snapshot for the live map; kept apart from the frozen evaluation snapshot
    OUT_ROOT = ROOT / "data" / "raw" / "alerts_in_ua_live"
URL = "https://api.alerts.in.ua/v1/regions/{uid}/alerts/month_ago.json"
WAIT_SECONDS = 35        # > 30 s keeps us under 2 history requests per minute
RETRY_WAIT_SECONDS = 65  # after a 429
TIMEOUT_SECONDS = 60

# alerts.in.ua UID -> region name as written in the volunteer file (for later joins)
UNITS = {
    3: "Khmelnytska oblast", 4: "Vinnytska oblast", 5: "Rivnenska oblast", 8: "Volynska oblast",
    9: "Dnipropetrovska oblast", 10: "Zhytomyrska oblast", 11: "Zakarpatska oblast",
    12: "Zaporizka oblast", 13: "Ivano-Frankivska oblast", 14: "Kyivska oblast",
    15: "Kirovohradska oblast", 17: "Mykolaivska oblast", 18: "Odeska oblast",
    19: "Poltavska oblast", 20: "Sumska oblast", 21: "Ternopilska oblast", 22: "Kharkivska oblast",
    23: "Khersonska oblast", 24: "Cherkaska oblast", 25: "Chernihivska oblast",
    26: "Chernivetska oblast", 27: "Lvivska oblast", 28: "Donetska oblast", 31: "Kyiv City",
}


def read_token():
    if not ENV_FILE.exists():
        sys.exit(f"{ENV_FILE} not found - create it with ALERTS_IN_UA_TOKEN=...")
    for line in ENV_FILE.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "ALERTS_IN_UA_TOKEN" and value.strip():
            return value.strip().strip("'\"")
    sys.exit("ALERTS_IN_UA_TOKEN is missing or empty in .env")


def fetch(uid, token):
    """Returns (http_status, headers dict, body bytes). Never raises on HTTP errors."""
    request = urllib.request.Request(URL.format(uid=uid), headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "air-alerts-forecast (research project)",
    })
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers or {}), error.read()


def records_of(payload):
    """The docs example is malformed, so accept either {"alerts": [...]} or a bare list."""
    if isinstance(payload, dict):
        for key in ("alerts", "data", "items"):
            if isinstance(payload.get(key), list):
                return payload[key], f"dict['{key}']"
        return [], f"dict with keys {sorted(payload)}"
    if isinstance(payload, list):
        return payload, "list"
    return [], type(payload).__name__


def summarize(body):
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        print(f"    could not parse JSON: {error}")
        return 0
    records, shape = records_of(payload)
    print(f"    response shape: {shape}, records: {len(records)}")
    if not records:
        return 0
    starts = sorted(r.get("started_at") for r in records if r.get("started_at"))
    print(f"    started_at min / max: {starts[0] if starts else None} / {starts[-1] if starts else None}")
    print(f"    finished_at missing (ongoing?): {sum(1 for r in records if not r.get('finished_at'))}")
    print(f"    location_type: {dict(Counter(r.get('location_type') for r in records))}")
    print(f"    alert_type: {dict(Counter(r.get('alert_type') for r in records))}")
    print(f"    fields of first record: {sorted(records[0])}")
    return len(records)


def main():
    token = read_token()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"snapshot folder: {out_dir.relative_to(ROOT)}  (units: {len(UNITS)}, ~{len(UNITS) * WAIT_SECONDS // 60} min)",
          flush=True)

    manifest = []
    started = time.monotonic()
    for n, (uid, name) in enumerate(UNITS.items(), start=1):
        if n > 1:
            time.sleep(WAIT_SECONDS)
        requested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        status, headers, body = fetch(uid, token)
        if status == 429:
            print(f"[{n}/{len(UNITS)}] uid {uid} {name}: 429 too many requests - waiting {RETRY_WAIT_SECONDS} s, retry once",
                  flush=True)
            time.sleep(RETRY_WAIT_SECONDS)
            requested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            status, headers, body = fetch(uid, token)

        suffix = "json" if status == 200 else "error.txt"
        path = out_dir / f"{uid}.{suffix}"
        path.write_bytes(body)
        entry = {"uid": uid, "name": name, "url": URL.format(uid=uid), "requested_at_utc": requested_at,
                 "http_status": status, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                 "last_modified": headers.get("Last-Modified"), "file": path.name, "records": None}
        print(f"[{n}/{len(UNITS)}] uid {uid} {name}: HTTP {status}, {len(body):,} bytes", flush=True)

        if status == 200:
            entry["records"] = summarize(body)
        else:
            text = body[:300].decode("utf-8", "replace").replace(token, "[REDACTED]")
            print(f"    error body (first 300 chars): {text}")
        manifest.append(entry)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

        if status in (401, 403):
            print("\nSTOP: 401 = token missing/wrong/not active yet; 403 = IP blocked or not available.")
            print("Nothing else is requested. Try again later (each day of delay loses one day of history).")
            break
        if status == 429:
            print("\nSTOP: still 429 after waiting - stopping to respect the limits (terms 8.2).")
            break
        sys.stdout.flush()

    ok = sum(1 for e in manifest if e["http_status"] == 200)
    print(f"\ndone: {ok}/{len(UNITS)} units OK, {sum(e['records'] or 0 for e in manifest)} records, "
          f"{(time.monotonic() - started) / 60:.1f} min. Manifest: {(out_dir / 'manifest.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
