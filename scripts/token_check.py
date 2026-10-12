"""Fail (exit 1) if the alerts.in.ua token from .env, or its first 8 characters, appear in any file git would
commit (tracked + untracked, not ignored). Prints only file names, never the token.
Run from the project root:  python scripts/token_check.py"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
lines = [l for l in (ROOT / ".env").read_text().splitlines() if l.startswith("ALERTS_IN_UA_TOKEN=")]
tok = lines[0].split("=", 1)[1].strip().strip("'\"") if lines else ""
if len(tok) < 8:
    sys.exit("token check: no token in .env - nothing to compare")
files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], capture_output=True, text=True, cwd=ROOT).stdout.split()
hits = [f for f in files if (ROOT / f).is_file() and (tok.encode() in (b := (ROOT / f).read_bytes()) or tok[:8].encode() in b)]
print(f"token check: {len(files)} files, containing token or fragment: {hits}")
sys.exit(1 if hits else 0)
