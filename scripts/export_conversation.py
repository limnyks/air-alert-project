"""Export the Claude Code conversation (session transcript .jsonl) to a readable, redacted Markdown log.

- Keeps: user messages, assistant replies, and a one-line summary of each tool call plus a short excerpt of its result.
- Removes: every image / screenshot (replaced by a note), system reminders, internal bookkeeping records.
- Redacts: the alerts.in.ua token from .env and its first 8 characters, everywhere (case-insensitive).
  Fails if either is still present in the output.

Usage (from the project root):
    python scripts/export_conversation.py <path/to/session.jsonl> logs/conversation_log.md             (readable)
    python scripts/export_conversation.py <path/to/session.jsonl> logs/conversation_log_full.md --full (everything in full)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_CHARS = 400
FULL = "--full" in sys.argv          # every tool call and its output in full


def secrets():
    lines = [l for l in (ROOT / ".env").read_text().splitlines() if l.startswith("ALERTS_IN_UA_TOKEN=")]
    tok = lines[0].split("=", 1)[1].strip().strip("'\"") if lines else ""
    if len(tok) < 8:
        sys.exit("no token in .env - cannot verify redaction")
    return [tok, tok[:8]]


def redact(text, secs):
    for s in secs:                                              # full token first, then the fragment
        text = re.sub(re.escape(s), "[REDACTED]", text, flags=re.IGNORECASE)
    return text


def clean(text):
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.S)
    text = re.sub(r"<(local-command-caveat|command-name|command-message|command-args|local-command-stdout)>.*?</\1>", "", text, flags=re.S)
    text = re.sub(r"\[Image: source: [^\]]*\]", "[screenshot omitted]", text)
    return text.strip()


def blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content or []


def main(src, dst):
    secs = secrets()
    out = ["# Conversation log - Air alerts forecast (KSE Agentic AI School, Stage 2)", "",
           "Exported from the Claude Code session transcript. Screenshots and images are omitted; the API token "
           "and its fragment are replaced by [REDACTED]. Tool calls are summarised in one line with a short "
           "excerpt of the result.", ""]
    n_user = n_asst = n_img = 0
    for line in open(src, encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") not in ("user", "assistant") or d.get("isMeta"):
            continue
        msg, ts = d.get("message", {}), (d.get("timestamp") or "")[:16].replace("T", " ")
        for b in blocks(msg.get("content")):
            t = b.get("type")
            if t == "image":
                n_img += 1
                out.append("*[screenshot omitted]*\n")
            elif t == "text":
                txt = clean(b.get("text", ""))
                if not txt:
                    continue
                if d["type"] == "user":
                    n_user += 1
                    head = "### Summary of the earlier part (context compaction)" if d.get("isCompactSummary") else f"### User ({ts} UTC)"
                    out.append(f"{head}\n\n{txt}\n")
                else:
                    n_asst += 1
                    out.append(f"**Assistant:**\n\n{txt}\n")
            elif t == "tool_use":
                inp = b.get("input", {})
                if FULL:
                    body = json.dumps(inp, ensure_ascii=False, indent=1).replace("```", "'''")
                    out.append(f"**Tool call `{b.get('name')}`:**\n\n```\n{body}\n```\n")
                else:
                    desc = inp.get("description") or inp.get("file_path") or inp.get("url") or inp.get("action") or ""
                    out.append(f"> tool `{b.get('name')}`: {str(desc)[:200]}\n")
            elif t == "tool_result":
                c = b.get("content")
                if isinstance(c, list):
                    n_img += sum(1 for x in c if x.get("type") == "image")
                    c = " ".join(x.get("text", "") for x in c if x.get("type") == "text")
                c = clean(str(c or ""))
                if c and FULL:
                    body = c.replace("```", "'''")
                    out.append(f"**Tool result:**\n\n```\n{body}\n```\n")
                elif c:
                    ex = c[:RESULT_CHARS].replace("\n", " ")
                    out.append(f">   result: {ex}{' ...' if len(c) > RESULT_CHARS else ''}\n")
    text = redact("\n".join(out), secs)
    low = text.lower()
    if any(s.lower() in low for s in secs):
        sys.exit("STOP: a secret is still present after redaction")
    Path(dst).write_text(text, encoding="utf-8")
    print(f"user messages {n_user}, assistant messages {n_asst}, images omitted {n_img}; "
          f"{len(text) / 1e6:.2f} MB -> {dst}; token and fragment: 0 occurrences")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--full"]
    main(args[0], args[1])
