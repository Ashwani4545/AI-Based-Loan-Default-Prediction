import json
import os
from datetime import datetime
from pathlib import Path

# Use absolute path so it works regardless of CWD.
# When app.py is launched from the webapp/ subdirectory the relative path
# "logs/audit_log.json" resolves to webapp/logs/ instead of the project root.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIT_LOG_PATH = os.path.join(BASE_DIR, "logs", "audit_log.json")


def log_decision(record: dict) -> None:
    """Append a prediction record to the append-only audit log (capped at 10 000)."""
    Path(AUDIT_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(AUDIT_LOG_PATH) as f:
            logs = json.load(f)
        if not isinstance(logs, list):
            logs = []
    except Exception:
        logs = []

    # Stamp with write time if the caller didn't already include one
    entry = dict(record)
    entry.setdefault("audit_timestamp", datetime.utcnow().isoformat())

    logs.insert(0, entry)

    with open(AUDIT_LOG_PATH, "w") as f:
        json.dump(logs[:10_000], f, indent=2, default=str)
