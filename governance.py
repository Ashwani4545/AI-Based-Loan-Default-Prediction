import json
from datetime import datetime
from pathlib import Path

AUDIT_LOG_PATH = "logs/audit_log.json"

def log_decision(record):
    Path(AUDIT_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(AUDIT_LOG_PATH) as f:
            logs = json.load(f)
    except Exception:
        logs = []

    logs.insert(0, record)

    with open(AUDIT_LOG_PATH, "w") as f:
        json.dump(logs[:1000], f, indent=2)
