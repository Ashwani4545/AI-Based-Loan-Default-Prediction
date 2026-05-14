import json
<<<<<<< HEAD
from datetime import datetime
from pathlib import Path

AUDIT_LOG_PATH = "logs/audit_log.json"
=======
import os
from datetime import datetime
from pathlib import Path

# Use absolute path so it works regardless of CWD (same fix as Bug #7)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIT_LOG_PATH = os.path.join(BASE_DIR, "logs", "audit_log.json")
>>>>>>> 5d6f7cb80e94c9b1113dea84a0f86173cb1c2f46

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
