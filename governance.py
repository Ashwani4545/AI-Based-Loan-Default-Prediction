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
>>>>>>> 44ab82bb832d0cf735042468c185eb3463bf6a67

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
