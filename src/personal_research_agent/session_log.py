from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT


SESSION_DIR = PROJECT_ROOT / "sessions"
SESSION_FILE = SESSION_DIR / f"session-{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.jsonl"


def append_session_log(event_type: str, data: dict[str, Any]) -> None:
    """Append one process-level runtime event to sessions/*.jsonl."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    event = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "type": event_type,
        "data": data,
    }
    with SESSION_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
