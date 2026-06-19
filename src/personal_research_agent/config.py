import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = PROJECT_ROOT / "profiles"
MEMORY_DIR = PROJECT_ROOT / "memory"
SESSIONS_DIR = MEMORY_DIR / "sessions"
ARCHIVED_SESSIONS_DIR = MEMORY_DIR / "archived_sessions"
ACTIVE_SESSION_FILE = MEMORY_DIR / "active_session.json"
DEFAULT_PROFILE = "research_to_product"

load_dotenv(PROJECT_ROOT / ".env")


def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def require_env(name: str) -> str:
    value = get_env(name)
    if not value:
        raise RuntimeError(f"缺少环境变量 {name}，请在 .env 中配置。")
    return value


def get_postgres_uri() -> str:
    return require_env("POSTGRES_URI")
