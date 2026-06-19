from __future__ import annotations

import re
from pathlib import Path

from .config import PROJECT_ROOT


RESEARCH_OUTPUT_ROOT = PROJECT_ROOT / "test"
MAX_OUTPUT_DIR_NAME_CHARS = 80


def sanitize_output_name(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", str(value or "").strip())
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.strip(". ")
    return cleaned[:MAX_OUTPUT_DIR_NAME_CHARS] or "未命名复试资料"


def research_output_dir_name(school: str, major: str) -> str:
    return sanitize_output_name(f"{school}{major}")


def research_output_dir(school: str, major: str, *, create: bool = True) -> Path:
    directory = (RESEARCH_OUTPUT_ROOT / research_output_dir_name(school, major)).resolve()
    root = RESEARCH_OUTPUT_ROOT.resolve()
    if not directory.is_relative_to(root):
        raise ValueError("资料输出目录超出 test 根目录")
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def to_project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def default_draft_filename(school: str, major: str) -> str:
    return f"{research_output_dir_name(school, major)}复试资料草稿.md"
