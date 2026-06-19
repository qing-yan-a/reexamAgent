from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from personal_research_agent.config import PROJECT_ROOT

from .registry import register_tool


WORKSPACE_ROOT = PROJECT_ROOT
BLOCKED_NAMES = {".env", ".venv", "__pycache__", ".git", ".idea"}
DEFAULT_IGNORED_NAMES = {"sessions"}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".csv",
    ".log",
}
DEFAULT_MAX_CHARS = 4000
DEFAULT_MAX_RESULTS = 20
MAX_FILE_SIZE_BYTES = 500_000
MAX_WRITE_CHARS = 20_000


def resolve_workspace_path(path: str) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path 必须是非空字符串")
    raw_path = Path(path)
    if raw_path.is_absolute():
        raise ValueError("不允许访问绝对路径，只能访问当前工作区内的相对路径")
    candidate = (WORKSPACE_ROOT / raw_path).resolve()
    if not candidate.is_relative_to(WORKSPACE_ROOT):
        raise ValueError("路径超出工作区，不允许访问")
    return candidate


def ensure_safe_path(path: Path) -> None:
    relative_parts = path.relative_to(WORKSPACE_ROOT).parts
    if any(part in BLOCKED_NAMES for part in relative_parts):
        raise ValueError("路径包含禁止访问的目录或文件")


def to_workspace_relative(path: Path) -> str:
    return str(path.relative_to(WORKSPACE_ROOT))


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def is_default_ignored_path(path: Path, search_root: Path) -> bool:
    relative_parts = path.relative_to(WORKSPACE_ROOT).parts
    root_parts = search_root.relative_to(WORKSPACE_ROOT).parts
    for ignored_name in DEFAULT_IGNORED_NAMES:
        if ignored_name in relative_parts:
            return ignored_name not in root_parts
    return False


def prepare_text_file_for_write(path: str) -> Path:
    target = resolve_workspace_path(path)
    ensure_safe_path(target)
    if not is_text_file(target):
        raise ValueError(f"不是允许写入的文本文件类型：{target.suffix}")
    if target.exists() and not target.is_file():
        raise ValueError(f"不是文件：{path}")
    return target


def validate_write_content(content: str) -> None:
    if not isinstance(content, str):
        raise ValueError("content 必须是字符串")
    if len(content) > MAX_WRITE_CHARS:
        raise ValueError(f"写入内容过长：{len(content)} chars")


class ListFilesInput(BaseModel):
    path: str = Field(default=".", description="工作区内的相对目录路径。根目录使用 '.'。")
    recursive: bool = Field(default=False, description="是否递归列出子目录。")


@tool(args_schema=ListFilesInput)
def list_files(path: str = ".", recursive: bool = False) -> dict[str, Any]:
    """列出工作区内指定目录下的文件和文件夹。"""
    target = resolve_workspace_path(path)
    ensure_safe_path(target)
    if not target.exists():
        raise FileNotFoundError(f"路径不存在：{path}")
    if not target.is_dir():
        raise ValueError(f"不是目录：{path}")

    iterator = target.rglob("*") if recursive else target.iterdir()
    items: list[dict[str, Any]] = []
    for item in sorted(iterator, key=lambda p: to_workspace_relative(p)):
        try:
            ensure_safe_path(item)
        except ValueError:
            continue
        if recursive and is_default_ignored_path(item, target):
            continue
        info: dict[str, Any] = {
            "name": item.name,
            "path": to_workspace_relative(item),
            "type": "directory" if item.is_dir() else "file",
        }
        if item.is_file():
            info["size"] = item.stat().st_size
        items.append(info)

    return {"path": to_workspace_relative(target), "recursive": recursive, "items": items}


class ReadFileInput(BaseModel):
    path: str = Field(description="工作区内的相对文件路径。")
    max_chars: int = Field(default=DEFAULT_MAX_CHARS, description="最大返回字符数。")


@tool(args_schema=ReadFileInput)
def read_file(path: str, max_chars: int = DEFAULT_MAX_CHARS) -> dict[str, Any]:
    """读取工作区内的文本文件内容。"""
    target = resolve_workspace_path(path)
    ensure_safe_path(target)
    if not target.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    if not target.is_file():
        raise ValueError(f"不是文件：{path}")
    if not is_text_file(target):
        raise ValueError(f"不是允许读取的文本文件类型：{target.suffix}")
    size = target.stat().st_size
    if size > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"文件过大，不允许读取：{size} bytes")
    if not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("max_chars 必须是正整数")
    content = target.read_text(encoding="utf-8", errors="replace")
    return {
        "path": to_workspace_relative(target),
        "content": content[:max_chars],
        "truncated": len(content) > max_chars,
        "size": size,
    }


class SearchFileInput(BaseModel):
    query: str = Field(description="需要搜索的关键词、函数名、类名或文本内容。")
    path: str = Field(default=".", description="工作区内的相对路径。")
    max_results: int = Field(default=DEFAULT_MAX_RESULTS, description="最大返回结果数。")


@tool(args_schema=SearchFileInput)
def search_file_content(query: str, path: str = ".", max_results: int = DEFAULT_MAX_RESULTS) -> dict[str, Any]:
    """在工作区内的文本文件内容中搜索关键词，并返回匹配行。"""
    if not isinstance(query, str) or not query:
        raise ValueError("query 必须是非空字符串")
    if not isinstance(max_results, int) or max_results <= 0:
        raise ValueError("max_results 必须是正整数")
    target = resolve_workspace_path(path)
    ensure_safe_path(target)
    if not target.exists():
        raise FileNotFoundError(f"路径不存在：{path}")

    candidates = [target] if target.is_file() else target.rglob("*")
    matches: list[dict[str, Any]] = []
    for file_path in candidates:
        if len(matches) >= max_results:
            break
        if not file_path.is_file():
            continue
        try:
            ensure_safe_path(file_path)
        except ValueError:
            continue
        if is_default_ignored_path(file_path, target):
            continue
        if not is_text_file(file_path) or file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
            continue
        for line_number, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if query in line:
                matches.append({"path": to_workspace_relative(file_path), "line": line_number, "text": line.strip()})
                if len(matches) >= max_results:
                    break

    return {
        "query": query,
        "path": to_workspace_relative(target),
        "max_results": max_results,
        "default_ignored": sorted(DEFAULT_IGNORED_NAMES),
        "matches": matches,
    }


class WriteTextFileInput(BaseModel):
    path: str = Field(description="工作区内的相对文件路径。")
    content: str = Field(description="要写入文件的文本内容。")
    overwrite: bool = Field(default=False, description="文件已存在时是否覆盖。")


@tool(args_schema=WriteTextFileInput)
def write_text_file(path: str, content: str, overwrite: bool = False) -> dict[str, Any]:
    """在工作区内创建或覆盖文本文件。"""
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite 必须是布尔值")
    target = prepare_text_file_for_write(path)
    validate_write_content(content)
    existed = target.exists()
    if existed and not overwrite:
        raise FileExistsError("文件已存在，如需覆盖请设置 overwrite=true")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"path": to_workspace_relative(target), "chars": len(content), "overwritten": existed}


class PatchTextFileInput(BaseModel):
    path: str = Field(description="工作区内的相对文件路径。")
    old_text: str = Field(description="文件中真实存在且只出现一次的完整片段。")
    new_text: str = Field(description="替换后的新文本片段。")


@tool(args_schema=PatchTextFileInput)
def patch_text_file(path: str, old_text: str, new_text: str) -> dict[str, Any]:
    """对工作区内的文本文件做局部替换。"""
    target = prepare_text_file_for_write(path)
    if not target.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    if target.stat().st_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"文件过大，不允许修改：{target.stat().st_size} bytes")
    if not old_text:
        raise ValueError("old_text 必须是非空字符串")
    validate_write_content(new_text)
    content = target.read_text(encoding="utf-8", errors="replace")
    match_count = content.count(old_text)
    if match_count == 0:
        raise ValueError("没有找到要替换的 old_text")
    if match_count > 1:
        raise ValueError(f"old_text 在文件中出现 {match_count} 次，不允许模糊替换")
    target.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
    return {"path": to_workspace_relative(target), "replacements": 1, "changed": True}


class AppendTextFileInput(BaseModel):
    path: str = Field(description="工作区内的相对文件路径。")
    content: str = Field(description="要追加到文件末尾的文本内容。")


@tool(args_schema=AppendTextFileInput)
def append_text_file(path: str, content: str) -> dict[str, Any]:
    """向工作区内的文本文件末尾追加内容。"""
    target = prepare_text_file_for_write(path)
    validate_write_content(content)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(content)
    return {"path": to_workspace_relative(target), "chars": len(content), "appended": True}


register_tool(list_files, "low")
register_tool(read_file, "low")
register_tool(search_file_content, "low")
register_tool(write_text_file, "high")
register_tool(patch_text_file, "high")
register_tool(append_text_file, "high")
