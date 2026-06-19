from __future__ import annotations

import subprocess
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .file_tools import WORKSPACE_ROOT, resolve_workspace_path
from .registry import register_tool


ALLOWED_COMMANDS = {"python", "py"}
MAX_OUTPUT_CHARS = 4000
COMMAND_TIMEOUT_SECONDS = 20


def validate_python_script_path(path: str) -> None:
    target = resolve_workspace_path(path)
    if not target.exists():
        raise FileNotFoundError(f"脚本不存在：{path}")
    if not target.is_file():
        raise ValueError(f"不是文件：{path}")
    if target.suffix.lower() != ".py":
        raise ValueError("只允许执行 .py 文件")


def validate_command(command: list[str]) -> None:
    if not isinstance(command, list) or not command:
        raise ValueError("command 必须是非空列表")
    if not all(isinstance(part, str) and part.strip() for part in command):
        raise ValueError("command 中的每一项都必须是非空字符串")
    executable = command[0]
    if executable not in ALLOWED_COMMANDS:
        raise ValueError(f"不允许执行该命令：{executable}")
    if len(command) == 2:
        validate_python_script_path(command[1])
        return
    if len(command) == 4 and command[1] == "-m" and command[2] == "py_compile":
        validate_python_script_path(command[3])
        return
    raise ValueError("只允许运行 python 文件，或 python -m py_compile 文件")


class RunCommandInput(BaseModel):
    command: list[str] = Field(description="命令数组，例如 ['python', 'test.py']。")


@tool(args_schema=RunCommandInput)
def run_command(command: list[str]) -> dict[str, Any]:
    """在工作区根目录运行白名单命令，用于验证代码。"""
    validate_command(command)
    result = subprocess.run(
        command,
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=COMMAND_TIMEOUT_SECONDS,
        shell=False,
    )
    return {
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout[:MAX_OUTPUT_CHARS],
        "stderr": result.stderr[:MAX_OUTPUT_CHARS],
        "stdout_truncated": len(result.stdout) > MAX_OUTPUT_CHARS,
        "stderr_truncated": len(result.stderr) > MAX_OUTPUT_CHARS,
    }


register_tool(run_command, "high")
