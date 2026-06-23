from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .file_tools import WORKSPACE_ROOT, resolve_workspace_path
from .registry import RiskLevel, register_tool, register_tool_call_risk


CommandPolicy = Literal["allow", "ask", "deny"]

MAX_OUTPUT_CHARS = 4000
COMMAND_TIMEOUT_SECONDS = 20
PYTHON_EXECUTABLES = {"python", "python3", "py"}
SHELL_EXECUTABLES = {"powershell", "powershell.exe", "pwsh", "pwsh.exe", "cmd", "cmd.exe", "bash", "sh", "zsh", "fish"}
DENIED_EXECUTABLES = SHELL_EXECUTABLES | {"rm", "del", "erase", "remove-item", "curl", "wget", "npx"}
SENSITIVE_PATH_NAMES = {".env", "config.yaml", "config.yml", ".npmrc", ".pypirc"}
RG_DENIED_FLAGS = {"--no-ignore", "--hidden", "-uu", "-uuu"}
GIT_AUTO_SUBCOMMANDS = {"status", "diff", "log", "show"}
GIT_DENIED_SUBCOMMANDS = {"reset", "clean"}
NPM_ASK_COMMANDS = {"test", "build", "lint"}
PIP_DENIED_SUBCOMMANDS = {"install", "uninstall"}
PIP_ASK_SUBCOMMANDS = {"freeze", "list", "show", "check"}


def validate_python_script_path(path: str) -> None:
    target = resolve_workspace_path(path)
    if not target.exists():
        raise FileNotFoundError(f"脚本不存在：{path}")
    if not target.is_file():
        raise ValueError(f"不是文件：{path}")
    if target.suffix.lower() != ".py":
        raise ValueError("只允许执行 .py 文件")


def executable_name(value: str) -> str:
    name = Path(value).name.lower()
    return name.removesuffix(".exe")


def contains_sensitive_path(command: list[str]) -> bool:
    for part in command[1:]:
        lowered = part.replace("\\", "/").lower()
        pieces = [piece for piece in lowered.split("/") if piece]
        if any(piece in SENSITIVE_PATH_NAMES for piece in pieces):
            return True
        if lowered in SENSITIVE_PATH_NAMES:
            return True
    return False


def contains_shell_operator(command: list[str]) -> bool:
    operators = {"|", "&&", "||", ";", ">", ">>", "<", "<<", "`"}
    return any(part in operators or "$(" in part for part in command)


def validate_no_workspace_escape(command: list[str]) -> None:
    for part in command[1:]:
        if part.startswith("-") or not looks_like_path(part):
            continue
        resolve_workspace_path(part)


def validate_executable_reference(command: list[str]) -> None:
    if looks_like_path(command[0]):
        resolve_workspace_path(command[0])


def looks_like_path(value: str) -> bool:
    return (
        "/" in value
        or "\\" in value
        or value.startswith(".")
        or value.endswith((".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".log"))
    )


def classify_python_command(command: list[str]) -> CommandPolicy:
    if len(command) == 2:
        validate_python_script_path(command[1])
        return "ask"

    if len(command) >= 2 and command[1] in {"-c", "-m"}:
        if command[1] == "-c":
            return "deny"
        if len(command) < 3:
            return "deny"
        module = command[2]
        if module == "py_compile" and len(command) == 4:
            validate_python_script_path(command[3])
            return "allow"
        if module == "unittest":
            validate_no_workspace_escape(command)
            return "allow"
        if module == "pytest":
            validate_no_workspace_escape(command)
            return "ask"
        if module == "uvicorn":
            return "ask"
        if module == "pip":
            return classify_pip_command(command[3:])
    return "deny"


def classify_pip_command(args: list[str]) -> CommandPolicy:
    if not args:
        return "deny"
    subcommand = args[0].lower()
    if subcommand in PIP_DENIED_SUBCOMMANDS:
        return "deny"
    if subcommand in PIP_ASK_SUBCOMMANDS:
        return "ask"
    return "deny"


def classify_git_command(command: list[str]) -> CommandPolicy:
    if len(command) < 2:
        return "deny"
    subcommand = command[1].lower()
    if subcommand in GIT_AUTO_SUBCOMMANDS:
        validate_no_workspace_escape(command)
        return "allow"
    if subcommand in GIT_DENIED_SUBCOMMANDS:
        return "deny"
    if subcommand == "push" and any(part in {"--force", "-f"} or part.startswith("--force") for part in command[2:]):
        return "deny"
    return "ask"


def classify_rg_command(command: list[str]) -> CommandPolicy:
    if any(part in RG_DENIED_FLAGS for part in command[1:]):
        return "deny"
    validate_no_workspace_escape(command)
    return "allow"


def classify_npm_command(command: list[str]) -> CommandPolicy:
    if len(command) < 2:
        return "deny"
    subcommand = command[1].lower()
    if subcommand in {"install", "i", "add"}:
        return "deny"
    if subcommand in NPM_ASK_COMMANDS:
        return "ask"
    if subcommand == "run" and len(command) >= 3 and command[2].lower() in NPM_ASK_COMMANDS:
        return "ask"
    return "deny"


def classify_command(command: list[str]) -> CommandPolicy:
    if not isinstance(command, list) or not command:
        raise ValueError("command 必须是非空列表")
    if not all(isinstance(part, str) and part.strip() for part in command):
        raise ValueError("command 中的每一项都必须是非空字符串")
    validate_executable_reference(command)
    executable = executable_name(command[0])
    if executable in DENIED_EXECUTABLES:
        return "deny"
    if contains_shell_operator(command):
        return "deny"
    if contains_sensitive_path(command):
        return "deny"
    if executable in PYTHON_EXECUTABLES:
        return classify_python_command(command)
    if executable == "git":
        return classify_git_command(command)
    if executable == "rg":
        return classify_rg_command(command)
    if executable == "pytest":
        validate_no_workspace_escape(command)
        return "ask"
    if executable == "uvicorn":
        return "ask"
    if executable == "pip":
        return classify_pip_command(command[1:])
    if executable == "npm":
        return classify_npm_command(command)
    validate_no_workspace_escape(command)
    return "ask"


def command_tool_risk(args: dict[str, Any]) -> RiskLevel:
    command = args.get("command")
    return "high" if classify_command(command) == "ask" else "low"


def validate_command(command: list[str]) -> None:
    policy = classify_command(command)
    if policy == "deny":
        executable = command[0] if isinstance(command, list) and command else command
        raise ValueError(f"不允许执行该命令：{executable}")


class RunCommandInput(BaseModel):
    command: list[str] = Field(
        description=(
            "命令数组，不经过 shell，例如 ['python', '-m', 'unittest', 'discover', '-s', 'tests']。"
            "只读/验证命令会自动执行；其他开发命令通常会请求用户审批；shell 包壳、安装命令、"
            "破坏性 git 和敏感配置路径会被拒绝。"
        )
    )


@tool(args_schema=RunCommandInput)
def run_command(command: list[str]) -> dict[str, Any]:
    """在工作区根目录运行开发命令，按三层策略自动执行、请求审批或拒绝。"""
    validate_command(command)
    policy = classify_command(command)
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
        "policy": policy,
        "exit_code": result.returncode,
        "stdout": result.stdout[:MAX_OUTPUT_CHARS],
        "stderr": result.stderr[:MAX_OUTPUT_CHARS],
        "stdout_truncated": len(result.stdout) > MAX_OUTPUT_CHARS,
        "stderr_truncated": len(result.stderr) > MAX_OUTPUT_CHARS,
    }


register_tool(run_command, "low")
register_tool_call_risk("run_command", command_tool_risk)
