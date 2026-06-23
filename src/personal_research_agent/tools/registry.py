from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from langchain_core.tools import BaseTool


RiskLevel = Literal["low", "medium", "high"]

_TOOLS: list[BaseTool] = []
_RISK_BY_NAME: dict[str, RiskLevel] = {}
_RISK_BY_CALL: dict[str, Callable[[dict[str, Any]], RiskLevel]] = {}


def register_tool(tool: BaseTool, risk: RiskLevel) -> BaseTool:
    # LangChain 的 tool 对象只描述“怎么调用工具”。
    # 风险等级是我们自己的生产安全策略，用于 graph.py 里的 human_approval 路由。
    if risk not in {"low", "medium", "high"}:
        raise ValueError(f"Unsupported tool risk: {risk}")
    _TOOLS.append(tool)
    _RISK_BY_NAME[tool.name] = risk
    return tool


def get_registered_tools() -> list[BaseTool]:
    return list(_TOOLS)


def get_tool_risk(tool_name: str) -> RiskLevel:
    # 未登记工具默认视为 high，宁可多审批，也不让未知工具静默执行。
    return _RISK_BY_NAME.get(tool_name, "high")


def register_tool_call_risk(tool_name: str, risk_fn: Callable[[dict[str, Any]], RiskLevel]) -> None:
    # 少数工具的风险取决于参数，例如 run_command。
    _RISK_BY_CALL[tool_name] = risk_fn


def get_tool_call_risk(tool_name: str, args: dict[str, Any] | None = None) -> RiskLevel:
    risk_fn = _RISK_BY_CALL.get(tool_name)
    if risk_fn is None:
        return get_tool_risk(tool_name)
    try:
        return risk_fn(args or {})
    except Exception:
        return "high"


def requires_approval(tool_name: str) -> bool:
    return get_tool_risk(tool_name) in {"medium", "high"}


def tool_call_requires_approval(tool_name: str, args: dict[str, Any] | None = None) -> bool:
    return get_tool_call_risk(tool_name, args) in {"medium", "high"}


def all_tool_risks() -> dict[str, RiskLevel]:
    return dict(_RISK_BY_NAME)
