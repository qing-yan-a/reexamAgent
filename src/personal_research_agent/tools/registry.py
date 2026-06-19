from __future__ import annotations

from typing import Literal

from langchain_core.tools import BaseTool


RiskLevel = Literal["low", "medium", "high"]

_TOOLS: list[BaseTool] = []
_RISK_BY_NAME: dict[str, RiskLevel] = {}


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


def requires_approval(tool_name: str) -> bool:
    return get_tool_risk(tool_name) in {"medium", "high"}


def all_tool_risks() -> dict[str, RiskLevel]:
    return dict(_RISK_BY_NAME)
