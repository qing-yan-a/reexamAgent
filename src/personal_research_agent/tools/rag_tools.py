from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from personal_research_agent.rag_db import RAG_SOURCE_DIR, rebuild_rag_index, retrieve_hybrid_rag

from .registry import register_tool


class SearchMemoryInput(BaseModel):
    query: str = Field(description="要在本地资料索引中检索的问题或关键词。")
    top_k: int = Field(default=5, description="最多返回多少条结果。")


@tool(args_schema=SearchMemoryInput)
def search_memory(query: str, top_k: int = 5) -> dict[str, Any]:
    """检索 PostgreSQL 中的本地 RAG 资料索引。"""
    if not query.strip():
        raise ValueError("query 必须是非空字符串")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k 必须是正整数")
    try:
        results = retrieve_hybrid_rag(query, top_k=min(top_k, 20))
    except Exception as exc:
        return {
            "query": query,
            "results": [],
            "index_available": False,
            "message": f"数据库 RAG 暂不可用：{exc}",
        }
    return {
        "query": query,
        "results": results,
        "index_available": True,
        "message": f"返回 {len(results)} 条数据库 RAG 检索结果。",
    }


@tool
def rebuild_memory_index() -> dict[str, Any]:
    """扫描 test/**/*.md，重建 PostgreSQL RAG chunk 和向量索引。"""
    try:
        result = rebuild_rag_index(RAG_SOURCE_DIR)
        result["index_available"] = True
        return result
    except Exception as exc:
        return {
            "source_dir": str(RAG_SOURCE_DIR),
            "index_available": False,
            "message": f"重建数据库 RAG 索引失败：{exc}",
        }


register_tool(search_memory, "low")
register_tool(rebuild_memory_index, "medium")
