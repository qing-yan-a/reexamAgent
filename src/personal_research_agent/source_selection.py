from __future__ import annotations

from typing import Any

from personal_research_agent.session_manager import read_research_session, utc_now, write_research_session


def source_key(item: dict[str, Any]) -> str:
    """为候选来源生成稳定 key，Web 勾选和后端筛选都使用同一套规则。"""
    return "|".join(
        [
            str(item.get("query_id") or ""),
            str(item.get("source_index") or ""),
            str(item.get("url") or ""),
        ]
    )


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _normalize_index(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _normalize_url(value: Any) -> str:
    return str(value or "").strip()


def _selected_source_payload(
    source: dict[str, Any],
    reviewed: dict[str, Any],
    *,
    key: str,
    selection_method: str,
) -> dict[str, Any]:
    return {
        "source_key": key,
        "source_index": source.get("source_index"),
        "title": reviewed.get("title") or source.get("title", ""),
        "url": reviewed.get("url") or source.get("url", ""),
        "source": reviewed.get("source") or source.get("source", ""),
        "relevance": reviewed.get("relevance", "unknown"),
        "credibility_hint": reviewed.get("credibility_hint", "unknown"),
        "risk_flags": reviewed.get("risk_flags", []),
        "next_action": reviewed.get("next_action", "needs_user_check"),
        "query_id": source.get("query_id", ""),
        "query_type": source.get("query_type", ""),
        "selected_at": utc_now(),
        "selection_method": selection_method,
    }


def select_sources_from_session(
    session: dict[str, Any],
    *,
    source_keys: list[str] | None = None,
    source_indexes: list[int | str] | None = None,
    urls: list[str] | None = None,
    selection_method: str = "manual",
) -> dict[str, Any]:
    """按 source_key/source_index/url 从候选来源中筛出用户确认保留的来源。"""
    candidate_sources = _as_dict_list(session.get("candidate_sources", []))
    reviewed_sources = _as_dict_list(session.get("reviewed_sources", []))

    candidate_by_key = {source_key(source): source for source in candidate_sources}
    reviewed_by_key = {source_key(review): review for review in reviewed_sources}

    # source_index 在多轮搜索中可能重复。CLI 输入 index 时默认选择最近一次出现的同名 index。
    latest_key_by_index: dict[str, str] = {}
    latest_key_by_url: dict[str, str] = {}
    for source in reversed(candidate_sources):
        key = source_key(source)
        index_token = _normalize_index(source.get("source_index"))
        url_token = _normalize_url(source.get("url"))
        if index_token and index_token not in latest_key_by_index:
            latest_key_by_index[index_token] = key
        if url_token and url_token not in latest_key_by_url:
            latest_key_by_url[url_token] = key

    requested_keys = [str(item).strip() for item in source_keys or [] if str(item).strip()]
    requested_indexes = [_normalize_index(item) for item in source_indexes or [] if _normalize_index(item)]
    requested_urls = [_normalize_url(item) for item in urls or [] if _normalize_url(item)]

    selected_keys: list[str] = []
    missing: list[dict[str, Any]] = []

    for key in requested_keys:
        if key in candidate_by_key:
            selected_keys.append(key)
        else:
            missing.append({"type": "source_key", "value": key})

    for index in requested_indexes:
        key = latest_key_by_index.get(index)
        if key:
            selected_keys.append(key)
        else:
            missing.append({"type": "source_index", "value": index})

    for url in requested_urls:
        key = latest_key_by_url.get(url)
        if key:
            selected_keys.append(key)
        else:
            missing.append({"type": "url", "value": url})

    deduped_keys = list(dict.fromkeys(selected_keys))
    selected_sources = [
        _selected_source_payload(
            candidate_by_key[key],
            reviewed_by_key.get(key, {}),
            key=key,
            selection_method=selection_method,
        )
        for key in deduped_keys
    ]

    return {
        "selected_sources": selected_sources,
        "selected_count": len(selected_sources),
        "missing": missing,
        "requested": {
            "source_keys": requested_keys,
            "source_indexes": requested_indexes,
            "urls": requested_urls,
        },
    }


def select_sources_for_session(
    session_id: str,
    *,
    source_keys: list[str] | None = None,
    source_indexes: list[int | str] | None = None,
    urls: list[str] | None = None,
    selection_method: str = "manual",
) -> dict[str, Any]:
    """读取 session、更新 selected_sources，并把选择结果写回 research_session.json。"""
    session = read_research_session(session_id)
    result = select_sources_from_session(
        session,
        source_keys=source_keys,
        source_indexes=source_indexes,
        urls=urls,
        selection_method=selection_method,
    )
    session["selected_sources"] = result["selected_sources"]
    write_research_session(session_id, session)
    return {
        "path": f"memory/sessions/{session_id}/research_session.json",
        "session": session,
        **result,
    }
