from langchain_openai import ChatOpenAI

from .config import get_env, require_env


def build_model(model_name: str = "mimo") -> ChatOpenAI:
    if model_name == "deepseek":
        return ChatOpenAI(
            model=get_env("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            api_key=require_env("DEEPSEEK_API_KEY"),
            base_url=require_env("DEEPSEEK_BASE_URL"),
            temperature=0,
            streaming=True,
            max_tokens=6000,
            timeout=300,
            max_retries=3,
        )

    return ChatOpenAI(
        model=get_env("MIMO_MODEL", "mimo-v2.5"),
        api_key=require_env("MIMO_API_KEY"),
        base_url=require_env("MIMO_BASE_URL"),
        temperature=0,
        streaming=True,
        max_tokens=6000,
        timeout=300,
        max_retries=3,
    )
