from .config import DEFAULT_PROFILE, PROFILE_DIR
from .session_manager import get_active_working_memory_path, get_active_working_summary_path


CORE_PROMPT = (
    "你是用户的 Personal Research-to-Product Agent，名字是张硕儿。面向用户自己的资料商品生产流程。目前主要搜索整理复试资料"
    "你可以使用工具读取工作区、执行受限验证命令、发现公开资料、筛选来源、抽取已确认来源并维护研究状态。"
    "你必须遵守工具边界：文件信息只能通过工具获取；高风险工具会进入人工审批；尽量去搜索课件、电子书、机构资料或笔记标记后提示进行人工审核；"
    "如果用户需要草稿，先在对话中生成可复制的 Markdown 内容；保存草稿应优先交给前端草稿编辑器或草稿保存接口。"
)


def load_profile_prompt(profile_name: str = DEFAULT_PROFILE) -> str:
    path = PROFILE_DIR / f"{profile_name}.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_working_memory() -> str:
    try:
        path = get_active_working_memory_path()
    except RuntimeError:
        return ""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:3000]


def load_working_summary() -> str:
    try:
        path = get_active_working_summary_path()
    except RuntimeError:
        return ""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:3000]
