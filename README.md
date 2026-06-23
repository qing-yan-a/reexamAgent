# reexamAgent

一个基于 LangGraph 的研究生复试资料搜索与整理 Agent 学习项目。

项目目标是把“手搓 Agent Loop”逐步迁移成可观察、可中断、可恢复的 LangGraph 图流程。当前重点能力是：围绕某个学校和专业的复试资料，执行缺口驱动的搜索循环，并把候选来源、初筛结果和资料缺口记录到本地 research session。

## 核心能力

- LangGraph 主流程：普通对话、工具调用、人工审批、checkpoint 压缩、长期记忆保存。
- 复试资料搜索循环：解析学校/专业/年份，判断资料缺口，生成 query，搜索公开网页，初筛来源，等待用户决定继续补搜或进入来源确认。
- 资料输出目录：每个复试任务自动创建 `test/<学校><专业>/`，用户要求保存的资料和草稿默认放入该目录。
- 本地 RAG：扫描本地 `test/**/*.md`，切 chunk，向量化后写入 PostgreSQL `rag_chunks` 表，查询时做 dense + BM25 混合检索。
- 长期记忆 Store：过滤用户长期事实，写入 LangGraph PostgresStore，并在碎片过多时压缩成 `memory_summary`。
- 安全边界：medium/high 风险工具通过 `interrupt()` 暂停，由用户审批后继续。

## 项目结构

```text
src/personal_research_agent/
  cli.py                  # 命令行入口
  graph.py                # LangGraph 主图
  reexam_search_flow.py   # 复试资料搜索循环逻辑
  rag_db.py               # PostgreSQL + pgvector RAG 层
  memory.py               # checkpoint 摘要和长期 store 压缩
  tools/                  # 文件、搜索、RAG、研究会话等工具

profiles/
  research_to_product.md  # 当前 Agent profile

tests/
  test_*.py               # 单元测试
```

`memory/`、`test/`、`.env`、`.venv` 是本地运行数据或隐私配置，默认不提交。

## 环境准备

建议使用 Python 3.11+。

```powershell
cd E:\study\LangGraph
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

需要在项目根目录创建 `.env`，按你的模型和数据库配置填写：

```env
MIMO_API_KEY=...
MIMO_BASE_URL=...
MIMO_MODEL=mimo-v2.5

DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=...
DEEPSEEK_MODEL=deepseek-v4-flash

POSTGRES_URI=postgresql://postgres:你的密码@localhost:5432/postgres?sslmode=disable
VOYAGE_API_KEY=...
USER_ID=qingyan
```

如果要使用数据库版 RAG，需要本地 PostgreSQL 已启用 `pgvector`：

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## 启动

CLI 入口：

```powershell
cd E:\study\LangGraph
.\.venv\Scripts\python.exe -m personal_research_agent.cli
```

只想临时测试、不连接 PostgreSQL：

```powershell
.\.venv\Scripts\python.exe -m personal_research_agent.cli --memory-only
```

Web 工作台入口：

```powershell
cd E:\study\LangGraph
.\.venv\Scripts\uvicorn.exe personal_research_agent.api.app:app --reload --host 127.0.0.1 --port 8000
```

然后打开：

```text
http://127.0.0.1:8000
```

启动后可以输入：

```text
帮我搜索昆明理工大学计算机复试资料
```

命中复试资料搜索意图后，图会进入专用循环：

```text
parse_reexam_goal
-> ensure_research_session
-> evaluate_reexam_gaps
-> generate_gap_queries
-> run_one_web_search
-> review_sources
-> record_search_iteration
-> ask_user_next_step
```

`ask_user_next_step` 会通过 `interrupt()` 暂停，等待你输入：

```text
continue 继续补搜
next     进入来源确认
stop     停止并保留 session
```

同一次任务会自动创建资料输出目录，例如：

```text
test/昆明理工大学计算机/
```

后续用户明确要求保存的资料或草稿，默认会写入这个目录，方便重新构建 RAG 索引。

## RAG 索引

RAG 原始资料默认读取本地：

```text
E:\study\LangGraph\test/**/*.md
```

该目录默认不提交到 GitHub，避免泄露本地资料或个人信息。

在 Agent 中请求“重建本地资料索引”时，会调用 `rebuild_memory_index`。该工具风险等级为 `medium`，会先进入人工审批；批准后才会扫描 Markdown、调用 embedding API，并写入 PostgreSQL `rag_chunks` 表。

## 测试

```powershell
cd E:\study\LangGraph
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

当前测试覆盖：

- checkpoint 摘要策略
- 长期 store 压缩摘要
- RAG chunk 和混合排序
- 复试资料搜索循环纯逻辑
- 工具风险和路由
- 文件安全边界

## 注意

- `web_search` 依赖本地 Tavily CLI：`~/.local/bin/tvly.exe`。
- 搜索结果只是候选来源，不等于已核验事实。
- PDF 或正文抽取必须由用户确认后再执行。
- 不自动生成可售卖最终资料，不自动发布或上架。
