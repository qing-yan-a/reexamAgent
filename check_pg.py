from dotenv import load_dotenv
from pathlib import Path
import os
import psycopg

load_dotenv(Path("E:/Claude/codex/agent-langgraph-lab/.env"))

conn = psycopg.connect(os.getenv("POSTGRES_URI"))
cur = conn.cursor()

cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")
rows = cur.fetchall()
print("pgvector:", rows if rows else "未安装")

cur.execute("SELECT version()")
print("PostgreSQL:", cur.fetchone()[0])

conn.close()
