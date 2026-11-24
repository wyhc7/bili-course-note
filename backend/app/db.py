"""
SQLite 持久化 + 任务模型。
串行队列：同一时间只有一个任务在跑（2核4G 不能并发）。
"""
import sqlite3
import json
import time
import threading
from pathlib import Path
from typing import Optional

from . import config

_LOCK = threading.RLock()

# 任务状态机
QUEUED = "queued"          # 排队中
DOWNLOADING = "downloading"
TRANSCRIBING = "transcribing"
EXTRACTING = "extracting"  # 抽帧 + 视觉
SUMMARIZING = "summarizing"
DONE = "done"
FAILED = "failed"


def _conn():
    conn = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _LOCK, _conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            title TEXT,
            status TEXT NOT NULL DEFAULT 'queued',
            progress INTEGER NOT NULL DEFAULT 0,
            stage_message TEXT,
            duration_sec REAL,
            output_md TEXT,
            error TEXT,
            options TEXT,
            collection_id TEXT,
            page_index INTEGER,
            page_title TEXT,
            part_index INTEGER,
            part_total INTEGER,
            retry_count INTEGER NOT NULL DEFAULT 0,
            not_before REAL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """)
        conn.commit()
        # 兼容老库：补列
        cols = [r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()]
        for col, ddl in [
            ("options", "ALTER TABLE tasks ADD COLUMN options TEXT"),
            ("collection_id", "ALTER TABLE tasks ADD COLUMN collection_id TEXT"),
            ("page_index", "ALTER TABLE tasks ADD COLUMN page_index INTEGER"),
            ("page_title", "ALTER TABLE tasks ADD COLUMN page_title TEXT"),
            ("part_index", "ALTER TABLE tasks ADD COLUMN part_index INTEGER"),
            ("part_total", "ALTER TABLE tasks ADD COLUMN part_total INTEGER"),
            ("retry_count", "ALTER TABLE tasks ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0"),
            ("not_before", "ALTER TABLE tasks ADD COLUMN not_before REAL DEFAULT 0"),
        ]:
            if col not in cols:
                conn.execute(ddl)
        conn.commit()


def create_task(url: str, options: str | None = None,
                collection_id: str | None = None, page_index: int | None = None,
                page_title: str | None = None, part_index: int | None = None,
                part_total: int | None = None) -> int:
    now = time.time()
    with _LOCK, _conn() as conn:
        cur = conn.execute(
            "INSERT INTO tasks (url, status, progress, stage_message, options, "
            "collection_id, page_index, page_title, part_index, part_total, created_at, updated_at) "
            "VALUES (?, 'queued', 0, '排队中', ?, ?, ?, ?, ?, ?, ?, ?)",
            (url, options, collection_id, page_index, page_title, part_index, part_total, now, now),
        )
        conn.commit()
        return cur.lastrowid


def collection_tasks(collection_id: str) -> list[dict]:
    with _LOCK, _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE collection_id=? ORDER BY part_index ASC",
            (collection_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_task(task_id: int, **fields):
    if not fields:
        return
    fields["updated_at"] = time.time()
    cols = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [task_id]
    with _LOCK, _conn() as conn:
        conn.execute(f"UPDATE tasks SET {cols} WHERE id=?", vals)
        conn.commit()


def get_task(task_id: int) -> Optional[dict]:
    with _LOCK, _conn() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return dict(row) if row else None


def list_tasks(limit: int = 50) -> list[dict]:
    with _LOCK, _conn() as conn:
        rows = conn.execute(
            "SELECT id,url,title,status,progress,stage_message,duration_sec,error,"
            "collection_id,page_index,page_title,part_index,part_total,retry_count,created_at,updated_at "
            "FROM tasks ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def delete_task(task_id: int) -> bool:
    with _LOCK, _conn() as conn:
        cur = conn.execute("DELETE FROM tasks WHERE id=?", (task_id,))
        conn.commit()
        return cur.rowcount > 0


def requeue_task(task_id: int) -> bool:
    """把任务重置为排队（重新跑），清空 retry/错误/产物路径。"""
    with _LOCK, _conn() as conn:
        cur = conn.execute(
            "UPDATE tasks SET status='queued', progress=0, stage_message='重新排队', "
            "error=NULL, output_md=NULL, retry_count=0, not_before=0, updated_at=? WHERE id=?",
            (time.time(), task_id))
        conn.commit()
        return cur.rowcount > 0


def next_queued() -> Optional[dict]:
    now = time.time()
    with _LOCK, _conn() as conn:
        row = conn.execute(
            "SELECT * FROM tasks WHERE status='queued' AND COALESCE(not_before,0)<=? "
            "ORDER BY id ASC LIMIT 1", (now,)
        ).fetchone()
        return dict(row) if row else None


def reset_stale_on_boot():
    """服务重启时，把卡在中间态的任务重新排队，避免永久卡死。"""
    with _LOCK, _conn() as conn:
        conn.execute(
            "UPDATE tasks SET status='queued', stage_message='重启后重新排队', "
            "updated_at=? WHERE status IN "
            "('downloading','transcribing','extracting','summarizing')",
            (time.time(),),
        )
        conn.commit()
