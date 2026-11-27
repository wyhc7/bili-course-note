#!/usr/bin/env python3
"""
任务失败诊断：读 tasks.db，把最近失败的任务按阶段归类，并给出针对性建议。

用法（在仓库根目录）：
    python3 scripts/diagnose.py          # 最近 200 条任务
    python3 scripts/diagnose.py 500      # 最近 500 条

只读数据库，不改任何东西。
"""
import re
import sqlite3
import sys
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

try:
    from app import config
    db_path = config.DB_PATH
except Exception:
    db_path = ROOT / "backend" / "data" / "tasks.db"

if not Path(db_path).exists():
    print(f"找不到数据库：{db_path}")
    print("如果你改过 DATA_DIR，请先 cd 到仓库根目录再运行，或手动指定：")
    print("  DB_PATH=/your/data/tasks.db python3 scripts/diagnose.py")
    sys.exit(1)

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 200

# 失败原因归类：按关键词从"最具体"到"最笼统"匹配
RULES = OrderedDict([
    ("磁盘空间不足", (r"磁盘空间不足|No space left", "清理 DATA_DIR 下的旧任务；或调低 DISK_GB_PER_HOUR")),
    ("B站接口/反爬", (r"view API|playurl|412|403|风控|未取到 cid|未返回音频流",
                    "换网络/加 BILI_COOKIE；服务器 IP 被风控时可挂代理")),
    ("下载中断", (r"下载.*失败|不完整|peer reset|timed out|timeout|Connection",
                "已支持断点续传，重试即可；频繁中断可调大 DOWNLOAD_ATTEMPTS")),
    ("音频转换", (r"ffmpeg 转 wav|封装视频失败", "多为下载文件不完整，重跑通常能好；持续失败请检查磁盘")),
    ("转写(语音识别)", (r"转写|whisper|OOM|内存|未识别到任何语音|音频",
                     "调小 WHISPER_CHUNK_SEC(如 180)、换 base 模型，或 TRANSCRIBE_PROVIDER=cloud")),
    ("LLM 总结", (r"LLM|总结|SummaryError|API key|token|429|rate limit",
                "已自动重试 3 次；仍失败多为额度/并发限制，可调 LLM_RETRY_ATTEMPTS 或换模型")),
    ("抽帧/视觉", (r"截帧|抽帧|FrameError|视觉", "截图会退化为无图笔记，不影响成稿；检查 ffmpeg")),
])


def classify(text: str):
    for name, (pat, _hint) in RULES.items():
        if re.search(pat, text, re.I):
            return name
    return "其他/未归类"


conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT id, title, status, stage_message, error, retry_count, duration_sec "
    "FROM tasks ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
conn.close()

failed = [r for r in rows if r["status"] == "failed"]
total = len(rows)
print(f"数据库：{db_path}")
print(f"检查范围：最近 {total} 条任务 | 失败 {len(failed)} 条"
      f"（{len(failed) / total * 100:.1f}%）\n" if total else "没有任务记录\n")

if not rows:
    sys.exit(0)

print("── 状态分布 ──")
for st, n in Counter(r["status"] for r in rows).most_common():
    print(f"  {st:<10} {n:>4}")

if not failed:
    print("\n没有失败任务。")
    sys.exit(0)

print("\n── 失败原因归类 ──")
buckets = Counter()
for r in failed:
    buckets[classify(f"{r['stage_message'] or ''} {r['error'] or ''}")] += 1
for name, n in buckets.most_common():
    hint = dict(RULES).get(name, ("", "看下面明细里的 error"))[1]
    print(f"  {name:<14} {n:>4} 条  → {hint}")

print("\n── 最近 8 条失败明细 ──")
for r in failed[:8]:
    title = (r["title"] or "").strip()[:36] or "(无标题)"
    dur = int((r["duration_sec"] or 0) // 60)
    stage = (r["stage_message"] or "").replace("\n", " ")[:70]
    err = (r["error"] or "").replace("\n", " ")
    err_head = re.split(r"\n?Traceback", err)[0][:160].strip()
    print(f"\n  #{r['id']}  {title}  [{dur}分钟, 重试{r['retry_count'] or 0}次]")
    print(f"    阶段：{stage}")
    if err_head:
        print(f"    错误：{err_head}")

print("\n提示：失败但 status 已回到 queued 的，是还在等重试，不算最终失败。")
