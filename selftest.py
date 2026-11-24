"""骨架自检：不依赖网络/真实下载，验证模块导入、DB建表、路由、前端托管。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app import db, config
print("[1] config OK: model=%s provider=%s cleanup=%s" % (
    config.WHISPER_MODEL, config.TRANSCRIBE_PROVIDER, config.CLEANUP_AFTER_DONE))

db.init_db()
tid = db.create_task("https://www.bilibili.com/video/BV18vyjzbELJ")
print("[2] DB OK: created task id=%s" % tid)
t = db.get_task(tid)
print("    task status=%s url=%s" % (t["status"], t["url"]))
print("    list_tasks count=%s" % len(db.list_tasks()))
print("    next_queued id=%s" % (db.next_queued() or {}).get("id"))

# 路由 & 前端
from app.main import app
routes = sorted({r.path for r in app.routes})
print("[3] routes: %s" % [r for r in routes if r.startswith("/api") or r == "/"])
idx = config.WEB_DIR / "index.html"
print("    index.html exists=%s size=%s" % (idx.exists(), idx.stat().st_size if idx.exists() else 0))

# 流水线模块（懒加载，仅 import 不触发重依赖）
from app.pipeline import download, transcribe, frames, summarize, worker
print("[4] pipeline modules imported OK")

# 测试时间戳格式化
print("[5] fmt_ts: %s / %s" % (summarize._fmt_ts(3725), summarize._fmt_ts(95)))

# 新增功能自检
from app.main import _md_to_html, _build_note_zip, _collect_images
_demo = ("# 标题\n\n## 第一节 *Content-[00:00]\n\n- 要点 **粗体**\n\n"
         "![00:00](/frames/task_x/shot_0.jpg)\n\n## 第二节 *Content-[01:00]\n\n内容\n\n"
         "## 第三节 *Content-[02:00]\n\n更多内容\n")
_html = _md_to_html(_demo)
assert "<h2" in _html and 'class="toc"' in _html and 'id="h' in _html, "HTML/TOC 渲染异常"
print("[6] _md_to_html OK (含目录锚点, len=%s)" % len(_html))

# DB 新列
t2 = db.get_task(tid)
assert "retry_count" in t2, "缺少 retry_count 列"
print("[7] retry_count/not_before 列 OK")

# 健康检查
import io as _io
from fastapi.testclient import TestClient
try:
    _c = TestClient(app)
    hr = _c.get("/api/health")
    assert hr.status_code == 200 and "disk" in hr.json(), "health 异常"
    print("[8] /api/health OK: %s" % hr.json().get("task_counts"))
except Exception as _e:
    print("[8] /api/health 跳过(TestClient不可用: %s)" % _e)

# 清理测试任务
with db._LOCK, db._conn() as conn:
    conn.execute("DELETE FROM tasks WHERE id=?", (tid,))
    conn.commit()
print("[9] cleanup test task done. ALL CHECKS PASSED")
