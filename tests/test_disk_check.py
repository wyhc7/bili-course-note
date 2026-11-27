"""磁盘预检的离线验证：估算数值 + 真实 worker 失败路径（不重试、状态落 FAILED）。

运行：python3 tests/test_disk_check.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

tmp = Path(tempfile.mkdtemp())
from app import config

config.DATA_DIR = tmp / "data"
config.OUTPUT_DIR = tmp / "data" / "output"
config.AUDIO_DIR = tmp / "data" / "audio"
config.FRAMES_DIR = tmp / "data" / "frames"
config.DB_PATH = tmp / "data" / "tasks.db"
for _d in (config.DATA_DIR, config.OUTPUT_DIR, config.AUDIO_DIR, config.FRAMES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

from app import db
from app.pipeline import download, worker

GB = 1e9

print("=== 1) 估算：按时长 × 系数 + 余量，且不低于绝对下限 ===")
for dur, want in ((0, 1.0), (600, 1.0), (3600, 1.5), (7200, 2.5)):
    got = download.estimate_need_bytes(dur) / GB
    print(f"    时长 {dur:>5}s -> 需 {got:.2f}GB（期望 {want}）")
    assert abs(got - want) < 0.01, (dur, got, want)

print("=== 2) 空间不足抛 DiskFullError，且文案可读 ===")
_orig_du = shutil.disk_usage


def _fake_du(free_gb):
    return lambda p: type("U", (), {
        "total": 100 * GB, "used": (100 - free_gb) * GB, "free": free_gb * GB})()


try:
    shutil.disk_usage = _fake_du(0.4)
    try:
        download._check_disk(7200)
        raise AssertionError("空间不足却没拦下来")
    except download.DiskFullError as e:
        msg = str(e)
        print("    ", msg)
        assert "磁盘空间不足" in msg and "请清理" in msg and "分钟" in msg
        assert isinstance(e, download.DownloadError), "必须是 DownloadError 子类"

    print("=== 3) 空间充足放行 ===")
    shutil.disk_usage = _fake_du(50)
    download._check_disk(7200)
    print("    50GB 可用 → 放行 ✅")

    print("=== 4) 开关关闭时不拦 ===")
    config.DISK_CHECK_ENABLED = False
    shutil.disk_usage = _fake_du(0.1)
    download._check_disk(7200)
    print("    DISK_CHECK_ENABLED=false → 不拦 ✅")
finally:
    shutil.disk_usage = _orig_du
    config.DISK_CHECK_ENABLED = True

print("=== 5) 真实 worker 路径：磁盘错误不重试，直接 FAILED ===")
db.init_db()
tid = db.create_task("https://www.bilibili.com/video/BV1xx411c7mD")
task = db.get_task(tid)
print("    建任务 id =", tid, "| 初始 retry_count =", task.get("retry_count"), "| MAX_RETRY =", config.MAX_RETRY)
assert int(task.get("retry_count") or 0) == 0


def _boom(*a, **k):
    raise download.DiskFullError("磁盘空间不足：剩余 0.4GB，预计需要 2.5GB。请清理 data 下的旧任务后重试。")


_orig_dl = download.download
download.download = _boom
try:
    worker._run_one(task)
finally:
    download.download = _orig_dl

after = db.get_task(tid)
print("    status =", after["status"], "| retry_count =", after["retry_count"], "| stage =", after["stage_message"])
assert after["status"] == db.FAILED, f"应以 FAILED 结束，实际 {after['status']}"
assert int(after["retry_count"] or 0) == 0, "磁盘错误不该消耗重试次数"
assert "磁盘空间不足" in after["stage_message"]
assert "请清理" in (after.get("error") or "")

print("\n=== 6) 对照：普通下载错误仍然会重试 ===")
tid2 = db.create_task("https://www.bilibili.com/video/BV1xx411c7mD")
task2 = db.get_task(tid2)


def _boom2(*a, **k):
    raise download.DownloadError("网络断了")


download.download = _boom2
try:
    worker._run_one(task2)
finally:
    download.download = _orig_dl
after2 = db.get_task(tid2)
print("    status =", after2["status"], "| retry_count =", after2["retry_count"])
assert after2["status"] == db.QUEUED, "普通错误应回到排队等待重试"
assert int(after2["retry_count"]) == 1

print("\n全部通过 ✅")
shutil.rmtree(tmp, ignore_errors=True)
