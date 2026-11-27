"""稳定性加固的离线验证：LLM 重试 / 下载续传 / 失败保留产物。

运行：python3 tests/test_resilience.py
不联网、不调真实 API。
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

tmp = Path(tempfile.mkdtemp())
from app import config

config.DATA_DIR = tmp / "data"
config.FRAMES_DIR = tmp / "data" / "frames"
config.AUDIO_DIR = tmp / "data" / "audio"
config.OUTPUT_DIR = tmp / "data" / "output"
config.DB_PATH = tmp / "data" / "tasks.db"
for _d in (config.DATA_DIR, config.FRAMES_DIR, config.AUDIO_DIR, config.OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

from app.pipeline import download, summarize

config.LLM_RETRY_BACKOFF_SEC = 0  # 测试不真等
config.DOWNLOAD_RETRY_BACKOFF_SEC = 0


# ---------------------------------------------------------------- 1. LLM 重试
print("=== 1) LLM 前两次失败、第三次成功 ===")


class _Delta:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class FakeLLM:
    def __init__(self, scripts):
        self.scripts = scripts  # 每次调用：list[str] 表示成功分片；Exception 表示失败
        self.calls = 0

    class _Completions:
        def __init__(self, outer):
            self.outer = outer

        def create(self, **kw):
            idx = self.outer.calls
            self.outer.calls += 1
            script = self.outer.scripts[min(idx, len(self.outer.scripts) - 1)]
            if isinstance(script, Exception):
                raise script

            def gen():
                for piece in script:
                    if isinstance(piece, Exception):
                        raise piece  # 模拟流到一半断开
                    yield _Chunk(piece)
            return gen()

    @property
    def chat(self):
        outer = self

        class _Chat:
            completions = FakeLLM._Completions(outer)
        return _Chat()


llm = FakeLLM([[ConnectionError("reset")], [TimeoutError("timeout")], ["笔记", "正文"]])
out = summarize._chat(llm, "sys", "user")
print(f"    调用 {llm.calls} 次，返回 {out!r}")
assert llm.calls == 3 and out == "笔记正文"

print("=== 2) 全部失败但已收到足够内容 -> 接受半成品并标注 ===")
config.LLM_PARTIAL_MIN_CHARS = 4
llm2 = FakeLLM([["前半段内容", ConnectionError("断流")]])
out2 = summarize._chat(llm2, "s", "u")
print("    返回末尾:", repr(out2[-40:]))
assert "提前结束" in out2 and "前半段内容" in out2

print("=== 3) 全部失败且内容太少 -> 抛 SummaryError ===")
config.LLM_PARTIAL_MIN_CHARS = 400
llm3 = FakeLLM([ConnectionError("x")])
try:
    summarize._chat(llm3, "s", "u")
    raise AssertionError("应该抛错")
except summarize.SummaryError as e:
    print("    ", str(e)[:60])


# ------------------------------------------------------------ 4. 下载断点续传
print("=== 4) 下载中断后按 Range 续传，最终文件完整 ===")
payload = b"A" * 100 + b"B" * 100


class _Resp:
    def __init__(self, status, headers, chunks, fail_at=None):
        self.status_code = status
        self.headers = headers
        self._chunks = chunks
        self._fail_at = fail_at

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_bytes(self, chunk_size=0):
        for i, c in enumerate(self._chunks):
            if self._fail_at is not None and i >= self._fail_at:
                raise ConnectionError("peer reset")
            yield c


class FakeClient:
    def __init__(self):
        self.calls = []

    def stream(self, method, url, headers=None, timeout=None):
        self.calls.append(dict(headers or {}))
        n = len(self.calls)
        if n == 1:
            # 声称 200 字节，只给前 100 就断
            return _Resp(200, {"content-length": "200"}, [payload[:100]], fail_at=1)
        # 第二次：必须带 Range，返回剩余 100
        rng = (headers or {}).get("Range")
        assert rng == "bytes=100-", f"续传没带 Range，实际 {rng}"
        return _Resp(206, {"content-length": "100"}, [payload[100:]])


out_file = tmp / "stream.bin"
cli = FakeClient()
download._download_stream(cli, "http://x/y", out_file, "http://ref", tag="音频")
got = out_file.read_bytes()
print(f"    调用 {len(cli.calls)} 次，文件 {len(got)} 字节，内容一致={got == payload}")
assert got == payload and len(cli.calls) == 2

print("=== 5) 声称长度与实际不符（无异常）也要重试 ===")
class ShortClient(FakeClient):
    def stream(self, method, url, headers=None, timeout=None):
        self.calls.append(dict(headers or {}))
        return _Resp(200, {"content-length": "200"}, [payload[:100]])


try:
    out_file2 = tmp / "short.bin"
    download._download_stream(ShortClient(), "http://x/y", out_file2, "http://ref", tag="音频")
    raise AssertionError("短文件应该被判失败")
except download.DownloadError as e:
    print("    ", str(e)[:50])
    assert "不完整" in str(e) or "失败" in str(e)


# ------------------------------------------------------ 6. 失败重试复用产物
print("=== 6) 断点续跑：meta.json + audio.wav 在 -> 直接复用 ===")
work = config.DATA_DIR / "task_99"
work.mkdir(parents=True, exist_ok=True)
(work / "audio.wav").write_bytes(b"x" * 20000)
(work / "meta.json").write_text(json.dumps({
    "audio_path": str(work / "audio.wav"), "video_path": None,
    "title": "复用测试", "duration_sec": 123.0, "bvid": "BV1xx411c7mD"}), encoding="utf-8")

config.EXTRACT_FRAMES_NEED_VIDEO = False
hit = download._load_reusable(work, "BV1xx411c7mD")
print("    命中:", bool(hit), "| title:", (hit or {}).get("title"))
assert hit and hit["duration_sec"] == 123.0

print("    需要抽帧但视频缺失 -> 不复用（否则这篇没截图）")
config.EXTRACT_FRAMES_NEED_VIDEO = True
assert download._load_reusable(work, "BV1xx411c7mD") is None
config.EXTRACT_FRAMES_NEED_VIDEO = False

print("    bvid 不一致 -> 不复用")
assert download._load_reusable(work, "BV_other") is None

print("    关闭 RESUME_ON_RETRY -> 不复用")
config.RESUME_ON_RETRY = False
assert download._load_reusable(work, "BV1xx411c7mD") is None
config.RESUME_ON_RETRY = True


# --------------------------------------------------- 7. 失败保留 / 成功清理
print("=== 7) 还有重试机会时不清理，最终才清 ===")
from app.pipeline import worker

work2 = config.DATA_DIR / "task_98"
work2.mkdir(parents=True, exist_ok=True)
(work2 / "audio.wav").write_bytes(b"y" * 20000)
frames98 = config.FRAMES_DIR / "task_98"
frames98.mkdir(parents=True, exist_ok=True)
(frames98 / "f_00001.jpg").write_bytes(b"j")

worker._cleanup(98, keep_imgs=set(), keep_files=True)
print("    keep_files=True  -> task 目录存在:", work2.exists(), "| 帧存在:", frames98.exists())
assert work2.exists() and frames98.exists()

worker._cleanup(98, keep_imgs=set(), keep_files=False)
print("    keep_files=False -> task 目录存在:", work2.exists(), "| 帧存在:", frames98.exists())
assert not work2.exists() and not frames98.exists()

print("\n全部通过 ✅")
shutil.rmtree(tmp, ignore_errors=True)
