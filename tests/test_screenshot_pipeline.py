"""
截图链路离线回归测试（不联网、不依赖真实视频）。

运行：python3 tests/test_screenshot_pipeline.py
覆盖：dHash 去重 / 网格序号解析 / 分块时间窗 / 标记吸附 / 只留引用帧 /
      超 1 小时时间戳 / 输出 token 自适应 / 视觉缓存 / 视觉重试 / 真实视频抽帧
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from PIL import Image, ImageDraw

TMP = Path(tempfile.mkdtemp(prefix="bcn_test_"))

from app import config

config.DATA_DIR = TMP / "data"
config.FRAMES_DIR = config.DATA_DIR / "frames"
config.AUDIO_DIR = config.DATA_DIR / "audio"
config.OUTPUT_DIR = config.DATA_DIR / "output"
for _d in (config.DATA_DIR, config.FRAMES_DIR, config.AUDIO_DIR, config.OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 固定与清理/视觉相关的配置，避免测试结果受宿主 .env（如 VISION_ENABLED=true）影响
config.CLEANUP_AFTER_DONE = True
config.KEEP_FRAMES = False
config.VISION_ENABLED = False

from app.pipeline import frames, summarize, worker


def case(title):
    print(f"— {title}")


def img_at(path: Path, color="white", text=None):
    im = Image.new("RGB", (320, 180), color)
    if text:
        ImageDraw.Draw(im).text((10, 10), text, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    return str(path)


# ---------------------------------------------------------------- 1. dHash
case("dHash：同画面不同质量判同，换页判不同")
p1, p2, p3 = (TMP / "a.jpg"), (TMP / "b.jpg"), (TMP / "c.jpg")
im = Image.new("RGB", (960, 540), "white")
d = ImageDraw.Draw(im)
d.text((40, 40), "Theorem 1.7 Cramer", fill="black")
d.rectangle([100, 200, 500, 400], outline="black", width=3)
im.save(p1, quality=90)
im.save(p2, quality=40)
im2 = Image.new("RGB", (960, 540), "black")
ImageDraw.Draw(im2).text((40, 40), "Next slide completely different", fill="white")
im2.save(p3, quality=90)
same = frames._hamming(frames._dhash(p1), frames._dhash(p2))
diff = frames._hamming(frames._dhash(p1), frames._dhash(p3))
print(f"   同画面距离={same}  换页距离={diff}")
assert same <= config.FRAME_DEDUPE_DISTANCE and diff > config.FRAME_DEDUPE_DISTANCE

# ------------------------------------------------- 2. 网格序号 / 时间戳解析
case("网格输出解析：三种格式容错")
ts = [0, 10, 20, 30, 40, 50, 60, 70, 80]
assert frames._parse_grid_reply("#1 一页PPT\n#3 公式\n#9 人物", ts) == \
    {0: "一页PPT", 20: "公式", 80: "人物"}
assert frames._parse_grid_reply("00:10 描述", ts).get(10) == "描述"
assert frames._parse_grid_reply("#2 12:30 带时间", ts).get(10) == "带时间"
assert frames._parse_grid_reply("第一格\n第二格", ts[:2]) == {0: "第一格", 10: "第二格"}

# ------------------------------------------------------- 3. 分块与时间窗
case("分块起止时间 + 候选时间窗过滤")
assert summarize._chunk_span("[00:00] 开场\n[03:20] 定理\n[07:45] 例题") == (0, 465)
assert summarize._chunk_span("[01:02:05] 长视频") == (3725, 3725)
vision = [{"ts": t, "desc": f"画面{t}", "img_url": f"/frames/task_1/f_{t}.jpg"}
          for t in (0, 30, 300, 600)]
assert summarize._vlines(vision, -30, 230) == ["- S1　00:00：画面0", "- S2　00:30：画面30"]
assert summarize._vlines([{"ts": 5, "desc": "[网格视觉失败: x]", "img_url": "/x"}]) == []

# --------------------------------------------- 4. 超 1 小时的时间戳（P0）
case("超 1 小时标记：hh:mm:ss 与三位分钟")
doc = ("## 一节 *Content-[01:02:05]\n\n*Screenshot-[01:02:30]\n\n"
       "## 二节 *Content-[62:05]\n\n*Screenshot-[63:20]\n\n*Screenshot-[00:30]\n")
assert summarize.extract_screenshot_marks(doc) == [
    ("*Screenshot-[01:02:30]", 3750),
    ("*Screenshot-[63:20]", 3800),
    ("*Screenshot-[00:30]", 30),
]
norm = summarize._CONTENT_RE.sub(summarize._norm_content, doc)
assert norm.count("*Content-[01:02:05]") == 2 and "]:05]" not in norm

# --------------------------------------------------- 5. 输出 token 自适应
case("输出 token：下限 1200 / 硬顶可配")
assert summarize._out_tokens("x" * 10, "y" * 1000) == 1200
assert summarize._out_tokens("x" * 10, "y" * 30000) == config.LLM_MAX_OUTPUT_TOKENS

# ------------------------------------------------ 6. 分块只给本段候选
case("分块总结：每段只拿本段时间窗的候选，合并阶段不再塞清单")
config.LLM_MAX_CHARS = 200
segs = [{"start": t, "text": "内容" * 60} for t in range(0, 160, 20)]
vis = [{"ts": t, "desc": f"第{t}秒画面", "img_url": f"/frames/task_1/f_{t:05d}.jpg"}
       for t in range(0, 160, 20)]
calls = []


def fake_chat(client, system, user, max_tokens=None):
    calls.append((system, user, max_tokens))
    return "## 章节 *Content-[00:20]\n\n要点\n\n*Screenshot-[00:35]"


_orig_chat = summarize._chat
summarize._client = lambda: object()
summarize._chat = fake_chat
out = summarize.summarize(segs, vis, {"title": "测试课", "duration_sec": 160,
                                      "style": "网课", "screenshot": True,
                                      "link": True, "toc": False, "summary": True})
assert len(calls) >= 2
for system, user, _ in calls[:-1]:
    first = next(ln for ln in user.splitlines() if ln.startswith("["))
    mm, ss = first[1:6].split(":")
    if int(mm) * 60 + int(ss) >= 80:
        assert "00:00：" not in system, "后段仍拿到第一段候选"
assert "8b)" not in calls[-1][0], "合并阶段不该塞候选"
assert "*Screenshot-[00:35]" in out
summarize._chat = _orig_chat
config.LLM_MAX_CHARS = 48000

# ------------------------------------------- 7. 吸附 + 只保留引用帧（P0）
case("标记吸附复用候选帧，清理时只留被引用文件")
tdir = config.FRAMES_DIR / "task_9"
for t in (30, 300):
    img_at(tdir / f"f_{t:05d}.jpg", "gray")
vis2 = [{"ts": 30, "desc": "d30", "img_url": "/frames/task_9/f_00030.jpg"},
        {"ts": 300, "desc": "d300", "img_url": "/frames/task_9/f_00300.jpg"}]
doc2 = ("# t\n\n*Screenshot-[00:33]\n\n正文\n\n*Screenshot-[04:55]\n\n"
        "*Screenshot-[08:00]\n")
out2, used = summarize.fill_screenshots(doc2, "/nonexistent/video.mp4", 9, vision=vis2)
assert "/frames/task_9/f_00030.jpg" in out2 and "/frames/task_9/f_00300.jpg" in out2
assert "08:00" not in out2, "无候选且不临近的标记应被清掉"
assert used == {"/frames/task_9/f_00030.jpg", "/frames/task_9/f_00300.jpg"}

for name in ("f_00001.jpg", "grid_000.jpg", "shot_99.jpg"):
    (tdir / name).write_bytes(b"x")
worker._cleanup(9, keep_imgs=used)
left = sorted(p.name for p in tdir.iterdir())
print(f"   清理后保留: {left}")
assert left == ["f_00030.jpg", "f_00300.jpg"]

# --------------------------------------------------- 8. 视觉缓存 + 重试
case("视觉缓存：二跑 0 次调用；重试：抖动后成功")
config.VISION_ENABLED = True
config.VISION_API_KEY = "test"
config.VISION_RETRY_BACKOFF_SEC = 0
_orig_vision = frames._vision_chat
n_calls = {"n": 0}


def fake_vision(client, model, text, b64, max_tokens=300):
    n_calls["n"] += 1
    return "\n".join(f"#{i} 描述{i}" for i in range(1, 10))


frames._vision_chat = fake_vision
ts_list = [i * 10 for i in range(18)]


def make_frames(task_id):
    d = config.FRAMES_DIR / f"task_{task_id}"
    out = []
    for t in ts_list:
        out.append({"path": img_at(d / f"f_{t:05d}.jpg", (t * 5 % 255, 80, 120)), "ts": t})
    return out


v1 = frames.describe_frames(make_frames(1), task_id=1, duration_sec=180, cache_key="BV1_1")
assert n_calls["n"] == 2 and v1[0]["desc"] == "描述1"
before = n_calls["n"]
v2 = frames.describe_frames(make_frames(2), task_id=2, duration_sec=180, cache_key="BV1_1")
print(f"   二跑新增调用={n_calls['n'] - before}，img_url={v2[0]['img_url']}")
assert n_calls["n"] == before, "缓存未命中"
assert v2[0]["img_url"].startswith("/frames/task_2/"), "img_url 未映射到新任务"
frames._vision_chat = _orig_vision


class Flaky:
    def __init__(self, fail_times):
        self.n = 0
        self.fail = fail_times
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.n += 1
        if self.n <= self.fail:
            raise RuntimeError("502 Bad Gateway")
        return iter([SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="好的"))])])


c = Flaky(fail_times=2)
assert frames._vision_chat(c, "m", "t", "b64") == "好的" and c.n == 3

# --------------------------------- 9. 图文对应：编号引用 / 时间倒序 / 错位解析
case("图文对应：编号直取候选；时间倒序不再丢图；网格不错位")
vis3 = [{"ts": t, "desc": f"第{i}页", "img_url": f"/frames/task_9/{n}"}
        for i, (t, n) in enumerate(
            [(60, "f_00001.jpg"), (600, "f_00003.jpg"), (1800, "f_00005.jpg")], 1)]

# 时间倒序（合并后章节回顾引用早期画面）：旧实现 `ts - last < gap` 会把后两个全删
d1 = "A *Screenshot-[30:00]\n\nB *Screenshot-[01:00]\n\nC *Screenshot-[10:00]\n"
o1, _ = summarize.fill_screenshots(d1, "/none.mp4", 9, vision=vis3)
assert o1.count("![") == 3, o1

# 编号引用：直接锁定那一条候选，不做任何时间吸附
d2 = "讲行列式 *Screenshot-[S3]\n\n讲例题 *Screenshot-[S2]\n"
o2, _ = summarize.fill_screenshots(d2, "/none.mp4", 9, vision=vis3)
assert "/frames/task_9/f_00005.jpg" in o2 and "/frames/task_9/f_00003.jpg" in o2

# 幻觉编号删标记、同一编号只插一次
d3 = "*Screenshot-[S99]\n\n*Screenshot-[S1]\n\n*Screenshot-[S1]\n"
o3, _ = summarize.fill_screenshots(d3, "/none.mp4", 9, vision=vis3)
assert o3.count("![") == 1 and "Screenshot" not in o3, o3

# 加粗/多余星号包裹也要认，且不残留星号
o4, _ = summarize.fill_screenshots("**Screenshot-[S2]**\n", "/none.mp4", 9, vision=vis3)
assert o4.count("![") == 1 and "*" not in o4, o4

# 标题行里的标记被挪到下一行，避免"标题配图"
o5 = summarize._HEADING_MARK_RE.sub(r"\1\n\n\2", "## 一、定义 *Screenshot-[S1]\n")
assert o5.splitlines()[0] == "## 一、定义", o5

# 网格：h:mm:ss 解析正确；缺序号且行数不匹配时宁可不给，也不能整体错一格
assert frames._parse_grid_reply("1:15:30 板书推导", [4530]) == {4530: "板书推导"}
assert frames._parse_grid_reply("好的，下面逐格：\n1. A\n2. B\n3. C", [0, 10, 20]) == {}

# ------------------------------------------- 10. 补洞（与远端 8c393b0 的融合）
case("补洞：相邻截图空档过大时从候选帧补图，关闭时不补")
vis4 = [{"ts": t, "desc": f"p{t}", "img_url": f"/frames/task_9/g_{t:05d}.jpg"}
        for t in (0, 100, 200, 300)]
d6 = "# t\n\n*Screenshot-[00:00]\n\n*Screenshot-[05:00]\n"
_old_gap = config.SCREENSHOT_MAX_GAP_SEC
config.SCREENSHOT_MAX_GAP_SEC = 120
o6, _ = summarize.fill_screenshots(d6, "/none.mp4", 9, vision=vis4)
print(f"   开补洞(120s)：{o6.count('![')} 张")
assert o6.count("![") == 4, o6          # 0 → 100 → 200 → 300
config.SCREENSHOT_MAX_GAP_SEC = 0
o7, _ = summarize.fill_screenshots(d6, "/none.mp4", 9, vision=vis4)
assert o7.count("![") == 2, o7          # 关闭补洞只剩两个锚点
config.SCREENSHOT_MAX_GAP_SEC = _old_gap

# ------------------------------------------------- 11. 真实视频抽帧（可选）
case("真实视频：静止段去重、换页帧保留")
if shutil.which("ffmpeg"):
    vid = TMP / "t.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=white:s=640x360:d=40",
                    "-vf", "drawbox=x=0:y=0:w=320:h=180:color=black:t=fill:enable='gte(t,20)'",
                    str(vid), "-loglevel", "error"], check=True)
    config.DENSE_FRAME_INTERVAL_SEC = 5
    frms = frames.extract_frames(str(vid), 11, 40)
    kept = [f["ts"] for f in frms]
    print(f"   40s 视频 8 帧 → 保留 {kept}")
    assert kept == [0, 20], kept
else:
    print("   跳过（没有 ffmpeg）")

shutil.rmtree(TMP, ignore_errors=True)
print("\nALL OK")
