"""
关键帧抽取 + 多模态视觉描述（可选）。
网课优化：优先取画面变化大的帧（翻页/写板书），而不是等间隔盲抽。

抽帧  : 等间隔密集采样（默认 10s）→ 感知哈希（dHash）去重，替代失效的 MD5 去重。
视觉  : 3x3 网格拼图，每格左上角烙印 `#序号 mm:ss`，模型按 `#序号` 回填描述。
截图  : 两阶段——标记 → 吸附到最近的已抽帧并复用该文件（不二次 ffmpeg 截帧）。
"""
import base64
import json
import re
import subprocess
import time
from pathlib import Path

from .. import config


class FrameError(Exception):
    pass


def grab_frame_at(video_path: str, task_id: int, ts_sec: float) -> str:
    """
    在指定秒数精确截取一帧（兜底路径：候选帧没覆盖到该时间点时才走这里）。
    返回该帧的 web 相对 URL（/frames/task_x/shot_<秒>.jpg）。

    用 "-ss 前置 1s + 后置精确定位" 的混合 seek：既快（只解码 1 秒）又准
    （-ss 放在 -i 前是快速 seek，会对齐到关键帧，画面可能差好几秒）。
    """
    out_dir = config.FRAMES_DIR / f"task_{task_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"shot_{int(ts_sec)}.jpg"
    out = out_dir / fname
    if not out.exists():
        pre = max(0.0, float(ts_sec) - 1.0)
        post = max(0.0, float(ts_sec) - pre)
        vf = []
        if config.SHOT_WIDTH and config.SHOT_WIDTH > 0:
            vf = ["-vf", f"scale={config.SHOT_WIDTH}:-2:flags=lanczos"]
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{pre:.3f}",
            "-i", video_path,
            "-ss", f"{post:.3f}",
            "-frames:v", "1",
            *vf,
            "-q:v", "2",
            str(out), "-loglevel", "error",
        ]
        p = subprocess.run(cmd, capture_output=True, timeout=120)
        if p.returncode != 0 or not out.exists():
            raise FrameError(f"截帧失败 ts={ts_sec}: {(p.stderr or b'')[-200:]}")
    return _frame_web_url(str(out))


def extract_frames(video_path: str, task_id: int, duration_sec: float,
                   progress_cb=None) -> list[dict]:
    """
    抽候选帧，返回 [{path, ts}]，ts 精确到秒。

    1) 按 DENSE_FRAME_INTERVAL_SEC 密集等间隔抽（默认 10s）
    2) 感知哈希 dHash 去重：与"上一张保留帧"的汉明距离 <= 阈值即视为同一画面丢弃
       （旧实现用文件 MD5，同一张 PPT 在不同时刻解码出来的 JPEG 压缩噪声不同，
        哈希必然不等，等于完全没去重）
    """
    out_dir = config.FRAMES_DIR / f"task_{task_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    for sub in out_dir.glob("scene"):
        if sub.is_dir():
            for f in sub.glob("*"):
                f.unlink()

    frames: list[dict] = []
    if progress_cb:
        progress_cb("抽取候选帧…")

    interval = max(1, config.DENSE_FRAME_INTERVAL_SEC)
    # 候选帧按最终插图同样的宽度抽，命中吸附后直接进笔记，清晰度不分叉
    src_w = config.SHOT_WIDTH if config.SHOT_WIDTH and config.SHOT_WIDTH > 0 else 960
    if duration_sec > 0:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"fps=1/{interval},scale={src_w}:-2",
            "-q:v", "3",
            str(out_dir / "f_%05d.jpg"),
            "-loglevel", "error",
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=3600)
        except Exception:
            pass

        prev_sig = None
        idx = 0
        for p in sorted(out_dir.glob("f_*.jpg")):
            ts = idx * interval
            idx += 1
            if config.FRAME_DEDUPE:
                sig = _dhash(p)
                if sig is not None and prev_sig is not None \
                        and _hamming(sig, prev_sig) <= config.FRAME_DEDUPE_DISTANCE:
                    p.unlink(missing_ok=True)
                    continue
                prev_sig = sig
            frames.append({"path": str(p), "ts": ts})

    if not frames:
        raise FrameError("抽帧失败，可能是视频无有效画面")

    if progress_cb:
        progress_cb(f"抽到 {len(frames)} 帧候选（去重后）")
    return frames


# ---- 感知哈希：dHash（无需新依赖，PIL 足够）----
def _dhash(path, size: int = 8) -> int | None:
    """8x8 差分哈希 → 64 bit。对 JPEG 噪声不敏感，对翻页/换板敏感。"""
    try:
        from PIL import Image
        img = Image.open(path).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
        px = list(img.getdata())
        bits = 0
        for row in range(size):
            base = row * (size + 1)
            for col in range(size):
                bits = (bits << 1) | (1 if px[base + col] > px[base + col + 1] else 0)
        return bits
    except Exception:
        return None


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _sample_frames(frames: list[dict], maxn: int) -> list[dict]:
    """均匀采样到 maxn 帧（保证去重）。"""
    if len(frames) > maxn:
        step = len(frames) / maxn
        picked = [frames[int(i * step)] for i in range(maxn)]
    else:
        picked = frames
    seen, uniq = set(), []
    for fr in picked:
        if fr["path"] in seen:
            continue
        seen.add(fr["path"])
        uniq.append(fr)
    return uniq


def _vision_frame_budget(duration_sec: float) -> int:
    """候选帧预算：按时长自适应（约每 20 秒一个点位），上下都有钳制。"""
    base = max(1, config.VISION_MAX_FRAMES)
    cap = max(base, config.VISION_MAX_FRAMES_CAP)
    if not duration_sec or duration_sec <= 0:
        return base
    want = int(duration_sec / 60 * config.VISION_FRAMES_PER_MIN)
    return max(base, min(cap, want))


def _fmt_mmss(sec: float) -> str:
    sec = int(sec or 0)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _build_grid_image(cells: list[dict], out_path: str) -> None:
    """
    把若干帧拼成一张 cols x rows 网格图，每格左上角烙印 `#序号 mm:ss`。
    序号让模型可以用 `#1` 指代格子，避免它复述时间戳时看错数字。
    """
    from PIL import Image, ImageDraw, ImageFont

    cols, rows = config.VISION_GRID_COLS, config.VISION_GRID_ROWS
    cw, ch = config.VISION_CELL_W, config.VISION_CELL_H
    try:
        font = ImageFont.truetype("arial.ttf", max(18, ch // 12))
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(18, ch // 12))
        except Exception:
            font = ImageFont.load_default()

    canvas = Image.new("RGB", (cw * cols, ch * rows), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for i, fr in enumerate(cells[: cols * rows]):
        try:
            img = Image.open(fr["path"]).convert("RGB").resize(
                (cw, ch), Image.Resampling.LANCZOS)
        except Exception:
            continue
        x, y = (i % cols) * cw, (i // cols) * ch
        canvas.paste(img, (x, y))
        label = f"#{i + 1} {_fmt_mmss(fr.get('ts', 0))}"
        # 黑底 + 黄色字，保证任何画面上都清晰可读
        try:
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = 90, 20
        draw.rectangle([x, y, x + tw + 16, y + th + 14], fill=(0, 0, 0))
        draw.text((x + 8, y + 5), label, fill=(255, 235, 0), font=font)
    canvas.save(out_path, quality=config.VISION_GRID_QUALITY)


def describe_frames(frames: list[dict], progress_cb=None, task_id: int = 0,
                    duration_sec: float = 0, cache_key: str | None = None) -> list[dict]:
    """
    对帧做多模态描述。VISION_ENABLED=false 时直接跳过返回空。

    网格拼图模式（默认）：候选帧按 3x3 拼成网格图，每格左上角烙印 `#序号 mm:ss`，
    一次视觉调用看 9 帧（9 个候选点 + 时间线上下文），按 `#序号` 回填每格描述。
    相比逐帧调用：调用次数除以 9，且模型能看到相邻画面的变化，选图更准。

    cache_key（bvid_cid）命中时按时间戳复用上次的描述，只对缺失的网格调视觉 ——
    换风格重跑 / 失败重试不必重复付费。图片 URL 每次按当前任务目录重新映射。
    """
    if not config.VISION_ENABLED:
        return []

    base = config.VISION_BASE_URL or config.LLM_BASE_URL
    key = config.VISION_API_KEY or config.LLM_API_KEY
    model = config.VISION_MODEL or config.LLM_MODEL
    if not key:
        return []

    from openai import OpenAI
    client = OpenAI(base_url=base, api_key=key)

    picked = _sample_frames(frames, _vision_frame_budget(duration_sec))
    cached = load_vision_cache(cache_key)
    hit_n = sum(1 for fr in picked if cached.get(int(fr["ts"])))
    if hit_n and progress_cb:
        progress_cb(f"视觉缓存命中 {hit_n}/{len(picked)} 帧")

    # 预置每帧结果：命中缓存的先填上，img_url 指向本次任务的帧文件
    by_path = {fr["path"]: {"path": fr["path"], "ts": fr["ts"],
                            "desc": cached.get(int(fr["ts"]), ""),
                            "img_url": _frame_web_url(fr["path"])}
               for fr in picked}

    if not config.VISION_USE_GRID:
        # 旧逐帧模式：保留但不再默认启用
        for i, fr in enumerate(picked):
            if by_path[fr["path"]]["desc"]:
                continue
            if progress_cb:
                progress_cb(f"视觉理解 {i+1}/{len(picked)}")
            by_path[fr["path"]]["desc"] = _describe_one(client, model, fr["path"])
        _persist_vision(cache_key, by_path)
        return [by_path[fr["path"]] for fr in picked]

    # --- 网格拼图模式 ---
    per_grid = config.VISION_GRID_COLS * config.VISION_GRID_ROWS
    grids = [picked[i:i + per_grid] for i in range(0, len(picked), per_grid)]
    grid_dir = config.FRAMES_DIR / f"task_{task_id}"
    grid_dir.mkdir(parents=True, exist_ok=True)

    todo = [(gi, cells) for gi, cells in enumerate(grids)
            if not all(by_path[c["path"]]["desc"] for c in cells)]
    if not todo and progress_cb and grids:
        progress_cb("视觉全部命中缓存，跳过调用")

    for gi, cells in todo:
        if progress_cb:
            progress_cb(f"视觉理解 网格 {gi+1}/{len(grids)}（每格 {len(cells)} 帧）")
        grid_path = str(grid_dir / f"grid_{gi:03d}.jpg")
        try:
            _build_grid_image(cells, grid_path)
        except Exception:
            # 拼图失败降级为逐帧
            for fr in cells:
                if not by_path[fr["path"]]["desc"]:
                    by_path[fr["path"]]["desc"] = _describe_one(client, model, fr["path"])
            continue
        prompt = (
            f"下面是一张课程视频画面的网格拼图，共 {len(cells)} 格。"
            "每格左上角烙印了该画面的序号和时间（形如 #3 12:30），序号从左上角开始、按行排列。\n"
            "请逐格用一句话中文客观描述画面关键信息（是 PPT/板书/公式/图表/代码/还是人物；"
            "如有文字或公式尽量转述要点）。严格按如下格式逐行输出，不要遗漏任何一格：\n"
            "#序号 描述内容\n"
            "例如：#1 这是一页 PPT，标题是「克拉默法则」，给出系数行列式 D≠0 时方程组有唯一解的定理。\n"
            "只依据画面里真实存在的内容，不要编造。"
        )
        try:
            b64 = _img_b64(grid_path)
            text = _vision_chat(client, model, prompt, b64, max_tokens=90 * len(cells) + 120)
        except Exception as e:
            text = f"[网格视觉失败: {e}]"
        desc_map = _parse_grid_reply(text, [c.get("ts", 0) for c in cells])
        for fr in cells:
            if by_path[fr["path"]]["desc"]:
                continue  # 缓存已有的不覆盖
            d = desc_map.get(int(fr.get("ts", 0)))
            by_path[fr["path"]]["desc"] = d or (text[:120] if text else "(画面无描述)")

    _persist_vision(cache_key, by_path)
    return [by_path[fr["path"]] for fr in picked]


def _persist_vision(cache_key: str | None, by_path: dict) -> None:
    """把本次成功拿到的描述并入缓存（失败占位不写）。"""
    got = {int(v["ts"]): v["desc"] for v in by_path.values()
           if v.get("desc") and not str(v["desc"]).startswith("[")}
    save_vision_cache(cache_key, got)


# ---- 视觉描述缓存：{ts: desc}，与转写缓存同一套 key（bvid_cid）----
def _vision_cache_path(cache_key: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(cache_key))
    d = config.DATA_DIR / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"vision_{safe}.json"


def load_vision_cache(cache_key: str | None) -> dict[int, str]:
    if not cache_key:
        return {}
    try:
        raw = json.loads(_vision_cache_path(cache_key).read_text(encoding="utf-8"))
        return {int(k): str(v) for k, v in raw.items() if str(v).strip()}
    except Exception:
        return {}


def save_vision_cache(cache_key: str | None, desc_by_ts: dict[int, str]) -> None:
    if not cache_key or not desc_by_ts:
        return
    try:
        path = _vision_cache_path(cache_key)
        merged: dict[int, str] = {}
        if path.exists():
            merged = {int(k): str(v) for k, v in
                      json.loads(path.read_text(encoding="utf-8")).items()}
        merged.update({int(k): v for k, v in desc_by_ts.items() if v})
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({str(k): merged[k] for k in sorted(merged)},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


def _describe_one(client, model: str, path: str) -> str:
    prompt = ("这是一段课程视频的截图。请用一两句中文客观描述画面关键信息："
              "是 PPT/板书/公式/图表/代码/还是人物？如有文字或公式，尽量转述其要点。"
              "不要编造画面里没有的内容。")
    try:
        return _vision_chat(client, model, prompt, _img_b64(path), max_tokens=250)
    except Exception as e:
        return f"[视觉失败: {e}]"


def _vision_chat(client, model: str, text: str, b64: str, max_tokens: int = 300) -> str:
    """
    单次多模态调用（端点强制 SSE，用 stream 收 delta）。
    带指数退避重试：一次 5xx/超时不该让整格 9 帧的候选全部报废。
    """
    attempts = max(1, config.VISION_RETRY_ATTEMPTS)
    last: Exception | None = None
    for i in range(attempts):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
                max_tokens=max_tokens,
                stream=True,
            )
            parts = []
            for chunk in stream:
                try:
                    d = chunk.choices[0].delta
                    if getattr(d, "content", None):
                        parts.append(d.content)
                except (IndexError, AttributeError):
                    continue
            out = "".join(parts).strip()
            if not out:
                raise RuntimeError("视觉返回空内容")
            return out
        except Exception as e:
            last = e
            if i + 1 >= attempts:
                break
            time.sleep(max(0, config.VISION_RETRY_BACKOFF_SEC) * (2 ** i))
    raise last if last else RuntimeError("视觉调用失败")


_GRID_IDX_RE = re.compile(r"^#\s*(\d{1,2})\s*[.、:：,，]?\s*(.+)$")
_GRID_TS_RE = re.compile(r"[【\[]?\s*(\d{1,3}):(\d{2})\s*[】\]]?\s*[:：]?\s*(.+)")


def _parse_grid_reply(text: str, ts_list: list[float]) -> dict[int, str]:
    """
    把模型逐行输出解析成 {秒: 描述}。三层容错：
    1) `#n 描述`      —— 主路径（序号最不容易看错）
    2) `mm:ss 描述`   —— 老格式
    3) 无标记则按行顺序对齐
    """
    out: dict[int, str] = {}
    if not text:
        return out
    valid = [int(t) for t in ts_list if t is not None]

    for line in text.splitlines():
        line = line.strip().lstrip("-•·* ").strip()
        m = _GRID_IDX_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(valid):
            desc = re.sub(r"^\d{1,3}:\d{2}\s*", "", m.group(2)).strip()
            if desc:
                out[valid[idx]] = desc
    if out:
        return out

    for line in text.splitlines():
        line = line.strip().lstrip("-•·* ").strip()
        if not line:
            continue
        m = _GRID_TS_RE.match(line)
        if m:
            sec = int(m.group(1)) * 60 + int(m.group(2))
            desc = m.group(3).strip()
            if desc:
                out[sec] = desc
    if out:
        return out

    rows = [ln.strip().lstrip("-•·* ").strip() for ln in text.splitlines() if ln.strip()]
    for sec, ln in zip(valid, rows):
        out[sec] = ln
    return out


def _frame_web_url(path: str) -> str:
    """把帧的绝对路径转成 web 静态路由相对路径 /frames/<相对 FRAMES_DIR>。"""
    try:
        rel = Path(path).resolve().relative_to(config.FRAMES_DIR.resolve())
        return "/frames/" + rel.as_posix()
    except Exception:
        return ""


def _img_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
