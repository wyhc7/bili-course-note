"""
关键帧抽取 + 多模态视觉描述（可选）。
网课优化：优先取画面变化大的帧（翻页/写板书），而不是等间隔盲抽。
视觉模型走 OpenAI 兼容多模态接口，不占本地算力。
"""
import base64
import re
import subprocess
from pathlib import Path

from .. import config


class FrameError(Exception):
    pass


def grab_frame_at(video_path: str, task_id: int, ts_sec: float) -> str:
    """
    在指定秒数精确截取一帧（BiliNote 两阶段截图法）。
    返回该帧的 web 相对 URL（/frames/task_x/shot_<秒>.jpg）。
    """
    out_dir = config.FRAMES_DIR / f"task_{task_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"shot_{int(ts_sec)}.jpg"
    out = out_dir / fname
    if not out.exists():
        cmd = [
            "ffmpeg", "-y", "-ss", str(max(0, ts_sec)),
            "-i", video_path, "-frames:v", "1",
            "-vf", "scale=960:-1", "-q:v", "3",
            str(out), "-loglevel", "error",
        ]
        p = subprocess.run(cmd, capture_output=True, timeout=120)
        if p.returncode != 0 or not out.exists():
            raise FrameError(f"截帧失败 ts={ts_sec}: {(p.stderr or b'')[-200:]}")
    return _frame_web_url(str(out))


def extract_frames(video_path: str, task_id: int, duration_sec: float,
                   progress_cb=None) -> list[dict]:
    """
    抽候选帧（参考 BiliNote：密集等间隔抽 + 相邻去重）。
    网课以 PPT/板书为主，画面变化才是关键，所以：
    1) 按 DENSE_INTERVAL 秒密集抽帧（默认 15s，2核4G 兼顾速度）
    2) 相邻帧 MD5 相同的丢弃（没翻页/没变化就跳过）
    返回 [{path, ts}]，ts 精确到秒。
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

    interval = config.DENSE_FRAME_INTERVAL_SEC
    if duration_sec > 0:
        # 逐帧抽（带精确时间戳），用 fps 过滤器一次性导出
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"fps=1/{interval},scale=960:-1",
            "-q:v", "3",
            str(out_dir / "f_%05d.jpg"),
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=3600)
        except Exception:
            pass

        prev_hash = None
        idx = 0
        for p in sorted(out_dir.glob("f_*.jpg")):
            ts = idx * interval
            idx += 1
            # 相邻去重：与上一张相同则删掉
            if config.FRAME_DEDUPE:
                h = _md5(p)
                if h == prev_hash:
                    p.unlink(missing_ok=True)
                    continue
                prev_hash = h
            frames.append({"path": str(p), "ts": ts})

    if not frames:
        raise FrameError("抽帧失败，可能是视频无有效画面")

    if progress_cb:
        progress_cb(f"抽到 {len(frames)} 帧候选（去重后）")
    return frames


def _md5(path) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _fmt_mmss(sec: float) -> str:
    sec = int(sec or 0)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _build_grid_image(cells: list[dict], out_path: str) -> None:
    """
    把若干帧拼成一张 cols x rows 网格图，并在每格左上角烙印 mm:ss 时间戳
    （参考 BiliNote VideoReader.concat_images）。
    只允许传图像库导入失败时降级为单帧。
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
        label = _fmt_mmss(fr.get("ts", 0))
        # 半透明黑底 + 黄色时间戳，保证任何画面上都清晰可读
        try:
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = 40, 20
        draw.rectangle([x, y, x + tw + 12, y + th + 12], fill=(0, 0, 0))
        draw.text((x + 6, y + 4), label, fill=(255, 235, 0), font=font)
    canvas.save(out_path, quality=config.VISION_GRID_QUALITY)


def describe_frames(frames: list[dict], progress_cb=None, task_id: int = 0) -> list[dict]:
    """
    对帧做多模态描述。VISION_ENABLED=false 时直接跳过返回空。

    网格拼图模式（默认，参考 BiliNote）：把候选帧按 3x3 拼成网格图，每格左上角烙印 mm:ss，
    一次视觉调用看整张网格（9 个候选点+时间线上下文），按时间戳返回每格描述。
    相比逐帧调用：调用次数除以 9，且模型能看到相邻画面的变化，选图更准。
    每帧结果仍带 web 可访问的相对 URL（供笔记嵌入截图）。
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

    picked = _sample_frames(frames, max(1, config.VISION_MAX_FRAMES))
    # 预置每帧结果（默认无描述），描述按时间戳回填
    by_path = {fr["path"]: {"path": fr["path"], "ts": fr["ts"],
                            "desc": "", "img_url": _frame_web_url(fr["path"])}
               for fr in picked}

    if not config.VISION_USE_GRID:
        # 旧逐帧模式：保留但不再默认启用
        for i, fr in enumerate(picked):
            if progress_cb:
                progress_cb(f"视觉理解 {i+1}/{len(picked)}")
            desc = _describe_one(client, model, fr["path"])
            by_path[fr["path"]]["desc"] = desc
        return [by_path[fr["path"]] for fr in picked]

    # --- 网格拼图模式 ---
    per_grid = config.VISION_GRID_COLS * config.VISION_GRID_ROWS
    grids = [picked[i:i + per_grid] for i in range(0, len(picked), per_grid)]
    grid_dir = config.FRAMES_DIR / f"task_{task_id}"
    grid_dir.mkdir(parents=True, exist_ok=True)

    for gi, cells in enumerate(grids):
        if progress_cb:
            progress_cb(f"视觉理解 网格 {gi+1}/{len(grids)}（每格 {len(cells)} 帧）")
        grid_path = str(grid_dir / f"grid_{gi:03d}.jpg")
        try:
            _build_grid_image(cells, grid_path)
        except Exception as e:
            # 拼图失败降级为逐帧
            for fr in cells:
                by_path[fr["path"]]["desc"] = _describe_one(client, model, fr["path"])
            continue
        ts_list = "、".join(_fmt_mmss(c.get("ts", 0)) for c in cells)
        prompt = (
            f"下面是一张课程视频画面的网格拼图，共 {len(cells)} 格，每格左上角标了该画面的视频时间（mm:ss）。"
            f"这些格的时间依次为：{ts_list}。\n"
            "请逐格用一句话中文客观描述画面关键信息（是 PPT/板书/公式/图表/代码/还是人物；"
            "如有文字或公式尽量转述要点）。严格按如下格式逐行输出，不要遗漏任何一格：\n"
            "【mm:ss】描述内容\n"
            "只依据画面里真实存在的内容，不要编造。"
        )
        try:
            b64 = _img_b64(grid_path)
            text = _vision_chat(client, model, prompt, b64, max_tokens=60 * len(cells) + 120)
        except Exception as e:
            text = f"[网格视觉失败: {e}]"
        desc_map = _parse_grid_reply(text, [c.get("ts", 0) for c in cells])
        for fr in cells:
            d = desc_map.get(int(fr.get("ts", 0)))
            by_path[fr["path"]]["desc"] = d or (text[:120] if text else "(画面无描述)")

    return [by_path[fr["path"]] for fr in picked]


def _describe_one(client, model: str, path: str) -> str:
    prompt = ("这是一段课程视频的截图。请用一两句中文客观描述画面关键信息："
              "是 PPT/板书/公式/图表/代码/还是人物？如有文字或公式，尽量转述其要点。"
              "不要编造画面里没有的内容。")
    try:
        return _vision_chat(client, model, prompt, _img_b64(path), max_tokens=250)
    except Exception as e:
        return f"[视觉失败: {e}]"


def _vision_chat(client, model: str, text: str, b64: str, max_tokens: int = 300) -> str:
    """单次多模态调用（端点强制 SSE，用 stream 收 delta）。"""
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
    return "".join(parts).strip()


def _parse_grid_reply(text: str, ts_list: list[float]) -> dict[int, str]:
    """把模型逐行【mm:ss】描述 解析成 {秒: 描述}，容错多种括号/写法。"""
    out: dict[int, str] = {}
    if not text:
        return out
    pat = re.compile(r"[【\[]?\s*(\d{1,3}):(\d{2})\s*[】\]]?\s*[:：]?\s*(.+)")
    for line in text.splitlines():
        line = line.strip().lstrip("-•·* ").strip()
        if not line:
            continue
        m = pat.match(line)
        if m:
            sec = int(m.group(1)) * 60 + int(m.group(2))
            desc = m.group(3).strip()
            if desc:
                out[sec] = desc
    # 若模型没用时间戳格式，退化为按行顺序对齐
    if not out:
        valid = [int(t) for t in ts_list if t is not None]
        rows = [ln.strip().lstrip("-•·* ").strip()
                for ln in text.splitlines() if ln.strip()]
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
