"""
LLM 总结：转写文字 → 结构化 Markdown 笔记。
对齐 BiliNote 的两阶段截图法：
  阶段1：LLM 只在需要截图处输出占位标记 `*Screenshot-[mm:ss]`（不给图、不让它编 URL）
  阶段2：扫标记 → 用 ffmpeg 在该秒精确截帧 → 替换成图片（见 worker 调用 fill_screenshots）
同时支持 9 种笔记风格、章节标题带 *Content-[mm:ss] 时间戳、长视频分段合并。
"""
from pathlib import Path
import re
from .. import config


class SummaryError(Exception):
    pass


def _fmt_ts(sec: float) -> str:
    sec = int(sec or 0)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _transcript_text(segments: list[dict]) -> str:
    lines = []
    for s in segments:
        lines.append(f"[{_fmt_ts(s['start'])}] {s['text']}")
    return "\n".join(lines)


def _client():
    from openai import OpenAI
    if not config.LLM_API_KEY:
        raise SummaryError("未配置 LLM_API_KEY，无法总结")
    return OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)


def _chat(client, system: str, user: str, max_tokens: int = 4000) -> str:
    # 该 LLM 端点强制返回 SSE 流式，必须用 stream=True 收 delta 再拼接
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.3,
        max_tokens=max_tokens,
        stream=True,
    )
    parts = []
    for chunk in resp:
        try:
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                parts.append(delta.content)
        except (IndexError, AttributeError):
            continue
    return "".join(parts).strip()


def _chunk_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks, buf, size = [], [], 0
    for line in text.splitlines():
        if size + len(line) + 1 > limit and buf:
            chunks.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += len(line) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks


# ---- 9 种笔记风格（借鉴 BiliNote，针对中文场景精简）----
STYLE_MAP = {
    "精简": "仅记录最重要内容，简洁明了，要点为主。",
    "详细": "尽可能完整记录视频内容，保留细节、例子与推理过程。",
    "学术": "结构严谨、用词正式，适合讲座/报告/科研，术语准确，引用理论。",
    "教程": "教程优化：步骤清晰，逐步记录关键操作、命令、关键点与结论。",
    "小红书": "活泼生动，用流行表达与 emoji，标题有吸引力，口语化。",
    "生活向": "日常化、有情感色彩，像 vlog 旁白。",
    "任务导向": "侧重待办、操作清单、目标与步骤。",
    "商业": "商业逻辑清晰、表达精准，适合财经/市场分析。",
    "会议纪要": "结构标准、语言正式，突出结论、决议、待办。",
    "网课": "课程学习笔记：抓核心概念/定义、公式、推导、例题、易错点、结论，条理清晰。",
}


def _style_desc(style: str) -> str:
    return STYLE_MAP.get(style, STYLE_MAP["网课"])


def summarize(segments: list[dict], vision: list[dict], meta: dict,
              progress_cb=None) -> str:
    client = _client()
    style = meta.get("style") or config.NOTE_STYLE
    want_screenshot = meta.get("screenshot", True)
    want_link = meta.get("link", True)
    want_toc = meta.get("toc", False)
    want_summary = meta.get("summary", True)
    transcript = _transcript_text(segments)

    rules = [
        "1) 开头用二级标题写视频主题，下面一句话概括。",
        "2) 按内容逻辑划分章节（## 章节名），每节用要点列表提炼知识。",
        "3) 务必提炼：核心概念/定义、公式、推导步骤、例题与解法、易错点、结论。",
        "4) 语言精简准确，不要空洞套话。数学公式用 LaTeX（$...$ 或 $$...$$）。",
        "5) 不要编造视频里没有的内容。",
        f"6) 笔记风格：{_style_desc(style)}",
    ]
    if want_link:
        rules.append(
            "7) 【章节时间戳】每个二级标题(##)后，用 `*Content-[mm:ss]` 标注该章节开始的视频时间，"
            "例如 `## 一、缘起 *Content-[00:15]`。")
    if want_screenshot:
        # 按视频时长算"目标截图数"下限（约每 DENSE 分钟一张），保证不稀疏
        dur = meta.get("duration_sec", 0) or 0
        target = max(4, int(dur // config.SCREENSHOT_EVERY_SEC) + 1) if dur else 6
        # 同时不超过可用候选帧数
        avail = len([v for v in vision if v.get("img_url")])
        if avail:
            target = min(target, avail)
        rules.append(
            f"8) 【截图占位 · 重要】这是一节网课，画面里的 PPT/板书/公式/图表/代码/推导步骤都很重要。"
            f"请在正文中**充分插入截图**：每当讲到一处新的板书/PPT 页/公式/图表/例题步骤时，"
            f"就在该处末尾插入截图占位标记 `*Screenshot-[mm:ss]`（mm:ss 是画面出现的视频时间）。"
            f"本节时长约 {int(dur//60)} 分钟，**至少插入 {target} 处截图**，宁多勿少（同一页不要重复）。"
            f"只写标记，不要自己编图片 URL。")
        if vision:
            # 提供"候选画面清单（时间戳 + 画面描述）"，让 LLM 有的放矢地选时间点。
            # 网格拼图模式下这些描述是按格回填的，天然带时间线顺序。
            vlines = []
            for v in vision:
                if not v.get("img_url"):
                    continue
                ts = _fmt_ts(v["ts"]) if v.get("ts") is not None else "?"
                d = (v.get('desc', '') or '').strip()
                if d:
                    vlines.append(f"- {ts}：{d[:70]}")
            if vlines:
                rules.append(
                    "8b) 【候选截图时间点（带画面描述）】以下是本次抽取的带时间戳画面清单，"
                    "请优先从这些时间点里挑选最有代表性的插入截图（同一画面不要重复）：\n"
                    + "\n".join(vlines[:60]))
    if want_toc:
        rules.append("9) 【目录】在正文开头根据 ## 标题生成一个标题目录。")
    if want_summary:
        rules.append("10) 【AI 总结】在笔记最后加 `## AI 总结`，用一段话概括全片。")

    system = (
        "你是一位专业的视频笔记助手，把视频转录内容整理成清晰、有条理、信息丰富的 Markdown 笔记。\n"
        "输出只能是 Markdown 正文本身，不要用 ``` 代码块包裹整体。\n"
        "注意：编号标题不要写成有序列表（避免 `1. **内容**` 被误解析），用 `## 1. 内容` 形式。\n"
        "规则（每条都要严格遵守）：\n" + "\n".join(rules)
    )

    chunks = _chunk_text(transcript, config.LLM_MAX_CHARS)
    title = meta.get("title", "课程视频")
    dur = meta.get("duration_sec", 0)

    def _user_block(seg_text: str, extra: str = "") -> str:
        u = f"视频标题：{title}\n总时长：{_fmt_ts(dur)}\n\n"
        u += f"【逐字稿（带时间戳）】\n{seg_text}\n\n"
        if extra:
            u += extra + "\n\n"
        u += "请按要求生成结构化 Markdown 笔记。"
        return u

    if len(chunks) == 1:
        if progress_cb:
            progress_cb("LLM 生成笔记…")
        body = _chat(client, system, _user_block(transcript))
    else:
        partials = []
        for i, ch in enumerate(chunks):
            if progress_cb:
                progress_cb(f"长视频分段总结 {i+1}/{len(chunks)}")
            p = _chat(
                client,
                "这是一段长课程视频的逐字稿切片（带时间戳）。按要求提炼本段要点，"
                "保留 *Content-[mm:ss] 与 *Screenshot-[mm:ss] 标记，供后续合并。",
                _user_block(ch),
            )
            partials.append(p)
        if progress_cb:
            progress_cb("合并分段笔记…")
        merged = "\n\n---\n\n".join(partials)
        merge_user = (
            f"视频标题：{title}\n\n下面是同一视频的多段笔记，请合并去重成一份完整笔记：\n"
            "- 只合并去重，不发明新内容；保持 Markdown 层级\n"
            "- 保留所有 *Content-[mm:ss] 与 *Screenshot-[mm:ss] 标记\n"
            f"- 遵循以下规则：\n{system}\n\n【分段笔记】\n{merged}"
        )
        body = _chat(client, system, merge_user, max_tokens=6000)

    header = (
        f"# {title}\n\n"
        f"> 来源：{meta.get('url','')}  \n"
        f"> 时长：{_fmt_ts(dur)}　BVID：{meta.get('bvid','')}  \n"
        f"> 风格：{style}　生成时间：{meta.get('generated_at','')}\n\n"
    )
    doc = header + body

    # 规范化 Content 标记（去掉 LLM 常写的包裹星号，统一成 *Content-[mm:ss]）
    doc = re.sub(r"\*?Content-\s*\[?(\d{1,2}:\d{2})\]?\*?", r"*Content-[\1]", doc)

    # 视觉描述：截图模式下图片已内嵌正文，不再在末尾重复堆一遍画面说明；
    # 仅在没开截图/没插入截图时，作为"关键画面说明"附在末尾供参考。
    has_marks = bool(_SCREENSHOT_RE.search(doc))
    if vision and not (want_screenshot and has_marks):
        doc += "\n\n---\n\n## 附：关键画面说明\n\n"
        for v in vision:
            ts = _fmt_ts(v["ts"]) if v.get("ts") is not None else "画面"
            doc += f"- **{ts}**　{v.get('desc','')}\n"
    return doc


# ---- 两阶段截图：标记 -> 精确截帧 -> 替换 ----
_SCREENSHOT_RE = re.compile(r"\*?Screenshot-\s*\[?(\d{1,2}):(\d{2})\]?\*?")


def extract_screenshot_marks(doc: str) -> list[tuple[str, int]]:
    """返回 [(原始标记文本, 秒数), ...]"""
    out = []
    for m in _SCREENSHOT_RE.finditer(doc):
        sec = int(m.group(1)) * 60 + int(m.group(2))
        out.append((m.group(0), sec))
    return out


def fill_screenshots(doc: str, video_path: str, task_id: int, progress_cb=None) -> str:
    """
    扫出所有 Screenshot 标记，逐一定点截帧并替换成图片。
    同秒只截一次。截帧失败则删掉标记（不破坏正文）。
    """
    from . import frames as frames_mod
    marks = extract_screenshot_marks(doc)
    if not marks:
        return doc

    cache = {}  # sec -> img_url
    # 相邻时间点太近（<5秒）的截图去重，并限制总数，防止 LLM 失控刷屏
    marks_clean = []
    last_sec = -999
    for marker, sec in marks:
        if sec - last_sec < config.SCREENSHOT_MIN_GAP_SEC:
            marks_clean.append((marker, None))  # 标记为跳过
            continue
        marks_clean.append((marker, sec))
        last_sec = sec
    # 超过上限的截断
    kept = [m for m in marks_clean if m[1] is not None][: config.MAX_SCREENSHOTS]
    kept_secs = {m[0] for m in kept}

    total = len(kept)
    done_i = 0
    for marker, sec in marks_clean:
        if sec is None or marker not in kept_secs:
            # 跳过或超限：删掉标记
            doc = doc.replace(marker, "", 1)
            continue
        done_i += 1
        if progress_cb:
            progress_cb(f"生成截图 {done_i}/{total}（{_fmt_ts(sec)}）")
        try:
            if sec not in cache:
                cache[sec] = frames_mod.grab_frame_at(video_path, task_id, sec)
            url = cache[sec]
            img = f"![{_fmt_ts(sec)}]({url})\n\n（{_fmt_ts(sec)} 画面）"
            doc = doc.replace(marker, img, 1)
        except Exception:
            doc = doc.replace(marker, "", 1)
    return doc


def save_markdown(task_id: int, content: str) -> str:
    path = config.OUTPUT_DIR / f"task_{task_id}.md"
    path.write_text(content, encoding="utf-8")
    return str(path)
