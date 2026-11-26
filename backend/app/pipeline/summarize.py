"""
LLM 总结：转写文字 → 结构化 Markdown 笔记。
对齐 BiliNote 的两阶段截图法：
  阶段1：LLM 只在需要截图处输出占位标记 `*Screenshot-[mm:ss]`（不给图、不让它编 URL）
  阶段2：扫标记 → 优先复用已抽的候选帧（吸附到最近时间点）→ 未覆盖才 ffmpeg 定点截帧
同时支持多种笔记风格、章节标题带 *Content-[mm:ss] 时间戳、长视频分段合并。
"""
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


def _out_tokens(system: str, user: str) -> int:
    """
    按输入体量估输出上限：笔记通常与逐字稿同量级，但不超过配置硬顶。
    固定 4000/6000 会让「详细」风格的长课写到一半被截断（还不报错）。
    """
    cap = max(512, config.LLM_MAX_OUTPUT_TOKENS)
    want = int((len(system) + len(user)) * 0.9)
    return max(1200, min(cap, want))


def _chat(client, system: str, user: str, max_tokens: int | None = None) -> str:
    # 该 LLM 端点强制返回 SSE 流式，必须用 stream=True 收 delta 再拼接
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.3,
        max_tokens=max_tokens or _out_tokens(system, user),
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


_TS_LINE_RE = re.compile(r"^\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]")


def _chunk_span(chunk: str) -> tuple[int, int]:
    """从切片的 `[mm:ss]` 行首时间戳里取该片段的起止秒数。"""
    t0 = t1 = None
    for line in chunk.splitlines():
        m = _TS_LINE_RE.match(line.strip())
        if not m:
            continue
        a, b, c = m.groups()
        t = (int(a) * 3600 + int(b) * 60 + int(c)) if c else (int(a) * 60 + int(b))
        if t0 is None:
            t0 = t
        t1 = t
    if t0 is None:
        return (0, 10 ** 9)
    return (t0, t1)


# ---- 笔记风格（借鉴 BiliNote，针对中文场景精简）----
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


def _vlines(vision: list[dict] | None, t0: int = 0, t1: int = 10 ** 9,
            limit: int = 60) -> list[str]:
    """候选画面清单（时间 + 描述），按时间窗过滤，视觉失败的占位直接丢掉。"""
    out = []
    for v in (vision or []):
        if not v.get("img_url") or v.get("ts") is None:
            continue
        ts = int(v["ts"])
        if ts < t0 or ts > t1:
            continue
        d = (v.get("desc") or "").strip()
        if not d or d.startswith("[") or d.startswith("("):
            continue
        out.append(f"- {_fmt_ts(ts)}：{d[:70]}")
        if len(out) >= limit:
            break
    return out


def _target_count(span_sec: int) -> int:
    """该片段建议的最少截图数（约每 SCREENSHOT_EVERY_SEC 一张，有上下限）。"""
    span = max(1, int(span_sec))
    n = int(span // max(1, config.SCREENSHOT_EVERY_SEC)) + 1
    return max(1, min(config.MAX_SCREENSHOTS, n))


def _build_rules(style: str, want_link: bool, want_screenshot: bool,
                 want_toc: bool, want_summary: bool, dur: int,
                 target: int | None, vlines: list[str] | None) -> list[str]:
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
        head = (
            "8) 【截图占位 · 重要】这是一节网课，画面里的 PPT/板书/公式/图表/代码/推导步骤都很重要。"
            "请在正文中**充分插入截图**：每当讲到一处新的板书/PPT 页/公式/图表/例题步骤时，"
            "就在该处末尾插入截图占位标记 `*Screenshot-[mm:ss]`（mm:ss 是画面出现的视频时间）。"
        )
        if target:
            head += (f"本段时长约 {max(1, int(dur // 60))} 分钟，**至少插入 {target} 处截图**，"
                     "宁多勿少（同一页不要重复）。")
        head += "只写标记，不要自己编图片 URL。"
        rules.append(head)
        if vlines:
            rules.append(
                "8b) 【本段可用的画面时间点（带画面描述）】**只能**从下面这些时间点里挑，"
                "不要自己编时间；优先选信息量最大的 PPT/板书/公式页，同一画面不要重复：\n"
                + "\n".join(vlines))
    if want_toc:
        rules.append("9) 【目录】在正文开头根据 ## 标题生成一个标题目录。")
    if want_summary:
        rules.append("10) 【AI 总结】在笔记最后加 `## AI 总结`，用一段话概括全片。")
    return rules


_SYS_HEAD = (
    "你是一位专业的视频笔记助手，把视频转录内容整理成清晰、有条理、信息丰富的 Markdown 笔记。\n"
    "输出只能是 Markdown 正文本身，不要用 ``` 代码块包裹整体。\n"
    "注意：编号标题不要写成有序列表（避免 `1. **内容**` 被误解析），用 `## 1. 内容` 形式。\n"
    "规则（每条都要严格遵守）：\n"
)


def _system_prompt(rules: list[str]) -> str:
    return _SYS_HEAD + "\n".join(rules)


def summarize(segments: list[dict], vision: list[dict], meta: dict,
              progress_cb=None) -> str:
    client = _client()
    style = meta.get("style") or config.NOTE_STYLE
    want_screenshot = meta.get("screenshot", True)
    want_link = meta.get("link", True)
    want_toc = meta.get("toc", False)
    want_summary = meta.get("summary", True)
    transcript = _transcript_text(segments)

    chunks = _chunk_text(transcript, config.LLM_MAX_CHARS)
    title = meta.get("title", "课程视频")
    dur = meta.get("duration_sec", 0) or 0

    def _user_block(seg_text: str) -> str:
        return (f"视频标题：{title}\n总时长：{_fmt_ts(dur)}\n\n"
                f"【逐字稿（带时间戳）】\n{seg_text}\n\n"
                "请按要求生成结构化 Markdown 笔记。")

    def _rules_for(t0: int, t1: int, with_candidates: bool = True):
        span = max(1, (t1 - t0) if t1 < 10 ** 9 else (dur or 600))
        return _build_rules(
            style, want_link, want_screenshot, want_toc, want_summary, span,
            _target_count(span) if want_screenshot else None,
            _vlines(vision, t0 - 30, t1 + 30) if (with_candidates and want_screenshot) else None,
        )

    if len(chunks) == 1:
        if progress_cb:
            progress_cb("LLM 生成笔记…")
        body = _chat(client, _system_prompt(_rules_for(0, 10 ** 9)), _user_block(transcript))
    else:
        partials = []
        for i, ch in enumerate(chunks):
            if progress_cb:
                progress_cb(f"长视频分段总结 {i+1}/{len(chunks)}")
            t0, t1 = _chunk_span(ch)
            # 关键：每段只给本段时间窗内的候选画面，避免第 3 段挑到第 1 段的时间点
            system = _system_prompt(_rules_for(t0, t1)) + (
                "\n这是一段长课程视频的逐字稿切片（带时间戳），只总结本段内容，不要跨度到本段之外；"
                "保留 *Content-[mm:ss] 与 *Screenshot-[mm:ss] 标记，供后续合并。")
            partials.append(_chat(client, system, _user_block(ch)))
        if progress_cb:
            progress_cb("合并分段笔记…")
        merged = "\n\n---\n\n".join(partials)
        merge_rules = _build_rules(
            style, want_link, want_screenshot, want_toc, want_summary, dur, None, None)
        merge_user = (
            f"视频标题：{title}\n\n下面是同一视频的多段笔记，请合并去重成一份完整笔记：\n"
            "- 只合并去重，不发明新内容；保持 Markdown 层级\n"
            "- 保留所有 *Content-[mm:ss] 与 *Screenshot-[mm:ss] 标记\n"
            f"- 遵循以下规则：\n{_system_prompt(merge_rules)}\n\n【分段笔记】\n{merged}"
        )
        body = _chat(client, _system_prompt(merge_rules), merge_user)

    header = (
        f"# {title}\n\n"
        f"> 来源：{meta.get('url','')}  \n"
        f"> 时长：{_fmt_ts(dur)}　BVID：{meta.get('bvid','')}  \n"
        f"> 风格：{style}　生成时间：{meta.get('generated_at','')}\n\n"
    )
    doc = header + body

    # 规范化 Content 标记（去掉 LLM 常写的包裹星号，统一成 *Content-[mm:ss] / [hh:mm:ss]）
    doc = _CONTENT_RE.sub(_norm_content, doc)

    # 视觉描述：截图模式下图片已内嵌正文，不再在末尾重复堆一遍画面说明；
    # 仅在没开截图/没插入截图时，作为"关键画面说明"附在末尾供参考。
    has_marks = bool(_SCREENSHOT_RE.search(doc))
    if vision and not (want_screenshot and has_marks):
        doc += "\n\n---\n\n## 附：关键画面说明\n\n"
        for v in vision:
            desc = (v.get("desc") or "").strip()
            if not desc or desc.startswith("["):
                continue
            ts = _fmt_ts(v["ts"]) if v.get("ts") is not None else "画面"
            doc += f"- **{ts}**　{desc}\n"
    return doc


# ---- 两阶段截图：标记 -> 吸附/精确截帧 -> 替换 ----
# 同时兼容 mm:ss 与 hh:mm:ss（超过 1 小时的课，模型会照抄转写里的 hh:mm:ss；
# 旧正则只吃到 "01:02" 就把 62 分钟当成 62 秒，后半程截图会全部堆到开头）
_SCREENSHOT_RE = re.compile(r"\*?Screenshot-\s*\[?(\d{1,3}):(\d{2})(?::(\d{2}))?\]?\*?")
_CONTENT_RE = re.compile(r"\*?Content-\s*\[?(\d{1,3}):(\d{2})(?::(\d{2}))?\]?\*?")


def _groups_to_sec(groups) -> int:
    a, b, c = groups
    return (int(a) * 3600 + int(b) * 60 + int(c)) if c else (int(a) * 60 + int(b))


def _norm_content(m: re.Match) -> str:
    return f"*Content-[{_fmt_ts(_groups_to_sec(m.groups()))}]"


def extract_screenshot_marks(doc: str) -> list[tuple[str, int]]:
    """返回 [(原始标记文本, 秒数), ...]"""
    out = []
    for m in _SCREENSHOT_RE.finditer(doc):
        out.append((m.group(0), _groups_to_sec(m.groups())))
    return out


def _snap(cand: list[dict], sec: int) -> dict | None:
    """把标记秒数吸附到最近的候选帧（容差 SNAP_TOLERANCE_SEC 内）。"""
    if not cand:
        return None
    lo, hi = 0, len(cand) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if int(cand[mid]["ts"]) < sec:
            lo = mid + 1
        else:
            hi = mid - 1
    best = None
    for j in (lo - 1, lo):
        if 0 <= j < len(cand):
            d = abs(int(cand[j]["ts"]) - sec)
            if best is None or d < best[0]:
                best = (d, cand[j])
    if best and best[0] <= config.SNAP_TOLERANCE_SEC:
        return best[1]
    return None


def fill_screenshots(doc: str, video_path: str, task_id: int,
                     vision: list[dict] | None = None,
                     progress_cb=None) -> tuple[str, set[str]]:
    """
    扫出所有 Screenshot 标记，替换成图片。返回 (新正文, 被引用的图片 URL 集合)。

    与旧实现的差别：
      - 标记先吸附到最近的候选帧（容差内），直接复用已抽好的帧文件 —— 截图画面
        就是"当初给模型看的那一帧"，且省掉一次 ffmpeg
      - 只有候选没覆盖到的秒数才走 grab_frame_at 定点截帧
      - 同一秒只截一次；相邻过近的合并；超上限截断
    """
    marks = extract_screenshot_marks(doc)
    if not marks:
        return doc, set()

    cand = sorted(
        [v for v in (vision or []) if v.get("img_url") and v.get("ts") is not None],
        key=lambda v: int(v["ts"]),
    )

    # 1) 吸附
    plan: list[list] = []
    for marker, sec in marks:
        hit = _snap(cand, sec)
        plan.append([marker, int(hit["ts"]) if hit else sec, hit])

    # 2) 时间过近去重 + 总量上限
    keep_idx, last = [], -10 ** 9
    for i, item in enumerate(plan):
        if item[1] - last < config.SCREENSHOT_MIN_GAP_SEC:
            continue
        keep_idx.append(i)
        last = item[1]
    keep_idx = keep_idx[: config.MAX_SCREENSHOTS]
    keep_set = set(keep_idx)

    from . import frames as frames_mod

    total, done, cache = len(keep_set), 0, {}
    for i, (marker, eff, hit) in enumerate(plan):
        if i not in keep_set:
            doc = doc.replace(marker, "", 1)  # 跳过或超限：删掉标记，不破坏正文
            continue
        done += 1
        if progress_cb:
            progress_cb(f"生成截图 {done}/{total}（{_fmt_ts(eff)}）")
        try:
            if hit:
                url = hit["img_url"]
            else:
                if eff not in cache:
                    cache[eff] = frames_mod.grab_frame_at(video_path, task_id, eff)
                url = cache[eff]
            doc = doc.replace(
                marker, f"![{_fmt_ts(eff)}]({url})\n\n（{_fmt_ts(eff)} 画面）", 1)
        except Exception:
            doc = doc.replace(marker, "", 1)

    used = set(re.findall(r"!\[[^\]]*\]\((/frames/[^)]+)\)", doc))
    return doc, used


def save_markdown(task_id: int, content: str) -> str:
    path = config.OUTPUT_DIR / f"task_{task_id}.md"
    path.write_text(content, encoding="utf-8")
    return str(path)
