"""
LLM 总结：转写文字 → 结构化 Markdown 笔记。
对齐 BiliNote 的两阶段截图法：
  阶段1：LLM 只在需要截图处输出占位标记 `*Screenshot-[mm:ss]`（不给图、不让它编 URL）
  阶段2：扫标记 → 优先复用已抽的候选帧（吸附到最近时间点）→ 未覆盖才 ffmpeg 定点截帧
同时支持多种笔记风格、章节标题带 *Content-[mm:ss] 时间戳、长视频分段合并。
"""
import re
import time

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
    """
    调 LLM 收流式输出，带重试。

    为什么必须重试：2 小时课要分段调 8 次 + 合并 1 次，旧实现一次超时/429/断流
    就让整个任务失败，而且失败后还要从零重下视频。单次抖动 3% 累积到 9 次约 24%。

    最后一次仍失败时，若已收到足够内容（LLM_PARTIAL_MIN_CHARS），接受这份半成品
    并在文末标注 —— 比整篇笔记丢掉强，也避免用户以为生成完整。
    """
    attempts = max(1, config.LLM_RETRY_ATTEMPTS)
    best = ""
    last: Exception | None = None

    for i in range(attempts):
        parts: list[str] = []
        try:
            # 该 LLM 端点强制返回 SSE 流式，必须用 stream=True 收 delta 再拼接
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.3,
                max_tokens=max_tokens or _out_tokens(system, user),
                stream=True,
            )
            for chunk in resp:
                try:
                    delta = chunk.choices[0].delta
                    if getattr(delta, "content", None):
                        parts.append(delta.content)
                except (IndexError, AttributeError):
                    continue
            text = "".join(parts).strip()
            if text:
                return text
            raise SummaryError("LLM 返回空内容")
        except Exception as e:
            last = e
            got = "".join(parts).strip()
            if len(got) > len(best):
                best = got  # 断流前收到的内容，留作兜底
            if i + 1 >= attempts:
                break
            wait = config.LLM_RETRY_BACKOFF_SEC * (2 ** i)
            time.sleep(max(0.0, wait))

    if len(best) >= max(1, config.LLM_PARTIAL_MIN_CHARS):
        return (best.rstrip()
                + "\n\n> ⚠️ 本次生成因网络/接口中断提前结束，"
                  "以上为已生成部分。可点「重跑」重新生成完整笔记。")

    raise SummaryError(f"LLM 调用失败（已尝试 {attempts} 次）：{last}")


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
            limit: int = 80) -> list[str]:
    """
    候选画面清单（编号 S1..Sn + 时间 + 描述），按时间窗过滤，视觉失败的占位直接丢掉。

    编号 = vision 原始下标 + 1，跨分段稳定。让模型引用编号而不是时间，是为了根治
    "模型把时间写偏一位 → 吸附到隔壁那一页 → 图文对不上"：编号直接定位到那一帧，
    中间不再有任何按秒数猜的环节。
    """
    out = []
    for i, v in enumerate(vision or []):
        if not v.get("img_url") or v.get("ts") is None:
            continue
        ts = int(v["ts"])
        if ts < t0 or ts > t1:
            continue
        d = (v.get("desc") or "").strip()
        if not d or d.startswith("[") or d.startswith("("):
            continue
        out.append(f"- S{i + 1}　{_fmt_ts(ts)}：{d[:90]}")
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
        if vlines:
            head = (
                "8) 【截图占位 · 重要】这是一节网课，画面里的 PPT/板书/公式/图表/代码/推导步骤都很重要。"
                "请在正文中**充分插入截图**：讲到一处新的板书/PPT 页/公式/图表/例题步骤时，"
                "就在描述该画面的那段文字末尾、**单独一行**插入 `*Screenshot-[S编号]`，"
                "例如 `*Screenshot-[S7]`。编号只能从 8b) 的清单里原样照抄，**不要自己写时间**。"
                "标记不要写进 `##` 标题行，也不要放在与该画面无关的段落里。"
            )
        else:
            head = (
                "8) 【截图占位 · 重要】这是一节网课，画面里的 PPT/板书/公式/图表/代码/推导步骤都很重要。"
                "请在正文中**充分插入截图**：讲到一处新的板书/PPT 页/公式/图表/例题步骤时，"
                "就在该处末尾、单独一行插入 `*Screenshot-[mm:ss]`（mm:ss 是画面出现的视频时间）。"
            )
        if target:
            head += (f"本段时长约 {max(1, int(dur // 60))} 分钟，**至少插入 {target} 处截图**，"
                     "宁多勿少（同一页不要重复）。")
        head += "只写标记，不要自己编图片 URL。"
        rules.append(head)
        if vlines:
            rules.append(
                "8b) 【本段可用画面（编号 + 时间 + 描述）】**只能**从下面这些编号里选，"
                "编号原样照抄（S 后面是数字，不要改成别的编号）；优先选信息量最大的 PPT/板书/公式页，"
                "同一个编号只引用一次：\n" + "\n".join(vlines))
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
        # 合并阶段不给候选清单：既不让它新写时间标记（没有清单撑着，写出来只能靠吸附猜），
        # 也不让它改写已有标记 —— 分段里已经是 `*Screenshot-[S12]`，原样搬过来即可。
        merge_rules = _build_rules(
            style, want_link, False, want_toc, want_summary, dur, None, None)
        merge_user = (
            f"视频标题：{title}\n\n下面是同一视频的多段笔记，请合并去重成一份完整笔记：\n"
            "- 只合并去重，不发明新内容；保持 Markdown 层级\n"
            "- **原样保留**所有 `*Content-[...]` 与 `*Screenshot-[...]` 标记，"
            "不要新增标记、不要改编号、不要把标记挪到别的段落\n"
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
    # 模型常把截图标记塞在标题行末（`## 一、xxx *Screenshot-[S3]`），替换后会变成
    # "标题配图"，看起来就是图与章节不对应。统一挪到标题的下一行。
    doc = _HEADING_MARK_RE.sub(r"\1\n\n\2", doc)

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


# ---- 两阶段截图：标记 -> 定位到候选帧 -> 替换 ----
# 标记有两种写法：
#   *Screenshot-[S12]      —— 推荐。直接引用候选清单里的编号，画面与文字零误差
#   *Screenshot-[mm:ss]    —— 兼容。没开视觉/老笔记只剩时间，走吸附兜底
# 时间部分同时兼容 mm:ss 与 hh:mm:ss（超过 1 小时的课，模型会照抄转写里的 hh:mm:ss；
# 旧正则只吃到 "01:02" 就把 62 分钟当成 62 秒，后半程截图会全部堆到开头）
_SHOT_MARK = (
    r"\*{0,2}[ \t]*Screenshot-[ \t]*\[?[ \t]*"
    r"(?:S(?P<sid>\d{1,3})|(?P<a>\d{1,3}):(?P<b>\d{2})(?::(?P<c>\d{2}))?)"
    r"[ \t]*\]?[ \t]*\*{0,2}"
)
_SCREENSHOT_RE = re.compile(_SHOT_MARK, re.IGNORECASE)
# 标题行里夹带的标记：`## 一、缘起 *Screenshot-[S3]`
_HEADING_MARK_RE = re.compile(
    r"^([ \t]*#{1,6}[^\n]*?)[ \t]*(\*{0,2}[ \t]*Screenshot-[^\n]*?)[ \t]*$",
    re.MULTILINE | re.IGNORECASE)
_CONTENT_RE = re.compile(r"\*?Content-\s*\[?(\d{1,3}):(\d{2})(?::(\d{2}))?\]?\*?")


def _groups_to_sec(groups) -> int:
    a, b, c = groups
    return (int(a) * 3600 + int(b) * 60 + int(c)) if c else (int(a) * 60 + int(b))


def _norm_content(m: re.Match) -> str:
    return f"*Content-[{_fmt_ts(_groups_to_sec(m.groups()))}]"


def extract_screenshot_marks(doc: str) -> list[tuple[str, int]]:
    """返回 [(原始标记文本, 秒数), ...]。

    编号标记（`*Screenshot-[S12]`）没有秒数概念，秒数位置返回 -1。
    """
    out = []
    for m in _SCREENSHOT_RE.finditer(doc):
        if m.group("sid"):
            out.append((m.group(0), -1))
            continue
        out.append((m.group(0), _groups_to_sec((m.group("a"), m.group("b"), m.group("c")))))
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

    定位优先级：
      1) 编号标记 `*Screenshot-[S12]` —— 直接取第 12 条候选帧，不做任何时间换算，
         图文必然对应；编号越界（模型幻觉）就删掉标记，不乱截
      2) 时间标记 `*Screenshot-[mm:ss]` —— 吸附到容差内最近的候选帧，没覆盖才定点截帧

    去重按"已插入帧的时间集合"判断，不依赖前一个标记的时间：分段合并后时间经常
    不单调（章节回顾会引用早期画面），旧写法 `ts - last < gap` 在时间倒退时会把
    后面所有标记误删成空。

    补洞：两张相邻截图间隔超过 SCREENSHOT_MAX_GAP_SEC 时，从候选帧里按中间点补图，
    避免大段无图（LLM 选点偏稀时兜底，保证有效密度）。
    """
    if not _SCREENSHOT_RE.search(doc):
        return doc, set()

    vision = vision or []
    cand = sorted(
        [v for v in vision if v.get("img_url") and v.get("ts") is not None],
        key=lambda v: int(v["ts"]),
    )
    total = len(_SCREENSHOT_RE.findall(doc))

    from . import frames as frames_mod

    state: dict = {"done": 0, "times": [], "ids": set(), "cache": {}, "last": None}
    max_gap = int(config.SCREENSHOT_MAX_GAP_SEC or 0)
    min_gap = int(config.SCREENSHOT_MIN_GAP_SEC)

    def _too_close(eff: int) -> bool:
        return any(abs(eff - t) < min_gap for t in state["times"])

    def _emit(sec: int, url: str, quiet: bool = False) -> str:
        state["times"].append(sec)
        state["done"] += 1
        if progress_cb and not quiet:
            progress_cb(f"生成截图 {state['done']}/{total}（{_fmt_ts(sec)}）")
        # 图片独立成块。不追加"（xx 画面）"这类注释 —— 它写的是帧时间，
        # 与正文表述稍有偏差就会让人觉得图配错了。
        return f"\n\n![{_fmt_ts(sec)}]({url})\n\n"

    def _fill_gap(a: int, b: int) -> str:
        """a→b 间隔过大时，从候选帧里按 max_gap 步长补图（只补区间内、未被用过的帧）。"""
        block = ""
        if not max_gap or max_gap <= 0 or b - a <= max_gap:
            return block
        cursor = a
        while b - cursor > max_gap and state["done"] < config.MAX_SCREENSHOTS:
            target = cursor + max_gap
            best, bd = None, 10 ** 9
            for c in cand:
                ct = int(c["ts"])
                if ct in state["times"]:
                    continue
                if not (cursor + min_gap <= ct <= b - min_gap):
                    continue
                d = abs(ct - target)
                if d < bd:
                    bd, best = d, c
            if not best:
                break
            block += _emit(int(best["ts"]), best["img_url"], quiet=True)
            cursor = int(best["ts"])
        return block

    def _repl(m: re.Match) -> str:
        if state["done"] >= config.MAX_SCREENSHOTS:
            return ""

        sid = m.group("sid")
        if sid:
            idx = int(sid) - 1
            if not (0 <= idx < len(vision)):
                return ""                        # 编号是幻觉，删标记而不是乱截
            v = vision[idx]
            if not v.get("img_url") or v.get("ts") is None:
                return ""
            key = f"S{int(sid)}"
            if key in state["ids"]:
                return ""                        # 同一编号重复引用
            eff = int(v["ts"])
            if _too_close(eff):
                return ""                        # 不同编号落到同一画面
            state["ids"].add(key)
            url = v["img_url"]
        else:
            sec = _groups_to_sec((m.group("a"), m.group("b"), m.group("c")))
            hit = _snap(cand, sec)
            eff = int(hit["ts"]) if hit else sec
            if _too_close(eff):
                return ""
            if hit:
                url = hit["img_url"]
            else:
                try:
                    if eff not in state["cache"]:
                        state["cache"][eff] = frames_mod.grab_frame_at(
                            video_path, task_id, eff)
                    url = state["cache"][eff]
                except Exception:
                    return ""

        block = ""
        if state["last"] is not None:
            block = _fill_gap(state["last"], eff)
        block += _emit(eff, url)
        state["last"] = eff
        return block

    doc = _SCREENSHOT_RE.sub(_repl, doc)
    # 删掉的标记会留下多余空行/行尾空格，收敛一下
    doc = re.sub(r"[ \t]+\n", "\n", doc)
    doc = re.sub(r"\n{3,}", "\n\n", doc)
    used = set(re.findall(r"!\[[^\]]*\]\((/frames/[^)]+)\)", doc))
    return doc, used




def save_markdown(task_id: int, content: str) -> str:
    path = config.OUTPUT_DIR / f"task_{task_id}.md"
    path.write_text(content, encoding="utf-8")
    return str(path)
