"""
B站下载（API 优先架构）。

背景：数据中心 IP 直连 bilibili 网页会触发 HTTP 412 反爬，yt-dlp 走网页抓取会失败。
但 B站 Web API 只要带 buvid3 cookie 就能正常返回。所以：
  1) 访问首页自动获取 buvid3 cookie（无需登录）
  2) view API  -> 标题/时长/cid/bvid
  3) playurl API -> dash 音频流 / 视频流 baseUrl
  4) 带 Referer 直接下载音频流（本项目转写只需音频；抽帧才需视频流）
yt-dlp 保留为 fallback（有 BILI_COOKIE 时可能可用）。
"""
import subprocess
import json
import re
import shutil
import time
from pathlib import Path

import httpx

from .. import config


class DownloadError(Exception):
    pass


class DiskFullError(DownloadError):
    """剩余空间不足。故意单独建类：worker 见到它不重试（重试也不会变出空间）。"""


_API_VIEW = "https://api.bilibili.com/x/web-interface/view"
_API_PLAYURL = "https://api.bilibili.com/x/player/playurl"
_BILI_HOME = "https://www.bilibili.com/"

# 进程级缓存 buvid3，避免每次任务都重新抓
_cookie_cache: dict = {}


def _headers(referer: str = _BILI_HOME) -> dict:
    return {
        "User-Agent": config.BILI_UA,
        "Referer": referer,
        "Accept-Language": "zh-CN,zh;q=0.9",
    }


def _get_buvid3(client: httpx.Client) -> str:
    """访问首页，从 Set-Cookie 里取 buvid3。失败则回退匿名内置值。"""
    if _cookie_cache.get("buvid3"):
        return _cookie_cache["buvid3"]
    # 若用户手动配了 BILI_COOKIE 且含 buvid3，优先用
    if config.BILI_COOKIE and "buvid3=" in config.BILI_COOKIE:
        m = re.search(r"buvid3=([^;]+)", config.BILI_COOKIE)
        if m:
            _cookie_cache["buvid3"] = m.group(1)
            return m.group(1)
    try:
        r = client.get(_BILI_HOME, headers=_headers())
        for c in r.cookies.jar:
            if c.name == "buvid3":
                _cookie_cache["buvid3"] = c.value
                _cookie_cache["b_nut"] = next(
                    (j.value for j in r.cookies.jar if j.name == "b_nut"), "")
                return c.value
    except Exception:
        pass
    return ""


def _extract_bvid(url: str) -> str:
    m = re.search(r"(BV[0-9A-Za-z]{6,})", url)
    if m:
        return m.group(1)
    # av 号
    m = re.search(r"av(\d+)", url)
    if m:
        return "av" + m.group(1)
    return ""


def resolve_short_url(url: str) -> str:
    url = url.strip()
    if "b23.tv" not in url:
        return url
    try:
        r = httpx.get(url, follow_redirects=False, timeout=15, headers=_headers())
        loc = r.headers.get("location")
        if loc:
            return loc
    except Exception:
        pass
    return url


def _pick_audio(audios: list) -> dict:
    """优先最高码率音频。"""
    return sorted(audios, key=lambda a: a.get("bandwidth", 0), reverse=True)[0]


def _pick_video(videos: list, max_height: int) -> dict | None:
    """选 <=max_height 的最高清视频流（抽帧用）。"""
    cand = [v for v in videos if v.get("height", 0) <= max_height] or videos
    if not cand:
        return None
    return sorted(cand, key=lambda v: v.get("height", 0), reverse=True)[0]


def _download_stream(client, url: str, out: Path, referer: str, progress_cb=None, tag=""):
    """
    分段下载 + 断点续传 + 完整性校验。

    B站 CDN 下 1.5GB 大文件断流很常见，旧实现一断就抛 DownloadError，
    导致整个任务重试（还要从 view API 重新走一遍）。这里改成：
      - 已下载的字节数作为 Range 起点续传（CDN 支持 206 时追加写）
      - 服务器不支持 Range（回 200）时自动退回覆盖写，结果依然正确
      - 收尾比对 Content-Length，短了算失败并重试，绝不让半截文件流到 ffmpeg
    """
    attempts = max(1, config.DOWNLOAD_ATTEMPTS)
    last: Exception | None = None

    for i in range(attempts):
        have = out.stat().st_size if out.exists() else 0
        headers = _headers(referer)
        if have > 0:
            headers["Range"] = f"bytes={have}-"
        try:
            with client.stream("GET", url, headers=headers, timeout=600) as r:
                if r.status_code == 416:
                    return  # 服务端认为已经给全了
                r.raise_for_status()
                if r.status_code == 206:
                    total = int(r.headers.get("content-length", 0)) + have
                    mode, done = "ab", have
                else:
                    total = int(r.headers.get("content-length", 0))
                    mode, done = "wb", 0
                with open(out, mode) as f:
                    for chunk in r.iter_bytes(chunk_size=1 << 16):
                        f.write(chunk)
                        done += len(chunk)
                        if progress_cb and total:
                            progress_cb(f"下载{tag} {int(done / total * 100)}%")
            got = out.stat().st_size if out.exists() else 0
            if total and got < total:
                raise DownloadError(f"{tag}流不完整：{got}/{total} 字节")
            return
        except Exception as e:
            last = e
            if i + 1 >= attempts:
                break
            wait = config.DOWNLOAD_RETRY_BACKOFF_SEC * (2 ** i)
            if progress_cb:
                got_mb = (out.stat().st_size if out.exists() else 0) / 1e6
                progress_cb(f"下载{tag}中断（已存 {got_mb:.0f}MB），{wait:.0f}s 后从断点续传…")
            time.sleep(wait)
    raise DownloadError(f"下载{tag}失败（已尝试 {attempts} 次）：{last}")


def list_pages(url: str) -> dict:
    """
    列出多 P 视频的所有分 P 信息。
    返回 {bvid, title, pages: [{page, cid, part, duration}], total}
    """
    url = resolve_short_url(url)
    bvid = _extract_bvid(url)
    if not bvid:
        raise DownloadError("无法从链接解析出 BVID")
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        buvid = _get_buvid3(client)
        cookie = f"buvid3={buvid}" if buvid else ""
        if config.BILI_COOKIE:
            cookie = (config.BILI_COOKIE + ("; " + cookie if cookie else "")).strip("; ")
        h = _headers()
        if cookie:
            h["Cookie"] = cookie
        r = client.get(_API_VIEW, params={"bvid": bvid}, headers=h)
        r.raise_for_status()
        view = r.json()
    if view.get("code") != 0:
        raise DownloadError(f"view API 错误 code={view.get('code')} {view.get('message')}")
    d = view["data"]
    pages = []
    for p in d.get("pages", []):
        pages.append({
            "page": int(p.get("page", 0)),
            "cid": p.get("cid"),
            "part": (p.get("part") or "").strip(),
            "duration": float(p.get("duration", 0)),
        })
    return {"bvid": bvid, "title": d.get("title", bvid),
            "pages": pages, "total": len(pages)}


def estimate_need_bytes(duration_sec: float) -> int:
    """按视频时长估算本次任务峰值占用（视频流 + wav + 候选帧）+ 安全余量。"""
    hours = max(0.0, float(duration_sec or 0)) / 3600.0
    need = config.DISK_GB_PER_HOUR * hours + config.DISK_SAFETY_GB
    return max(int(config.DISK_MIN_FREE_GB * 1e9), int(need * 1e9))


def _check_disk(duration_sec: float):
    """
    下载前预检磁盘。不足直接抛 DiskFullError —— 宁可现在就失败，
    也不要下到 1.8GB 时 IO 报错白跑一趟（2 小时课的下载时间很贵）。
    时长未知（0）时退回绝对下限判断，不会因为拿不到时长就放行。
    """
    if not config.DISK_CHECK_ENABLED:
        return
    try:
        du = shutil.disk_usage(str(config.DATA_DIR))
    except OSError:
        return  # 查不到就不拦，别让预检本身成为故障点
    need = estimate_need_bytes(duration_sec)
    if du.free >= need:
        return
    mins = int(duration_sec // 60) if duration_sec else 0
    span = f"，本视频时长约 {mins} 分钟" if mins else ""
    raise DiskFullError(
        f"磁盘空间不足：剩余 {du.free / 1e9:.1f}GB，预计需要 {need / 1e9:.1f}GB"
        f"（估算含视频流、音频、抽帧）{span}。"
        f"请清理 {config.DATA_DIR} 下的旧任务后重试。"
    )


def _load_reusable(work: Path, bvid: str) -> dict | None:
    """
    断点续跑：上一次失败后保留了 task 目录，这里直接复用，避免重下 1.5GB + 重转 wav。
    只在 audio.wav 完好、bvid 一致、且（需要抽帧时）视频文件也在的情况下命中。
    """
    if not config.RESUME_ON_RETRY:
        return None
    mf, aw = work / "meta.json", work / "audio.wav"
    if not (mf.exists() and aw.exists()) or aw.stat().st_size < 10240:
        return None
    try:
        meta = json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return None
    if meta.get("bvid") != bvid:
        return None
    vp = meta.get("video_path")
    video_path = vp if (vp and Path(vp).exists()) else None
    # 需要抽帧但画面流丢了：不复用，让它重下，否则这篇会没有截图
    if config.EXTRACT_FRAMES_NEED_VIDEO and not video_path:
        return None
    return {
        "audio_path": str(aw),
        "video_path": video_path,
        "title": meta.get("title") or bvid,
        "duration_sec": float(meta.get("duration_sec") or 0),
        "bvid": bvid,
    }


def download(url: str, task_id: int, progress_cb=None,
             page_cid: int | None = None, page_title: str | None = None) -> dict:
    """
    返回 {audio_path, video_path(可能None), title, duration_sec, bvid}。
    page_cid 指定时下该分P，否则下 P1（第一个 cid）。
    """
    url = resolve_short_url(url)
    bvid = _extract_bvid(url)
    if not bvid:
        raise DownloadError("无法从链接解析出 BVID，请贴标准 bilibili 视频链接")

    work = config.DATA_DIR / f"task_{task_id}"
    work.mkdir(parents=True, exist_ok=True)

    # 断点续跑：失败重试时复用上次已下载的音频/视频，不重复走 API 和下载
    reused = _load_reusable(work, bvid)
    if reused:
        if progress_cb:
            progress_cb("复用上次已下载的文件，跳过下载")
        return reused

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        if progress_cb:
            progress_cb("获取 B站视频信息…")
        buvid = _get_buvid3(client)
        cookie = f"buvid3={buvid}" if buvid else ""
        # 合并用户手动 cookie
        if config.BILI_COOKIE:
            cookie = (config.BILI_COOKIE + ("; " + cookie if cookie else "")).strip("; ")

        def api_get(url_, params):
            h = _headers()
            if cookie:
                h["Cookie"] = cookie
            r = client.get(url_, params=params, headers=h)
            r.raise_for_status()
            return r.json()

        # 1) view
        try:
            view = api_get(_API_VIEW, {"bvid": bvid})
        except Exception as e:
            raise DownloadError(f"view API 失败：{e}")
        if view.get("code") != 0:
            raise DownloadError(f"view API 返回错误 code={view.get('code')} {view.get('message')}")
        d = view["data"]
        title = d.get("title", bvid)
        cid = d.get("cid")
        # 多P：若未显式给 page_cid，尝试从 URL 的 ?p=N 里解析分P号
        if page_cid is None:
            m_p = re.search(r"[?&]p=(\d+)", url)
            if m_p:
                want_page = int(m_p.group(1))
                matched = next(
                    (p for p in d.get("pages", []) if int(p.get("page", 0)) == want_page),
                    None)
                if matched and matched.get("cid"):
                    page_cid = matched["cid"]
                    if not page_title:
                        page_title = (matched.get("part") or "").strip() or None
        # 多P：若指定 page_cid 则用该分P，标题带上分P名
        if page_cid:
            cid = page_cid
            matched = next((p for p in d.get("pages", []) if p.get("cid") == page_cid), None)
            duration = float(matched.get("duration", 0)) if matched else float(d.get("duration", 0))
            if page_title:
                title = f"{title} · {page_title}"
        else:
            duration = float(d.get("duration", 0))
        if not cid:
            raise DownloadError("未取到 cid")

        # 1.5) 磁盘预检：时长已拿到，此时估算最准（短课不会被误拦）
        _check_disk(duration)

        # 2) playurl（fnval=16 拿 dash）
        try:
            pu = api_get(_API_PLAYURL,
                         {"bvid": bvid, "cid": cid, "fnval": 16, "fourk": 1})
        except Exception as e:
            raise DownloadError(f"playurl API 失败：{e}")
        if pu.get("code") != 0:
            raise DownloadError(f"playurl 返回错误 code={pu.get('code')} {pu.get('message')}")
        dash = pu["data"].get("dash") or {}
        audios = dash.get("audio") or []
        if not audios:
            raise DownloadError("playurl 未返回音频流（可能是付费/受限视频）")

        referer = f"https://www.bilibili.com/video/{bvid}"
        audio_stream = _pick_audio(audios)
        raw_audio = work / "audio.m4s"
        _download_stream(client, audio_stream["baseUrl"], raw_audio, referer,
                         progress_cb, tag="音频")

        # 3) 可选：下低清视频流（抽帧用）。抽帧失败不致命。
        video_path = None
        vstream = _pick_video(dash.get("video") or [], config.MAX_HEIGHT)
        raw_video = None
        if vstream and config.EXTRACT_FRAMES_NEED_VIDEO:
            try:
                raw_video = work / "video.m4s"
                _download_stream(client, vstream["baseUrl"], raw_video, referer,
                                 progress_cb, tag="画面")
                video_path = raw_video
            except Exception:
                video_path = None

    # 4) ffmpeg 转 wav（m4s 是裸流，ffmpeg 可直接读）
    if progress_cb:
        progress_cb("提取音频…")
    audio_wav = work / "audio.wav"
    _to_wav(raw_audio, audio_wav)

    # 5) 若下了视频流，合成 mp4 供抽帧
    final_video = None
    if video_path and video_path.exists():
        try:
            mp4 = work / "video.mp4"
            _mux_video(video_path, mp4)
            if mp4.exists():
                final_video = mp4
        except Exception:
            final_video = None

    # 用实际媒体时长校正（API 的 duration 在多P/合集场景常不准，曾出现 29 小时）
    real_dur = _probe_duration(final_video or audio_wav)
    if real_dur and real_dur > 0:
        duration = real_dur

    result = {
        "audio_path": str(audio_wav),
        "video_path": str(final_video) if final_video else None,
        "title": title,
        "duration_sec": duration,
        "bvid": bvid,
    }
    # 落一份元数据，供失败重试时断点续跑（见 _load_reusable）
    try:
        (work / "meta.json").write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return result


def _probe_duration(path) -> float:
    """用 ffprobe 取实际媒体时长（秒）。失败返回 0。"""
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=60)
        s = (p.stdout or "").strip()
        return float(s) if s else 0.0
    except Exception:
        return 0.0


def _to_wav(src: Path, dst: Path):
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
           "-c:a", "pcm_s16le", str(dst)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not dst.exists():
        raise DownloadError(f"ffmpeg 转 wav 失败：{(p.stderr or '')[-400:]}")


def _mux_video(src: Path, dst: Path):
    cmd = ["ffmpeg", "-y", "-i", str(src), "-c", "copy", str(dst)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not dst.exists():
        raise DownloadError("ffmpeg 封装视频失败")
