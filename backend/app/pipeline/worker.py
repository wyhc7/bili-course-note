"""
串行任务 worker：从队列取一个任务，跑完整条流水线。
2核4G 只允许单任务串行，绝不同时跑两个（否则必 OOM）。
"""
import time
import threading
import traceback
import shutil
import json
from pathlib import Path

from .. import db, config
from . import download, transcribe, frames as frames_mod, summarize


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---- 转写缓存：同一视频（bvid+cid）二次处理时跳过下载+转写，省钱省时 ----
def _cache_path(bvid: str, cid) -> Path:
    d = config.DATA_DIR / "cache"
    d.mkdir(parents=True, exist_ok=True)
    key = f"{bvid}_{cid or 'p1'}"
    key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    return d / f"transcript_{key}.json"


def _load_cache(bvid: str, cid) -> list[dict] | None:
    p = _cache_path(bvid, cid)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        segs = data.get("segments")
        if segs:
            return segs
    except Exception:
        pass
    return None


def _save_cache(bvid: str, cid, segments: list[dict]) -> None:
    try:
        p = _cache_path(bvid, cid)
        p.write_text(json.dumps({"segments": segments}, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _cleanup(tid: int, keep_imgs: set[str] | None = None):
    """
    任务完成后清理大文件：删下载的视频/音频。
    帧目录只删"没被笔记引用"的候选帧（f_*.jpg）和网格图（grid_*.jpg），
    保留下文字引用的 shot_*.jpg —— 否则网页上的截图会全部 404。
    """
    if not config.CLEANUP_AFTER_DONE:
        return
    work = config.DATA_DIR / f"task_{tid}"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    audio = config.AUDIO_DIR / f"task_{tid}.wav"
    if audio.exists():
        audio.unlink(missing_ok=True)

    frames_dir = config.FRAMES_DIR / f"task_{tid}"
    if not frames_dir.exists() or config.KEEP_FRAMES:
        return
    # /frames/task_x/shot_30.jpg -> shot_30.jpg
    keep = {Path(u[len("/frames/"):]).name for u in (keep_imgs or set())
            if u.startswith("/frames/")}
    for f in frames_dir.iterdir():
        if f.is_dir():
            shutil.rmtree(f, ignore_errors=True)
        elif f.name not in keep:
            f.unlink(missing_ok=True)
    try:
        if not any(frames_dir.iterdir()):
            frames_dir.rmdir()
    except Exception:
        pass


def _run_one(task: dict):
    tid = task["id"]
    url = task["url"]
    used_imgs: set[str] = set()
    try:
        def setp(status, progress, msg):
            db.update_task(tid, status=status, progress=progress, stage_message=msg)

        # 1) 下载（多P时可指定分P cid）
        page_cid = task.get("page_index")  # 这里 page_index 字段存 cid（多P用）
        p_cid = int(page_cid) if page_cid else None

        # 先算 bvid，看能否命中转写缓存（命中则跳过下载+转写，但要下视频流做截图）
        from .download import _extract_bvid
        bvid = _extract_bvid(url) or ""
        cached_segments = _load_cache(bvid, p_cid) if bvid else None

        # 提前解析任务选项（转写模型等要早用）
        _opts = {}
        try:
            if task.get("options"):
                _opts = json.loads(task["options"])
        except Exception:
            _opts = {}
        asr_model = _opts.get("whisper_model") or None   # None=用 .env 默认

        meta = download.download(
            url, tid,
            progress_cb=lambda m: setp(db.DOWNLOADING, 10, m),
            page_cid=p_cid,
            page_title=task.get("page_title"),
        )
        db.update_task(tid, title=meta["title"], duration_sec=meta["duration_sec"])

        # 2) 转写（优先用缓存）
        if cached_segments:
            segments = cached_segments
            setp(db.TRANSCRIBING, 65, f"命中转写缓存，跳过转写（{len(segments)} 段）")
        else:
            setp(db.TRANSCRIBING, 15, "开始转写…")
            segments = transcribe.transcribe(
                meta["audio_path"], tid,
                progress_cb=lambda m: setp(db.TRANSCRIBING, 20, m),
                model_size=asr_model,
            )
            if not segments:
                raise RuntimeError("转写结果为空")
            if bvid:
                _save_cache(bvid, p_cid, segments)

        # 3) 抽帧 + 视觉（没有视频流则跳过）
        vision = []
        if meta.get("video_path"):
            setp(db.EXTRACTING, 70, "抽取关键帧…")
            try:
                frms = frames_mod.extract_frames(
                    meta["video_path"], tid, meta["duration_sec"],
                    progress_cb=lambda m: setp(db.EXTRACTING, 75, m),
                )
                vision = frames_mod.describe_frames(
                    frms, progress_cb=lambda m: setp(db.EXTRACTING, 80, m), task_id=tid,
                    duration_sec=meta.get("duration_sec", 0),
                    cache_key=(f"{bvid}_{p_cid or 'p1'}" if bvid else None))
            except Exception as e:
                vision = []
                db.update_task(tid, stage_message=f"抽帧/视觉失败，跳过：{e}")
        else:
            setp(db.EXTRACTING, 72, "跳过画面（仅转写模式）")

        # 4) 总结（两阶段：先出 Screenshot 标记）
        setp(db.SUMMARIZING, 85, "生成笔记…")
        meta["url"] = url
        meta["generated_at"] = _now_str()
        opts = _opts  # 复用顶部已解析的选项
        meta["style"] = opts.get("style") or config.NOTE_STYLE
        meta["screenshot"] = opts.get("screenshot", True)
        meta["link"] = opts.get("link", True)
        meta["toc"] = opts.get("toc", False)
        meta["summary"] = opts.get("summary", True)

        md = summarize.summarize(
            segments, vision, meta,
            progress_cb=lambda m: setp(db.SUMMARIZING, 88, m),
        )

        # 5) 两阶段截图：标记 -> 吸附到已有候选帧（未覆盖才定点截帧）-> 替换
        if meta["screenshot"] and meta.get("video_path"):
            md, used_imgs = summarize.fill_screenshots(
                md, meta["video_path"], tid, vision=vision,
                progress_cb=lambda m: setp(db.SUMMARIZING, 95, m),
            )
        md_path = summarize.save_markdown(tid, md)

        db.update_task(tid, status=db.DONE, progress=100,
                       stage_message="完成", output_md=md_path)
        # 合集任务完成后刷新总览页
        if task.get("collection_id"):
            try:
                _rebuild_collection_index(task["collection_id"])
            except Exception:
                pass
    except Exception as e:
        tb = traceback.format_exc()
        retry_count = int(task.get("retry_count") or 0)
        # 磁盘不足重试也不会变出空间：直接判失败，别白等两轮退避（30s/60s）
        no_retry = isinstance(e, download.DiskFullError)
        if not no_retry and retry_count < config.MAX_RETRY:
            # 未超上限：设 not_before 让 worker 到点再取（不阻塞循环、不立即空转重试）
            wait = config.RETRY_BACKOFF_SEC * (2 ** retry_count)
            db.update_task(tid, status=db.QUEUED,
                           stage_message=f"失败，{wait}s 后自动重试（{retry_count+1}/{config.MAX_RETRY}）：{str(e)[:80]}",
                           error=f"{e}\n\n{tb[-1200:]}",
                           retry_count=retry_count + 1,
                           not_before=time.time() + wait,
                           progress=0)
        else:
            stage = "失败：磁盘空间不足（需手动清理）" if no_retry else "失败（重试次数已用尽）"
            db.update_task(tid, status=db.FAILED, stage_message=stage,
                           error=f"{e}\n\n{tb[-1500:]}")
    finally:
        # 无论成功失败都清理大文件，防止磁盘被撑爆（2核4G 关键）
        # 只保留笔记正文引用到的截图，候选帧与网格图删掉
        try:
            _cleanup(tid, keep_imgs=used_imgs)
        except Exception:
            pass


def _rebuild_collection_index(collection_id: str):
    """根据合集下所有分P任务，重建总览页 md（含每P状态、链接）。"""
    tasks = db.collection_tasks(collection_id)
    if not tasks:
        return
    base_title = next((t.get("title") for t in tasks if t.get("title")), "") or "多P合集"
    # title 里可能带 " · 分P名"，截断到主标题
    main_title = base_title.split(" · ")[0]
    lines = [f"# {main_title}", "",
             f"> 共 {len(tasks)} 讲　BVID：{tasks[0].get('url','')}", ""]
    lines.append("## 目录")
    lines.append("")
    done_cnt = 0
    for t in tasks:
        pi = t.get("part_index") or 0
        ptitle = t.get("page_title") or (t.get("title") or "").split(" · ")[-1]
        status = t.get("status")
        if status == db.DONE:
            done_cnt += 1
            md_file = f"task_{t['id']}.md"
            dur = ""
            if t.get("duration_sec"):
                sec = int(t["duration_sec"])
                dur = f"（{sec//60}分{sec%60}秒）" if sec >= 60 else f"（{sec}秒）"
            lines.append(f"{pi}. [{ptitle}]({md_file})　✅ 完成{dur}")
        elif status == db.FAILED:
            lines.append(f"{pi}. {ptitle}　❌ 失败")
        else:
            lines.append(f"{pi}. {ptitle}　⏳ 处理中（{t.get('stage_message','')}）")
    lines.append("")
    lines.append(f"---")
    lines.append(f"进度：{done_cnt}/{len(tasks)} 讲已完成")
    content = "\n".join(lines)
    path = config.OUTPUT_DIR / f"collection_{collection_id}.md"
    path.write_text(content, encoding="utf-8")
    return str(path)


def worker_loop(stop_event: threading.Event):
    """常驻循环：取任务 → 跑 → 空转等待。"""
    db.reset_stale_on_boot()
    while not stop_event.is_set():
        task = db.next_queued()
        if not task:
            stop_event.wait(3)
            continue
        _run_one(task)
        time.sleep(1)


def start_worker() -> threading.Event:
    stop = threading.Event()
    t = threading.Thread(target=worker_loop, args=(stop,), daemon=True)
    t.start()
    return stop
