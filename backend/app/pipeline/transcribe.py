"""
语音转写：本地 faster-whisper（默认），分段处理防 OOM，可切云端。
输出统一为 segments: [{start, end, text}]，方便后续做时间戳笔记。
"""
import gc
import os
import subprocess
from pathlib import Path

from .. import config


class TranscribeError(Exception):
    pass


def _audio_duration(audio_path: Path) -> float:
    """用 ffprobe 拿时长。"""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            capture_output=True, text=True, timeout=30,
        )
        return float((out.stdout or "0").strip() or 0)
    except Exception:
        return 0.0


def transcribe(audio_path: str, task_id: int, progress_cb=None, model_size: str | None = None) -> list[dict]:
    provider = config.TRANSCRIBE_PROVIDER
    if provider == "cloud":
        return _transcribe_cloud(audio_path, task_id, progress_cb)
    return _transcribe_local(audio_path, task_id, progress_cb, model_size=model_size)


def _transcribe_local(audio_path: str, task_id: int, progress_cb=None,
                      model_size: str | None = None) -> list[dict]:
    from faster_whisper import WhisperModel

    audio = Path(audio_path)
    if not audio.exists():
        raise TranscribeError("音频文件不存在")

    use_model = model_size or config.WHISPER_MODEL
    total = _audio_duration(audio) or 0.0
    if progress_cb:
        progress_cb(f"加载本地模型 {use_model}({config.WHISPER_COMPUTE})…")

    model = WhisperModel(
        use_model,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE,
        cpu_threads=config.WHISPER_CPU_THREADS,
        num_workers=1,
    )

    # 关键防 OOM 策略：先把长音频用 ffmpeg 切成一个个小文件（切片本身几乎不吃内存），
    # 再逐个小文件转写，转完立刻删掉该片段。这样模型每次只处理一小段，内存峰值可控。
    chunk = config.WHISPER_CHUNK_SEC
    segments_out: list[dict] = []
    n_chunks = max(1, int((total + chunk - 1) // chunk)) if total else 1

    seg_dir = audio.parent / f"chunks_{task_id}"
    seg_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_chunks):
        start = i * chunk
        dur = chunk if total == 0 else min(chunk, max(0.0, total - start))
        if dur <= 0:
            continue
        if progress_cb:
            pct = int((i / n_chunks) * 90) + 5
            progress_cb(f"转写中 第 {i+1}/{n_chunks} 段（{pct}%）")

        seg_file = seg_dir / f"part_{i:04d}.wav"
        try:
            # ffmpeg 切出该段小文件（-ss 在 -i 前，快速定位）
            cut = subprocess.run(
                ["ffmpeg", "-y", "-ss", str(start), "-t", str(dur),
                 "-i", str(audio), "-ar", "16000", "-ac", "1",
                 "-c:a", "pcm_s16le", str(seg_file), "-loglevel", "error"],
                capture_output=True, timeout=300,
            )
            if cut.returncode != 0 or not seg_file.exists():
                raise TranscribeError("ffmpeg 切片失败")

            segs, _info = model.transcribe(
                str(seg_file),
                language=config.WHISPER_LANG,
                initial_prompt="以下是一段中文课程或讲解的音频。",
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                beam_size=1,  # 低配用 beam=1 速度快
                condition_on_previous_text=False,
            )
            for s in segs:
                txt = (s.text or "").strip()
                if txt:
                    segments_out.append({
                        "start": round(float(s.start) + start, 2),
                        "end": round(float(s.end) + start, 2),
                        "text": txt,
                    })
        except Exception as e:
            segments_out.append({"start": start, "end": start + dur,
                                 "text": f"[该段转写失败: {e}]"})
        finally:
            # 立刻删片段文件 + gc，压住内存峰值
            try:
                if seg_file.exists():
                    seg_file.unlink()
            except Exception:
                pass
            gc.collect()

    # 释放模型内存
    del model
    gc.collect()

    if progress_cb:
        progress_cb("转写完成，正在整理…")
    return segments_out


def _transcribe_cloud(audio_path: str, task_id: int, progress_cb=None) -> list[dict]:
    """
    云端转写预留口（OpenAI whisper-1 兼容）。
    你以后不想让 2核4G 硬扛，可在 .env 设 TRANSCRIBE_PROVIDER=cloud 并配 CLOUD_ASR_*。
    """
    from openai import OpenAI
    base = os.getenv("CLOUD_ASR_BASE_URL", "https://api.openai.com/v1")
    key = os.getenv("CLOUD_ASR_API_KEY", "")
    model = os.getenv("CLOUD_ASR_MODEL", "whisper-1")
    if not key:
        raise TranscribeError("云端转写未配置 CLOUD_ASR_API_KEY")
    client = OpenAI(base_url=base, api_key=key)
    if progress_cb:
        progress_cb("上传音频到云端转写…")
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=model, file=f, response_format="verbose_json",
            language="zh",
        )
    segs = []
    for s in getattr(resp, "segments", []) or []:
        segs.append({"start": round(float(s.start), 2),
                     "end": round(float(s.end), 2),
                     "text": (s.text or "").strip()})
    if not segs and getattr(resp, "text", ""):
        segs.append({"start": 0, "end": 0, "text": resp.text})
    return segs
