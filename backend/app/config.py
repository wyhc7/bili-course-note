"""
统一配置。所有可调项走环境变量 / .env，针对 2核4G 低配服务器做了默认值。
"""
import os
from pathlib import Path

# 自动加载项目根目录的 .env（需 python-dotenv）
try:
    from dotenv import load_dotenv
    _BASE = Path(__file__).resolve().parent.parent.parent
    load_dotenv(_BASE / ".env")
    load_dotenv(_BASE.parent / ".env")  # 兼容 .env 放在 /root/bili-note 上层
except Exception:
    pass

# ---- 路径 ----
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # 项目根
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", DATA_DIR / "output"))
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", DATA_DIR / "audio"))
FRAMES_DIR = Path(os.getenv("FRAMES_DIR", DATA_DIR / "frames"))
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "tasks.db"))
WEB_DIR = BASE_DIR / "web"

for d in (DATA_DIR, OUTPUT_DIR, AUDIO_DIR, FRAMES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- 服务器 ----
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# ---- 下载 ----
# B站部分大会员/高清内容需要 Cookie。把浏览器导出的 cookie 字符串填这里，
# 格式: "SESSDATA=xxx; bili_jct=yyy; ..."  留空则尝试匿名下载 720p 以内。
BILI_COOKIE = os.getenv("BILI_COOKIE", "")
# 下载清晰度上限，越低文件越小、越快（低配机建议 480/720）
MAX_HEIGHT = int(os.getenv("MAX_HEIGHT", "720"))
# 伪装浏览器 UA —— 解 B站 412 反爬的关键，别乱改
BILI_UA = os.getenv(
    "BILI_UA",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
)

# ---- 转写 ----
# 本地 faster-whisper。2核4G 建议 base + int8；想更准但慢可换 small。
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")
WHISPER_LANG = os.getenv("WHISPER_LANG", "zh")
# 分段转写的每段秒数——这是 2核4G 防 OOM 的关键，不要调太大（120s 更保险）
WHISPER_CHUNK_SEC = int(os.getenv("WHISPER_CHUNK_SEC", "120"))
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "2"))

# 转写后端: local(本地,默认) | cloud(云端,预留)
TRANSCRIBE_PROVIDER = os.getenv("TRANSCRIBE_PROVIDER", "local")

# ---- 抽帧 / 视觉 ----
# 抽帧 / 视觉
# 密集抽帧间隔（秒）——越小候选越多、截图越密，但抽帧更久。
# 10s 采样 + 感知哈希去重，既能抓住翻页，又不会让候选爆炸
DENSE_FRAME_INTERVAL_SEC = int(os.getenv("DENSE_FRAME_INTERVAL_SEC", "10"))
# 相邻帧去重：感知哈希（dHash）汉明距离 <= 该值视为同一画面（0=关闭去重）
# 64 位哈希，0-4 几乎同图，5-8 同页轻微动效，>12 基本是换页
FRAME_DEDUPE_DISTANCE = int(os.getenv("FRAME_DEDUPE_DISTANCE", "6"))
# 是否开启相邻帧去重
FRAME_DEDUPE = os.getenv("FRAME_DEDUPE", "true").lower() == "true"
# 目标截图密度：约每多少秒一张（用于给 LLM 数量下限）——120s=每2分钟一张
SCREENSHOT_EVERY_SEC = int(os.getenv("SCREENSHOT_EVERY_SEC", "120"))
# 相邻截图最小间隔（秒），太近的合并，防止同页重复
SCREENSHOT_MIN_GAP_SEC = int(os.getenv("SCREENSHOT_MIN_GAP_SEC", "8"))
# 单篇最多截图数（防 LLM 失控）
MAX_SCREENSHOTS = int(os.getenv("MAX_SCREENSHOTS", "40"))
# 截图时间点吸附容差（秒）：标记落在某个已抽帧附近就复用那张，不再二次 ffmpeg
SNAP_TOLERANCE_SEC = int(os.getenv("SNAP_TOLERANCE_SEC", "20"))
# 最终插图宽度（px），0=保持原片分辨率；1280 对板书/公式足够，体积减半
SHOT_WIDTH = int(os.getenv("SHOT_WIDTH", "1280"))
# 旧的等间隔兜底（保留兼容，现已不用）
FRAME_INTERVAL_SEC = int(os.getenv("FRAME_INTERVAL_SEC", "60"))
# 是否需要下载视频流来抽帧（false 则只转写，最省流量和磁盘，2核4G 推荐先关）
EXTRACT_FRAMES_NEED_VIDEO = os.getenv("EXTRACT_FRAMES_NEED_VIDEO", "true").lower() == "true"
# 视觉理解开关（需要多模态模型；默认复用 LLM 端点/key/模型）
VISION_ENABLED = os.getenv("VISION_ENABLED", "false").lower() == "true"
# 默认复用 LLM 的模型（需支持视觉）；留空则用 LLM_MODEL
VISION_MODEL = os.getenv("VISION_MODEL", "")
# 留空则复用 LLM_BASE_URL / LLM_API_KEY
VISION_BASE_URL = os.getenv("VISION_BASE_URL", "")
VISION_API_KEY = os.getenv("VISION_API_KEY", "")
# 最多分析多少帧（作为选图参考喂给 LLM 的下限；越多越准但越贵）
VISION_MAX_FRAMES = int(os.getenv("VISION_MAX_FRAMES", "36"))
# 按时长自适应：每分钟取多少帧（20 分钟课 ≈ 60 帧 ≈ 7 张网格）
VISION_FRAMES_PER_MIN = int(os.getenv("VISION_FRAMES_PER_MIN", "3"))
# 自适应的硬上限（2 小时课最多 240 帧 = 27 次视觉调用）
VISION_MAX_FRAMES_CAP = int(os.getenv("VISION_MAX_FRAMES_CAP", "240"))
# ---- 网格拼图（参考 BiliNote VideoReader：把多帧拼成一张网格图，左上角烙时间戳）----
# 每张网格图的列数/行数：3x3=9 帧拼一张，一次视觉调用看 9 个候选点（省调用、看得到时间线上下文）
VISION_GRID_COLS = int(os.getenv("VISION_GRID_COLS", "3"))
VISION_GRID_ROWS = int(os.getenv("VISION_GRID_ROWS", "3"))
# 网格里每一格缩略图的宽/高（px）——太小读不出公式下标，768x432 是清晰度与 token 的平衡点
VISION_CELL_W = int(os.getenv("VISION_CELL_W", "768"))
VISION_CELL_H = int(os.getenv("VISION_CELL_H", "432"))
# 网格图 JPEG 质量
VISION_GRID_QUALITY = int(os.getenv("VISION_GRID_QUALITY", "80"))
# 是否用网格拼图模式（true=拼图一次看9帧；false=旧的逐帧调用）
VISION_USE_GRID = os.getenv("VISION_USE_GRID", "true").lower() == "true"
# 单次视觉调用的重试：网络抖动时不要整格候选报废（总尝试次数，含首次）
VISION_RETRY_ATTEMPTS = int(os.getenv("VISION_RETRY_ATTEMPTS", "3"))
VISION_RETRY_BACKOFF_SEC = int(os.getenv("VISION_RETRY_BACKOFF_SEC", "2"))

# ---- LLM 总结 ----
# OpenAI 兼容接口。默认 DeepSeek，可换通义/豆包等。
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_MAX_CHARS = int(os.getenv("LLM_MAX_CHARS", "48000"))  # 单次喂给 LLM 的上限
# 单次输出上限（token）。实际取 min(本值, 按输入体量估算)，避免长课笔记被腰斩
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "8000"))

# 笔记风格: 网课/学术/口语/重点
NOTE_STYLE = os.getenv("NOTE_STYLE", "网课")

# ---- 失败重试 ----
# 任务失败（下载/转写/总结抛异常）自动重试次数（不含首次）。2核4G 建议 1~2，避免死循环烧 CPU。
MAX_RETRY = int(os.getenv("MAX_RETRY", "2"))
# 重试前等待秒数（指数退避：base * (2^已重试次数)）
RETRY_BACKOFF_SEC = int(os.getenv("RETRY_BACKOFF_SEC", "30"))

# ---- 磁盘清理（2核4G 关键，跑多了不清理会爆盘）----
CLEANUP_AFTER_DONE = os.getenv("CLEANUP_AFTER_DONE", "true").lower() == "true"
# 关键帧是否全量保留：
#   false(默认)=只保留笔记正文真正引用的 shot_*.jpg，删掉候选帧 f_*.jpg 与网格图
#   true = 整个 task 目录全留（体积大，一般不需要）
# 注意：默认值已保证"笔记里的图不会 404"，不需要再为了图去开 true
KEEP_FRAMES = os.getenv("KEEP_FRAMES", "false").lower() == "true"
