# B站视频内容提取 · 网课笔记生成

粘贴 B 站链接 → 后台排队 → 本地 Whisper 转写 → 自动生成结构化 **Markdown 课程笔记**。
针对 **2核4G 无 GPU 云服务器 + 长视频（网课）** 优化：串行排队、分段转写防 OOM、完成后自动清理磁盘。

---

## 一、它做什么

一条流水线：
1. **下载** B 站视频（支持 `b23.tv` 短链和 `BV` 号）
2. **抽音频** → 本地 **faster-whisper** 转写（带时间戳）
3. **抽关键帧**（场景变化 + 等间隔），可选调用**多模态视觉 AI** 描述画面
4. **LLM 总结**成 Markdown 课程笔记：章节 / 知识点 / 公式 / 例题 / 易错点，关键处带 `（见 hh:mm:ss）` 时间戳可回看
5. **下载（含图 zip）**：一键打包笔记，解压即看带图笔记（见下）

### 下载"含图 zip"怎么用

网页上的「下载(含图 zip)」按钮下载的是 `note_N.zip`，解压后包含：
- `note.html` —— **双击即可看**的完整笔记（图片已 base64 内嵌，无需联网、无需 Markdown 软件，图片和文字交替排版）
- `note.md` —— 原始 Markdown（图片路径已改成相对 `images/`，供你导入 Typora/Obsidian 等编辑器）
- `images/` —— 全部截图文件

> 为什么要打包：网页里图片是服务器路径 `/frames/...`，单独下 `.md` 在本地打开会"看不到图"。打包 zip 后图片随笔记一起带走，彻底离线可看。合集总览同理（`collection_xxx.zip`：总览.md + 各分P的 .md/.html + 图片）。

## 一、多P网课（合集）支持

粘贴多P视频链接（如一门 39 讲的网课）后：
- 自动**检测分P数**，弹出分P选择面板（可全选/全不选/任意勾选，看得到每P标题和时长）
- **每P生成一篇独立笔记** + 一个**合集总览页**（列出所有讲、状态、时长，点标题可跳进对应笔记）
- 串行处理（2核4G 一次只跑一集，队列自动推进）

## 二、磁盘要求

| 项 | 占用 |
|---|---|
| Python 依赖 | ~2 GB |
| Whisper 模型（base） | ~150 MB（small ~500MB） |
| **单条视频临时文件** | ≈ 视频体积 × 2（下载 mp4 + wav 音频） |

**建议服务器留 20–30GB 可用空间。** 任务完成后会自动删除视频/音频（`.env` 里 `CLEANUP_AFTER_DONE=true`），只保留笔记 `.md`。

---

## 三、部署（Docker，推荐）

```bash
# 1. 拷代码到服务器，进目录
cd bili-note

# 2. 配置
cp .env.example .env
nano .env        # 至少填 LLM_API_KEY；要视觉理解再填 VISION_*

# 3. 启动
docker compose up -d --build

# 4. 打开浏览器
# http://服务器IP:8000
```

首次启动会下载 Whisper 模型到卷里（只下一次）。

## 四、部署（不用 Docker，直接跑）

```bash
# 系统依赖
apt install -y ffmpeg python3 python3-pip       # 服务器
# Windows: 下载 ffmpeg 静态版解压，把 bin 目录加进 PATH

pip install -r backend/requirements.txt

cp .env.example .env    # 填 LLM_API_KEY
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

---

## 五、配置要点（`.env`）

| 变量 | 说明 |
|---|---|
| `LLM_API_KEY` | **必填**。默认 DeepSeek，可换通义/豆包（改 `LLM_BASE_URL`+`LLM_MODEL`） |
| `BILI_COOKIE` | 可选。下载大会员/高清内容才需要 |
| `MAX_HEIGHT` | 下载清晰度上限，低配建议 `480`/`720` |
| `WHISPER_MODEL` | 2核4G 用 `base`；要更准换 `small`（更慢更吃内存） |
| `WHISPER_CHUNK_SEC` | 分段秒数，**防 OOM 关键**，默认 300 |
| `VISION_ENABLED` | 视觉理解开关，需配 `VISION_API_KEY`（如通义 qwen-vl-max） |
| `TRANSCRIBE_PROVIDER` | `local` 本地 / `cloud` 云端（不想让 2核4G 硬扛时切） |
| `CLEANUP_AFTER_DONE` | 完成后删临时文件，**务必 true** |
| `DENSE_FRAME_INTERVAL_SEC` | 密集抽帧间隔，默认 15 秒（越小候选越多、截图越密，但抽帧更久） |
| `SCREENSHOT_EVERY_SEC` | 目标截图密度，默认 120 秒/张（给 LLM 数量下限，控制一篇的截图数） |
| `MAX_SCREENSHOTS` | 单篇截图上限，默认 40（防失控） |
| `VISION_MAX_FRAMES` | 送去视觉分析的帧数上限，默认 36（越多 LLM 选图越准，越耗 API） |
| `VISION_USE_GRID` | 网格拼图模式（参考 BiliNote），默认 `true`。见下方"网格拼图" |
| `VISION_GRID_COLS` / `VISION_GRID_ROWS` | 网格列/行，默认 `3`×`3`=每张拼 9 帧 |
| `VISION_CELL_W` / `VISION_CELL_H` | 网格每格缩略图尺寸，默认 `640`×`360`（越小越省 token） |

### 网格拼图（借鉴 BiliNote VideoReader）

默认开启。做法：把候选帧按 **3×3 拼成一张网格图，每格左上角烙印 `mm:ss` 时间戳**，然后**一次视觉调用看整张网格**（9 个候选点 + 时间线上下文），模型按时间戳逐格返回画面描述，再喂给 LLM 选插入点。

相比逐帧调用：**视觉调用次数 ÷ 9**（一节 20 分钟课从 36 次降到 4 次），且模型能看到相邻画面变化，选图更准更快。实测 P6「克拉默法则」：36 次调用 → **4 次网格调用**，整篇处理从 ~308s 降到 ~203s，产出 16 张精准截图。

## 六、2核4G 现实预期（重要）

- **base + int8** 转 1 小时音频：约 0.5–1.5 小时；2 小时视频约 **1.5–4 小时**。
- 慢是常态，**放着后台跑即可**，界面能看到进度。
- **防 OOM 机制**：转写时用 ffmpeg 把音频切成 120 秒小片段、逐段转写、转完立即删片段并 gc。实测 29 分钟视频峰值内存可控，不会因内存爆掉被系统 kill。
- **服务守护**：`guard.sh` 循环检测，uvicorn 若崩溃 5 秒内自动重启（守护日志在 `logs/guard.log`）。
- 若嫌慢且愿花点钱：`.env` 设 `TRANSCRIBE_PROVIDER=cloud` 填云端 ASR key，代码无需改。

## 七、目录结构

```
backend/app/
  main.py          FastAPI 接口 + 托管前端
  config.py        全部配置
  db.py            SQLite 任务表 + 串行队列
  pipeline/
    download.py    yt-dlp 下载 + ffmpeg 抽音频
    transcribe.py  本地 faster-whisper（分段/可切云端）
    frames.py      抽帧 + 多模态视觉
    summarize.py   LLM 转 Markdown 笔记
    worker.py      串行任务编排 + 磁盘清理
web/index.html     单页前端
Dockerfile / docker-compose.yml / .env.example
```
