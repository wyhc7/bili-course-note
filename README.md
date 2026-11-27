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
| `DENSE_FRAME_INTERVAL_SEC` | 密集抽帧间隔，默认 **10** 秒（越小候选越多，再靠感知哈希去重压回来） |
| `FRAME_DEDUPE_DISTANCE` | 相邻帧感知哈希（dHash）距离阈值，默认 6；`0`=关闭去重 |
| `SCREENSHOT_EVERY_SEC` | 目标截图密度，默认 120 秒/张（给 LLM 数量下限，控制一篇的截图数） |
| `SCREENSHOT_MIN_GAP_SEC` / `MAX_SCREENSHOTS` | 相邻截图最小间隔 8s / 单篇上限 40 张（防同页重复与失控） |
| `SNAP_TOLERANCE_SEC` | 标记点吸附容差，默认 20s：落在已抽帧附近就直接复用那张图，不再二次截帧 |
| `SHOT_WIDTH` | 最终插图宽度，默认 1280（`0`=原片分辨率） |
| `VISION_MAX_FRAMES` / `VISION_FRAMES_PER_MIN` | 候选帧预算：下限 36，且按时长每分钟 3 帧自适应（上限 `VISION_MAX_FRAMES_CAP`=240） |
| `VISION_USE_GRID` | 网格拼图模式（参考 BiliNote），默认 `true`。见下方"网格拼图" |
| `VISION_GRID_COLS` / `VISION_GRID_ROWS` | 网格列/行，默认 `3`×`3`=每张拼 9 帧 |
| `VISION_CELL_W` / `VISION_CELL_H` | 网格每格缩略图尺寸，默认 `768`×`432`（太小读不出公式下标） |
| `VISION_RETRY_ATTEMPTS` / `VISION_RETRY_BACKOFF_SEC` | 视觉调用重试次数与退避基数（默认 3 次 / 2s），防网络抖动导致整格候选报废 |
| `LLM_MAX_OUTPUT_TOKENS` | 单次输出上限，默认 8000；实际取 `min(本值, 按输入体量估算)`，防止长课笔记被截断 |

### 网格拼图（借鉴 BiliNote VideoReader）

默认开启。做法：把候选帧按 **3×3 拼成一张网格图，每格左上角烙印 `#序号 mm:ss`**，然后**一次视觉调用看整张网格**（9 个候选点 + 时间线上下文），模型按 `#序号` 逐格返回画面描述，再喂给 LLM 选插入点。

相比逐帧调用：**视觉调用次数 ÷ 9**（一节 20 分钟课从 36 次降到 4 次），且模型能看到相邻画面变化，选图更准更快。实测 P6「克拉默法则」：36 次调用 → **4 次网格调用**，整篇处理从 ~308s 降到 ~203s，产出 16 张精准截图。

> 用 `#序号` 而不是让模型复述 `mm:ss`，是因为模型最容易把画面角上的数字看错一位；序号错了会整格错位。

### 两阶段截图（编号引用 → 落图）

1. 视觉阶段给每个候选画面编号 `S1..Sn`（编号 = 候选帧下标，跨分段稳定），清单形如
   `- S7　03:20：一页 PPT，标题「克拉默法则」，给出 D≠0 时方程组有唯一解的定理`；
2. LLM 只输出 `*Screenshot-[S7]` 占位，**不写时间、不碰图片 URL**；
3. 落图时直接取第 7 条候选的帧文件，**不做任何按秒吸附** —— 模型挑的是哪一页，插进来的就是哪一页；
4. 兼容老格式 `*Screenshot-[mm:ss]`：吸附到 `SNAP_TOLERANCE_SEC` 内的候选帧，候选没覆盖的时间点才 `grab_frame_at` 定点截帧（混合 seek：前 1s 快速定位 + 后 1s 精确解码，既快又不会偏到关键帧）；
5. 编号越界（模型幻觉）直接删标记不乱截；同一编号只插一次；时间相近的不同编号合并；超 `MAX_SCREENSHOTS` 截断；
6. 任务结束后只保留正文引用到的截图，候选帧与网格图删掉，**不会出现图片 404**。

> **为什么要改成编号**：时间只能精确到秒，模型把 `03:20` 写成 `03:25`，就近吸附就可能吸到隔壁那一页 —— PPT 每 30 秒翻页时几乎必然错配。编号是离散的，没有"差一点"的中间态，从机制上消除了图文不对应。
>
> 另外两处会静默错配的坑也一并修了：① 合并后的笔记时间经常不单调（章节回顾会引用早期画面），旧实现用 `ts - last < gap` 去重，时间一倒退就把后面所有标记删空；② 超 1 小时的课，网格烙印是 `1:15:30`，旧解析按 `mm:ss` 读成 75 秒，九格描述全部挂到错误时间上。

### 公式渲染

笔记里的 `$...$` / `$$...$$` 在网页端和导出的单文件 HTML 里都由 **KaTeX** 渲染：

- 网页端：`web/vendor/katex/` 本地打包（含 woff2 字体），离线可用，不依赖 CDN；公式在 `\n → <br>` 之前就被抽出保护，所以 `\begin{cases}`、`\begin{vmatrix}` 这类跨行公式不会被拆坏；
- 导出 HTML：仅在该篇出现公式时注入 KaTeX（字体已 base64 内联），双击即可离线看，单篇体积约 +600KB；无公式的笔记体积不变；
- KaTeX 不可用时退化为等宽原文，不会把 `$` 当正文、也不会被斜体规则吃掉。

> 视觉描述按 `bvid_cid` 缓存在 `data/cache/vision_*.json`：换风格重跑、失败重试只补缺失的时间点，不重复调视觉 API。缓存只存 `{时间戳: 描述}`，图片 URL 每次按当前任务目录重新映射，不会指向旧任务。

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
