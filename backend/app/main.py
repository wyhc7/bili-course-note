"""
FastAPI 应用：提交任务、查进度、看/下载 Markdown、托管单页前端。
"""
from contextlib import asynccontextmanager
from pathlib import Path
import uuid
import io
import re
import zipfile

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, db
from .pipeline import worker, download as dl


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    stop = worker.start_worker()
    yield
    stop.set()


app = FastAPI(title="B站视频内容提取", lifespan=lifespan)

# 关键帧截图静态目录：/frames/<task_x>/xxx.jpg
app.mount("/frames", StaticFiles(directory=str(config.FRAMES_DIR)), name="frames")


class SubmitReq(BaseModel):
    url: str
    style: str | None = None
    screenshot: bool = True
    link: bool = True
    toc: bool = False
    summary: bool = True
    whisper_model: str | None = None   # base / small；空=用 .env 默认


@app.post("/api/tasks")
def submit(req: SubmitReq):
    url = (req.url or "").strip()
    if not url or ("bilibili.com" not in url and "b23.tv" not in url and not url.startswith("BV")):
        raise HTTPException(400, "请输入有效的 B 站视频链接（b23.tv 或 bilibili.com）")
    if url.startswith("BV"):
        url = f"https://www.bilibili.com/video/{url}"
    import json as _json
    opts = _json.dumps({
        "style": req.style,
        "screenshot": req.screenshot,
        "link": req.link,
        "toc": req.toc,
        "summary": req.summary,
        "whisper_model": req.whisper_model or "",
    }, ensure_ascii=False)
    tid = db.create_task(url, opts)
    return {"id": tid, "status": db.QUEUED}


@app.get("/api/tasks")
def tasks():
    return db.list_tasks()


# ---------- 多P / 合集 ----------
class PartsReq(BaseModel):
    url: str


@app.post("/api/parts")
def list_parts(req: PartsReq):
    """列出多P视频的所有分P，供前端选择要处理哪些。"""
    url = (req.url or "").strip()
    if url.startswith("BV"):
        url = f"https://www.bilibili.com/video/{url}"
    try:
        info = dl.list_pages(url)
        return info
    except Exception as e:
        raise HTTPException(400, f"获取分P失败：{e}")


class CollectionReq(BaseModel):
    url: str
    parts: list[int] | None = None      # 要处理的分P序号列表（1-based）；None=全部
    range_start: int | None = None      # 或指定范围
    range_end: int | None = None
    style: str | None = None
    screenshot: bool = True
    link: bool = True
    toc: bool = False
    summary: bool = True
    whisper_model: str | None = None


@app.post("/api/collection")
def submit_collection(req: CollectionReq):
    url = (req.url or "").strip()
    if url.startswith("BV"):
        url = f"https://www.bilibili.com/video/{url}"
    if not url or ("bilibili.com" not in url and "b23.tv" not in url):
        raise HTTPException(400, "请输入有效的 B 站链接")
    try:
        info = dl.list_pages(url)
    except Exception as e:
        raise HTTPException(400, f"获取分P失败：{e}")
    pages = info["pages"]
    if not pages:
        raise HTTPException(400, "该视频没有分P")

    # 计算要处理的 page 列表
    if req.parts:
        wanted = set(req.parts)
        selected = [p for p in pages if p["page"] in wanted]
    elif req.range_start or req.range_end:
        s = req.range_start or 1
        e = req.range_end or len(pages)
        selected = [p for p in pages if s <= p["page"] <= e]
    else:
        selected = list(pages)

    if not selected:
        raise HTTPException(400, "没有选中任何分P")

    import json as _json
    opts = _json.dumps({
        "style": req.style, "screenshot": req.screenshot,
        "link": req.link, "toc": req.toc, "summary": req.summary,
        "whisper_model": req.whisper_model or "",
    }, ensure_ascii=False)

    collection_id = uuid.uuid4().hex[:8]
    total = len(selected)
    created = []
    for i, p in enumerate(selected):
        tid = db.create_task(
            url, opts,
            collection_id=collection_id,
            page_index=p["cid"],          # 存 cid
            page_title=p["part"] or f"P{p['page']}",
            part_index=p["page"],
            part_total=total,
        )
        created.append({"task_id": tid, "page": p["page"], "part": p["part"]})
    return {"collection_id": collection_id, "total": total, "tasks": created}


@app.get("/api/collections/{cid}/markdown", response_class=PlainTextResponse)
def collection_markdown(cid: str):
    path = config.OUTPUT_DIR / f"collection_{cid}.md"
    if not path.exists():
        raise HTTPException(404, "总览页尚未生成")
    return path.read_text(encoding="utf-8")


@app.get("/api/collections/{cid}/tasks")
def collection_task_list(cid: str):
    return db.collection_tasks(cid)


@app.get("/api/collections/{cid}/download")
def collection_download(cid: str):
    path = config.OUTPUT_DIR / f"collection_{cid}.md"
    if not path.exists():
        raise HTTPException(404, "总览页尚未生成")
    return FileResponse(path, media_type="text/markdown", filename=f"collection_{cid}.md")


@app.get("/api/collections/{cid}/bundle")
def collection_bundle(cid: str):
    """合集总览打包下载：总览 .md + 各分P笔记 + 全部图片（图片改相对路径）。"""
    path = config.OUTPUT_DIR / f"collection_{cid}.md"
    if not path.exists():
        raise HTTPException(404, "总览页尚未生成")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        # 总览
        z.writestr("总览.md", path.read_text(encoding="utf-8").encode("utf-8"))
        # 各分P笔记 + 图片
        for t in db.collection_tasks(cid):
            md_path = t.get("output_md")
            if not md_path or not Path(md_path).exists():
                continue
            md = Path(md_path).read_text(encoding="utf-8")
            note_name = f"笔记_{t.get('part_index') or t['id']}_{(t.get('page_title') or t['id'])}.md"
            note_name = re.sub(r'[\\/:*?"<>|]', "_", note_name)
            img_map = _collect_images(md)
            new_md = _IMG_RE.sub(
                lambda m: f"![{m.group(1)}]({img_map.get(m.group(2), m.group(2))})", md)
            z.writestr(note_name, new_md.encode("utf-8"))
            # 自包含 HTML（base64 内嵌图片，双击可看）
            html_name = note_name.rsplit(".", 1)[0] + ".html"
            z.writestr(html_name, _md_to_html(md).encode("utf-8"))
            for web, rel_zip in img_map.items():
                src = config.FRAMES_DIR / web[len("/frames/"):]
                if src.exists():
                    try:
                        z.write(str(src), rel_zip)
                    except Exception:
                        pass
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="collection_{cid}.zip"'},
    )


@app.get("/api/collections/{cid}/html")
def collection_html(cid: str, inline: int = 0):
    """合集总览单文件 HTML：目录页；分P标题链接指向各分P的 HTML（手机浏览器直接看）。"""
    path = config.OUTPUT_DIR / f"collection_{cid}.md"
    if not path.exists():
        raise HTTPException(404, "总览页尚未生成")
    md = path.read_text(encoding="utf-8")

    # 把总览里的 (task_N.md) 链接改指向该分P的 HTML 页面
    def _rewrite(href: str) -> str:
        m = re.search(r"task_(\d+)\.md", href)
        if m:
            return f"/api/tasks/{m.group(1)}/html?inline=1"
        return href

    html_doc = _md_to_html(md, link_rewrite=_rewrite)
    disp = "inline" if inline else f'attachment; filename="collection_{cid}.html"'
    return Response(content=html_doc.encode("utf-8"), media_type="text/html",
                    headers={"Content-Disposition": disp})



@app.post("/api/tasks/{tid}/rerun")
def task_rerun(tid: int):
    """重新解析：重置为排队重跑（保留原链接和选项）。"""
    t = db.get_task(tid)
    if not t:
        raise HTTPException(404, "任务不存在")
    db.requeue_task(tid)
    return {"id": tid, "status": db.QUEUED}


def _delete_task_artifacts(tid: int, output_md: str | None):
    """删除单个任务的产物（笔记 md、帧图目录）。"""
    import shutil as _sh
    try:
        if output_md:
            Path(output_md).unlink(missing_ok=True)
        (config.OUTPUT_DIR / f"task_{tid}.md").unlink(missing_ok=True)
    except Exception:
        pass
    try:
        fdir = config.FRAMES_DIR / f"task_{tid}"
        if fdir.exists():
            _sh.rmtree(fdir, ignore_errors=True)
    except Exception:
        pass


@app.delete("/api/tasks/{tid}")
def task_delete(tid: int):
    """删除任务及其产物（笔记 md、帧图目录）。"""
    t = db.get_task(tid)
    if not t:
        raise HTTPException(404, "任务不存在")
    _delete_task_artifacts(tid, t.get("output_md"))
    db.delete_task(tid)
    return {"deleted": tid}


@app.post("/api/collections/{cid}/rerun")
def collection_rerun(cid: str):
    """重跑整个合集：把所有分P重置为排队。"""
    tasks = db.collection_tasks(cid)
    if not tasks:
        raise HTTPException(404, "合集不存在")
    for t in tasks:
        db.requeue_task(t["id"])
    return {"rerun_collection": cid, "tasks": len(tasks)}


@app.delete("/api/collections/{cid}")
def collection_delete(cid: str):
    """删除整个合集：所有分P任务 + 各自产物 + 总览文件。"""
    tasks = db.collection_tasks(cid)
    if not tasks:
        raise HTTPException(404, "合集不存在")
    n = 0
    for t in tasks:
        _delete_task_artifacts(t["id"], t.get("output_md"))
        db.delete_task(t["id"])
        n += 1
    # 删总览 md
    try:
        (config.OUTPUT_DIR / f"collection_{cid}.md").unlink(missing_ok=True)
    except Exception:
        pass
    return {"deleted_collection": cid, "tasks_removed": n}


@app.get("/api/health")
def health():
    """健康面板：磁盘、内存、队列概况，方便自查。"""
    import shutil as _sh
    try:
        du = _sh.disk_usage(str(config.DATA_DIR))
        disk = {"total_gb": round(du.total / 1e9, 1),
                "used_gb": round(du.used / 1e9, 1),
                "free_gb": round(du.free / 1e9, 1),
                "free_pct": round(du.free / du.total * 100, 1)}
    except Exception:
        disk = {}
    mem = {}
    try:
        with open("/proc/meminfo") as f:
            kv = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    kv[parts[0].rstrip(":")] = int(parts[1])
            mt = kv.get("MemTotal", 0)
            ma = kv.get("MemAvailable", 0)
            if mt:
                mem = {"total_mb": mt // 1024, "avail_mb": ma // 1024,
                       "avail_pct": round(ma / mt * 100, 1)}
    except Exception:
        pass
    tasks = db.list_tasks(200)
    counts = {}
    for t in tasks:
        counts[t["status"]] = counts.get(t["status"], 0) + 1
    return {"disk": disk, "memory": mem, "task_counts": counts,
            "retry_max": config.MAX_RETRY}


@app.get("/api/tasks/{tid}")
def task_detail(tid: int):
    t = db.get_task(tid)
    if not t:
        raise HTTPException(404, "任务不存在")
    return t


# ---------- 打包下载（笔记 .md + 图片，图片改相对路径，解压即可离线看）----------
_IMG_RE = re.compile(r"!\[([^\]]*)\]\((/frames/[^)]+)\)")


def _collect_images(md_text: str):
    """返回 (img_map, 无冲突文件名列表)：web路径 /frames/... -> zip内路径 images/xxx.jpg"""
    img_map, used = {}, set()
    for m in _IMG_RE.finditer(md_text):
        web = m.group(2)
        if web in img_map:
            continue
        rel = web[len("/frames/"):]
        fname = Path(rel).name
        if fname in used:
            fname = rel.replace("/", "_")
        used.add(fname)
        img_map[web] = f"images/{fname}"
    return img_map


def _md_to_html(md_text: str, link_rewrite=None) -> str:
    """
    极简 Markdown -> 单文件 HTML，图片用 base64 内嵌（完全自包含，双击即可离线看，无需 Markdown 软件）。
    覆盖本项目笔记会用到的语法：# / ## / ### 标题、列表、**粗体**、`code`、> 引用、表格、图片、LaTeX 原样保留。
    """
    import base64
    import html as _html

    def _img_tag(m):
        alt, web = m.group(1), m.group(2)
        src = config.FRAMES_DIR / web[len("/frames/"):]
        if src.exists():
            try:
                b64 = base64.b64encode(src.read_bytes()).decode()
                return f'<img alt="{_html.escape(alt)}" src="data:image/jpeg;base64,{b64}" />'
            except Exception:
                pass
        return f'<img alt="{_html.escape(alt)}" src="{_html.escape(web)}" />'

    # 先替换图片为占位，避免后续转义破坏
    imgs = []
    def _stash(m):
        imgs.append(_img_tag(m))
        return f"\x00IMG{len(imgs)-1}\x00"
    text = _IMG_RE.sub(_stash, md_text)

    lines = text.split("\n")
    out, in_ul, in_ol, in_table = [], False, False, False
    toc = []          # [(level, id, 纯文本标题)]，用于生成目录锚点
    hidx = [0]        # 稳定 id 计数

    def _inline(s):
        s = _html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        if link_rewrite:
            def _lnk(m):
                txt_, href_ = m.group(1), m.group(2)
                return f'<a href="{link_rewrite(href_)}">{txt_}</a>'
            s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _lnk, s)
        else:
            s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
        # 还原图片占位（图片已是 HTML，不能再被转义）
        s = re.sub(r"\x00IMG(\d+)\x00", lambda m: imgs[int(m.group(1))], s)
        return s

    def _heading(level: int, text_): 
        """生成带锚点的标题，并登记到 toc（去掉 *Content-[mm:ss] 标记用于目录显示）。"""
        hidx[0] += 1
        anchor = f"h{hidx[0]}"
        clean = re.sub(r"\*?Content-\s*\[?\d{1,2}:\d{2}\]?\*?", "", text_).strip()
        if level == 2:      # 只把二级标题放进目录（网课章节层级刚好）
            toc.append((level, anchor, clean))
        return f'<h{level} id="{anchor}">{_inline(text_)}</h{level}>'

    def _close():
        nonlocal in_ul, in_ol, in_table
        if in_ul: out.append("</ul>"); in_ul = False
        if in_ol: out.append("</ol>"); in_ol = False
        if in_table: out.append("</table>"); in_table = False

    for raw in lines:
        line = raw.rstrip()
        # 图片占位单独成行
        if re.fullmatch(r"\x00IMG\d+\x00", line.strip()):
            _close(); out.append(_inline(line.strip())); continue
        if not line.strip():
            _close(); continue
        if line.startswith("### "):
            _close(); out.append(_heading(3, line[4:]))
        elif line.startswith("## "):
            _close(); out.append(_heading(2, line[3:]))
        elif line.startswith("# "):
            _close(); out.append(_heading(1, line[2:]))
        elif line.startswith(">"):
            _close(); out.append(f"<blockquote>{_inline(line.lstrip('> ').strip())}</blockquote>")
        elif re.match(r"^\s*[-*]\s+", line):
            if in_ol: out.append("</ol>"); in_ol = False
            if not in_ul: out.append("<ul>"); in_ul = True
            item = re.sub(r"^\s*[-*]\s+", "", line)
            out.append("<li>" + _inline(item) + "</li>")
        elif re.match(r"^\s*\d+[.)]\s+", line):
            if in_ul: out.append("</ul>"); in_ul = False
            if not in_ol: out.append("<ol>"); in_ol = True
            item = re.sub(r"^\s*\d+[.)]\s+", "", line)
            out.append("<li>" + _inline(item) + "</li>")
        else:
            _close(); out.append(f"<p>{_inline(line)}</p>")
    _close()

    body = "\n".join(out)

    # 生成目录（二级标题）。用 <details> 可折叠，手机端不占地。
    toc_html = ""
    if len(toc) >= 3:
        items = "".join(
            f'<li><a href="#{aid}">{_html.escape(t)}</a></li>'
            for _lvl, aid, t in toc)
        toc_html = (
            '<details class="toc" open><summary>📑 目录（点击跳转）</summary>'
            f'<ol>{items}</ol></details>')

    return (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>网课笔记</title><style>"
        "body{max-width:820px;margin:24px auto;padding:0 18px;font-family:-apple-system,"
        "'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;line-height:1.75;color:#222}"
        "h1{font-size:1.7em;border-bottom:2px solid #eee;padding-bottom:.3em}"
        "h2{font-size:1.35em;margin-top:1.6em;border-left:4px solid #4b8bf5;padding-left:.5em;"
        "scroll-margin-top:12px}h3{font-size:1.12em;color:#333;margin-top:1.2em;scroll-margin-top:12px}"
        "img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:6px;"
        "margin:10px 0;display:block}"
        "code{background:#f3f4f6;padding:.1em .35em;border-radius:4px;font-size:.92em}"
        "blockquote{color:#555;border-left:3px solid #ccc;margin:.6em 0;padding:.2em .8em;"
        "background:#fafafa}ul,ol{padding-left:1.5em}li{margin:.25em 0}"
        "strong{color:#000}"
        ".toc{background:#f7f9fc;border:1px solid #e3e9f2;border-radius:8px;padding:10px 14px;margin:14px 0}"
        ".toc summary{cursor:pointer;font-weight:600;color:#2b5dbf}"
        ".toc ol{margin:8px 0 0;font-size:.95em}.toc a{color:#2b5dbf;text-decoration:none}"
        "</style></head><body>" + toc_html + body + "</body></html>"
    )


def _build_note_zip(md_text: str, zip_name_note: str) -> bytes:
    """把 Markdown 里引用的 /frames/... 图片收进 images/ 目录，另附一份自包含 HTML，打成 zip。"""
    buf = io.BytesIO()
    img_map = _collect_images(md_text)

    # 改写 Markdown 图片路径为相对 images/
    def _repl(m):
        alt, web = m.group(1), m.group(2)
        return f"![{alt}]({img_map.get(web, web)})"
    new_md = _IMG_RE.sub(_repl, md_text)

    # 生成自包含 HTML（图片 base64 内嵌，双击即可看图）
    html_doc = _md_to_html(md_text)
    html_name = (zip_name_note.rsplit(".", 1)[0] if "." in zip_name_note else zip_name_note) + ".html"

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(zip_name_note, new_md.encode("utf-8"))
        z.writestr(html_name, html_doc.encode("utf-8"))
        for web, rel_zip in img_map.items():
            src = config.FRAMES_DIR / web[len("/frames/"):]
            if src.exists():
                try:
                    z.write(str(src), rel_zip)
                except Exception:
                    pass
    return buf.getvalue()


def _note_zip_response(md_text: str, note_name: str, zip_name: str) -> Response:
    data = _build_note_zip(md_text, note_name)
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


@app.get("/api/tasks/{tid}/bundle")
def task_bundle(tid: int):
    """打包下载：笔记 .md + images/ 图片目录（解压后本地可看带图笔记）。"""
    t = db.get_task(tid)
    if not t or not t.get("output_md") or not Path(t["output_md"]).exists():
        raise HTTPException(404, "笔记尚未生成")
    md = Path(t["output_md"]).read_text(encoding="utf-8")
    return _note_zip_response(md, "note.md", f"note_{tid}.zip")


@app.get("/api/tasks/{tid}/html")
def task_html(tid: int, inline: int = 0):
    """单文件 HTML（图片 base64 内嵌）——手机浏览器直接打开即看，无需解压。
    inline=1 时在浏览器内直接渲染（不触发下载）。"""
    t = db.get_task(tid)
    if not t or not t.get("output_md") or not Path(t["output_md"]).exists():
        raise HTTPException(404, "笔记尚未生成")
    md = Path(t["output_md"]).read_text(encoding="utf-8")
    html_doc = _md_to_html(md)
    disp = "inline" if inline else f'attachment; filename="note_{tid}.html"'
    return Response(content=html_doc.encode("utf-8"), media_type="text/html",
                    headers={"Content-Disposition": disp})


@app.get("/api/tasks/{tid}/markdown", response_class=PlainTextResponse)
def task_markdown(tid: int):
    t = db.get_task(tid)
    if not t:
        raise HTTPException(404, "任务不存在")
    if not t.get("output_md") or not Path(t["output_md"]).exists():
        raise HTTPException(404, "笔记尚未生成")
    return Path(t["output_md"]).read_text(encoding="utf-8")


@app.get("/api/tasks/{tid}/download")
def task_download(tid: int):
    t = db.get_task(tid)
    if not t or not t.get("output_md") or not Path(t["output_md"]).exists():
        raise HTTPException(404, "笔记尚未生成")
    safe = f"note_{tid}.md"
    return FileResponse(t["output_md"], media_type="text/markdown", filename=safe)


@app.get("/", response_class=HTMLResponse)
def index():
    idx = config.WEB_DIR / "index.html"
    if not idx.exists():
        return HTMLResponse("<h1>前端缺失</h1>")
    return HTMLResponse(idx.read_text(encoding="utf-8"))
