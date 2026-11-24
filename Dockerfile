FROM python:3.11-slim

# 系统依赖：ffmpeg 用于抽音频/抽帧
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY web ./web

# 数据目录（挂载卷持久化）
RUN mkdir -p /app/data/output /app/data/audio /app/data/frames /app/uploads

ENV PYTHONUNBUFFERED=1
WORKDIR /app
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
