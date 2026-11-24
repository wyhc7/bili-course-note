"""uvicorn 启动入口。用法：python -m backend.run"""
import uvicorn
from app import config

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT, reload=False)
