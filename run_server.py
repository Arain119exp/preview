#!/usr/bin/env python3
"""
Gemini API Proxy 启动脚本
支持本地开发和生产环境部署
"""

import os
import sys
import logging
import uvicorn
from api_server import app

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def main():
    """主启动函数"""
    # 获取端口配置
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"

    # 检测运行环境
    is_render = bool(os.getenv('RENDER_EXTERNAL_URL'))
    is_local = not is_render

    logger.info("🚀 Starting Gemini API Proxy...")
    logger.info(f"📍 Environment: {'Render' if is_render else 'Local'}")
    logger.info(f"🌐 Host: {host}")
    logger.info(f"🔌 Port: {port}")

    if is_render:
        logger.info(f"🔗 Render URL: {os.getenv('RENDER_EXTERNAL_URL')}")

    # 检查数据库
    try:
        from database import Database
        db = Database()
        logger.info("✅ Database initialized successfully")

        # 检查是否有API密钥
        available_keys = len(db.get_available_gemini_keys())
        if available_keys == 0:
            logger.warning("⚠️ No Gemini API keys configured!")
            logger.info("💡 Add keys via: POST /admin/config/gemini-key")
        else:
            logger.info(f"🔑 Found {available_keys} available API keys")

    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        sys.exit(1)

    # 启动配置
    config = {
        "host": host,
        "port": port,
        "log_level": "info",
        "access_log": True,
        "server_header": False,  # 隐藏服务器信息
        "date_header": False,  # 隐藏日期头
    }

    # 生产环境优化
    if is_render:
        config.update({
            "workers": 1,  # Render 推荐单worker
            "loop": "auto",
            "http": "auto",
            "ws": "auto",
            "lifespan": "on",
            "interface": "asgi3",
            "log_config": None,  # 使用自定义日志配置
        })
    else:
        # 开发环境配置
        config.update({
            "reload": True,  # 热重载
            "reload_dirs": ["."],
            "reload_excludes": ["*.db", "*.log"],
        })

    logger.info("🎯 Server configuration:")
    for key, value in config.items():
        logger.info(f"   {key}: {value}")

    try:
        # 启动服务器
        uvicorn.run(app, **config)
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"💥 Server crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()