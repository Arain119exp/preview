# api_routes.py
from fastapi import APIRouter, Depends, HTTPException, Request, Header, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, List, Any
from datetime import datetime
import os
import time
import uuid
import mimetypes
import base64
import logging

from database import Database
from dependencies import get_db, get_start_time, get_request_count, get_keep_alive_enabled, get_anti_detection
from api_utils import GeminiAntiDetectionInjector, validate_file_for_gemini, upload_file_to_gemini, delete_file_from_gemini, openai_to_gemini, get_actual_model_name, inject_prompt_to_messages
from api_services import stream_non_stream_keep_alive, stream_with_fast_failover, stream_with_failover, make_request_with_fast_failover, make_request_with_failover, should_use_fast_failover
from api_models import ChatCompletionRequest

logger = logging.getLogger(__name__)

router = APIRouter()

# 模拟文件存储
file_storage = {}
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

MAX_INLINE_SIZE = 20 * 1024 * 1024 - 1024  # 略小于20MB


@router.get("/", summary="服务根端点", tags=["通用"])
async def root(
    db: Database = Depends(get_db),
    keep_alive_enabled: bool = Depends(get_keep_alive_enabled)
):
    """
    **服务根端点**

    返回服务的基本信息、状态和功能列表。
    可用于快速检查服务是否正在运行。
    """
    return {
        "service": "Gemini API Proxy",
        "status": "running",
        "version": "1.4.2",
        "features": ["Gemini 2.5 Multimodal"],
        "keep_alive": keep_alive_enabled,
        "auto_cleanup": db.get_auto_cleanup_config()['enabled'],
        "anti_detection": db.get_config('anti_detection_enabled', 'true').lower() == 'true',
        "fast_failover": db.get_failover_config()['fast_failover_enabled'],
        "docs": "/docs",
        "health": "/health"
    }


@router.get("/health", summary="服务健康检查", tags=["通用"])
async def health_check(
    db: Database = Depends(get_db),
    start_time: float = Depends(get_start_time),
    request_count: int = Depends(get_request_count),
    keep_alive_enabled: bool = Depends(get_keep_alive_enabled)
):
    """
    **服务健康检查**

    提供详细的服务健康状况，包括：
    - 运行状态
    - 可用密钥数量
    - 运行环境
    - 运行时长
    - 请求总数
    - 各项高级功能（如保活、自动清理）的启用状态
    """
    available_keys = len(db.get_available_gemini_keys())
    uptime = time.time() - start_time

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "available_keys": available_keys,
        "environment": "render" if os.getenv('RENDER_EXTERNAL_URL') else "local",
        "uptime_seconds": int(uptime),
        "request_count": request_count,
        "version": "1.4.2",
        "multimodal_support": "Gemini 2.5 Optimized",
        "keep_alive_enabled": keep_alive_enabled,
        "auto_cleanup_enabled": db.get_auto_cleanup_config()['enabled'],
        "anti_detection_enabled": db.get_config('anti_detection_enabled', 'true').lower() == 'true',
        "fast_failover_enabled": db.get_failover_config()['fast_failover_enabled']
    }


@router.get("/wake", summary="服务唤醒", tags=["通用"])
async def wake_up(
    db: Database = Depends(get_db),
    keep_alive_enabled: bool = Depends(get_keep_alive_enabled)
):
    """
    **服务唤醒**

    用于在 Render 等平台从休眠状态唤醒服务。
    定期调用此端点可以保持服务持续在线（保活）。
    """
    return {
        "status": "awake",
        "timestamp": datetime.now().isoformat(),
        "message": "Service is active",
        "keep_alive_enabled": keep_alive_enabled,
        "auto_cleanup_enabled": db.get_auto_cleanup_config()['enabled'],
        "anti_detection_enabled": db.get_config('anti_detection_enabled', 'true').lower() == 'true',
        "fast_failover_enabled": db.get_failover_config()['fast_failover_enabled']
    }


@router.get("/status", summary="获取详细服务状态", tags=["通用"])
async def get_status(
    db: Database = Depends(get_db),
    start_time: float = Depends(get_start_time),
    request_count: int = Depends(get_request_count),
    keep_alive_enabled: bool = Depends(get_keep_alive_enabled),
    anti_detection: GeminiAntiDetectionInjector = Depends(get_anti_detection)
):
    """
    **获取详细服务状态**

    返回包括资源使用情况（内存、CPU）、Python版本、支持的模型列表等在内的详细技术状态。
    主要用于调试和监控。
    """
    import psutil
    import sys

    process = psutil.Process(os.getpid())

    return {
        "service": "Gemini API Proxy",
        "status": "running",
        "version": "1.4.2",
        "render_url": os.getenv('RENDER_EXTERNAL_URL'),
        "python_version": sys.version,
        "models": db.get_supported_models(),
        "active_keys": len(db.get_available_gemini_keys()),
        "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "uptime_seconds": int(time.time() - start_time),
        "total_requests": request_count,
        "thinking_enabled": db.get_thinking_config()['enabled'],
        "multimodal_optimized": True,
        "keep_alive_enabled": keep_alive_enabled,
        "auto_cleanup_enabled": db.get_auto_cleanup_config()['enabled'],
        "anti_detection_enabled": db.get_config('anti_detection_enabled', 'true').lower() == 'true',
        "anti_detection_stats": anti_detection.get_statistics(),
        "fast_failover_enabled": db.get_failover_config()['fast_failover_enabled']
    }


@router.get("/metrics", summary="获取服务指标", tags=["通用"])
async def get_metrics(
    db: Database = Depends(get_db),
    start_time: float = Depends(get_start_time),
    request_count: int = Depends(get_request_count),
    keep_alive_enabled: bool = Depends(get_keep_alive_enabled),
    anti_detection: GeminiAntiDetectionInjector = Depends(get_anti_detection)
):
    """
    **获取服务指标**

    提供可用于监控系统（如 Prometheus）的核心指标，包括：
    - 内存和CPU使用率
    - 活跃连接数
    - 数据库大小
    - 各项功能的启用状态
    """
    import psutil

    process = psutil.Process(os.getpid())

    return {
        "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "active_connections": len(db.get_available_gemini_keys()),
        "uptime_seconds": int(time.time() - start_time),
        "requests_count": request_count,
        "database_size_mb": os.path.getsize(db.db_path) / 1024 / 1024 if os.path.exists(db.db_path) else 0,
        "keep_alive_enabled": keep_alive_enabled,
        "auto_cleanup_enabled": db.get_auto_cleanup_config()['enabled'],
        "anti_detection_enabled": db.get_config('anti_detection_enabled', 'true').lower() == 'true',
        "anti_detection_stats": anti_detection.get_statistics(),
        "fast_failover_enabled": db.get_failover_config()['fast_failover_enabled']
    }


@router.get("/v1", summary="获取 v1 API 信息", tags=["通用"])
async def api_v1_info(
    db: Database = Depends(get_db),
    start_time: float = Depends(get_start_time),
    request_count: int = Depends(get_request_count),
    keep_alive_enabled: bool = Depends(get_keep_alive_enabled)
):
    """
    **获取 v1 API 信息**

    返回关于 v1 API 的详细信息，包括：
    - OpenAI 兼容性说明
    - 功能列表
    - 端点列表
    - 支持的模型
    - 多模态能力详情
    """
    available_keys = len(db.get_available_gemini_keys())
    supported_models = db.get_supported_models()
    thinking_config = db.get_thinking_config()
    cleanup_config = db.get_auto_cleanup_config()
    failover_config = db.get_failover_config()

    render_url = os.getenv('RENDER_EXTERNAL_URL')
    base_url = render_url if render_url else 'https://your-service.onrender.com'

    return {
        "service": "Gemini API Proxy",
        "version": "1.4.2",
        "api_version": "v1",
        "compatibility": "OpenAI API v1",
        "description": "A high-performance proxy for Gemini API with OpenAI compatibility.",
        "status": "operational",
        "base_url": base_url,
        "features": [
            "Multi-key polling & load balancing",
            "OpenAI API compatibility",
            "Rate limiting & usage analytics",
            "Thinking mode support",
            "Optimized Gemini 2.5 multimodal",
            "Streaming responses",
            "Fast failover",
            "Real-time monitoring",
            "Health checking",
            "Adaptive load balancing",
            "Auto keep-alive",
            "Auto-cleanup unhealthy keys",
            "Anti-automation detection"
        ],
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
            "files": "/v1/files",
            "api_info": "/v1",
            "health": "/health",
            "status": "/status",
            "admin": "/admin/*",
            "docs": "/docs"
        },
        "supported_models": supported_models,
        "service_status": {
            "active_gemini_keys": available_keys,
            "thinking_enabled": thinking_config.get('enabled', False),
            "thinking_budget": thinking_config.get('budget', -1),
            "uptime_seconds": int(time.time() - start_time),
            "total_requests": request_count,
            "keep_alive_enabled": keep_alive_enabled,
            "auto_cleanup_enabled": cleanup_config['enabled'],
            "auto_cleanup_threshold": cleanup_config['days_threshold'],
            "anti_detection_enabled": db.get_config('anti_detection_enabled', 'true').lower() == 'true',
            "fast_failover_enabled": failover_config['fast_failover_enabled'],

        },
        "multimodal_support": {
            "images": ["jpeg", "png", "gif", "webp", "bmp"],
            "audio": ["mp3", "wav", "ogg", "mp4", "flac", "aac"],
            "video": ["mp4", "avi", "mov", "webm", "quicktime"],
            "documents": ["pdf", "txt", "csv", "docx", "xlsx"]
        },
        "timestamp": datetime.now().isoformat()
    }


# 文件上传端点
@router.post("/v1/files", summary="上传文件", tags=["用户 API"])
async def upload_file(
        file: UploadFile = File(..., description="要上传的文件，最大100MB"),
        authorization: str = Header(None, description="用户访问密钥，格式为 'Bearer sk-...'"),
        db: Database = Depends(get_db)
):
    """
    **上传文件用于多模态对话**

    此端点用于上传文件（图片、音频、视频、文档），以便在 `/v1/chat/completions` 中引用。

    - **智能处理**:
        - 小于20MB的文件将作为内联数据处理。
        - 大于20MB的文件将自动上传到Gemini File API。
    - **返回**: 返回一个文件对象，其中包含一个唯一的 `file_id`，可在对话中引用。
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    api_key = authorization.replace("Bearer ", "")
    user_key = db.validate_user_key(api_key)

    if not user_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    file_content = await file.read()

    mime_type = file.content_type or mimetypes.guess_type(file.filename)[0]
    if not mime_type:
        ext = os.path.splitext(file.filename)[1].lower()
        mime_type_map = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
            '.gif': 'image/gif', '.webp': 'image/webp', '.bmp': 'image/bmp',
            '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.ogg': 'audio/ogg',
            '.mp4': 'video/mp4', '.avi': 'video/avi', '.mov': 'video/quicktime',
            '.pdf': 'application/pdf', '.txt': 'text/plain', '.csv': 'text/csv',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }
        mime_type = mime_type_map.get(ext, 'application/octet-stream')

    validation_result = validate_file_for_gemini(file_content, mime_type, file.filename)

    file_id = f"file-{uuid.uuid4().hex}"

    file_info = {
        "id": file_id,
        "object": "file",
        "bytes": validation_result["size"],
        "created_at": int(time.time()),
        "filename": file.filename,
        "purpose": "multimodal",
        "mime_type": mime_type,
        "use_inline": validation_result["use_inline"]
    }

    if validation_result["use_inline"]:
        # 小文件使用内联数据
        file_info["data"] = base64.b64encode(file_content).decode('utf-8')
        file_info["format"] = "inlineData"
    else:
        # 大文件上传到Gemini File API
        # 获取一个可用的Gemini Key用于文件上传
        gemini_keys = db.get_available_gemini_keys()
        if not gemini_keys:
            raise HTTPException(status_code=503, detail="No available Gemini keys for file upload")

        gemini_key = gemini_keys[0]['key']
        gemini_file_uri = await upload_file_to_gemini(file_content, mime_type, file.filename, gemini_key)

        if gemini_file_uri:
            file_info["gemini_file_uri"] = gemini_file_uri
            file_info["gemini_key_used"] = gemini_key
            file_info["format"] = "fileData"
            logger.info(f"File uploaded to Gemini File API: {gemini_file_uri}")
        else:
            # 如果上传到Gemini失败，回退到本地存储
            file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
            with open(file_path, "wb") as f:
                f.write(file_content)
            file_info["file_path"] = file_path
            file_info["file_uri"] = f"file://{os.path.abspath(file_path)}"
            file_info["format"] = "fileData"
            logger.warning(f"Failed to upload to Gemini, using local storage: {file_path}")

    file_storage[file_id] = file_info

    logger.info(
        f"File uploaded: {file_id}, size: {validation_result['size']} bytes, "
        f"type: {mime_type}, format: {file_info['format']}"
    )

    return {
        "id": file_id,
        "object": "file",
        "bytes": validation_result["size"],
        "created_at": file_info["created_at"],
        "filename": file.filename,
        "purpose": "multimodal",
        "format": file_info["format"]
    }


@router.get("/v1/files", summary="列出已上传的文件", tags=["用户 API"])
async def list_files(authorization: str = Header(None, description="用户访问密钥，格式为 'Bearer sk-...'"), db: Database = Depends(get_db)):
    """
    **列出已上传的文件**

    返回当前服务器上所有已上传文件的列表。
    文件会在24小时后自动清理。
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    api_key = authorization.replace("Bearer ", "")
    user_key = db.validate_user_key(api_key)

    if not user_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    files = []
    for file_id, file_info in file_storage.items():
        files.append({
            "id": file_id,
            "object": "file",
            "bytes": file_info["bytes"],
            "created_at": file_info["created_at"],
            "filename": file_info["filename"],
            "purpose": file_info["purpose"]
        })

    return {
        "object": "list",
        "data": files
    }


@router.get("/v1/files/{file_id}", summary="获取文件信息", tags=["用户 API"])
async def get_file(
    file_id: str,
    authorization: str = Header(None, description="用户访问密钥，格式为 'Bearer sk-...'"),
    db: Database = Depends(get_db)
):
    """
    **获取文件信息**

    根据文件ID获取指定文件的详细信息。
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    api_key = authorization.replace("Bearer ", "")
    user_key = db.validate_user_key(api_key)

    if not user_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")

    file_info = file_storage[file_id]
    return {
        "id": file_id,
        "object": "file",
        "bytes": file_info["bytes"],
        "created_at": file_info["created_at"],
        "filename": file_info["filename"],
        "purpose": file_info["purpose"]
    }


@router.delete("/v1/files/{file_id}", summary="删除文件", tags=["用户 API"])
async def delete_file(
    file_id: str,
    authorization: str = Header(None, description="用户访问密钥，格式为 'Bearer sk-...'"),
    db: Database = Depends(get_db)
):
    """
    **删除文件**

    从服务器删除指定的文件。
    如果文件已上传到Gemini File API，也会尝试从Gemini侧删除。
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    api_key = authorization.replace("Bearer ", "")
    user_key = db.validate_user_key(api_key)

    if not user_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")

    file_info = file_storage[file_id]

    # 如果文件存储在Gemini File API，先从Gemini删除
    if "gemini_file_uri" in file_info and "gemini_key_used" in file_info:
        await delete_file_from_gemini(file_info["gemini_file_uri"], file_info["gemini_key_used"])

    # 如果有本地文件，也删除
    if "file_path" in file_info and os.path.exists(file_info["file_path"]):
        os.remove(file_info["file_path"])

    del file_storage[file_id]

    logger.info(f"File deleted: {file_id}")

    return {
        "id": file_id,
        "object": "file",
        "deleted": True
    }

# chat_completions端点
@router.post("/v1/chat/completions", summary="创建聊天补全", tags=["用户 API"])
async def chat_completions(
        request: ChatCompletionRequest,
        authorization: str = Header(None, description="用户访问密钥，格式为 'Bearer sk-...'"),
        db: Database = Depends(get_db)
):
    """
    **创建聊天补全**

    这是核心的对话接口，与OpenAI的API完全兼容。
    它支持：
    - 文本对话
    - 多模态对话（引用已上传的文件）
    - 流式响应
    - 高级功能如思考模式和提示词注入
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    api_key = authorization.replace("Bearer ", "")
    user_key = db.validate_user_key(api_key)

    if not user_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    user_key_info = user_key

    if not request.messages or len(request.messages) == 0:
        raise HTTPException(status_code=422, detail="Messages cannot be empty")

    # 验证消息格式和多模态内容
    total_content_size = 0
    for msg in request.messages:
        if not hasattr(msg, 'role') or not hasattr(msg, 'content'):
            raise HTTPException(status_code=422, detail="Invalid message format")
        if msg.role not in ['system', 'user', 'assistant']:
            raise HTTPException(status_code=422, detail=f"Invalid role: {msg.role}")

        # 检查多模态内容大小
        if isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict) and item.get('type') in ['image', 'audio', 'video', 'document']:
                    inline_data = item.get('inline_data') or item.get('inlineData')
                    if inline_data and 'data' in inline_data:
                        total_content_size += len(inline_data['data']) * 3 // 4

    # 检查总请求大小（Gemini 2.5限制20MB）
    if total_content_size > MAX_INLINE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Total multimodal content size exceeds {MAX_INLINE_SIZE // (1024 * 1024)}MB limit"
        )

    actual_model_name = get_actual_model_name(request.model)
    request.messages = inject_prompt_to_messages(request.messages)

    # 使用增强版的转换函数，包含防检测功能
    gemini_request = openai_to_gemini(request, enable_anti_detection=True)

    has_multimodal = any(msg.has_multimodal_content() for msg in request.messages)
    if has_multimodal:
        logger.info(f"Processing multimodal request for model {actual_model_name}")

    # 记录防检测应用情况
    anti_detection_enabled = db.get_config('anti_detection_enabled', 'true').lower() == 'true'
    if anti_detection_enabled:
        logger.info(f"Anti-detection processing applied for user {user_key_info['name']}")


    
    # 获取管理者配置的流式模式
    stream_mode_config = db.get_stream_mode_config()
    stream_mode = stream_mode_config.get('mode', 'auto')

    # 获取与 Gemini 通信的流式模式
    stream_to_gemini_mode_config = db.get_stream_to_gemini_mode_config()
    stream_to_gemini_mode = stream_to_gemini_mode_config.get('mode', 'auto')
    
    # 检查是否有工具调用
    has_tool_calls = bool(request.tools or request.tool_choice)
    
    # 根据流式模式配置决定是否使用流式响应
    should_stream = request.stream  # 默认跟随用户请求
    logger.info(f"DEBUG: request.stream={request.stream}, stream_mode={stream_mode}, has_tool_calls={has_tool_calls}")
    
    # 如果启用了响应解密，则强制使用非流式模式
    decryption_enabled = db.get_response_decryption_config().get('enabled', False)
    if decryption_enabled:
        should_stream = False
        logger.info("Response decryption is enabled, forcing non-streaming mode.")

    # 工具调用强制使用非流式模式
    if has_tool_calls:
        should_stream = False
        logger.info("Tool calls detected, forcing non-streaming mode")
    elif stream_mode == 'stream':
        should_stream = True  # 强制流式
        logger.info("Stream mode forced to streaming")
    elif stream_mode == 'non_stream':
        should_stream = False  # 强制非流式
        logger.info("Stream mode forced to non-streaming")
    # stream_mode == 'auto' 时保持原有逻辑，跟随用户请求

    # ===== 计算向 Gemini 的流式模式 =====
    should_stream_to_gemini = True  # 默认向 Gemini 使用流式
    if has_tool_calls:
        should_stream_to_gemini = False
        logger.info("Tool calls detected, forcing non-streaming mode to Gemini")
    elif stream_to_gemini_mode == 'stream':
        should_stream_to_gemini = True
        logger.info("Gemini stream mode forced to streaming")
    elif stream_to_gemini_mode == 'non_stream':
        should_stream_to_gemini = False
        logger.info("Gemini stream mode forced to non-streaming")
    # auto 时保持默认

    logger.info(f"DEBUG: Final should_stream={should_stream}, should_stream_to_gemini={should_stream_to_gemini}")

    if should_stream:
        # 当客户端要求流式，但向 Gemini 使用非流式时，走 keep-alive 兼容路径
        if not should_stream_to_gemini:
            return StreamingResponse(
                stream_non_stream_keep_alive(
                    gemini_request,
                    request,
                    actual_model_name,
                    user_key_info=user_key_info
                ),
                media_type="text/event-stream; charset=utf-8"
            )
        if await should_use_fast_failover():
            return StreamingResponse(
                stream_with_fast_failover(
                    gemini_request,
                    request,
                    actual_model_name,
                    user_key_info=user_key_info,

                ),
                media_type="text/event-stream; charset=utf-8"
            )
        else:
            # 回退到传统故障转移逻辑
            return StreamingResponse(
                stream_with_failover(
                    gemini_request,
                    request,
                    actual_model_name,
                    user_key_info=user_key_info,

                ),
                media_type="text/event-stream; charset=utf-8"
            )
    else:
        logger.info("DEBUG: Using non-streaming response path")
        # 使用统一的流式架构（内部收集为完整响应）
        if await should_use_fast_failover():
            openai_response = await make_request_with_fast_failover(
                gemini_request,
                request,
                actual_model_name,
                user_key_info=user_key_info,

            )
        else:
            # 回退到传统故障转移逻辑
            openai_response = await make_request_with_failover(
                gemini_request,
                request,
                actual_model_name,
                user_key_info=user_key_info,

            )

        # 直接返回已经转换好的OpenAI格式响应
        return JSONResponse(content=openai_response)


@router.get("/v1/models", summary="列出可用模型", tags=["用户 API"])
async def list_models(db: Database = Depends(get_db)):
    """
    **列出可用模型**

    返回当前服务支持的所有模型列表，格式与OpenAI兼容。
    """
    models = db.get_supported_models()

    model_list = []
    for model in models:
        model_list.append({
            "id": model,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google"
        })

    return {"object": "list", "data": model_list}


# ===================================================================================
# 管理 API (Admin API)
# ===================================================================================

admin_router = APIRouter(prefix="/admin", tags=["管理 API"])

@admin_router.get("/gemini_keys", summary="获取所有 Gemini Keys", response_model=List[Dict[str, Any]])
async def get_gemini_keys(db: Database = Depends(get_db)):
    """获取数据库中所有 Gemini API Keys 的列表。"""
    return db.get_gemini_keys()

@admin_router.post("/gemini_keys", summary="添加或更新 Gemini Key")
async def add_or_update_gemini_key(key: str, db: Database = Depends(get_db)):
    """添加一个新的 Gemini API Key 或更新一个已存在的 Key。"""
    db.add_gemini_key(key)
    return {"message": "Gemini key added/updated successfully"}

@admin_router.delete("/gemini_keys/{key}", summary="删除 Gemini Key")
async def delete_gemini_key(key: str, db: Database = Depends(get_db)):
    """根据提供的 Key 字符串删除一个 Gemini API Key。"""
    db.delete_gemini_key(key)
    return {"message": "Gemini key deleted successfully"}

@admin_router.get("/user_keys", summary="获取所有用户 Keys", response_model=List[Dict[str, Any]])
async def get_user_keys(db: Database = Depends(get_db)):
    """获取所有用户访问密钥的列表。"""
    return db.get_user_keys()

@admin_router.post("/user_keys", summary="创建用户 Key")
async def create_user_key(name: str, db: Database = Depends(get_db)):
    """创建一个新的用户访问密钥。"""
    new_key = db.create_user_key(name)
    return {"message": "User key created successfully", "key": new_key}

@admin_router.delete("/user_keys/{key}", summary="删除用户 Key")
async def delete_user_key(key: str, db: Database = Depends(get_db)):
    """根据提供的 Key 字符串删除一个用户访问密钥。"""
    db.delete_user_key(key)
    return {"message": "User key deleted successfully"}

@admin_router.get("/config", summary="获取所有配置")
async def get_all_config(db: Database = Depends(get_db)):
    """获取所有系统配置项。"""
    return db.get_all_config()

@admin_router.post("/config", summary="更新配置")
async def update_config(key: str, value: str, db: Database = Depends(get_db)):
    """更新一个配置项。"""
    db.set_config(key, value)
    return {"message": f"Config '{key}' updated successfully"}

@admin_router.get("/logs", summary="获取服务日志")
async def get_logs(limit: int = 100, db: Database = Depends(get_db)):
    """获取最新的服务日志。"""
    # This is a placeholder. In a real app, you'd read from a log file or a logging service.
    return {"message": "Log fetching is not fully implemented in this version."}


@admin_router.get("/stats", summary="获取管理统计信息")
async def get_admin_stats(
    db: Database = Depends(get_db),
    start_time: float = Depends(get_start_time),
    request_count: int = Depends(get_request_count)
):
    """获取核心管理统计数据，用于仪表盘展示。"""
    health_summary = db.get_keys_health_summary()
    uptime = time.time() - start_time
    
    return {
        "total_requests": request_count,
        "uptime_seconds": int(uptime),
        "active_gemini_keys": len(db.get_round_robin_key()),
        "total_gemini_keys": len(db.get_all_gemini_keys()),
        "healthy_gemini_keys": health_summary.get('healthy', 0),
        "total_user_keys": len(db.get_all_user_keys()),
        "database_size_mb": os.path.getsize(db.db_path) / 1024 / 1024 if os.path.exists(db.db_path) else 0,
        "usage_stats": db.get_usage_stats(),
        "failover_config": db.get_failover_config()
    }


# 将管理路由包含到主路由中
router.include_router(admin_router)
