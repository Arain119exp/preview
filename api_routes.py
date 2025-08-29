# api_routes.py
import asyncio
import base64
import logging
import mimetypes
import os
import re
import time
import uuid
from datetime import datetime

import psutil
import sys
from fastapi import (APIRouter, Depends, File, Header, HTTPException,
                     UploadFile)
from fastapi.responses import JSONResponse, StreamingResponse

from api_models import ChatCompletionRequest
from api_services import (make_request_with_failover,
                          make_request_with_fast_failover,
                          should_use_fast_failover,
                          stream_non_stream_keep_alive,
                          stream_with_failover, stream_with_fast_failover)
from api_utils import (GeminiAntiDetectionInjector, check_gemini_key_health,
                       delete_file_from_gemini, get_actual_model_name,
                       inject_prompt_to_messages, openai_to_gemini,
                       upload_file_to_gemini, validate_file_for_gemini)
from database import Database
from dependencies import (get_anti_detection, get_db, get_keep_alive_enabled,
                          get_request_count, get_start_time)

logger = logging.getLogger(__name__)

# Main Router for user-facing APIs
router = APIRouter()

# Admin Router for management APIs
admin_router = APIRouter(prefix="/admin", tags=["管理 API"])


# Mock file storage (as in the original single-file app)
file_storage = {}
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

MAX_INLINE_SIZE = 20 * 1024 * 1024

# ===============================================================================
# General and v1 API Routes
# ===============================================================================

@router.get("/", summary="服务根端点", tags=["通用"])
async def root(
    db: Database = Depends(get_db),
    keep_alive_enabled: bool = Depends(get_keep_alive_enabled)
):
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
            "Multi-key polling & load balancing", "OpenAI API compatibility", "Rate limiting & usage analytics",
            "Thinking mode support", "Optimized Gemini 2.5 multimodal", "Streaming responses", "Fast failover",
            "Real-time monitoring", "Health checking", "Adaptive load balancing", "Auto keep-alive",
            "Auto-cleanup unhealthy keys", "Anti-automation detection"
        ],
        "endpoints": {
            "chat_completions": "/v1/chat/completions", "models": "/v1/models", "files": "/v1/files",
            "api_info": "/v1", "health": "/health", "status": "/status", "admin": "/admin/*", "docs": "/docs"
        },
        "supported_models": db.get_supported_models(),
        "service_status": {
            "active_gemini_keys": len(db.get_available_gemini_keys()),
            "thinking_enabled": db.get_thinking_config().get('enabled', False),
            "thinking_budget": db.get_thinking_config().get('budget', -1),
            "uptime_seconds": int(time.time() - start_time),
            "total_requests": request_count,
            "keep_alive_enabled": keep_alive_enabled,
            "auto_cleanup_enabled": db.get_auto_cleanup_config()['enabled'],
            "auto_cleanup_threshold": db.get_auto_cleanup_config()['days_threshold'],
            "anti_detection_enabled": db.get_config('anti_detection_enabled', 'true').lower() == 'true',
            "fast_failover_enabled": db.get_failover_config()['fast_failover_enabled'],
        },
        "multimodal_support": {
            "images": ["jpeg", "png", "gif", "webp", "bmp"],
            "audio": ["mp3", "wav", "ogg", "mp4", "flac", "aac"],
            "video": ["mp4", "avi", "mov", "webm", "quicktime"],
            "documents": ["pdf", "txt", "csv", "docx", "xlsx"]
        },
        "timestamp": datetime.now().isoformat()
    }

@router.post("/v1/files", summary="上传文件", tags=["用户 API"])
async def upload_file(
        file: UploadFile = File(..., description="要上传的文件，最大100MB"),
        authorization: str = Header(None, description="用户访问密钥，格式为 'Bearer sk-...'"),
        db: Database = Depends(get_db)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    user_key = db.validate_user_key(authorization.replace("Bearer ", ""))
    if not user_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    file_content = await file.read()
    mime_type = file.content_type or mimetypes.guess_type(file.filename)[0] or 'application/octet-stream'
    validation_result = validate_file_for_gemini(file_content, mime_type, file.filename)
    file_id = f"file-{uuid.uuid4().hex}"
    file_info = {
        "id": file_id, "object": "file", "bytes": validation_result["size"], "created_at": int(time.time()),
        "filename": file.filename, "purpose": "multimodal", "mime_type": mime_type,
        "use_inline": validation_result["use_inline"]
    }

    if validation_result["use_inline"]:
        file_info["data"] = base64.b64encode(file_content).decode('utf-8')
        file_info["format"] = "inlineData"
    else:
        gemini_keys = db.get_available_gemini_keys()
        if not gemini_keys:
            raise HTTPException(status_code=503, detail="No available Gemini keys for file upload")
        gemini_key = gemini_keys[0]['key']
        gemini_file_uri = await upload_file_to_gemini(file_content, mime_type, file.filename, gemini_key)
        if gemini_file_uri:
            file_info["gemini_file_uri"] = gemini_file_uri
            file_info["gemini_key_used"] = gemini_key
            file_info["format"] = "fileData"
        else:
            raise HTTPException(status_code=500, detail="Failed to upload file to Gemini File API")

    file_storage[file_id] = file_info
    return {
        "id": file_id, "object": "file", "bytes": validation_result["size"], "created_at": file_info["created_at"],
        "filename": file.filename, "purpose": "multimodal", "format": file_info["format"]
    }

@router.get("/v1/files", summary="列出已上传的文件", tags=["用户 API"])
async def list_files(authorization: str = Header(None), db: Database = Depends(get_db)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    if not db.validate_user_key(authorization.replace("Bearer ", "")):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    files = [{"id": fid, **finfo} for fid, finfo in file_storage.items()]
    return {"object": "list", "data": files}

@router.get("/v1/files/{file_id}", summary="获取文件信息", tags=["用户 API"])
async def get_file(file_id: str, authorization: str = Header(None), db: Database = Depends(get_db)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    if not db.validate_user_key(authorization.replace("Bearer ", "")):
        raise HTTPException(status_code=401, detail="Invalid API key")
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")
    return file_storage[file_id]

@router.delete("/v1/files/{file_id}", summary="删除文件", tags=["用户 API"])
async def delete_file(file_id: str, authorization: str = Header(None), db: Database = Depends(get_db)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    if not db.validate_user_key(authorization.replace("Bearer ", "")):
        raise HTTPException(status_code=401, detail="Invalid API key")
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = file_storage[file_id]
    if "gemini_file_uri" in file_info:
        await delete_file_from_gemini(file_info["gemini_file_uri"], file_info["gemini_key_used"])
    
    del file_storage[file_id]
    return {"id": file_id, "object": "file", "deleted": True}

@router.post("/v1/chat/completions", summary="创建聊天补全", tags=["用户 API"])
async def chat_completions(
        request: ChatCompletionRequest,
        authorization: str = Header(None),
        db: Database = Depends(get_db)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    user_key_info = db.validate_user_key(authorization.replace("Bearer ", ""))
    if not user_key_info:
        raise HTTPException(status_code=401, detail="Invalid API key")

    actual_model_name = get_actual_model_name(request.model)
    request.messages = inject_prompt_to_messages(request.messages)
    gemini_request = openai_to_gemini(request, enable_anti_detection=True)

    stream_mode = db.get_stream_mode_config().get('mode', 'auto')
    has_tool_calls = bool(request.tools or request.tool_choice)
    decryption_enabled = db.get_response_decryption_config().get('enabled', False)

    should_stream = request.stream
    if decryption_enabled or has_tool_calls:
        should_stream = False
    elif stream_mode == 'stream':
        should_stream = True
    elif stream_mode == 'non_stream':
        should_stream = False

    if should_stream:
        if await should_use_fast_failover():
            return StreamingResponse(stream_with_fast_failover(gemini_request, request, actual_model_name, user_key_info), media_type="text/event-stream")
        else:
            return StreamingResponse(stream_with_failover(gemini_request, request, actual_model_name, user_key_info), media_type="text/event-stream")
    else:
        if await should_use_fast_failover():
            response = await make_request_with_fast_failover(gemini_request, request, actual_model_name, user_key_info)
        else:
            response = await make_request_with_failover(gemini_request, request, actual_model_name, user_key_info)
        return JSONResponse(content=response)

@router.get("/v1/models", summary="列出可用模型", tags=["用户 API"])
async def list_models(db: Database = Depends(get_db)):
    models = db.get_supported_models()
    model_list = [{"id": model, "object": "model", "created": int(time.time()), "owned_by": "google"} for model in models]
    return {"object": "list", "data": model_list}

# ===============================================================================
# Admin API Routes
# ===============================================================================

@admin_router.post("/health/check-all", summary="一键健康检测")
async def check_all_keys_health_endpoint(db: Database = Depends(get_db)):
    active_keys = [key for key in db.get_all_gemini_keys() if key['status'] == 1]
    if not active_keys:
        return {"success": True, "message": "No active keys to check"}
    
    tasks = [check_gemini_key_health(key['key']) for key in active_keys]
    results = await asyncio.gather(*tasks)
    
    healthy_count = 0
    output_results = []
    for key, result in zip(active_keys, results):
        db.update_key_performance(key['id'], result['healthy'], result['response_time'])
        db.record_daily_health_status(key['id'], result['healthy'], result['response_time'])
        if result['healthy']:
            healthy_count += 1
        output_results.append({"key_id": key['id'], **result})
        
    return {
        "success": True, "message": f"Health check completed: {healthy_count}/{len(active_keys)} keys healthy",
        "results": output_results
    }

@admin_router.get("/health/summary", summary="获取健康状态汇总")
async def get_health_summary(db: Database = Depends(get_db)):
    return {"success": True, "summary": db.get_keys_health_summary()}

@admin_router.get("/cleanup/status", summary="获取自动清理状态")
async def get_cleanup_status(db: Database = Depends(get_db)):
    config = db.get_auto_cleanup_config()
    return {
        "success": True, **config,
        "at_risk_keys": db.get_at_risk_keys(config['days_threshold'])
    }

@admin_router.post("/cleanup/config", summary="更新自动清理配置")
async def update_cleanup_config(request: dict, db: Database = Depends(get_db)):
    db.set_auto_cleanup_config(
        enabled=request.get('enabled'),
        days_threshold=request.get('days_threshold'),
        min_checks_per_day=request.get('min_checks_per_day')
    )
    return {"success": True, "message": "Config updated"}

@admin_router.get("/config/failover", summary="获取故障转移配置")
async def get_failover_config_endpoint(db: Database = Depends(get_db)):
    return {"success": True, "config": db.get_failover_config()}

@admin_router.post("/config/failover", summary="更新故障转移配置")
async def update_failover_config_endpoint(request: dict, db: Database = Depends(get_db)):
    db.set_failover_config(
        fast_failover_enabled=request.get('fast_failover_enabled'),
        background_health_check=request.get('background_health_check'),
        health_check_delay=request.get('health_check_delay')
    )
    return {"success": True, "message": "Config updated"}

@admin_router.get("/config/anti-detection", summary="获取防检测配置")
async def get_anti_detection_config_endpoint(db: Database = Depends(get_db), anti_detection: GeminiAntiDetectionInjector = Depends(get_anti_detection)):
    return {
        "success": True,
        "anti_detection_enabled": db.get_config('anti_detection_enabled', 'true').lower() == 'true',
        "disable_for_tools": db.get_config('anti_detection_disable_for_tools', 'true').lower() == 'true',
        "token_threshold": int(db.get_config('anti_detection_token_threshold', '5000')),
        "statistics": anti_detection.get_statistics()
    }

@admin_router.post("/config/anti-detection", summary="更新防检测配置")
async def update_anti_detection_config_endpoint(request: dict, db: Database = Depends(get_db)):
    if (v := request.get('enabled')) is not None: db.set_config('anti_detection_enabled', str(v).lower())
    if (v := request.get('disable_for_tools')) is not None: db.set_config('anti_detection_disable_for_tools', str(v).lower())
    if (v := request.get('token_threshold')) is not None: db.set_config('anti_detection_token_threshold', str(v))
    return {"success": True, "message": "Config updated"}

@admin_router.get("/config/anti-truncation", summary="获取防截断配置")
async def get_anti_truncation_config_endpoint(db: Database = Depends(get_db)):
    return {"success": True, "config": db.get_anti_truncation_config()}

@admin_router.post("/config/anti-truncation", summary="更新防截断配置")
async def update_anti_truncation_config_endpoint(request: dict, db: Database = Depends(get_db)):
    db.set_anti_truncation_config(enabled=request.get('enabled'))
    return {"success": True, "message": "Config updated"}

@admin_router.get("/config/response-decryption", summary="获取响应解密配置")
async def get_response_decryption_config_endpoint(db: Database = Depends(get_db)):
    return {"success": True, "config": db.get_response_decryption_config()}

@admin_router.post("/config/response-decryption", summary="更新响应解密配置")
async def update_response_decryption_config_endpoint(request: dict, db: Database = Depends(get_db)):
    db.set_response_decryption_config(enabled=request.get('enabled'))
    return {"success": True, "message": "Config updated"}

@admin_router.get("/keys/gemini", summary="获取所有Gemini密钥")
async def get_gemini_keys_endpoint(db: Database = Depends(get_db)):
    return {"success": True, "keys": db.get_all_gemini_keys()}

@admin_router.post("/config/gemini-key", summary="添加Gemini密钥")
async def add_gemini_key_endpoint(request: dict, db: Database = Depends(get_db)):
    keys_str = request.get("key", "")
    keys = [k.strip() for k in re.split(r'[,;\s\n\r]+', keys_str) if k.strip()]
    added_count = 0
    for key in keys:
        if db.add_gemini_key(key):
            added_count += 1
    return {"success": True, "message": f"Added {added_count} keys."}

@admin_router.delete("/keys/gemini/{key_id}", summary="删除Gemini密钥")
async def delete_gemini_key_endpoint(key_id: int, db: Database = Depends(get_db)):
    if db.delete_gemini_key(key_id):
        return {"success": True, "message": "Key deleted"}
    raise HTTPException(404, "Key not found")

@admin_router.post("/keys/gemini/{key_id}/toggle", summary="切换Gemini密钥状态")
async def toggle_gemini_key_status_endpoint(key_id: int, db: Database = Depends(get_db)):
    if db.toggle_gemini_key_status(key_id):
        return {"success": True, "message": "Status toggled"}
    raise HTTPException(404, "Key not found")

@admin_router.get("/keys/user", summary="获取所有用户密钥")
async def get_user_keys_endpoint(db: Database = Depends(get_db)):
    return {"success": True, "keys": db.get_all_user_keys()}

@admin_router.post("/config/user-key", summary="生成用户密钥")
async def generate_user_key_endpoint(request: dict, db: Database = Depends(get_db)):
    name = request.get("name", "API User")
    key = db.generate_user_key(name)
    return {"success": True, "key": key, "name": name}

@admin_router.delete("/keys/user/{key_id}", summary="删除用户密钥")
async def delete_user_key_endpoint(key_id: int, db: Database = Depends(get_db)):
    if db.delete_user_key(key_id):
        return {"success": True, "message": "Key deleted"}
    raise HTTPException(404, "Key not found")

@admin_router.post("/keys/user/{key_id}/toggle", summary="切换用户密钥状态")
async def toggle_user_key_status_endpoint(key_id: int, db: Database = Depends(get_db)):
    if db.toggle_user_key_status(key_id):
        return {"success": True, "message": "Status toggled"}
    raise HTTPException(404, "Key not found")

@admin_router.get("/models", summary="列出所有模型配置")
async def list_model_configs_endpoint(db: Database = Depends(get_db)):
    return {"success": True, "models": db.get_all_model_configs()}

@admin_router.get("/models/{model_name}", summary="获取模型配置")
async def get_model_config_endpoint(model_name: str, db: Database = Depends(get_db)):
    config = db.get_model_config(model_name)
    if not config:
        raise HTTPException(404, "Model not found")
    return {"success": True, "model_name": model_name, **config}

@admin_router.post("/models/{model_name}", summary="更新模型配置")
async def update_model_config_endpoint(model_name: str, request: dict, db: Database = Depends(get_db)):
    allowed = ['single_api_rpm_limit', 'single_api_tpm_limit', 'single_api_rpd_limit', 'status']
    update_data = {k: v for k, v in request.items() if k in allowed}
    if not update_data:
        raise HTTPException(422, "No valid fields to update")
    if db.update_model_config(model_name, **update_data):
        return {"success": True, "message": "Model config updated"}
    raise HTTPException(500, "Failed to update model config")

@admin_router.get("/config", summary="获取所有系统配置")
async def get_all_config_endpoint(db: Database = Depends(get_db)):
    # This is the line that caused the 500 error, now fixed.
    return {
        "success": True,
        "system_configs": db.get_all_configs(),
        "thinking_config": db.get_thinking_config(),
        "inject_config": db.get_inject_prompt_config(),
        "cleanup_config": db.get_auto_cleanup_config(),
        "failover_config": db.get_failover_config(),
        "anti_detection_config": {'enabled': db.get_config('anti_detection_enabled', 'true').lower() == 'true'},
        "stream_mode_config": db.get_stream_mode_config(),
        "stream_to_gemini_mode_config": db.get_stream_to_gemini_mode_config(),
    }

@admin_router.get("/stats", summary="获取管理统计信息")
async def get_admin_stats_endpoint(db: Database = Depends(get_db), anti_detection: GeminiAntiDetectionInjector = Depends(get_anti_detection), keep_alive_enabled: bool = Depends(get_keep_alive_enabled)):
    health_summary = db.get_keys_health_summary()
    return {
        "gemini_keys": len(db.get_all_gemini_keys()),
        "active_gemini_keys": len(db.get_available_gemini_keys()),
        "healthy_gemini_keys": health_summary.get('healthy', 0),
        "user_keys": len(db.get_all_user_keys()),
        "active_user_keys": len([k for k in db.get_all_user_keys() if k['status'] == 1]),
        "supported_models": db.get_supported_models(),
        "usage_stats": db.get_all_usage_stats(),
        "thinking_config": db.get_thinking_config(),
        "inject_config": db.get_inject_prompt_config(),
        "cleanup_config": db.get_auto_cleanup_config(),
        "health_summary": health_summary,
        "keep_alive_enabled": keep_alive_enabled,
        "anti_detection_enabled": db.get_config('anti_detection_enabled', 'true').lower() == 'true',
        "anti_detection_stats": anti_detection.get_statistics(),
        "stream_mode_config": db.get_stream_mode_config(),
        "stream_to_gemini_mode_config": db.get_stream_to_gemini_mode_config(),
        "failover_config": db.get_failover_config()
    }
