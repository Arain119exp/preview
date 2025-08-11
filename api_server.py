import asyncio
import json
import time
import uuid
import logging
import os
import sys
import base64
import mimetypes
import random
import hashlib
import io  # 新增，用于 google-genai 文件上传
import itertools  # 轮询计数器
_rr_counter = itertools.count()  # 全局递增计数器
_rr_lock = asyncio.Lock()  # 轮询锁
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncGenerator, Union, Any
from contextlib import asynccontextmanager

from google import genai
from google.genai import types
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Header, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError, validator
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from database import Database

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 全局变量
request_count = 0
start_time = time.time()  # 服务启动时间

# -----------------------
# GenAI Client 缓存，避免每次请求重复实例化导致 AFC 日志刷屏
# -----------------------
_client_cache: Dict[str, genai.Client] = {}


def get_cached_client(api_key: str) -> genai.Client:
    """按 key 复用 google-genai Client，减小握手 & 日志开销"""
    client = _client_cache.get(api_key)
    if client is None:
        client = genai.Client(api_key=api_key)
        _client_cache[api_key] = client
    return client

# 防自动化检测注入器
class GeminiAntiDetectionInjector:
    """
    防自动化检测的 Unicode 字符注入器
    """
    def __init__(self):
        # Unicode符号库
        self.safe_symbols = [
            # 数学符号
            '∙', '∘', '∞', '≈', '≠', '≤', '≥', '±', '∓', '×', '÷', '∂', '∆', '∇',

            # 几何图形
            '○', '●', '◯', '◦', '◉', '◎', '⦿', '⊙', '⊚', '⊛', '⊜', '⊝',
            '□', '■', '▢', '▣', '▤', '▥', '▦', '▧', '▨', '▩', '▪', '▫',
            '△', '▲', '▴', '▵', '▶', '▷', '▸', '▹', '►', '▻', '▼', '▽',
            '◀', '◁', '◂', '◃', '◄', '◅', '◆', '◇', '◈', '◉', '◊',

            # 星号和装饰符号
            '☆', '★', '⭐', '✦', '✧', '✩', '✪', '✫', '✬', '✭', '✮', '✯',
            '✰', '✱', '✲', '✳', '✴', '✵', '✶', '✷', '✸', '✹', '✺', '✻',

            # 箭头符号
            '→', '←', '↑', '↓', '↔', '↕', '↖', '↗', '↘', '↙', '↚', '↛',
            '⇒', '⇐', '⇑', '⇓', '⇔', '⇕', '⇖', '⇗', '⇘', '⇙', '⇚', '⇛',

            # 标点和分隔符
            '‖', '‗', '‰', '‱', '′', '″', '‴', '‵', '‶', '‷', '‸', '‹', '›',
            '‼', '‽', '‾', '‿', '⁀', '⁁', '⁂', '⁃', '⁆', '⁇', '⁈', '⁉',

            # 特殊标记
            '※', '⁎', '⁑', '⁒', '⁓', '⁔', '⁕', '⁖', '⁗', '⁘', '⁙', '⁚',

            # 带圆圈的符号
            '⊕', '⊖', '⊗', '⊘', '⊙', '⊚', '⊛', '⊜', '⊝', '⊞', '⊟', '⊠',

            # 小型符号
            '⋄', '⋅', '⋆', '⋇', '⋈', '⋉', '⋊', '⋋', '⋌', '⋍', '⋎', '⋏'
        ]

        # 隐身符号
        self.invisible_symbols = [
            '\u200B',  # 零宽度空格
            '\u200C',  # 零宽度非连接符
            '\u2060',  # 单词连接符
        ]

        # 请求历史去重
        self.request_history = set()
        self.max_history_size = 5000

    def inject_symbols(self, text: str, strategy: str = 'auto') -> str:
        """注入随机符号到文本中"""
        if not text.strip():
            return text

        # 随机选择符号数量 (2-4个，确保<5)
        symbol_count = random.randint(2, 4)

        # 选择符号类型
        if strategy == 'invisible':
            symbols = random.sample(self.invisible_symbols, min(2, len(self.invisible_symbols)))
        elif strategy == 'mixed':
            # 混合可见和不可见符号
            visible_count = random.randint(1, 2)
            invisible_count = 1
            symbols = (random.sample(self.safe_symbols, visible_count) +
                       random.sample(self.invisible_symbols, invisible_count))
        else:
            symbols = random.sample(self.safe_symbols, min(symbol_count, len(self.safe_symbols)))

        # 随机选择注入策略
        strategies = ['prefix', 'suffix', 'wrap']
        if strategy == 'auto':
            strategy = random.choice(strategies)

        if strategy == 'prefix':
            return ''.join(symbols) + ' ' + text
        elif strategy == 'suffix':
            return text + ' ' + ''.join(symbols)
        elif strategy == 'wrap':
            mid = len(symbols) // 2
            prefix = ''.join(symbols[:mid])
            suffix = ''.join(symbols[mid:])
            return f"{prefix} {text} {suffix}" if prefix and suffix else f"{text} {suffix}"
        else:
            return text + ' ' + ''.join(symbols)

    def process_content(self, content: Union[str, List]) -> Union[str, List]:
        """处理各种格式的内容"""
        content_hash = hashlib.md5(str(content).encode()).hexdigest()

        # 检查是否已处理过相同内容
        if content_hash in self.request_history:
            # 强制使用不同策略
            strategy = random.choice(['mixed', 'invisible', 'prefix', 'suffix'])
        else:
            strategy = 'auto'

        self.request_history.add(content_hash)

        # 限制历史记录大小
        if len(self.request_history) > self.max_history_size:
            old_records = list(self.request_history)
            self.request_history = set(old_records[self.max_history_size // 2:])

        if isinstance(content, str):
            return self.inject_symbols(content, strategy)
        elif isinstance(content, list):
            # 处理消息列表
            processed = []
            for item in content:
                if isinstance(item, dict):
                    processed_item = item.copy()

                    # 处理文本内容
                    if 'text' in processed_item:
                        processed_item['text'] = self.inject_symbols(processed_item['text'], strategy)

                    processed.append(processed_item)
                else:
                    processed.append(item)
            return processed

        return content

    def get_statistics(self) -> Dict:
        """获取使用统计"""
        return {
            'available_symbols': len(self.safe_symbols),
            'invisible_symbols': len(self.invisible_symbols),
            'request_history_size': len(self.request_history),
            'max_history_size': self.max_history_size
        }


# 思考配置模型
class ThinkingConfig(BaseModel):
    thinking_budget: Optional[int] = None  # 0-32768, 0=禁用思考, None=自动
    include_thoughts: Optional[bool] = False  # 是否在响应中包含思考过程

    class Config:
        extra = "allow"

    @validator('thinking_budget')
    def validate_thinking_budget(cls, v):
        if v is not None:
            if not isinstance(v, int) or v < 0 or v > 32768:
                raise ValueError("thinking_budget must be an integer between 0 and 32768")
        return v


# 文件数据模型
class InlineData(BaseModel):
    """内联数据模型 - 用于小文件(<20MB)"""
    mime_type: Optional[str] = None  # 兼容旧字段名
    mimeType: Optional[str] = None  # Gemini 2.5标准字段名
    data: str  # base64编码的文件数据

    def __init__(self, **data):
        # 确保两种字段名都支持
        if 'mime_type' in data and 'mimeType' not in data:
            data['mimeType'] = data['mime_type']
        elif 'mimeType' in data and 'mime_type' not in data:
            data['mime_type'] = data['mimeType']
        super().__init__(**data)


class FileData(BaseModel):
    """文件引用模型 - 用于已上传的文件"""
    mime_type: Optional[str] = None  # 兼容旧字段名
    mimeType: Optional[str] = None  # Gemini 2.5标准字段名
    file_uri: Optional[str] = None  # 兼容旧字段名
    fileUri: Optional[str] = None  # Gemini 2.5标准字段名

    def __init__(self, **data):
        # 确保两种字段名都支持
        if 'mime_type' in data and 'mimeType' not in data:
            data['mimeType'] = data['mime_type']
        elif 'mimeType' in data and 'mime_type' not in data:
            data['mime_type'] = data['mimeType']

        if 'file_uri' in data and 'fileUri' not in data:
            data['fileUri'] = data['file_uri']
        elif 'fileUri' in data and 'file_uri' not in data:
            data['file_uri'] = data['fileUri']
        super().__init__(**data)


# 多模态内容
class ContentPart(BaseModel):
    type: str  # "text", "image", "audio", "video", "document"
    text: Optional[str] = None

    # Gemini 2.5标准格式
    inlineData: Optional[InlineData] = None
    fileData: Optional[FileData] = None

    # 向后兼容的字段
    inline_data: Optional[InlineData] = None
    file_data: Optional[FileData] = None

    def __init__(self, **data):
        # 处理字段名兼容性
        if 'inline_data' in data and 'inlineData' not in data:
            data['inlineData'] = data['inline_data']
        elif 'inlineData' in data and 'inline_data' not in data:
            data['inline_data'] = data['inlineData']

        if 'file_data' in data and 'fileData' not in data:
            data['fileData'] = data['file_data']
        elif 'fileData' in data and 'file_data' not in data:
            data['file_data'] = data['fileData']

        super().__init__(**data)

# 请求/响应
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Union[str, Dict[str, Any], ContentPart]]]

    class Config:
        extra = "allow"

    @validator('content')
    def validate_content(cls, v):
        """验证并标准化content字段"""
        if isinstance(v, str):
            return v
        elif isinstance(v, list):
            return v
        else:
            raise ValueError("content must be string or array of content objects")

    def get_text_content(self) -> str:
        """获取纯文本内容"""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            text_parts = []
            for item in self.content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    if item.get('type') == 'text' and 'text' in item:
                        text_parts.append(item['text'])
                    elif 'text' in item:
                        text_parts.append(item['text'])
            return ' '.join(text_parts) if text_parts else ""
        else:
            return str(self.content)

    def has_multimodal_content(self) -> bool:
        """检查是否包含多模态内容"""
        if isinstance(self.content, list):
            for item in self.content:
                if isinstance(item, dict) and item.get('type') in ['image', 'audio', 'video', 'document']:
                    return True
        return False


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None
    thinking_config: Optional[ThinkingConfig] = None

    # OpenAI Compatible 工具调用字段
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    class Config:
        extra = "allow"

    def __init__(self, **data):
        # 参数范围验证
        if 'temperature' in data and data['temperature'] is not None:
            data['temperature'] = max(0.0, min(2.0, data['temperature']))
        if 'top_p' in data and data['top_p'] is not None:
            data['top_p'] = max(0.0, min(1.0, data['top_p']))
        if 'n' in data and data['n'] is not None:
            data['n'] = max(1, min(4, data['n']))
        if 'max_tokens' in data and data['max_tokens'] is not None:
            data['max_tokens'] = max(1, data['max_tokens'])

        super().__init__(**data)

# 内存缓存用于RPM/TPM限制
class RateLimitCache:
    def __init__(self, max_entries: int = 10000):
        self.cache: Dict[str, Dict[str, List[tuple]]] = {}
        self.max_entries = max_entries
        self.lock = asyncio.Lock()

    async def cleanup_expired(self, window_seconds: int = 60):
        """定期清理过期缓存"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds

        async with self.lock:
            for model_name in list(self.cache.keys()):
                if model_name in self.cache:
                    self.cache[model_name]['requests'] = [
                        (t, v) for t, v in self.cache[model_name]['requests']
                        if t > cutoff_time
                    ]
                    self.cache[model_name]['tokens'] = [
                        (t, v) for t, v in self.cache[model_name]['tokens']
                        if t > cutoff_time
                    ]

    async def add_usage(self, model_name: str, requests: int = 1, tokens: int = 0):
        async with self.lock:
            if model_name not in self.cache:
                self.cache[model_name] = {'requests': [], 'tokens': []}

            current_time = time.time()
            self.cache[model_name]['requests'].append((current_time, requests))
            self.cache[model_name]['tokens'].append((current_time, tokens))

    async def get_current_usage(self, model_name: str, window_seconds: int = 60) -> Dict[str, int]:
        async with self.lock:
            if model_name not in self.cache:
                return {'requests': 0, 'tokens': 0}

            current_time = time.time()
            cutoff_time = current_time - window_seconds

            # 清理过期记录
            self.cache[model_name]['requests'] = [
                (t, v) for t, v in self.cache[model_name]['requests']
                if t > cutoff_time
            ]
            self.cache[model_name]['tokens'] = [
                (t, v) for t, v in self.cache[model_name]['tokens']
                if t > cutoff_time
            ]

            # 计算总和
            total_requests = sum(v for _, v in self.cache[model_name]['requests'])
            total_tokens = sum(v for _, v in self.cache[model_name]['tokens'])

            return {'requests': total_requests, 'tokens': total_tokens}


# 健康检测功能
async def check_gemini_key_health(api_key: str, timeout: int = 10) -> Dict[str, Any]:
    """使用 google-genai SDK 检测单个 Gemini Key 的健康状态"""
    start_time = time.time()
    try:
        # 复用缓存客户端，避免频繁创建导致 AFC 日志刷屏
        client = get_cached_client(api_key)
        # SDK 默认 httpx 超时较高，这里通过 asyncio.wait_for 施加整体超时
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents="Hello",
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            ),
            timeout=timeout
        )
        response_time = time.time() - start_time
        # 成功调用即视为健康
        return {
            "healthy": True,
            "response_time": response_time,
            "status_code": 200,
            "error": None
        }
    except asyncio.TimeoutError:
        return {
            "healthy": False,
            "response_time": timeout,
            "status_code": None,
            "error": "Timeout"
        }
    except Exception as e:
        return {
            "healthy": False,
            "response_time": time.time() - start_time,
            "status_code": None,
            "error": str(e)
        }


# 保活功能
async def keep_alive_ping():
    """保活函数：定期ping自己的健康检查端点"""
    try:
        render_url = os.getenv('RENDER_EXTERNAL_URL')
        target_url = f"{render_url}/wake" if render_url else "http://localhost:8000/wake"
        
        # 使用标准库 aiohttp 替代 httpx
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(target_url, timeout=30) as response:
                    if response.status == 200:
                        logger.info(f"🟢 Keep-alive ping successful: {response.status}")
                    else:
                        logger.warning(f"🟡 Keep-alive ping warning: {response.status}")
        except ImportError:
            # 备选：使用 urllib 标准库
            import urllib.request
            with urllib.request.urlopen(target_url, timeout=30) as response:
                if response.status == 200:
                    logger.info(f"🟢 Keep-alive ping successful: {response.status}")
                else:
                    logger.warning(f"🟡 Keep-alive ping warning: {response.status}")
    except Exception as e:
        logger.warning(f"🔴 Keep-alive ping failed: {e}")


# 每小时健康检测函数
async def record_hourly_health_check():
    """每小时记录一次健康检测结果"""
    try:
        available_keys = db.get_available_gemini_keys()

        for key_info in available_keys:
            key_id = key_info['id']

            # 执行健康检测
            health_result = await check_gemini_key_health(key_info['key'])

            # 记录到历史表
            db.record_daily_health_status(
                key_id,
                health_result['healthy'],
                health_result['response_time']
            )

            # 更新性能指标
            db.update_key_performance(
                key_id,
                health_result['healthy'],
                health_result['response_time']
            )

        logger.info(f"✅ Hourly health check completed for {len(available_keys)} keys")

    except Exception as e:
        logger.error(f"❌ Hourly health check failed: {e}")


# 自动清理函数
async def auto_cleanup_failed_keys():
    """每日自动清理连续异常的API key"""
    try:
        # 获取配置
        cleanup_config = db.get_auto_cleanup_config()

        if not cleanup_config['enabled']:
            logger.info("🔒 Auto cleanup is disabled")
            return

        days_threshold = cleanup_config['days_threshold']
        min_checks_per_day = cleanup_config['min_checks_per_day']

        # 执行自动清理
        removed_keys = db.auto_remove_failed_keys(days_threshold, min_checks_per_day)

        if removed_keys:
            logger.warning(
                f"🗑️ Auto-removed {len(removed_keys)} failed keys after {days_threshold} consecutive unhealthy days:")
            for key in removed_keys:
                logger.warning(f"   - Key #{key['id']}: {key['key']} (failed for {key['consecutive_days']} days)")
        else:
            logger.info(f"✅ No keys need cleanup (threshold: {days_threshold} days)")

    except Exception as e:
        logger.error(f"❌ Auto cleanup failed: {e}")

# 快速故障转移函数
async def update_key_performance_background(key_id: int, success: bool, response_time: float):
    """
    在后台异步更新key性能指标，不阻塞主请求流程
    """
    try:
        db.update_key_performance(key_id, success, response_time)

        # 如果失败，启动后台健康检测任务
        if not success:
            asyncio.create_task(schedule_health_check(key_id))

    except Exception as e:
        logger.error(f"Background performance update failed for key {key_id}: {e}")


async def schedule_health_check(key_id: int):
    """
    调度后台健康检测任务
    """
    try:
        # 获取配置中的延迟时间
        config = db.get_failover_config()
        delay = config.get('health_check_delay', 5)

        # 延迟指定时间后执行健康检测，避免立即重复检测
        await asyncio.sleep(delay)

        key_info = db.get_gemini_key_by_id(key_id)
        if key_info and key_info.get('status') == 1:  # 只检测激活的key
            health_result = await check_gemini_key_health(key_info['key'])

            # 更新健康状态
            db.update_key_performance(
                key_id,
                health_result['healthy'],
                health_result['response_time']
            )

            # 记录健康检测历史
            db.record_daily_health_status(
                key_id,
                health_result['healthy'],
                health_result['response_time']
            )

            status = "healthy" if health_result['healthy'] else "unhealthy"
            logger.info(f"Background health check for key #{key_id}: {status}")

    except Exception as e:
        logger.error(f"Background health check failed for key {key_id}: {e}")


async def log_usage_background(gemini_key_id: int, user_key_id: int, model_name: str, requests: int, tokens: int):
    """
    在后台异步记录使用量，不阻塞主请求流程
    """
    try:
        db.log_usage(gemini_key_id, user_key_id, model_name, requests, tokens)
    except Exception as e:
        logger.error(f"Background usage logging failed: {e}")


async def collect_gemini_response_directly(
        gemini_key: str,
        key_id: int,
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        model_name: str
) -> Dict:
    """
    从Google API收集完整响应
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse"
    
    # 确定超时时间
    has_tool_calls = bool(openai_request.tools or openai_request.tool_choice)
    is_fast_failover = await should_use_fast_failover()
    if has_tool_calls:
        timeout = 60.0
    elif is_fast_failover:
        timeout = 60.0
    else:
        timeout = float(db.get_config('request_timeout', '60'))

    logger.info(f"Starting direct collection from: {url}")
    
    complete_content = ""
    thinking_content = ""
    total_tokens = 0
    finish_reason = "stop"
    processed_lines = 0
    start_time = time.time()

    try:
        client = get_cached_client(gemini_key)
        # 使用 google-genai 的流式接口
        genai_stream = client.aio.models.generate_content_stream(
            model=model_name,
            body=gemini_request
        )
        async with asyncio.timeout(timeout):
            async for chunk in genai_stream:
                # chunk.candidates 列表结构与 REST 回包保持一致
                data = chunk.to_dict()  # SDK 对象转为 dict，字段与官方 REST 保持同名
                for candidate in data.get("candidates", []):
                    content_data = candidate.get("content", {})
                    parts = content_data.get("parts", [])
                    for part in parts:
                        if "text" in part:
                            text = part["text"]
                            if not text:
                                continue
                            total_tokens += len(text.split())
                            is_thought = part.get("thought", False)
                            if is_thought and not (openai_request.thinking_config and openai_request.thinking_config.include_thoughts):
                                thinking_content += text
                            else:
                                if is_thought and not thinking_content:
                                    complete_content += "**Thinking Process:**\n"
                                elif not is_thought and thinking_content and not complete_content.endswith("**Response:**\n"):
                                    complete_content += "\n\n**Response:**\n"
                                complete_content += text
                    finish_reason = candidate.get("finishReason", "stop")
                    if finish_reason:
                        finish_reason = map_finish_reason(finish_reason)

                    processed_lines += 1

                response_time = time.time() - start_time
                asyncio.create_task(
                    update_key_performance_background(key_id, True, response_time)
                )

    except asyncio.TimeoutError as e:
        logger.warning(f"Direct request timeout/connection error: {str(e)}")
        response_time = time.time() - start_time
        asyncio.create_task(
            update_key_performance_background(key_id, False, response_time)
        )
        raise Exception(f"Direct request failed: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected direct request error: {str(e)}")
        response_time = time.time() - start_time
        asyncio.create_task(
            update_key_performance_background(key_id, False, response_time)
        )
        raise

    # 检查是否收集到内容
    if not complete_content.strip():
        logger.error(f"No content collected directly. Processed {processed_lines} lines")
        raise HTTPException(
            status_code=502,
            detail="No content received from Google API"
        )

    # 计算token使用量
    prompt_tokens = len(str(openai_request.messages).split())
    completion_tokens = len(complete_content.split())

    # 构建最终响应
    openai_response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": openai_request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": complete_content.strip()
            },
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }

    logger.info(f"Successfully collected direct response: {len(complete_content)} chars, {completion_tokens} tokens")
    return openai_response


async def make_gemini_request_single_attempt(
        gemini_key: str,
        key_id: int,
        gemini_request: Dict,
        model_name: str,
        timeout: float = 60.0
) -> Dict:
    start_time = time.time()

    try:
        client = get_cached_client(gemini_key)
        async with asyncio.timeout(timeout):
            response_obj = await client.aio.models.generate_content(
                model=model_name,
                **gemini_request
            )
        response_time = time.time() - start_time
        # SDK 对象转 dict
        response_dict = response_obj.to_dict() if hasattr(response_obj, "to_dict") else json.loads(response_obj.model_dump_json())
        asyncio.create_task(
            update_key_performance_background(key_id, True, response_time)
        )
        return response_dict

    except asyncio.TimeoutError:
        response_time = time.time() - start_time
        asyncio.create_task(
            update_key_performance_background(key_id, False, response_time)
        )
        logger.warning(f"Key #{key_id} timeout after {response_time:.2f}s")
        raise HTTPException(status_code=504, detail="Request timeout")

    except Exception as e:
        response_time = time.time() - start_time
        asyncio.create_task(
            update_key_performance_background(key_id, False, response_time)
        )
        # google-genai 会在异常中封装详细信息
        err_msg = str(e)
        if "rate_limit" in err_msg.lower() or "status: 429" in err_msg:
            logger.warning(f"Key #{key_id} is rate-limited (429). Marking as 'rate_limited'.")
            db.update_gemini_key_status(key_id, 'rate_limited')
            raise HTTPException(status_code=429, detail="Rate limited")
        logger.error(f"Key #{key_id} request error: {err_msg}")
        raise HTTPException(status_code=500, detail=err_msg)


async def make_request_with_fast_failover(
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        model_name: str,
        user_key_info: Dict = None,
        max_key_attempts: int = None
) -> Dict:
    """
    快速故障转移请求处理
    """
    available_keys = db.get_available_gemini_keys()

    if not available_keys:
        logger.error("No available keys for request")
        raise HTTPException(
            status_code=503,
            detail="No available API keys"
        )

    if max_key_attempts is None:
        max_key_attempts = len(available_keys)
    max_key_attempts = min(max_key_attempts, len(available_keys))

    logger.info(f"Starting fast failover with up to {max_key_attempts} key attempts for model {model_name}")

    failed_keys = []
    last_error = None

    for attempt in range(max_key_attempts):
        try:
            # 选择下一个可用的key（排除已失败的）
            selection_result = await select_gemini_key_and_check_limits(
                model_name,
                excluded_keys=set(failed_keys)
            )

            if not selection_result:
                logger.warning(f"No more available keys after {attempt} attempts")
                break

            key_info = selection_result['key_info']
            logger.info(f"Fast failover attempt {attempt + 1}: Using key #{key_info['id']}")

            try:
                # 确定超时时间：工具调用或快速响应模式使用60秒，其他使用配置值
                has_tool_calls = bool(openai_request.tools or openai_request.tool_choice)
                is_fast_failover = await should_use_fast_failover()
                if has_tool_calls:
                    timeout_seconds = 60.0  # 工具调用强制60秒超时
                    logger.info("Using extended 60s timeout for tool calls")
                elif is_fast_failover:
                    timeout_seconds = 60.0  # 快速响应模式使用60秒超时
                    logger.info("Using extended 60s timeout for fast response mode")
                else:
                    timeout_seconds = float(db.get_config('request_timeout', '60'))
                
                # 从Google API收集完整响应
                logger.info(f"Using direct collection for non-streaming request with key #{key_info['id']}")
                
                # 收集响应
                response = await collect_gemini_response_directly(
                    key_info['key'],
                    key_info['id'],
                    gemini_request,
                    openai_request,
                    model_name
                )
                
                logger.info(f"✅ Request successful with key #{key_info['id']} on attempt {attempt + 1}")

                # 从响应中获取token使用量
                usage = response.get('usage', {})
                total_tokens = usage.get('completion_tokens', 0)
                prompt_tokens = usage.get('prompt_tokens', 0)

                # 记录使用量
                if user_key_info:
                    # 在后台记录使用量，不阻塞响应
                    asyncio.create_task(
                        log_usage_background(
                            key_info['id'],
                            user_key_info['id'],
                            model_name,
                            1,
                            total_tokens
                        )
                    )

                # 更新速率限制
                await rate_limiter.add_usage(model_name, 1, total_tokens)
                return response

            except HTTPException as e:
                failed_keys.append(key_info['id'])
                last_error = e

                logger.warning(f"❌ Key #{key_info['id']} failed: {e.detail}")

                # 记录失败的使用量
                if user_key_info:
                    asyncio.create_task(
                        log_usage_background(
                            key_info['id'],
                            user_key_info['id'],
                            model_name,
                            1,
                            0
                        )
                    )

                await rate_limiter.add_usage(model_name, 1, 0)

                # 如果是客户端错误（4xx），不继续尝试其他key
                if 400 <= e.status_code < 500:
                    logger.warning(f"Client error {e.status_code}, stopping failover")
                    raise e

                # 服务器错误或网络错误，继续尝试下一个key
                continue

        except Exception as e:
            logger.error(f"Unexpected error during failover attempt {attempt + 1}: {str(e)}")
            last_error = HTTPException(status_code=500, detail=str(e))
            continue

    # 所有key都失败了
    failed_count = len(failed_keys)
    logger.error(f"❌ All {failed_count} attempted keys failed for {model_name}")

    if last_error:
        raise last_error
    else:
        raise HTTPException(
            status_code=503,
            detail=f"All {failed_count} available API keys failed"
        )

async def stream_gemini_response_single_attempt(
        gemini_key: str,
        key_id: int,
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        model_name: str
) -> AsyncGenerator[bytes, None]:
    """
    单次流式请求尝试，失败立即抛出异常，使用 google-genai SDK 实现
    """
    # 确定超时时间：工具调用或快速响应模式使用60秒，其他使用配置值
    has_tool_calls = bool(openai_request.tools or openai_request.tool_choice)
    is_fast_failover = await should_use_fast_failover()
    if has_tool_calls:
        timeout = 60.0  # 工具调用强制60秒超时
        logger.info("Using extended 60s timeout for tool calls in streaming")
    elif is_fast_failover:
        timeout = 60.0  # 快速响应模式使用60秒超时
        logger.info("Using extended 60s timeout for fast response mode in streaming")
    else:
        timeout = float(db.get_config('request_timeout', '60'))

    logger.info(f"Starting single stream request to model: {model_name}")

    start_time = time.time()

    try:
        client = get_cached_client(gemini_key)
        async with asyncio.timeout(timeout):
            contents = gemini_request["contents"]
            # 流式接口直接使用contents和body参数
            genai_stream = client.aio.models.generate_content_stream(
                model=model_name,
                body=gemini_request
            )

            if False:  # legacy httpx code disabled after migration to google-genai
                    response_time = time.time() - start_time
                    asyncio.create_task(
                        update_key_performance_background(key_id, False, response_time)
                    )

                    error_text = await response.aread()
                    error_msg = error_text.decode() if error_text else f"HTTP {response.status_code}"
                    logger.error(f"Stream request failed with status {response.status_code}: {error_msg}")
                    raise Exception(f"Stream request failed: {error_msg}")

            stream_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created = int(time.time())
            total_tokens = 0
            thinking_sent = False
            has_content = False
            processed_lines = 0

            logger.info(f"Stream response started, status: {response.status_code}")

            try:
                    async for line in response.aiter_lines():
                        processed_lines += 1

                        if not line:
                            continue

                        if processed_lines <= 5:
                            logger.debug(f"Stream line {processed_lines}: {line[:100]}...")

                        if line.startswith("data: "):
                            json_str = line[6:]

                            if json_str.strip() == "[DONE]":
                                logger.info("Received [DONE] signal from stream")
                                break

                            if not json_str.strip():
                                continue

                            try:
                                data = json.loads(json_str)

                                for candidate in data.get("candidates", []):
                                    content_data = candidate.get("content", {})
                                    parts = content_data.get("parts", [])

                                    for part in parts:
                                        if "text" in part:
                                            text = part["text"]
                                            if not text:
                                                continue

                                            total_tokens += len(text.split())
                                            has_content = True

                                            is_thought = part.get("thought", False)

                                            if is_thought and not (openai_request.thinking_config and
                                                                   openai_request.thinking_config.include_thoughts):
                                                continue

                                            if is_thought and not thinking_sent:
                                                thinking_header = {
                                                    "id": stream_id,
                                                    "object": "chat.completion.chunk",
                                                    "created": created,
                                                    "model": openai_request.model,
                                                    "choices": [{
                                                        "index": 0,
                                                        "delta": {"content": "**Thinking Process:**\n"},
                                                        "finish_reason": None
                                                    }]
                                                }
                                                yield f"data: {json.dumps(thinking_header, ensure_ascii=False)}\n\n".encode(
                                                    'utf-8')
                                                thinking_sent = True
                                                logger.debug("Sent thinking header")
                                            elif not is_thought and thinking_sent:
                                                response_header = {
                                                    "id": stream_id,
                                                    "object": "chat.completion.chunk",
                                                    "created": created,
                                                    "model": openai_request.model,
                                                    "choices": [{
                                                        "index": 0,
                                                        "delta": {"content": "\n\n**Response:**\n"},
                                                        "finish_reason": None
                                                    }]
                                                }
                                                yield f"data: {json.dumps(response_header, ensure_ascii=False)}\n\n".encode(
                                                    'utf-8')
                                                thinking_sent = False
                                                logger.debug("Sent response header")

                                            chunk_data = {
                                                "id": stream_id,
                                                "object": "chat.completion.chunk",
                                                "created": created,
                                                "model": openai_request.model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {"content": text},
                                                    "finish_reason": None
                                                }]
                                            }
                                            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n".encode(
                                                'utf-8')

                                    finish_reason = candidate.get("finishReason")
                                    if finish_reason:
                                        finish_chunk = {
                                            "id": stream_id,
                                            "object": "chat.completion.chunk",
                                            "created": created,
                                            "model": openai_request.model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {},
                                                "finish_reason": map_finish_reason(finish_reason)
                                            }]
                                        }
                                        yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n".encode(
                                            'utf-8')
                                        yield "data: [DONE]\n\n".encode('utf-8')

                                        logger.info(
                                            f"Stream completed with finish_reason: {finish_reason}, tokens: {total_tokens}")

                                        response_time = time.time() - start_time
                                        asyncio.create_task(
                                            update_key_performance_background(key_id, True, response_time)
                                        )
                                        await rate_limiter.add_usage(model_name, 1, total_tokens)
                                        return

                            except json.JSONDecodeError as e:
                                logger.warning(f"JSON decode error: {e}, line: {json_str[:200]}...")
                                continue

                        elif line.startswith("event: ") or line.startswith("id: ") or line.startswith("retry: "):
                            continue

                    # 如果正常结束但没有内容，抛出异常
                    if not has_content:
                        logger.warning(f"Stream ended without content after processing {processed_lines} lines")
                        raise Exception("Stream response had no content")

                    # 正常结束，发送完成信号
                    if has_content:
                        finish_chunk = {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": openai_request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        }
                        yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n".encode('utf-8')
                        yield "data: [DONE]\n\n".encode('utf-8')

                        logger.info(
                            f"Stream ended naturally, processed {processed_lines} lines, tokens: {total_tokens}")

                        response_time = time.time() - start_time
                        asyncio.create_task(
                            update_key_performance_background(key_id, True, response_time)
                        )

                    await rate_limiter.add_usage(model_name, 1, total_tokens)

                # except Exception as e:  # 原 httpx 异常已移除
                # Legacy httpx branch disabled after migration to google-genai
            except Exception as e:
                    logger.warning(f"Stream connection error: {str(e)}")
                    response_time = time.time() - start_time
                    asyncio.create_task(
                        update_key_performance_background(key_id, False, response_time)
                    )
                    raise Exception(f"Stream connection error: {str(e)}")

    # except Exception as e:  # 原 httpx 超时连接异常移除
    # Legacy httpx branch disabled after migration to google-genai
    except Exception as e:
        logger.warning(f"Stream timeout/connection error: {str(e)}")
        response_time = time.time() - start_time
        asyncio.create_task(
            update_key_performance_background(key_id, False, response_time)
        )
        raise Exception(f"Stream connection failed: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected stream error: {str(e)}")
        response_time = time.time() - start_time
        asyncio.create_task(
            update_key_performance_background(key_id, False, response_time)
        )
        raise


async def stream_with_fast_failover(
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        model_name: str,
        user_key_info: Dict = None,
        max_key_attempts: int = None
) -> AsyncGenerator[bytes, None]:
    """
    流式响应快速故障转移
    """
    available_keys = db.get_available_gemini_keys()

    if not available_keys:
        error_data = {
            'error': {
                'message': 'No available API keys',
                'type': 'service_unavailable',
                'code': 503
            }
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode('utf-8')
        yield "data: [DONE]\n\n".encode('utf-8')
        return

    if max_key_attempts is None:
        max_key_attempts = len(available_keys)
    max_key_attempts = min(max_key_attempts, len(available_keys))

    logger.info(f"Starting stream fast failover with up to {max_key_attempts} key attempts for {model_name}")

    failed_keys = []

    for attempt in range(max_key_attempts):
        try:
            selection_result = await select_gemini_key_and_check_limits(
                model_name,
                excluded_keys=set(failed_keys)
            )

            if not selection_result:
                break

            key_info = selection_result['key_info']
            logger.info(f"Stream fast failover attempt {attempt + 1}: Using key #{key_info['id']}")

            success = False
            total_tokens = 0

            try:
                async for chunk in stream_gemini_response_single_attempt(
                        key_info['key'],
                        key_info['id'],
                        gemini_request,
                        openai_request,
                        model_name
                ):
                    yield chunk
                    success = True

                if success:
                    # 在后台记录使用量
                    if user_key_info:
                        asyncio.create_task(
                            log_usage_background(
                                key_info['id'],
                                user_key_info['id'],
                                model_name,
                                1,
                                total_tokens
                            )
                        )

                    await rate_limiter.add_usage(model_name, 1, total_tokens)
                    return

            except Exception as e:
                failed_keys.append(key_info['id'])
                logger.warning(f"Stream key #{key_info['id']} failed: {str(e)}")

                # 在后台更新性能指标
                asyncio.create_task(
                    update_key_performance_background(key_info['id'], False, 0.0)
                )

                # 记录失败的使用量
                if user_key_info:
                    asyncio.create_task(
                        log_usage_background(
                            key_info['id'],
                            user_key_info['id'],
                            model_name,
                            1,
                            0
                        )
                    )

                if attempt < max_key_attempts - 1:
                    logger.info(f"Key #{key_info['id']} failed, trying next key...")
                    # retry_msg = {
                    #     'error': {
                    #         'message': f'Key #{key_info["id"]} failed, trying next key...',
                    #         'type': 'retry_info',
                    #         'retry_attempt': attempt + 1
                    #     }
                    # }
                    # yield f"data: {json.dumps(retry_msg, ensure_ascii=False)}\n\n".encode('utf-8')
                    continue
                else:
                    break

        except Exception as e:
            logger.error(f"Stream failover error on attempt {attempt + 1}: {str(e)}")
            continue

    # 所有key都失败了
    error_data = {
        'error': {
            'message': f'All {len(failed_keys)} available API keys failed',
            'type': 'all_keys_failed',
            'code': 503,
            'failed_keys': failed_keys
        }
    }
    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode('utf-8')
    yield "data: [DONE]\n\n".encode('utf-8')


# 配置管理函数
async def should_use_fast_failover() -> bool:
    """检查是否应该使用快速故障转移"""
    config = db.get_failover_config()
    return config.get('fast_failover_enabled', True)




# 全局变量
db = Database()
rate_limiter = RateLimitCache()
anti_detection = GeminiAntiDetectionInjector()  # 防检测注入器实例
scheduler = None
keep_alive_enabled = False

# 文件存储配置
UPLOAD_DIR = "uploads"
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_INLINE_SIZE = 20 * 1024 * 1024  # 20MB - Gemini 2.5 内联数据限制

# Gemini 2.5 支持的MIME类型
SUPPORTED_MIME_TYPES = {
    # 图片
    'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp', 'image/bmp',

    # 音频
    'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/ogg', 'audio/mp4', 'audio/flac',
    'audio/aac', 'audio/webm',

    # 视频
    'video/mp4', 'video/avi', 'video/mov', 'video/wmv', 'video/webm', 'video/quicktime',
    'video/x-msvideo', 'video/mpeg',

    # 文档
    'application/pdf',
    'text/plain', 'text/csv', 'text/xml', 'text/html',
    'application/json',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # docx
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',  # xlsx
    'application/vnd.ms-excel',  # xls
    'application/msword',  # doc
}

# 确保上传目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 文件存储字典（内存存储，生产环境建议使用数据库）
file_storage: Dict[str, Dict] = {}

# Gemini File API 基础URL
GEMINI_FILE_API_BASE = "https://generativelanguage.googleapis.com/v1beta/files"


# 初始化防检测配置
def init_anti_detection_config():
    """初始化防检测配置"""
    try:
        # 确保配置表中有防检测设置
        db.set_config('anti_detection_enabled', 'true')
        logger.info("✅ Anti-detection system initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize anti-detection system: {e}")


async def upload_file_to_gemini(file_content: bytes, mime_type: str, filename: str, gemini_key: str) -> Optional[str]:
    """上传文件到Gemini File API并返回fileUri（使用 google-genai SDK）"""
    try:
        client = get_cached_client(gemini_key)
        # 使用 BytesIO 包装文件内容以便 SDK 读取
        file_stream = io.BytesIO(file_content)
        upload_result = await client.aio.files.upload(
            file=file_stream,
            config={
                "mimeType": mime_type,
                "displayName": filename,
                "name": f"files/{uuid.uuid4().hex}_{filename}"
            }
        )
        file_uri = getattr(upload_result, "uri", None)
        if file_uri:
            logger.info(f"File uploaded to Gemini successfully: {file_uri}")
            return file_uri
        else:
            logger.error("No URI returned from google-genai upload result")
            return None
        url = f"{GEMINI_FILE_API_BASE}?key={gemini_key}"  # pyright: ignore[reportUnreachable]

        # 准备multipart/form-data
        files = {
            'metadata': (None, json.dumps({
                'name': f"files/{uuid.uuid4().hex}_{filename}",
                'displayName': filename
            }), 'application/json'),
            'data': (filename, file_content, mime_type)
        }

        # async with httpx.AsyncClient(timeout=60.0) as client:  # 已弃用
        response = await client.post(url, files=files)

        if response.status_code == 200:
            result = response.json()
            file_uri = result.get('uri')
            if file_uri:
                logger.info(f"File uploaded to Gemini successfully: {file_uri}")
                return file_uri
            else:
                logger.error(f"No URI returned from Gemini File API: {result}")
                return None
        else:
            logger.error(f"Failed to upload file to Gemini: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        logger.error(f"Error uploading file to Gemini: {str(e)}")
        return None


async def delete_file_from_gemini(file_uri: str, gemini_key: str) -> bool:
    """从Gemini File API删除文件（使用 google-genai SDK）"""
    try:
        client = genai.Client(api_key=gemini_key)
        await client.aio.files.delete(name=file_uri)
        logger.info(f"File deleted from Gemini successfully: {file_uri}")
        return True
    except Exception as e:
        logger.error(f"Error deleting file from Gemini: {str(e)}")
        return False
        # response = await client.delete(url) # 这一行似乎是多余的，因为上面已经使用了genai SDK删除文件

        if response.status_code == 200:
            logger.info(f"File deleted from Gemini successfully: {file_uri}")
            return True
        else:
            logger.warning(f"Failed to delete file from Gemini: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        logger.error(f"Error deleting file from Gemini: {str(e)}")
        return False


async def cleanup_expired_files():
    """清理过期的文件"""
    try:
        current_time = time.time()
        expired_files = []

        for file_id, file_info in list(file_storage.items()):
            # 检查文件是否超过1天
            file_age = current_time - file_info.get('created_at', 0)
            if file_age > 1 * 24 * 3600:
                expired_files.append(file_id)

        cleaned_count = 0
        for file_id in expired_files:
            try:
                file_info = file_storage[file_id]

                # 如果文件存储在Gemini，尝试删除
                if "gemini_file_uri" in file_info and "gemini_key_used" in file_info:
                    await delete_file_from_gemini(file_info["gemini_file_uri"], file_info["gemini_key_used"])

                # 删除本地文件
                if "file_path" in file_info and os.path.exists(file_info["file_path"]):
                    os.remove(file_info["file_path"])

                # 从存储中移除
                del file_storage[file_id]
                cleaned_count += 1

            except Exception as e:
                logger.error(f"Error cleaning up file {file_id}: {str(e)}")

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired files")

    except Exception as e:
        logger.error(f"Error in cleanup_expired_files: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler, keep_alive_enabled

    # 启动时的操作
    logger.info("Starting Gemini API Proxy with Anti-Detection...")
    logger.info(f"Available API keys: {len(db.get_available_gemini_keys())}")
    logger.info(f"Environment: {'Render' if os.getenv('RENDER_EXTERNAL_URL') else 'Local'}")
    logger.info("✅ Gemini 2.5 multimodal features optimized")

    # 初始化防检测系统
    init_anti_detection_config()

    # 启动时执行一次健康检测
    try:
        logger.info("🔍 Performing initial health check for all API keys...")
        await record_hourly_health_check()
        logger.info("✅ Initial health check completed")
    except Exception as e:
        logger.error(f"❌ Initial health check failed: {e}")

    # 检查是否启用保活功能
    enable_keep_alive = os.getenv('ENABLE_KEEP_ALIVE', 'true').lower() == 'true'
    keep_alive_interval = int(os.getenv('KEEP_ALIVE_INTERVAL', '10'))  # 默认10分钟

    if enable_keep_alive:
        try:
            scheduler = AsyncIOScheduler()

            # 添加保活任务
            scheduler.add_job(
                keep_alive_ping,
                'interval',
                minutes=keep_alive_interval,
                id='keep_alive',
                max_instances=1,  # 防止重叠执行
                coalesce=True,  # 合并延迟的任务
                misfire_grace_time=30  # 30秒的宽限时间
            )

            # 添加缓存清理任务
            scheduler.add_job(
                rate_limiter.cleanup_expired,
                'interval',
                minutes=5,
                id='cache_cleanup',
                max_instances=1
            )

            # 每小时健康检测任务
            scheduler.add_job(
                record_hourly_health_check,
                'interval',
                hours=1,
                id='hourly_health_check',
                max_instances=1,
                coalesce=True
            )

            # 每天凌晨2点自动清理任务
            scheduler.add_job(
                auto_cleanup_failed_keys,
                'cron',
                hour=2,  # 凌晨2点执行
                minute=0,
                id='daily_cleanup',
                max_instances=1,
                coalesce=True
            )

            # 每天凌晨3点清理过期文件
            scheduler.add_job(
                cleanup_expired_files,
                'cron',
                hour=3,  # 凌晨3点执行
                minute=0,
                id='file_cleanup',
                max_instances=1,
                coalesce=True
            )

            scheduler.start()
            keep_alive_enabled = True
            logger.info(f"✅ Scheduler started with auto-cleanup enabled (interval: {keep_alive_interval} minutes)")

            # 启动后立即执行一次保活
            await keep_alive_ping()

        except Exception as e:
            logger.error(f"❌ Failed to start scheduler: {e}")
            keep_alive_enabled = False
    else:
        logger.info("⚪ Keep-alive disabled (set ENABLE_KEEP_ALIVE=true to enable)")

    yield

    # 关闭时的操作
    if scheduler:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler shutdown")
    logger.info("API Server shutting down...")


app = FastAPI(
    title="Gemini API Proxy",
    description="",
    version="1.4",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求计数中间件
@app.middleware("http")
async def count_requests(request: Request, call_next):
    global request_count
    request_count += 1

    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # 记录请求日志
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")

    return response


# 全局异常处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """处理请求验证错误"""
    logger.warning(f"Request validation error: {exc}")

    error_details = []
    for error in exc.errors():
        field = " -> ".join(str(x) for x in error["loc"])
        msg = error["msg"]
        error_details.append(f"{field}: {msg}")

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": f"Request validation failed: {'; '.join(error_details)}",
                "type": "invalid_request_error",
                "code": "request_validation_error"
            }
        }
    )


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    """处理Pydantic验证错误"""
    logger.warning(f"Pydantic validation error: {exc}")

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": f"Data validation failed: {str(exc)}",
                "type": "invalid_request_error",
                "code": "data_validation_error"
            }
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
                "code": "server_error"
            }
        }
    )


# 辅助函数
def get_actual_model_name(request_model: str) -> str:
    """获取实际使用的模型名称"""
    supported_models = db.get_supported_models()

    if request_model in supported_models:
        logger.info(f"Using requested model: {request_model}")
        return request_model

    default_model = db.get_config('default_model_name', 'gemini-2.5-flash')
    logger.info(f"Unsupported model: {request_model}, using default: {default_model}")
    return default_model


def inject_prompt_to_messages(messages: List[ChatMessage]) -> List[ChatMessage]:
    """向消息中注入prompt"""
    inject_config = db.get_inject_prompt_config()

    if not inject_config['enabled'] or not inject_config['content']:
        return messages

    content = inject_config['content']
    position = inject_config['position']
    new_messages = messages.copy()

    if position == 'system':
        system_msg = None
        for i, msg in enumerate(new_messages):
            if msg.role == 'system':
                system_msg = msg
                break

        if system_msg:
            new_content = f"{content}\n\n{system_msg.get_text_content()}"
            new_messages[i] = ChatMessage(role='system', content=new_content)
        else:
            new_messages.insert(0, ChatMessage(role='system', content=content))

    elif position == 'user_prefix':
        for i, msg in enumerate(new_messages):
            if msg.role == 'user':
                original_content = msg.get_text_content()
                new_content = f"{content}\n\n{original_content}"
                new_messages[i] = ChatMessage(role='user', content=new_content)
                break

    elif position == 'user_suffix':
        for i in range(len(new_messages) - 1, -1, -1):
            if new_messages[i].role == 'user':
                original_content = new_messages[i].get_text_content()
                new_content = f"{original_content}\n\n{content}"
                new_messages[i] = ChatMessage(role='user', content=new_content)
                break

    return new_messages


def get_thinking_config(request: ChatCompletionRequest) -> Dict:
    """根据配置生成思考配置"""
    thinking_config = {}

    global_thinking_enabled = db.get_config('thinking_enabled', 'true').lower() == 'true'
    global_thinking_budget = int(db.get_config('thinking_budget', '-1'))
    global_include_thoughts = db.get_config('include_thoughts', 'false').lower() == 'true'

    if not global_thinking_enabled:
        return {"thinkingBudget": 0}

    if request.thinking_config:
        if request.thinking_config.thinking_budget is not None:
            thinking_config["thinkingBudget"] = request.thinking_config.thinking_budget
        elif global_thinking_budget >= 0:
            thinking_config["thinkingBudget"] = global_thinking_budget

        if request.thinking_config.include_thoughts is not None:
            thinking_config["includeThoughts"] = request.thinking_config.include_thoughts
        elif global_include_thoughts:
            thinking_config["includeThoughts"] = global_include_thoughts
    else:
        if global_thinking_budget >= 0:
            thinking_config["thinkingBudget"] = global_thinking_budget
        if global_include_thoughts:
            thinking_config["includeThoughts"] = global_include_thoughts

    return thinking_config


def process_multimodal_content(item: Dict) -> Optional[Dict]:
    """处理多模态内容"""
    try:
        # 检查是否有文件数据
        file_data = item.get('file_data') or item.get('fileData')
        inline_data = item.get('inline_data') or item.get('inlineData')

        if inline_data:
            # 内联数据格式
            mime_type = inline_data.get('mimeType') or inline_data.get('mime_type')
            data = inline_data.get('data')

            if mime_type and data:
                return {
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": data
                    }
                }
        elif file_data:
            # 文件引用格式
            mime_type = file_data.get('mimeType') or file_data.get('mime_type')
            file_uri = file_data.get('fileUri') or file_data.get('file_uri')

            if mime_type and file_uri:
                return {
                    "fileData": {
                        "mimeType": mime_type,
                        "fileUri": file_uri
                    }
                }

        # 处理通过文件ID引用的情况
        elif item.get('type') == 'file' and 'file_id' in item:
            file_id = item['file_id']
            if file_id in file_storage:
                file_info = file_storage[file_id]

                if file_info.get('format') == 'inlineData':
                    return {
                        "inlineData": {
                            "mimeType": file_info['mime_type'],
                            "data": file_info['data']
                        }
                    }
                elif file_info.get('format') == 'fileData':
                    if 'gemini_file_uri' in file_info:
                        # 使用Gemini File API的URI
                        return {
                            "fileData": {
                                "mimeType": file_info['mime_type'],
                                "fileUri": file_info['gemini_file_uri']
                            }
                        }
                    elif 'file_uri' in file_info:
                        # 回退到本地文件URI（不推荐，但作为备用）
                        logger.warning(f"Using local file URI for file {file_id}, this may not work with Gemini")
                        return {
                            "fileData": {
                                "mimeType": file_info['mime_type'],
                                "fileUri": file_info['file_uri']
                            }
                        }
            else:
                logger.warning(f"File ID {file_id} not found in storage")

        # 处理直接的图片URL格式（OpenAI兼容）
        if item.get('type') == 'image_url' and 'image_url' in item:
            image_url = item['image_url'].get('url', '')
            if image_url.startswith('data:'):
                try:
                    header, data = image_url.split(',', 1)
                    mime_type = header.split(';')[0].split(':')[1]
                    return {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": data
                        }
                    }
                except Exception as e:
                    logger.warning(f"Failed to parse data URL: {e}")
            else:
                logger.warning("HTTP URLs not supported for images, use file upload instead")

        logger.warning(f"Unsupported multimodal content format: {item}")
        return None

    except Exception as e:
        logger.error(f"Error processing multimodal content: {e}")
        return None


def estimate_token_count(text: str) -> int:
    """
    估算文本的Token数量（简单估算：1个Token约等于4个字符）
    """
    return len(text) // 4


def should_apply_anti_detection(request: ChatCompletionRequest, enable_anti_detection: bool = True) -> bool:
    """
    判断是否应该应用防检测
    """
    if not enable_anti_detection:
        return False
    
    # 检查全局防检测开关
    if not db.get_config('anti_detection_enabled', 'true').lower() == 'true':
        return False
    
    # 检查是否有工具调用且配置为禁用
    disable_for_tools = db.get_config('anti_detection_disable_for_tools', 'true').lower() == 'true'
    if disable_for_tools and (request.tools or request.tool_choice):
        logger.info("Anti-detection disabled for tool calls")
        return False
    
    # 检查Token阈值
    token_threshold = int(db.get_config('anti_detection_token_threshold', '5000'))
    total_tokens = 0
    
    for msg in request.messages:
        if isinstance(msg.content, str):
            total_tokens += estimate_token_count(msg.content)
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, str):
                    total_tokens += estimate_token_count(item)
                elif isinstance(item, dict) and item.get('type') == 'text':
                    total_tokens += estimate_token_count(item.get('text', ''))
    
    if total_tokens < token_threshold:
        logger.info(f"Anti-detection skipped: token count {total_tokens} below threshold {token_threshold}")
        return False
    
    logger.info(f"Anti-detection enabled: token count {total_tokens} exceeds threshold {token_threshold}")
    return True


def openai_to_gemini(request: ChatCompletionRequest, enable_anti_detection: bool = True) -> Dict:
    """
    将OpenAI格式转换为Gemini格式
    """
    contents = []

    # 检查是否应用防检测
    anti_detection_enabled = should_apply_anti_detection(request, enable_anti_detection)

    for msg in request.messages:
        parts = []

        if isinstance(msg.content, str):
            text_content = msg.content

            # 应用防检测处理 - 只对用户消息应用，避免影响系统消息
            if anti_detection_enabled and msg.role == 'user':
                text_content = anti_detection.inject_symbols(text_content)

            if msg.role == "system":
                parts.append({"text": f"[System]: {text_content}"})
            else:
                parts.append({"text": text_content})

        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, str):
                    text_content = item
                    if anti_detection_enabled and msg.role == 'user':
                        text_content = anti_detection.inject_symbols(text_content)
                    parts.append({"text": text_content})

                elif isinstance(item, dict):
                    if item.get('type') == 'text':
                        text_content = item.get('text', '')
                        if anti_detection_enabled and msg.role == 'user':
                            text_content = anti_detection.inject_symbols(text_content)
                        parts.append({"text": text_content})
                    elif item.get('type') in ['image', 'image_url', 'audio', 'video', 'document']:
                        multimodal_part = process_multimodal_content(item)
                        if multimodal_part:
                            parts.append(multimodal_part)

        role = "user" if msg.role in ["system", "user"] else "model"

        if parts:
            contents.append({
                "role": role,
                "parts": parts
            })

    thinking_config = get_thinking_config(request)

    thinking_cfg_obj = None
    if thinking_config:
        thinking_cfg_obj = types.ThinkingConfig(
            thinking_budget=thinking_config.get("thinkingBudget"),
            include_thoughts=thinking_config.get("includeThoughts")
        )

    generation_config = types.GenerationConfig(
        temperature=request.temperature,
        top_p=request.top_p,
        candidate_count=request.n,
        thinking_config=thinking_cfg_obj,
        max_output_tokens=request.max_tokens,
        stop_sequences=request.stop
    )

    gemini_request = {
        "contents": contents,
        "generation_config": generation_config
    }

    return gemini_request


def extract_thoughts_and_content(gemini_response: Dict, include_thoughts: bool = True) -> tuple[str, str]:
    """从Gemini响应中提取思考过程和最终内容"""
    thoughts = ""
    content = ""

    for candidate in gemini_response.get("candidates", []):
        parts = candidate.get("content", {}).get("parts", [])

        for part in parts:
            if "text" in part and part["text"]:  # 确保文本不为空
                is_thought = part.get("thought", False)

                if is_thought:
                    thoughts += part["text"]
                    # 当不包含思考内容时，将思考内容也加入到content中
                    if not include_thoughts:
                        content += part["text"]
                else:
                    # 所有非思考内容都添加到 content
                    content += part["text"]

    return thoughts, content

def gemini_to_openai(gemini_response: Dict, request: ChatCompletionRequest, usage_info: Dict = None) -> Dict:
    """将Gemini响应转换为OpenAI格式"""
    choices = []

    include_thoughts = request.thinking_config and request.thinking_config.include_thoughts
    thoughts, content = extract_thoughts_and_content(gemini_response, include_thoughts)

    for i, candidate in enumerate(gemini_response.get("candidates", [])):
        message_content = content if content else ""

        if thoughts and request.thinking_config and request.thinking_config.include_thoughts:
            message_content = f"**Thinking:**\n{thoughts}\n\n**Response:**\n{content}"

        choices.append({
            "index": i,
            "message": {
                "role": "assistant",
                "content": message_content
            },
            "finish_reason": map_finish_reason(candidate.get("finishReason", "STOP"))
        })

    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": choices,
        "usage": usage_info or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

    return response


def map_finish_reason(gemini_reason: str) -> str:
    """映射Gemini的结束原因到OpenAI格式"""
    mapping = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "OTHER": "stop"
    }
    return mapping.get(gemini_reason, "stop")


def validate_file_for_gemini(file_content: bytes, mime_type: str, filename: str) -> Dict[str, Any]:
    """验证文件是否符合Gemini 2.5要求"""
    file_size = len(file_content)

    if mime_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {mime_type}. Supported types: {', '.join(sorted(SUPPORTED_MIME_TYPES))}"
        )

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB"
        )

    use_inline = file_size <= MAX_INLINE_SIZE

    return {
        "size": file_size,
        "mime_type": mime_type,
        "use_inline": use_inline,
        "filename": filename
    }


async def select_gemini_key_and_check_limits(model_name: str, excluded_keys: set = None) -> Optional[Dict]:
    """自适应选择可用的Gemini Key并检查模型限制"""
    if excluded_keys is None:
        excluded_keys = set()

    available_keys = db.get_available_gemini_keys()
    available_keys = [k for k in available_keys if k['id'] not in excluded_keys]

    if not available_keys:
        logger.warning("No available Gemini keys found after exclusions")
        return None

    model_config = db.get_model_config(model_name)
    if not model_config:
        logger.error(f"Model config not found for: {model_name}")
        return None

    logger.info(
        f"Model {model_name} limits: RPM={model_config['total_rpm_limit']}, TPM={model_config['total_tpm_limit']}, RPD={model_config['total_rpd_limit']}")
    logger.info(f"Available API keys: {len(available_keys)}")

    current_usage = await rate_limiter.get_current_usage(model_name)

    if (current_usage['requests'] >= model_config['total_rpm_limit'] or
            current_usage['tokens'] >= model_config['total_tpm_limit']):
        logger.warning(
            f"Model {model_name} has reached rate limits: requests={current_usage['requests']}/{model_config['total_rpm_limit']}, tokens={current_usage['tokens']}/{model_config['total_tpm_limit']}")
        return None

    day_usage = db.get_usage_stats(model_name, 'day')
    if day_usage['requests'] >= model_config['total_rpd_limit']:
        logger.warning(
            f"Model {model_name} has reached daily request limit: {day_usage['requests']}/{model_config['total_rpd_limit']}")
        return None

    strategy = db.get_config('load_balance_strategy', 'adaptive')

    if strategy == 'round_robin':
        async with _rr_lock:
            idx = next(_rr_counter) % len(available_keys)
            selected_key = available_keys[idx]
    elif strategy == 'least_used':
        selected_key = available_keys[0]
    else:  # adaptive strategy
        best_key = None
        best_score = -1

        for key_info in available_keys:
            success_rate = key_info.get('success_rate', 1.0)
            avg_response_time = key_info.get('avg_response_time', 0.0)
            time_score = max(0, 1.0 - (avg_response_time / 10.0))
            score = success_rate * 0.7 + time_score * 0.3

            if score > best_score:
                best_score = score
                best_key = key_info

        selected_key = best_key if best_key else available_keys[0]

    logger.info(f"Selected API key #{selected_key['id']} for model {model_name} (strategy: {strategy})")

    return {
        'key_info': selected_key,
        'model_config': model_config
    }


# 传统故障转移函数 - 使用 google-genai 替代 httpx
async def make_gemini_request_with_retry(
        gemini_key: str,
        key_id: int,
        gemini_request: Dict,
        model_name: str,
        max_retries: int = 3,
        timeout: float = None
) -> Dict:
    """带重试的Gemini API请求，记录性能指标"""
    if timeout is None:
        timeout = float(db.get_config('request_timeout', '60'))

    for attempt in range(max_retries):
        start_time = time.time()
        try:
            # 使用官方 google-genai 库替代 httpx
            client = genai.Client(api_key=gemini_key)
            async with asyncio.timeout(timeout):
                genai_response = await client.aio.models.generate_content(
                    model=model_name,
                    **gemini_request
                )
                
                response_time = time.time() - start_time
                # 更新key性能
                db.update_key_performance(key_id, True, response_time)
                
                # 将genai响应格式化为与旧代码兼容的格式
                response_json = genai_response.to_dict()
                return response_json

        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            db.update_key_performance(key_id, False, response_time)
            if attempt == max_retries - 1:
                raise HTTPException(status_code=504, detail="Request timeout")
            else:
                logger.warning(f"Request timeout (attempt {attempt + 1}), retrying...")
                await asyncio.sleep(2 ** attempt)
                continue
        except Exception as e:
            response_time = time.time() - start_time
            db.update_key_performance(key_id, False, response_time)
            if attempt == max_retries - 1:
                # 提取错误消息
                error_message = str(e)
                status_code = 500
                
                # 尝试分析错误类型
                if "429" in error_message or "rate limit" in error_message.lower():
                    status_code = 429
                elif "403" in error_message or "permission" in error_message.lower():
                    status_code = 403
                elif "404" in error_message or "not found" in error_message.lower():
                    status_code = 404
                elif "400" in error_message or "invalid" in error_message.lower():
                    status_code = 400
                
                raise HTTPException(status_code=status_code, detail=error_message)
            else:
                logger.warning(f"Request failed (attempt {attempt + 1}): {str(e)}, retrying...")
                await asyncio.sleep(2 ** attempt)
                continue

    raise HTTPException(status_code=500, detail="Max retries exceeded")


async def make_request_with_failover(
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        model_name: str,
        user_key_info: Dict = None,
        max_key_attempts: int = None,
        excluded_keys: set = None
) -> Dict:
    """传统请求处理（保留用于兼容）"""
    if excluded_keys is None:
        excluded_keys = set()

    available_keys = db.get_available_gemini_keys()
    available_keys = [k for k in available_keys if k['id'] not in excluded_keys]

    if not available_keys:
        logger.error("No available keys for failover")
        raise HTTPException(
            status_code=503,
            detail="No available API keys"
        )

    if max_key_attempts is None:
        max_key_attempts = len(available_keys)
    else:
        max_key_attempts = min(max_key_attempts, len(available_keys))

    logger.info(f"Starting failover with {max_key_attempts} key attempts for model {model_name}")

    # 确定超时时间：工具调用或快速响应模式使用60秒，其他使用配置值
    has_tool_calls = bool(openai_request.tools or openai_request.tool_choice)
    is_fast_failover = await should_use_fast_failover()
    if has_tool_calls:
        timeout_seconds = 60.0  # 工具调用强制60秒超时
        logger.info("Using extended 60s timeout for tool calls in traditional failover")
    elif is_fast_failover:
        timeout_seconds = 60.0  # 快速响应模式使用60秒超时
        logger.info("Using extended 60s timeout for fast response mode in traditional failover")
    else:
        timeout_seconds = float(db.get_config('request_timeout', '60'))

    last_error = None
    failed_keys = []

    for attempt in range(max_key_attempts):
        try:
            selection_result = await select_gemini_key_and_check_limits(
                model_name,
                excluded_keys=excluded_keys.union(set(failed_keys))
            )

            if not selection_result:
                logger.warning(f"No more available keys after {attempt} attempts")
                break

            key_info = selection_result['key_info']
            model_config = selection_result['model_config']

            logger.info(f"Attempt {attempt + 1}: Using key #{key_info['id']} for {model_name}")

            try:
                # 直接从Google API收集完整响应（传统故障转移）
                logger.info(f"Using direct collection for non-streaming request with key #{key_info['id']} (traditional failover)")
                
                # 直接收集响应，避免SSE双重解析
                response = await collect_gemini_response_directly(
                    key_info['key'],
                    key_info['id'],
                    gemini_request,
                    openai_request,
                    model_name
                )

                logger.info(f"✅ Request successful with key #{key_info['id']} on attempt {attempt + 1}")

                # 从响应中获取token使用量
                usage = response.get('usage', {})
                total_tokens = usage.get('completion_tokens', 0)

                if user_key_info:
                    db.log_usage(
                        gemini_key_id=key_info['id'],
                        user_key_id=user_key_info['id'],
                        model_name=model_name,
                        requests=1,
                        tokens=total_tokens
                    )
                    logger.info(
                        f"📊 Logged usage: gemini_key_id={key_info['id']}, user_key_id={user_key_info['id']}, model={model_name}, tokens={total_tokens}")

                await rate_limiter.add_usage(model_name, 1, total_tokens)
                return response

            except HTTPException as e:
                failed_keys.append(key_info['id'])
                last_error = e

                db.update_key_performance(key_info['id'], False, 0.0)

                if user_key_info:
                    db.log_usage(
                        gemini_key_id=key_info['id'],
                        user_key_id=user_key_info['id'],
                        model_name=model_name,
                        requests=1,
                        tokens=0
                    )

                await rate_limiter.add_usage(model_name, 1, 0)

                logger.warning(f"❌ Key #{key_info['id']} failed with {e.status_code}: {e.detail}")

                if e.status_code < 500:
                    logger.warning(f"Client error {e.status_code}, stopping failover")
                    raise e

                continue

        except Exception as e:
            logger.error(f"Unexpected error during failover attempt {attempt + 1}: {str(e)}")
            last_error = HTTPException(status_code=500, detail=str(e))
            continue

    failed_count = len(failed_keys)
    logger.error(f"❌ All {failed_count} keys failed for {model_name}")

    if last_error:
        raise last_error
    else:
        raise HTTPException(
            status_code=503,
            detail=f"All {failed_count} available API keys failed"
        )


async def stream_with_failover(
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        model_name: str,
        user_key_info: Dict = None,
        max_key_attempts: int = None,
        excluded_keys: set = None
) -> AsyncGenerator[bytes, None]:
    """传统流式响应处理（保留用于兼容）"""
    if excluded_keys is None:
        excluded_keys = set()

    available_keys = db.get_available_gemini_keys()
    available_keys = [k for k in available_keys if k['id'] not in excluded_keys]

    if not available_keys:
        error_data = {
            'error': {
                'message': 'No available API keys',
                'type': 'service_unavailable',
                'code': 503
            }
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode('utf-8')
        yield "data: [DONE]\n\n".encode('utf-8')
        return

    if max_key_attempts is None:
        max_key_attempts = len(available_keys)
    else:
        max_key_attempts = min(max_key_attempts, len(available_keys))

    logger.info(f"Starting stream failover with {max_key_attempts} key attempts for {model_name}")

    failed_keys = []

    for attempt in range(max_key_attempts):
        try:
            selection_result = await select_gemini_key_and_check_limits(
                model_name,
                excluded_keys=excluded_keys.union(set(failed_keys))
            )

            if not selection_result:
                break

            key_info = selection_result['key_info']
            logger.info(f"Stream attempt {attempt + 1}: Using key #{key_info['id']}")

            success = False
            total_tokens = 0
            try:
                async for chunk in stream_gemini_response(
                        key_info['key'],
                        key_info['id'],
                        gemini_request,
                        openai_request,
                        key_info,
                        model_name
                ):
                    yield chunk
                    success = True

                if success:
                    if user_key_info:
                        db.log_usage(
                            gemini_key_id=key_info['id'],
                            user_key_id=user_key_info['id'],
                            model_name=model_name,
                            requests=1,
                            tokens=total_tokens
                        )
                        logger.info(
                            f"📊 Logged stream usage: gemini_key_id={key_info['id']}, user_key_id={user_key_info['id']}, model={model_name}")

                    await rate_limiter.add_usage(model_name, 1, total_tokens)
                    return

            except Exception as e:
                failed_keys.append(key_info['id'])
                logger.warning(f"Stream key #{key_info['id']} failed: {str(e)}")

                db.update_key_performance(key_info['id'], False, 0.0)

                if user_key_info:
                    db.log_usage(
                        gemini_key_id=key_info['id'],
                        user_key_id=user_key_info['id'],
                        model_name=model_name,
                        requests=1,
                        tokens=0
                    )

                if attempt < max_key_attempts - 1:
                    retry_msg = {
                        'error': {
                            'message': f'Key #{key_info["id"]} failed, trying next key...',
                            'type': 'retry_info',
                            'retry_attempt': attempt + 1
                        }
                    }
                    yield f"data: {json.dumps(retry_msg, ensure_ascii=False)}\n\n".encode('utf-8')
                    continue
                else:
                    break

        except Exception as e:
            logger.error(f"Stream failover error on attempt {attempt + 1}: {str(e)}")
            continue

    error_data = {
        'error': {
            'message': f'All {len(failed_keys)} available API keys failed',
            'type': 'all_keys_failed',
            'code': 503,
            'failed_keys': failed_keys
        }
    }
    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode('utf-8')
    yield "data: [DONE]\n\n".encode('utf-8')


async def stream_gemini_response(
        gemini_key: str,
        key_id: int,
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        key_info: Dict,
        model_name: str
) -> AsyncGenerator[bytes, None]:
    """处理Gemini的流式响应，记录性能指标"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse"
    
    # 确定超时时间：工具调用或快速响应模式使用60秒，其他使用配置值
    has_tool_calls = bool(openai_request.tools or openai_request.tool_choice)
    is_fast_failover = await should_use_fast_failover()
    if has_tool_calls:
        timeout = 60.0  # 工具调用强制60秒超时
        logger.info("Using extended 60s timeout for tool calls in traditional streaming")
    elif is_fast_failover:
        timeout = 60.0  # 快速响应模式使用60秒超时
        logger.info("Using extended 60s timeout for fast response mode in traditional streaming")
    else:
        timeout = float(db.get_config('request_timeout', '60'))
    
    max_retries = int(db.get_config('max_retries', '3'))

    logger.info(f"Starting stream request to: {url}")

    start_time = time.time()

    for attempt in range(max_retries):
        try:
            client = genai.Client(api_key=gemini_key)
            async with asyncio.timeout(timeout):
                genai_stream = client.aio.models.generate_content_stream(
                    model=model_name,
                    body=gemini_request
                )
                # 将 google-genai 流式响应包装为 SSE
                stream_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                created = int(time.time())
                total_tokens = 0
                thinking_sent = False
                processed_chunks = 0

                async for chunk in genai_stream:
                    processed_chunks += 1
                    choices = chunk.candidates or []
                    for candidate in choices:
                        content = candidate.content or {}
                        parts = content.parts or []
                        for part in parts:
                            if hasattr(part, "text"):
                                text = part.text
                                if not text:
                                    continue
                                total_tokens += len(text.split())
                                chunk_data = {
                                    "id": stream_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": openai_request.model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": text},
                                        "finish_reason": None
                                    }]
                                }
                                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n".encode('utf-8')

                        finish_reason = getattr(candidate, "finish_reason", None)
                        if finish_reason:
                            finish_chunk = {
                                "id": stream_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": openai_request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": map_finish_reason(finish_reason)
                                }]
                            }
                            yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n".encode('utf-8')
                            yield "data: [DONE]\n\n".encode('utf-8')

                            response_time = time.time() - start_time
                            db.update_key_performance(key_id, True, response_time)
                            await rate_limiter.add_usage(model_name, 1, total_tokens)
                            return
                    if response.status_code != 200:
                        response_time = time.time() - start_time
                        db.update_key_performance(key_id, False, response_time)

                        # 如果是429错误，则标记为速率受限
                        if response.status_code == 429:
                            logger.warning(f"Stream key #{key_id} is rate-limited (429). Marking as 'rate_limited'.")
                            db.update_gemini_key_status(key_id, 'rate_limited')

                        error_text = await response.aread()
                        error_msg = error_text.decode() if error_text else "Unknown error"
                        logger.error(f"Stream request failed with status {response.status_code}: {error_msg}")
                        yield f"data: {json.dumps({'error': {'message': error_msg, 'type': 'api_error', 'code': response.status_code}}, ensure_ascii=False)}\n\n".encode(
                            'utf-8')
                        yield "data: [DONE]\n\n".encode('utf-8')
                        return

                    stream_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                    created = int(time.time())
                    total_tokens = 0
                    thinking_sent = False
                    has_content = False
                    processed_lines = 0

                    logger.info(f"Stream response started, status: {response.status_code}")

                    try:
                        async for line in response.aiter_lines():
                            processed_lines += 1

                            if not line:
                                continue

                            if processed_lines <= 5:
                                logger.debug(f"Stream line {processed_lines}: {line[:100]}...")

                            if line.startswith("data: "):
                                json_str = line[6:]

                                if json_str.strip() == "[DONE]":
                                    logger.info("Received [DONE] signal from stream")
                                    break

                                if not json_str.strip():
                                    continue

                                try:
                                    data = json.loads(json_str)

                                    for candidate in data.get("candidates", []):
                                        content_data = candidate.get("content", {})
                                        parts = content_data.get("parts", [])

                                        for part in parts:
                                            if "text" in part:
                                                text = part["text"]
                                                if not text:
                                                    continue

                                                total_tokens += len(text.split())
                                                has_content = True

                                                is_thought = part.get("thought", False)

                                                if is_thought and not (openai_request.thinking_config and
                                                                       openai_request.thinking_config.include_thoughts):
                                                    continue

                                                if is_thought and not thinking_sent:
                                                    thinking_header = {
                                                        "id": stream_id,
                                                        "object": "chat.completion.chunk",
                                                        "created": created,
                                                        "model": openai_request.model,
                                                        "choices": [{
                                                            "index": 0,
                                                            "delta": {"content": "**Thinking Process:**\n"},
                                                            "finish_reason": None
                                                        }]
                                                    }
                                                    yield f"data: {json.dumps(thinking_header, ensure_ascii=False)}\n\n".encode(
                                                        'utf-8')
                                                    thinking_sent = True
                                                    logger.debug("Sent thinking header")
                                                elif not is_thought and thinking_sent:
                                                    response_header = {
                                                        "id": stream_id,
                                                        "object": "chat.completion.chunk",
                                                        "created": created,
                                                        "model": openai_request.model,
                                                        "choices": [{
                                                            "index": 0,
                                                            "delta": {"content": "\n\n**Response:**\n"},
                                                            "finish_reason": None
                                                        }]
                                                    }
                                                    yield f"data: {json.dumps(response_header, ensure_ascii=False)}\n\n".encode(
                                                        'utf-8')
                                                    thinking_sent = False
                                                    logger.debug("Sent response header")

                                                chunk_data = {
                                                    "id": stream_id,
                                                    "object": "chat.completion.chunk",
                                                    "created": created,
                                                    "model": openai_request.model,
                                                    "choices": [{
                                                        "index": 0,
                                                        "delta": {"content": text},
                                                        "finish_reason": None
                                                    }]
                                                }
                                                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n".encode(
                                                    'utf-8')

                                        finish_reason = candidate.get("finishReason")
                                        if finish_reason:
                                            finish_chunk = {
                                                "id": stream_id,
                                                "object": "chat.completion.chunk",
                                                "created": created,
                                                "model": openai_request.model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {},
                                                    "finish_reason": map_finish_reason(finish_reason)
                                                }]
                                            }
                                            yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n".encode(
                                                'utf-8')
                                            yield "data: [DONE]\n\n".encode('utf-8')

                                            logger.info(
                                                f"Stream completed with finish_reason: {finish_reason}, tokens: {total_tokens}")

                                            response_time = time.time() - start_time
                                            db.update_key_performance(key_id, True, response_time)
                                            await rate_limiter.add_usage(model_name, 1, total_tokens)
                                            return

                                except json.JSONDecodeError as e:
                                    logger.warning(f"JSON decode error: {e}, line: {json_str[:200]}...")
                                    continue

                            elif line.startswith("event: "):
                                continue
                            elif line.startswith("id: ") or line.startswith("retry: "):
                                continue

                        if has_content:
                            finish_chunk = {
                                "id": stream_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": openai_request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }]
                            }
                            yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n".encode('utf-8')
                            yield "data: [DONE]\n\n".encode('utf-8')

                            logger.info(
                                f"Stream ended naturally, processed {processed_lines} lines, tokens: {total_tokens}")

                            response_time = time.time() - start_time
                            db.update_key_performance(key_id, True, response_time)

                        if not has_content:
                            logger.warning(
                                f"Stream response had no content after processing {processed_lines} lines, falling back to non-stream")
                            try:
                                fallback_response = await make_gemini_request_with_retry(
                                    gemini_key, key_id, gemini_request, model_name, 1, timeout=timeout
                                )

                                include_thoughts_fallback = openai_request.thinking_config and openai_request.thinking_config.include_thoughts
                                thoughts, content = extract_thoughts_and_content(fallback_response, include_thoughts_fallback)

                                if thoughts and openai_request.thinking_config and openai_request.thinking_config.include_thoughts:
                                    full_content = f"**Thinking Process:**\n{thoughts}\n\n**Response:**\n{content}"
                                else:
                                    full_content = content

                                if full_content:
                                    chunk_data = {
                                        "id": stream_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": openai_request.model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"content": full_content},
                                            "finish_reason": None
                                        }]
                                    }
                                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n".encode('utf-8')

                                    finish_chunk = {
                                        "id": stream_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": openai_request.model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {},
                                            "finish_reason": "stop"
                                        }]
                                    }
                                    yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n".encode('utf-8')
                                    total_tokens = len(full_content.split())

                                    logger.info(f"Fallback completed, tokens: {total_tokens}")

                            except Exception as e:
                                logger.error(f"Fallback request failed: {e}")
                                response_time = time.time() - start_time
                                db.update_key_performance(key_id, False, response_time)
                                yield f"data: {json.dumps({'error': {'message': 'Failed to get response', 'type': 'server_error'}}, ensure_ascii=False)}\n\n".encode(
                                    'utf-8')

                        await rate_limiter.add_usage(model_name, 1, total_tokens)
                        yield "data: [DONE]\n\n".encode('utf-8')
                        return

                    except Exception as e:  # 原 httpx 异常移除
                        logger.warning(f"Stream connection error (attempt {attempt + 1}): {str(e)}")
                        response_time = time.time() - start_time
                        db.update_key_performance(key_id, False, response_time)
                        if attempt < max_retries - 1:
                            yield f"data: {json.dumps({'error': {'message': 'Connection interrupted, retrying...', 'type': 'connection_error'}}, ensure_ascii=False)}\n\n".encode(
                                'utf-8')
                            await asyncio.sleep(1)
                            continue
                        else:
                            yield f"data: {json.dumps({'error': {'message': 'Stream connection failed after retries', 'type': 'connection_error'}}, ensure_ascii=False)}\n\n".encode(
                                'utf-8')
                            yield "data: [DONE]\n\n".encode('utf-8')
                            return

        except Exception as e:  # 原 httpx 超时异常已移除
            logger.warning(f"Connection error (attempt {attempt + 1}): {str(e)}")
            response_time = time.time() - start_time
            db.update_key_performance(key_id, False, response_time)
            if attempt < max_retries - 1:
                yield f"data: {json.dumps({'error': {'message': f'Connection error, retrying... (attempt {attempt + 1})', 'type': 'connection_error'}}, ensure_ascii=False)}\n\n".encode(
                    'utf-8')
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                yield f"data: {json.dumps({'error': {'message': 'Connection failed after all retries', 'type': 'connection_error'}}, ensure_ascii=False)}\n\n".encode(
                    'utf-8')
                yield "data: [DONE]\n\n".encode('utf-8')
                return
        except Exception as e:
            logger.error(f"Unexpected error in stream (attempt {attempt + 1}): {str(e)}")
            response_time = time.time() - start_time
            db.update_key_performance(key_id, False, response_time)
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
            else:
                yield f"data: {json.dumps({'error': {'message': 'Unexpected error occurred', 'type': 'server_error'}}, ensure_ascii=False)}\n\n".encode(
                    'utf-8')
                yield "data: [DONE]\n\n".encode('utf-8')
                return


# API端点
@app.get("/", summary="服务根端点", tags=["通用"])
async def root():
    """
    **服务根端点**

    返回服务的基本信息、状态和功能列表。
    可用于快速检查服务是否正在运行。
    """
    return {
        "service": "Gemini API Proxy",
        "status": "running",
        "version": "1.4",
        "features": ["Gemini 2.5 Multimodal"],
        "keep_alive": keep_alive_enabled,
        "auto_cleanup": db.get_auto_cleanup_config()['enabled'],
        "anti_detection": db.get_config('anti_detection_enabled', 'true').lower() == 'true',
        "fast_failover": db.get_failover_config()['fast_failover_enabled'],
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", summary="服务健康检查", tags=["通用"])
async def health_check():
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
        "version": "1.4",
        "multimodal_support": "Gemini 2.5 Optimized",
        "keep_alive_enabled": keep_alive_enabled,
        "auto_cleanup_enabled": db.get_auto_cleanup_config()['enabled'],
        "anti_detection_enabled": db.get_config('anti_detection_enabled', 'true').lower() == 'true',
        "fast_failover_enabled": db.get_failover_config()['fast_failover_enabled']
    }


@app.get("/wake", summary="服务唤醒", tags=["通用"])
async def wake_up():
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


@app.get("/status", summary="获取详细服务状态", tags=["通用"])
async def get_status():
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
        "version": "1.4",
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


@app.get("/metrics", summary="获取服务指标", tags=["通用"])
async def get_metrics():
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


@app.get("/v1", summary="获取 v1 API 信息", tags=["通用"])
async def api_v1_info():
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
        "version": "1.4",
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
@app.post("/v1/files", summary="上传文件", tags=["用户 API"])
async def upload_file(
        file: UploadFile = File(..., description="要上传的文件，最大100MB"),
        authorization: str = Header(None, description="用户访问密钥，格式为 'Bearer sk-...'")
):
    """
    **上传文件用于多模态对话**

    此端点用于上传文件（图片、音频、视频、文档），以便在 `/v1/chat/completions` 中引用。

    - **智能处理**:
        - 小于20MB的文件将作为内联数据处理。
        - 大于20MB的文件将自动上传到Gemini File API。
    - **返回**: 返回一个文件对象，其中包含一个唯一的 `file_id`，可在对话中引用。
    """
    try:
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/files", summary="列出已上传的文件", tags=["用户 API"])
async def list_files(authorization: str = Header(None, description="用户访问密钥，格式为 'Bearer sk-...'")):
    """
    **列出已上传的文件**

    返回当前服务器上所有已上传文件的列表。
    文件会在24小时后自动清理。
    """
    try:
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List files failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/files/{file_id}", summary="获取文件信息", tags=["用户 API"])
async def get_file(
    file_id: str,
    authorization: str = Header(None, description="用户访问密钥，格式为 'Bearer sk-...'")
):
    """
    **获取文件信息**

    根据文件ID获取指定文件的详细信息。
    """
    try:
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get file failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/files/{file_id}", summary="删除文件", tags=["用户 API"])
async def delete_file(
    file_id: str,
    authorization: str = Header(None, description="用户访问密钥，格式为 'Bearer sk-...'")
):
    """
    **删除文件**

    从服务器删除指定的文件。
    如果文件已上传到Gemini File API，也会尝试从Gemini侧删除。
    """
    try:
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete file failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# chat_completions端点
@app.post("/v1/chat/completions", summary="创建聊天补全", tags=["用户 API"])
async def chat_completions(
        request: ChatCompletionRequest,
        authorization: str = Header(None, description="用户访问密钥，格式为 'Bearer sk-...'")
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
    try:
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
        
        # 检查是否有工具调用
        has_tool_calls = bool(request.tools or request.tool_choice)
        
        # 根据流式模式配置决定是否使用流式响应
        should_stream = request.stream  # 默认跟随用户请求
        logger.info(f"DEBUG: request.stream={request.stream}, stream_mode={stream_mode}, has_tool_calls={has_tool_calls}")
        
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

        logger.info(f"DEBUG: Final should_stream={should_stream}")

        if should_stream:
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models", summary="列出可用模型", tags=["用户 API"])
async def list_models():
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


# 健康检测相关端点
@app.post("/admin/health/check-all", summary="一键健康检测", tags=["管理 API：健康与状态"])
async def check_all_keys_health():
    """
    **一键健康检测**

    对所有已激活的Gemini API Key执行一次健康检查。
    - **操作**: 向每个Key发送一个测试请求。
    - **结果**: 更新每个Key的健康状态、响应时间和成功率。
    - **用途**: 在添加新Key或服务出现问题时，快速诊断所有Key的可用性。
    """
    try:
        all_keys = db.get_all_gemini_keys()
        active_keys = [key for key in all_keys if key['status'] == 1]

        if not active_keys:
            return {
                "success": True,
                "message": "No active keys to check",
                "results": []
            }

        results = []
        healthy_count = 0

        tasks = []
        for key_info in active_keys:
            task = check_gemini_key_health(key_info['key'])
            tasks.append((key_info['id'], task))

        for key_id, task in tasks:
            health_result = await task

            db.update_key_performance(
                key_id,
                health_result['healthy'],
                health_result['response_time']
            )

            # 同时记录到健康检测历史
            db.record_daily_health_status(
                key_id,
                health_result['healthy'],
                health_result['response_time']
            )

            if health_result['healthy']:
                healthy_count += 1

            results.append({
                "key_id": key_id,
                "healthy": health_result['healthy'],
                "response_time": health_result['response_time'],
                "error": health_result['error']
            })

        return {
            "success": True,
            "message": f"Health check completed: {healthy_count}/{len(active_keys)} keys healthy",
            "total_checked": len(active_keys),
            "healthy_count": healthy_count,
            "unhealthy_count": len(active_keys) - healthy_count,
            "results": results
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/health/summary", summary="获取健康状态汇总", tags=["管理 API：健康与状态"])
async def get_health_summary():
    """
    **获取健康状态汇总**

    返回所有Gemini API Key的健康状态统计信息，包括：
    - 总数
    - 激活数
    - 健康数
    - 异常数
    - 未知数
    """
    try:
        summary = db.get_keys_health_summary()
        return {
            "success": True,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Failed to get health summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 自动清理管理端点
@app.get("/admin/cleanup/status", summary="获取自动清理状态", tags=["管理 API：配置"])
async def get_cleanup_status():
    """
    **获取自动清理状态**

    返回自动清理功能的当前配置和状态，包括：
    - 是否启用
    - 清理阈值（连续异常天数）
    - 处于风险状态（可能被清理）的密钥列表
    """
    try:
        cleanup_config = db.get_auto_cleanup_config()
        at_risk_keys = db.get_at_risk_keys(cleanup_config['days_threshold'])

        return {
            "success": True,
            "auto_cleanup_enabled": cleanup_config['enabled'],
            "days_threshold": cleanup_config['days_threshold'],
            "min_checks_per_day": cleanup_config['min_checks_per_day'],
            "at_risk_keys": at_risk_keys,
            "next_cleanup": "Every day at 02:00 UTC"
        }

    except Exception as e:
        logger.error(f"Failed to get cleanup status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/cleanup/config", summary="更新自动清理配置", tags=["管理 API：配置"])
async def update_cleanup_config(request: dict):
    """
    **更新自动清理配置**

    允许管理员修改自动清理的参数。
    - **enabled**: `true` 或 `false`
    - **days_threshold**: 连续异常多少天后进行清理 (整数)
    - **min_checks_per_day**: 每日最少需要被检测多少次才纳入统计 (整数)
    """
    try:
        enabled = request.get('enabled')
        days_threshold = request.get('days_threshold')
        min_checks = request.get('min_checks_per_day')

        success = db.set_auto_cleanup_config(
            enabled=enabled,
            days_threshold=days_threshold,
            min_checks_per_day=min_checks
        )

        if success:
            logger.info(
                f"Updated auto cleanup config: enabled={enabled}, days={days_threshold}, min_checks={min_checks}")

            return {
                "success": True,
                "message": "Auto cleanup configuration updated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update auto cleanup configuration")

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update cleanup config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/cleanup/manual", summary="手动执行清理", tags=["管理 API：配置"])
async def manual_cleanup():
    """
    **手动执行清理**

    立即根据当前配置执行一次自动清理任务。
    主要用于测试或紧急处理。
    """
    try:
        await auto_cleanup_failed_keys()
        return {
            "success": True,
            "message": "Manual cleanup executed successfully"
        }
    except Exception as e:
        logger.error(f"Manual cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 故障转移配置管理端点
@app.get("/admin/config/failover", summary="获取故障转移配置", tags=["管理 API：配置"])
async def get_failover_config():
    """
    **获取故障转移配置**

    返回故障转移功能的当前配置，包括：
    - 是否启用快速故障转移
    - 后台健康检测设置
    - 当前可用和健康的密钥统计
    """
    try:
        config = db.get_failover_config()

        # 获取当前Key统计信息
        available_keys = db.get_available_gemini_keys()
        healthy_keys = db.get_healthy_gemini_keys()

        return {
            "success": True,
            "config": config,
            "stats": {
                "available_keys": len(available_keys),
                "healthy_keys": len(healthy_keys),
                "max_possible_attempts": min(len(available_keys), 20)
            }
        }
    except Exception as e:
        logger.error(f"Failed to get failover config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/config/failover", summary="更新故障转移配置", tags=["管理 API：配置"])
async def update_failover_config(request: dict):
    """
    **更新故障转移配置**

    允许管理员修改故障转移策略的参数。
    """
    try:
        fast_failover_enabled = request.get('fast_failover_enabled')

        background_health_check = request.get('background_health_check')
        health_check_delay = request.get('health_check_delay')

        success = db.set_failover_config(
            fast_failover_enabled=fast_failover_enabled,

            background_health_check=background_health_check,
            health_check_delay=health_check_delay
        )

        if success:
            logger.info(f"Updated failover config: {request}")
            return {
                "success": True,
                "message": "Failover configuration updated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update failover configuration")

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update failover config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/failover/stats", summary="获取故障转移统计", tags=["管理 API：健康与状态"])
async def get_failover_stats():
    """
    **获取故障转移统计信息**

    返回与故障转移相关的统计数据和建议，例如：
    - 密钥健康状态汇总
    - 基于当前状态的配置建议
    """
    try:
        # 获取Key健康状态统计
        health_summary = db.get_keys_health_summary()

        # 获取最近的故障转移统计（可以从使用日志中统计）
        return {
            "success": True,
            "health_summary": health_summary,
            "config": db.get_failover_config(),
            "recommendations": {
                "optimal_max_attempts": min(max(health_summary.get('healthy', 0), 2), 5),
                "fast_failover_recommended": health_summary.get('unhealthy', 0) > 0,
                "background_check_recommended": True
            }
        }
    except Exception as e:
        logger.error(f"Failed to get failover stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 防检测管理端点
@app.post("/admin/config/anti-detection", summary="更新防检测配置", tags=["管理 API：配置"])
async def update_anti_detection_config(request: dict):
    """
    **更新防检测配置**

    修改防自动化检测功能的参数。
    - **enabled**: `true` 或 `false`
    - **disable_for_tools**: 在工具调用时是否禁用
    - **token_threshold**: 触发防检测的最小Token阈值
    """
    try:
        enabled = request.get('enabled')
        disable_for_tools = request.get('disable_for_tools')
        token_threshold = request.get('token_threshold')

        success_count = 0
        
        if enabled is not None:
            if db.set_config('anti_detection_enabled', 'true' if enabled else 'false'):
                success_count += 1
                logger.info(f"Anti-detection enabled: {enabled}")
        
        if disable_for_tools is not None:
            if db.set_config('anti_detection_disable_for_tools', 'true' if disable_for_tools else 'false'):
                success_count += 1
                logger.info(f"Anti-detection disable for tools: {disable_for_tools}")
        
        if token_threshold is not None:
            if isinstance(token_threshold, (int, float)) and token_threshold >= 1000:
                if db.set_config('anti_detection_token_threshold', str(int(token_threshold))):
                    success_count += 1
                    logger.info(f"Anti-detection token threshold: {token_threshold}")
            else:
                raise HTTPException(status_code=422, detail="Token threshold must be a number >= 1000")

        if success_count > 0:
            return {
                "success": True,
                "message": "Anti-detection configuration updated successfully",
                "updated_fields": success_count
            }
        else:
            raise HTTPException(status_code=422, detail="No valid parameters provided")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update anti-detection config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/config/anti-detection", summary="获取防检测配置", tags=["管理 API：配置"])
async def get_anti_detection_config():
    """
    **获取防检测配置**

    返回防自动化检测功能的当前配置和统计信息。
    """
    try:
        enabled = db.get_config('anti_detection_enabled', 'true').lower() == 'true'
        disable_for_tools = db.get_config('anti_detection_disable_for_tools', 'true').lower() == 'true'
        token_threshold = int(db.get_config('anti_detection_token_threshold', '5000'))

        return {
            "success": True,
            "anti_detection_enabled": enabled,
            "disable_for_tools": disable_for_tools,
            "token_threshold": token_threshold,
            "statistics": anti_detection.get_statistics()
        }
    except Exception as e:
        logger.error(f"Failed to get anti-detection config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/test/anti-detection", summary="测试防检测功能", tags=["管理 API：配置"])
async def test_anti_detection():
    """
    **测试防检测功能**

    对示例文本执行一次防检测注入处理，并返回结果。
    用于验证注入功能是否正常工作。
    """
    try:
        test_texts = [
            "请帮我分析这个问题",
            "使用中文回复：",
            "请告诉我",
            "我想说："
        ]

        results = []
        for text in test_texts:
            processed = anti_detection.inject_symbols(text)
            results.append({
                "original": text,
                "processed": processed,
                "char_difference": len(processed) - len(text)
            })

        return {
            "success": True,
            "results": results,
            "total_symbols_available": len(anti_detection.safe_symbols)
        }

    except Exception as e:
        logger.error(f"Anti-detection test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 保活管理端点
@app.post("/admin/keep-alive/toggle", summary="切换保活状态", tags=["管理 API：配置"])
async def toggle_keep_alive():
    """
    **切换保活状态**

    启用或禁用服务的自动保活功能。
    """
    global scheduler, keep_alive_enabled

    try:
        if keep_alive_enabled and scheduler and scheduler.running:
            # 停用保活
            scheduler.shutdown(wait=False)
            scheduler = None
            keep_alive_enabled = False
            logger.info("🔴 Keep-alive disabled manually")
            return {
                "success": True,
                "message": "Keep-alive disabled",
                "enabled": False
            }
        else:
            # 启用保活
            keep_alive_interval = int(os.getenv('KEEP_ALIVE_INTERVAL', '10'))
            scheduler = AsyncIOScheduler()

            scheduler.add_job(
                keep_alive_ping,
                'interval',
                minutes=keep_alive_interval,
                id='keep_alive',
                max_instances=1,
                coalesce=True,
                misfire_grace_time=30
            )

            scheduler.add_job(
                rate_limiter.cleanup_expired,
                'interval',
                minutes=5,
                id='cache_cleanup',
                max_instances=1
            )

            # 重新添加健康检测和自动清理任务
            scheduler.add_job(
                record_hourly_health_check,
                'interval',
                hours=1,
                id='hourly_health_check',
                max_instances=1,
                coalesce=True
            )

            scheduler.add_job(
                auto_cleanup_failed_keys,
                'cron',
                hour=2,
                minute=0,
                id='daily_cleanup',
                max_instances=1,
                coalesce=True
            )

            scheduler.start()
            keep_alive_enabled = True

            # 立即执行一次保活
            await keep_alive_ping()

            logger.info(f"🟢 Keep-alive enabled manually (interval: {keep_alive_interval} minutes)")
            return {
                "success": True,
                "message": f"Keep-alive enabled (interval: {keep_alive_interval} minutes)",
                "enabled": True,
                "interval_minutes": keep_alive_interval
            }

    except Exception as e:
        logger.error(f"Failed to toggle keep-alive: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to toggle keep-alive: {str(e)}",
            "enabled": keep_alive_enabled
        }


@app.get("/admin/keep-alive/status", summary="获取保活状态", tags=["管理 API：配置"])
async def get_keep_alive_status():
    """
    **获取保活状态**

    返回保活功能的当前状态，包括是否启用、下次ping时间等。
    """
    global keep_alive_enabled

    next_run = None
    if scheduler and scheduler.running:
        try:
            job = scheduler.get_job('keep_alive')
            if job:
                next_run = job.next_run_time.isoformat() if job.next_run_time else None
        except:
            pass

    return {
        "enabled": keep_alive_enabled,
        "scheduler_running": scheduler.running if scheduler else False,
        "next_ping": next_run,
        "interval_minutes": int(os.getenv('KEEP_ALIVE_INTERVAL', '10')),
        "environment_enabled": os.getenv('ENABLE_KEEP_ALIVE', 'false').lower() == 'true'
    }


@app.post("/admin/keep-alive/ping", summary="手动执行保活ping", tags=["管理 API：配置"])
async def manual_keep_alive_ping():
    """
    **手动执行保活ping**

    立即执行一次保活ping操作。
    """
    try:
        await keep_alive_ping()
        return {
            "success": True,
            "message": "Keep-alive ping executed successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Manual keep-alive ping failed: {str(e)}")
        return {
            "success": False,
            "message": f"Keep-alive ping failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


# 密钥管理端点
@app.get("/admin/keys/gemini", summary="获取所有Gemini密钥", tags=["管理 API：密钥管理"])
async def get_gemini_keys():
    """
    **获取所有Gemini密钥**

    返回数据库中存储的所有Gemini API Key的详细列表，包括其状态和性能指标。
    """
    try:
        keys = db.get_all_gemini_keys()
        return {
            "success": True,
            "keys": keys
        }
    except Exception as e:
        logger.error(f"Failed to get Gemini keys: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/keys/user", summary="获取所有用户密钥", tags=["管理 API：密钥管理"])
async def get_user_keys():
    """
    **获取所有用户密钥**

    返回数据库中存储的所有用户访问密钥的列表。
    """
    try:
        keys = db.get_all_user_keys()
        return {
            "success": True,
            "keys": keys
        }
    except Exception as e:
        logger.error(f"Failed to get user keys: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/admin/keys/gemini/{key_id}", summary="删除Gemini密钥", tags=["管理 API：密钥管理"])
async def delete_gemini_key(key_id: int):
    """
    **删除指定的Gemini密钥**

    根据ID从数据库中永久删除一个Gemini API Key。
    """
    try:
        success = db.delete_gemini_key(key_id)
        if success:
            logger.info(f"Deleted Gemini key #{key_id}")
            return {
                "success": True,
                "message": f"Gemini key #{key_id} deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Key not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete Gemini key #{key_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/admin/keys/user/{key_id}", summary="删除用户密钥", tags=["管理 API：密钥管理"])
async def delete_user_key(key_id: int):
    """
    **删除指定的用户密钥**

    根据ID从数据库中永久删除一个用户访问密钥。
    """
    try:
        success = db.delete_user_key(key_id)
        if success:
            logger.info(f"Deleted user key #{key_id}")
            return {
                "success": True,
                "message": f"User key #{key_id} deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Key not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user key #{key_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/keys/gemini/{key_id}/toggle", summary="切换Gemini密钥状态", tags=["管理 API：密钥管理"])
async def toggle_gemini_key_status(key_id: int):
    """
    **切换Gemini密钥状态**

    切换指定Gemini API Key的激活/禁用状态。
    禁用的密钥将不会被用于处理请求。
    """
    try:
        success = db.toggle_gemini_key_status(key_id)
        if success:
            logger.info(f"Toggled Gemini key #{key_id} status")
            return {
                "success": True,
                "message": f"Gemini key #{key_id} status toggled successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Key not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle Gemini key #{key_id} status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/keys/user/{key_id}/toggle", summary="切换用户密钥状态", tags=["管理 API：密钥管理"])
async def toggle_user_key_status(key_id: int):
    """
    **切换用户密钥状态**

    切换指定用户访问密钥的激活/禁用状态。
    """
    try:
        success = db.toggle_user_key_status(key_id)
        if success:
            logger.info(f"Toggled user key #{key_id} status")
            return {
                "success": True,
                "message": f"User key #{key_id} status toggled successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Key not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle user key #{key_id} status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 管理端点
@app.get("/admin/models/{model_name}", summary="获取模型配置", tags=["管理 API：模型配置"])
async def get_model_config(model_name: str):
    """
    **获取指定模型的配置**

    返回指定模型的速率限制（RPM, RPD, TPM）和状态。
    """
    try:
        model_config = db.get_model_config(model_name)
        if not model_config:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        return {
            "success": True,
            "model_name": model_name,
            **model_config
        }
    except Exception as e:
        logger.error(f"Failed to get model config for {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/models/{model_name}", summary="更新模型配置", tags=["管理 API：模型配置"])
async def update_model_config(model_name: str, request: dict):
    """
    **更新指定模型的配置**

    修改指定模型的速率限制和状态。
    """
    try:
        if model_name not in db.get_supported_models():
            raise HTTPException(status_code=404, detail=f"Model {model_name} not supported")

        allowed_fields = ['single_api_rpm_limit', 'single_api_tpm_limit', 'single_api_rpd_limit', 'status']
        update_data = {}

        for field in allowed_fields:
            if field in request:
                update_data[field] = request[field]

        if not update_data:
            raise HTTPException(status_code=422, detail="No valid fields to update")

        success = db.update_model_config(model_name, **update_data)

        if success:
            logger.info(f"Updated model config for {model_name}: {update_data}")
            return {
                "success": True,
                "message": f"Model {model_name} configuration updated successfully",
                "updated_fields": update_data
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update model configuration")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update model config for {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/models", summary="列出所有模型配置", tags=["管理 API：模型配置"])
async def list_model_configs():
    """
    **获取所有模型的配置**

    返回所有支持的模型的配置列表。
    """
    try:
        model_configs = db.get_all_model_configs()
        return {
            "success": True,
            "models": model_configs
        }
    except Exception as e:
        logger.error(f"Failed to get model configs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/config/gemini-key", summary="添加Gemini密钥", tags=["管理 API：密钥管理"])
async def add_gemini_key(request: dict):
    """
    **添加Gemini密钥**

    向数据库中添加一个新的或多个Gemini API Key。
    支持通过逗号、分号、换行符等分隔符进行批量添加。
    """
    input_keys = request.get("key", "").strip()

    if not input_keys:
        return {"success": False, "message": "请提供API密钥"}

    separators = [',', ';', '\n', '\r\n', '\r', '\t']
    has_separator = any(sep in input_keys for sep in separators)

    if has_separator or '  ' in input_keys:
        lines = input_keys.replace('\r\n', '\n').replace('\r', '\n').split('\n')

        keys_to_add = []
        for line in lines:
            line_keys = []
            for sep in [',', ';', '\t']:
                if sep in line:
                    line_keys.extend([k.strip() for k in line.split(sep)])
                    break
            else:
                if '  ' in line:
                    line_keys.extend([k.strip() for k in line.split()])
                else:
                    line_keys.append(line.strip())

            keys_to_add.extend(line_keys)

        keys_to_add = [key for key in keys_to_add if key]

        logger.info(f"检测到批量添加模式，将添加 {len(keys_to_add)} 个密钥")

    else:
        keys_to_add = [input_keys]

    results = {
        "success": True,
        "total_processed": len(keys_to_add),
        "successful_adds": 0,
        "failed_adds": 0,
        "details": [],
        "invalid_keys": [],
        "duplicate_keys": []
    }

    for i, key in enumerate(keys_to_add, 1):
        key = key.strip()

        if not key:
            continue

        if not key.startswith('AIzaSy'):
            results["invalid_keys"].append(f"#{i}: {key[:20]}... (不是有效的Gemini API密钥格式)")
            results["failed_adds"] += 1
            continue

        if len(key) < 30 or len(key) > 50:
            results["invalid_keys"].append(f"#{i}: {key[:20]}... (密钥长度异常)")
            results["failed_adds"] += 1
            continue

        try:
            if db.add_gemini_key(key):
                results["successful_adds"] += 1
                results["details"].append(f"✅ #{i}: {key[:10]}...{key[-4:]} 添加成功")
                logger.info(f"成功添加Gemini密钥 #{i}")
            else:
                results["duplicate_keys"].append(f"#{i}: {key[:10]}...{key[-4:]} (密钥已存在)")
                results["failed_adds"] += 1
        except Exception as e:
            results["failed_adds"] += 1
            results["details"].append(f"❌ #{i}: {key[:10]}...{key[-4:]} 添加失败 - {str(e)}")
            logger.error(f"添加Gemini密钥 #{i} 失败: {str(e)}")

    if results["successful_adds"] > 0:
        message_parts = [f"成功添加 {results['successful_adds']} 个密钥"]

        if results["failed_adds"] > 0:
            message_parts.append(f"失败 {results['failed_adds']} 个")

        results["message"] = "、".join(message_parts)
        results["success"] = True
    else:
        results["success"] = False
        results["message"] = f"所有 {results['total_processed']} 个密钥添加失败"

    logger.info(
        f"批量添加结果: 处理{results['total_processed']}个，成功{results['successful_adds']}个，失败{results['failed_adds']}个")

    return results


@app.post("/admin/config/user-key", summary="生成用户密钥", tags=["管理 API：密钥管理"])
async def generate_user_key(request: dict):
    """
    **生成用户密钥**

    创建一个新的用户访问密钥 (sk-...)，用于API调用认证。
    可以为密钥指定一个名称以方便识别。
    """
    name = request.get("name", "API User")
    key = db.generate_user_key(name)
    logger.info(f"Generated new user key for: {name}")
    return {"success": True, "key": key, "name": name}


@app.post("/admin/config/thinking", summary="更新思考模式配置", tags=["管理 API：配置"])
async def update_thinking_config(request: dict):
    """
    **更新思考模式配置**

    修改思考模式的参数。
    - **enabled**: `true` 或 `false`
    - **budget**: 思考预算 (-1=自动, 0=禁用, >0=具体值)
    - **include_thoughts**: 是否在响应中包含思考过程
    """
    try:
        enabled = request.get('enabled')
        budget = request.get('budget')
        include_thoughts = request.get('include_thoughts')

        success = db.set_thinking_config(
            enabled=enabled,
            budget=budget,
            include_thoughts=include_thoughts
        )

        if success:
            logger.info(f"Updated thinking config: {request}")
            return {
                "success": True,
                "message": "Thinking configuration updated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update thinking configuration")

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update thinking config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/config/inject-prompt", summary="更新提示词注入配置", tags=["管理 API：配置"])
async def update_inject_prompt_config(request: dict):
    """
    **更新提示词注入配置**

    修改自动提示词注入的参数。
    - **enabled**: `true` 或 `false`
    - **content**: 要注入的提示词内容
    - **position**: 注入位置 ('system', 'user_prefix', 'user_suffix')
    """
    try:
        enabled = request.get('enabled')
        content = request.get('content')
        position = request.get('position')

        success = db.set_inject_prompt_config(
            enabled=enabled,
            content=content,
            position=position
        )

        if success:
            logger.info(f"Updated inject prompt config: enabled={enabled}, position={position}")
            return {
                "success": True,
                "message": "Inject prompt configuration updated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update inject prompt configuration")

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update inject prompt config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/config/stream-mode", summary="更新流式模式配置", tags=["管理 API：配置"])
async def update_stream_mode_config(request: dict):
    """
    **更新流式模式配置**

    修改API响应的全局流式行为。
    - **mode**: 'auto', 'stream', 'non_stream'
    """
    try:
        mode = request.get('mode')

        success = db.set_stream_mode_config(mode=mode)

        if success:
            logger.info(f"Updated stream mode config: mode={mode}")
            return {
                "success": True,
                "message": "Stream mode configuration updated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update stream mode configuration")

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update stream mode config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/config/load-balance", summary="更新负载均衡策略", tags=["管理 API：配置"])
async def update_load_balance_config(request: dict):
    """
    **更新负载均衡策略**

    修改选择Gemini API Key时使用的负载均衡算法。
    - **load_balance_strategy**: 'adaptive', 'least_used', 'round_robin'
    """
    try:
        strategy = request.get('load_balance_strategy')
        if strategy not in ['adaptive', 'least_used', 'round_robin']:
            raise ValueError("Invalid load balance strategy")

        success = db.set_config('load_balance_strategy', strategy)

        if success:
            logger.info(f"Updated load balance strategy: {strategy}")
            return {
                "success": True,
                "message": "Load balance strategy updated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update load balance strategy")

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update load balance strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/config", summary="获取所有系统配置", tags=["管理 API：配置"])
async def get_all_config():
    """
    **获取所有系统配置**

    返回一个包含所有可配置项的JSON对象，包括系统、思考、注入、清理、防检测、流式和故障转移等配置。
    """
    try:
        configs = db.get_all_configs()
        thinking_config = db.get_thinking_config()
        inject_config = db.get_inject_prompt_config()
        cleanup_config = db.get_auto_cleanup_config()
        failover_config = db.get_failover_config()

        # 添加防检测配置
        anti_detection_config = {
            'enabled': db.get_config('anti_detection_enabled', 'true').lower() == 'true'
        }
        
        # 添加流式模式配置
        stream_mode_config = db.get_stream_mode_config()

        return {
            "success": True,
            "system_configs": configs,
            "thinking_config": thinking_config,
            "inject_config": inject_config,
            "cleanup_config": cleanup_config,
            "anti_detection_config": anti_detection_config,
            "stream_mode_config": stream_mode_config,
            "failover_config": failover_config
        }
    except Exception as e:
        logger.error(f"Failed to get configs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/stats", summary="获取管理统计", tags=["管理 API：健康与状态"])
async def get_admin_stats():
    """
    **获取管理统计**

    返回一个全面的服务统计信息汇总，用于管理仪表盘。
    """
    health_summary = db.get_keys_health_summary()

    return {
        "gemini_keys": len(db.get_all_gemini_keys()),
        "active_gemini_keys": len(db.get_available_gemini_keys()),
        "healthy_gemini_keys": health_summary['healthy'],
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
        "failover_config": db.get_failover_config()
    }


# 运行服务器的函数
def run_api_server(port: int = 8000):
    """运行API服务器"""
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(
        f"Starting Gemini API Proxy with optimized multimodal support, auto keep-alive, auto-cleanup, anti-automation detection and fast failover on port {port}")
    run_api_server(port)
