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
import io
import itertools
import copy
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncGenerator, Union, Any

from google import genai
from google.genai import types
from fastapi import HTTPException

from database import Database
from api_models import ChatCompletionRequest, ChatMessage

# 配置日志
logger = logging.getLogger(__name__)

# GenAI Client 缓存
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
            '∙', '∘', '∞', '≈', '≠', '≤', '≥', '±', '∓', '×', '÷', '∂', '∆', '∇',
            '○', '●', '◯', '◦', '◉', '◎', '⦿', '⊙', '⊚', '⊛', '⊜', '⊝',
            '□', '■', '▢', '▣', '▤', '▥', '▦', '▧', '▨', '▩', '▪', '▫',
            '△', '▲', '▴', '▵', '▶', '▷', '▸', '▹', '►', '▻', '▼', '▽',
            '◀', '◁', '◂', '◃', '◄', '◅', '◆', '◇', '◈', '◉', '◊',
            '☆', '★', '⭐', '✦', '✧', '✩', '✪', '✫', '✬', '✭', '✮', '✯',
            '✰', '✱', '✲', '✳', '✴', '✵', '✶', '✷', '✸', '✹', '✺', '✻',
            '→', '←', '↑', '↓', '↔', '↕', '↖', '↗', '↘', '↙', '↚', '↛',
            '⇒', '⇐', '⇑', '⇓', '⇔', '⇕', '⇖', '⇗', '⇘', '⇙', '⇚', '⇛',
            '‖', '‗', '‰', '‱', '′', '″', '‴', '‵', '‶', '‷', '‸', '‹', '›',
            '‼', '‽', '‾', '‿', '⁀', '⁁', '⁂', '⁃', '⁆', '⁇', '⁈', '⁉',
            '※', '⁎', '⁑', '⁒', '⁓', '⁔', '⁕', '⁖', '⁗', '⁘', '⁙', '⁚',
            '⊕', '⊖', '⊗', '⊘', '⊙', '⊚', '⊛', '⊜', '⊝', '⊞', '⊟', '⊠',
            '⋄', '⋅', '⋆', '⋇', '⋈', '⋉', '⋊', '⋋', '⋌', '⋍', '⋎', '⋏'
        ]
        self.invisible_symbols = ['\u200B', '\u200C', '\u2060']
        self.request_history = set()
        self.max_history_size = 5000

    def inject_symbols(self, text: str, strategy: str = 'auto') -> str:
        if not text.strip(): return text
        symbol_count = random.randint(2, 4)
        if strategy == 'invisible':
            symbols = random.sample(self.invisible_symbols, min(2, len(self.invisible_symbols)))
        elif strategy == 'mixed':
            visible_count = random.randint(1, 2)
            invisible_count = 1
            symbols = (random.sample(self.safe_symbols, visible_count) +
                       random.sample(self.invisible_symbols, invisible_count))
        else:
            symbols = random.sample(self.safe_symbols, min(symbol_count, len(self.safe_symbols)))
        strategies = ['prefix', 'suffix', 'wrap']
        if strategy == 'auto': strategy = random.choice(strategies)
        if strategy == 'prefix': return ''.join(symbols) + ' ' + text
        elif strategy == 'suffix': return text + ' ' + ''.join(symbols)
        elif strategy == 'wrap':
            mid = len(symbols) // 2
            prefix = ''.join(symbols[:mid])
            suffix = ''.join(symbols[mid:])
            return f"{prefix} {text} {suffix}" if prefix and suffix else f"{text} {suffix}"
        else: return text + ' ' + ''.join(symbols)

    def process_content(self, content: Union[str, List]) -> Union[str, List]:
        content_hash = hashlib.md5(str(content).encode()).hexdigest()
        strategy = random.choice(['mixed', 'invisible', 'prefix', 'suffix']) if content_hash in self.request_history else 'auto'
        self.request_history.add(content_hash)
        if len(self.request_history) > self.max_history_size:
            self.request_history = set(list(self.request_history)[self.max_history_size // 2:])
        if isinstance(content, str): return self.inject_symbols(content, strategy)
        elif isinstance(content, list):
            processed = []
            for item in content:
                if isinstance(item, dict):
                    processed_item = item.copy()
                    if 'text' in processed_item:
                        processed_item['text'] = self.inject_symbols(processed_item['text'], strategy)
                    processed.append(processed_item)
                else: processed.append(item)
            return processed
        return content

    def get_statistics(self) -> Dict:
        return {
            'available_symbols': len(self.safe_symbols),
            'invisible_symbols': len(self.invisible_symbols),
            'request_history_size': len(self.request_history),
            'max_history_size': self.max_history_size
        }

def decrypt_response(hex_string: str) -> str:
    if not isinstance(hex_string, str) or not hex_string or len(hex_string) % 8 != 0: return hex_string
    try:
        if not all(c in '0123456789abcdef' for c in hex_string.lower()): return hex_string
        txt = ''
        for i in range(0, len(hex_string), 8):
            codepoint = 0
            hex_block = hex_string[i:i+8]
            for j in range(0, 8, 2):
                byte_hex = hex_block[j:j+2]
                byte_val = int(byte_hex, 16)
                decrypted_byte = byte_val ^ 0x5A
                codepoint = (codepoint << 8) | decrypted_byte
            txt += chr(codepoint)
        return txt
    except (ValueError, TypeError): return hex_string

class RateLimitCache:
    def __init__(self, max_entries: int = 10000):
        self.cache: Dict[str, Dict[str, List[tuple]]] = {}
        self.max_entries = max_entries
        self.lock = asyncio.Lock()

    async def cleanup_expired(self, window_seconds: int = 60):
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        async with self.lock:
            for model_name in list(self.cache.keys()):
                if model_name in self.cache:
                    self.cache[model_name]['requests'] = [(t, v) for t, v in self.cache[model_name]['requests'] if t > cutoff_time]
                    self.cache[model_name]['tokens'] = [(t, v) for t, v in self.cache[model_name]['tokens'] if t > cutoff_time]

    async def add_usage(self, model_name: str, requests: int = 1, tokens: int = 0):
        async with self.lock:
            if model_name not in self.cache: self.cache[model_name] = {'requests': [], 'tokens': []}
            current_time = time.time()
            self.cache[model_name]['requests'].append((current_time, requests))
            self.cache[model_name]['tokens'].append((current_time, tokens))

    async def get_current_usage(self, model_name: str, window_seconds: int = 60) -> Dict[str, int]:
        async with self.lock:
            if model_name not in self.cache: return {'requests': 0, 'tokens': 0}
            current_time = time.time()
            cutoff_time = current_time - window_seconds
            self.cache[model_name]['requests'] = [(t, v) for t, v in self.cache[model_name]['requests'] if t > cutoff_time]
            self.cache[model_name]['tokens'] = [(t, v) for t, v in self.cache[model_name]['tokens'] if t > cutoff_time]
            total_requests = sum(v for _, v in self.cache[model_name]['requests'])
            total_tokens = sum(v for _, v in self.cache[model_name]['tokens'])
            return {'requests': total_requests, 'tokens': total_tokens}

async def check_gemini_key_health(api_key: str, timeout: int = 10) -> Dict[str, Any]:
    start_time = time.time()
    try:
        client = get_cached_client(api_key)
        await asyncio.wait_for(
            client.aio.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents="Hello",
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                )
            ),
            timeout=timeout
        )
        return {"healthy": True, "response_time": time.time() - start_time, "status_code": 200, "error": None}
    except asyncio.TimeoutError:
        return {"healthy": False, "response_time": timeout, "status_code": None, "error": "Timeout"}
    except Exception as e:
        return {"healthy": False, "response_time": time.time() - start_time, "status_code": None, "error": str(e)}

async def keep_alive_ping():
    try:
        render_url = os.getenv('RENDER_EXTERNAL_URL')
        target_url = f"{render_url}/wake" if render_url else "http://localhost:8000/wake"
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(target_url, timeout=30) as response:
                    logger.info(f"Keep-alive ping {'successful' if response.status == 200 else 'warning'}: {response.status}")
        except ImportError:
            import urllib.request
            with urllib.request.urlopen(target_url, timeout=30) as response:
                logger.info(f"Keep-alive ping {'successful' if response.status == 200 else 'warning'}: {response.status}")
    except Exception as e:
        logger.warning(f"Keep-alive ping failed: {e}")

def init_anti_detection_config(db: Database):
    try:
        db.set_config('anti_detection_enabled', 'true')
        logger.info("Anti-detection system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize anti-detection system: {e}")

async def upload_file_to_gemini(file_content: bytes, mime_type: str, filename: str, gemini_key: str) -> Optional[str]:
    try:
        client = get_cached_client(gemini_key)
        file_stream = io.BytesIO(file_content)
        upload_result = await client.aio.files.upload(
            file=file_stream,
            config={"mimeType": mime_type, "displayName": filename, "name": f"files/{uuid.uuid4().hex}_{filename}"}
        )
        file_uri = getattr(upload_result, "uri", None)
        if file_uri:
            logger.info(f"File uploaded to Gemini successfully: {file_uri}")
            return file_uri
        else:
            logger.error("No URI returned from google-genai upload result")
            return None
    except Exception as e:
        logger.error(f"Error uploading file to Gemini: {str(e)}")
        return None

async def delete_file_from_gemini(file_uri: str, gemini_key: str) -> bool:
    try:
        client = get_cached_client(gemini_key)
        await client.aio.files.delete(name=file_uri)
        logger.info(f"File deleted from Gemini successfully: {file_uri}")
        return True
    except Exception as e:
        logger.error(f"Error deleting file from Gemini: {str(e)}")
        return False

def get_actual_model_name(db: Database, request_model: str) -> str:
    supported_models = db.get_supported_models()
    if request_model in supported_models:
        logger.info(f"Using requested model: {request_model}")
        return request_model
    default_model = db.get_config('default_model_name', 'gemini-2.5-flash-lite')
    logger.info(f"Unsupported model: {request_model}, using default: {default_model}")
    return default_model

def inject_prompt_to_messages(db: Database, messages: List[ChatMessage]) -> List[ChatMessage]:
    inject_config = db.get_inject_prompt_config()
    if not inject_config['enabled'] or not inject_config['content']: return messages
    content = inject_config['content']
    position = inject_config['position']
    new_messages = messages.copy()
    if position == 'system':
        system_msg = next((msg for msg in new_messages if msg.role == 'system'), None)
        if system_msg:
            system_msg.content = f"{content}\n\n{system_msg.get_text_content()}"
        else:
            new_messages.insert(0, ChatMessage(role='system', content=content))
    elif position == 'user_prefix':
        user_msg = next((msg for msg in new_messages if msg.role == 'user'), None)
        if user_msg: user_msg.content = f"{content}\n\n{user_msg.get_text_content()}"
    elif position == 'user_suffix':
        user_msg = next((msg for msg in reversed(new_messages) if msg.role == 'user'), None)
        if user_msg: user_msg.content = f"{user_msg.get_text_content()}\n\n{content}"
    anti_truncation_cfg = db.get_anti_truncation_config()
    if anti_truncation_cfg.get('enabled'):
        user_msg = next((msg for msg in reversed(new_messages) if msg.role == 'user'), None)
        if user_msg:
            suffix = "请以 [finish] 结尾"
            if isinstance(user_msg.content, str): user_msg.content = f"{user_msg.content}\n\n{suffix}"
            elif isinstance(user_msg.content, list): user_msg.content.append(suffix)
    return new_messages

def get_thinking_config(db: Database, request: ChatCompletionRequest) -> Dict:
    thinking_config = {}
    global_thinking_enabled = db.get_config('thinking_enabled', 'true').lower() == 'true'
    if not global_thinking_enabled: return {"thinkingBudget": 0}
    global_thinking_budget = int(db.get_config('thinking_budget', '-1'))
    global_include_thoughts = db.get_config('include_thoughts', 'false').lower() == 'true'
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
        if global_thinking_budget >= 0: thinking_config["thinkingBudget"] = global_thinking_budget
        if global_include_thoughts: thinking_config["includeThoughts"] = global_include_thoughts
    return thinking_config

def process_multimodal_content(item: Dict, file_storage: Dict) -> Optional[Dict]:
    try:
        file_data = item.get('file_data') or item.get('fileData')
        inline_data = item.get('inline_data') or item.get('inlineData')
        if inline_data:
            mime_type = inline_data.get('mimeType') or inline_data.get('mime_type')
            data = inline_data.get('data')
            if mime_type and data: return {"inlineData": {"mimeType": mime_type, "data": data}}
        elif file_data:
            mime_type = file_data.get('mimeType') or file_data.get('mime_type')
            file_uri = file_data.get('fileUri') or file_data.get('file_uri')
            if mime_type and file_uri: return {"fileData": {"mimeType": mime_type, "fileUri": file_uri}}
        elif item.get('type') == 'file' and 'file_id' in item:
            file_id = item['file_id']
            if file_id in file_storage:
                file_info = file_storage[file_id]
                if file_info.get('format') == 'inlineData':
                    return {"inlineData": {"mimeType": file_info['mime_type'], "data": file_info['data']}}
                elif file_info.get('format') == 'fileData':
                    if 'gemini_file_uri' in file_info:
                        return {"fileData": {"mimeType": file_info['mime_type'], "fileUri": file_info['gemini_file_uri']}}
                    elif 'file_uri' in file_info:
                        logger.warning(f"Using local file URI for file {file_id}, this may not work with Gemini")
                        return {"fileData": {"mimeType": file_info['mime_type'], "fileUri": file_info['file_uri']}}
            else: logger.warning(f"File ID {file_id} not found in storage")
        if item.get('type') == 'image_url' and 'image_url' in item:
            image_url = item['image_url'].get('url', '')
            if image_url.startswith('data:'):
                try:
                    header, data = image_url.split(',', 1)
                    mime_type = header.split(';')[0].split(':')[1]
                    return {"inlineData": {"mimeType": mime_type, "data": data}}
                except Exception as e: logger.warning(f"Failed to parse data URL: {e}")
            else: logger.warning("HTTP URLs not supported for images, use file upload instead")
        logger.warning(f"Unsupported multimodal content format: {item}")
        return None
    except Exception as e:
        logger.error(f"Error processing multimodal content: {e}")
        return None

def estimate_token_count(text: str) -> int:
    return len(text) // 4

def should_apply_anti_detection(db: Database, request: ChatCompletionRequest, anti_detection_injector: GeminiAntiDetectionInjector, enable_anti_detection: bool = True) -> bool:
    if not enable_anti_detection or not db.get_config('anti_detection_enabled', 'true').lower() == 'true': return False
    disable_for_tools = db.get_config('anti_detection_disable_for_tools', 'true').lower() == 'true'
    if disable_for_tools and (request.tools or request.tool_choice):
        logger.info("Anti-detection disabled for tool calls")
        return False
    token_threshold = int(db.get_config('anti_detection_token_threshold', '5000'))
    total_tokens = sum(estimate_token_count(item.get('text', '')) for msg in request.messages if isinstance(msg.content, list) for item in msg.content if isinstance(item, dict) and item.get('type') == 'text')
    if total_tokens < token_threshold:
        logger.info(f"Anti-detection skipped: token count {total_tokens} below threshold {token_threshold}")
        return False
    logger.info(f"Anti-detection enabled: token count {total_tokens} exceeds threshold {token_threshold}")
    return True

def openai_to_gemini(db: Database, request: ChatCompletionRequest, anti_detection_injector: GeminiAntiDetectionInjector, file_storage: Dict, enable_anti_detection: bool = True) -> Dict:
    contents = []
    anti_detection_enabled = should_apply_anti_detection(db, request, anti_detection_injector, enable_anti_detection)
    for msg in request.messages:
        parts = []
        if isinstance(msg.content, str):
            text_content = anti_detection_injector.inject_symbols(msg.content) if anti_detection_enabled and msg.role == 'user' else msg.content
            parts.append({"text": f"[System]: {text_content}" if msg.role == "system" else text_content})
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, str):
                    text_content = anti_detection_injector.inject_symbols(item) if anti_detection_enabled and msg.role == 'user' else item
                    parts.append({"text": text_content})
                elif isinstance(item, dict):
                    if item.get('type') == 'text':
                        text_content = anti_detection_injector.inject_symbols(item.get('text', '')) if anti_detection_enabled and msg.role == 'user' else item.get('text', '')
                        parts.append({"text": text_content})
                    elif item.get('type') in ['image', 'image_url', 'audio', 'video', 'document']:
                        multimodal_part = process_multimodal_content(item, file_storage)
                        if multimodal_part: parts.append(multimodal_part)
        if parts: contents.append({"role": "user" if msg.role in ["system", "user"] else "model", "parts": parts})
    thinking_config = get_thinking_config(db, request)
    thinking_cfg_obj = types.ThinkingConfig(**thinking_config) if thinking_config else None
    afc_obj = types.AutomaticFunctionCallingConfig(disable=True)
    generation_config = types.GenerateContentConfig(
        temperature=request.temperature, top_p=request.top_p, candidate_count=request.n,
        thinking_config=thinking_cfg_obj, max_output_tokens=request.max_tokens,
        stop_sequences=request.stop, automatic_function_calling=afc_obj
    )
    return {"contents": contents, "generation_config": generation_config}

def extract_thoughts_and_content(gemini_response: Dict, include_thoughts: bool = True) -> tuple[str, str]:
    thoughts, content = "", ""
    for candidate in gemini_response.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part and part["text"]:
                if part.get("thought", False):
                    thoughts += part["text"]
                    if not include_thoughts: content += part["text"]
                else: content += part["text"]
    return thoughts, content

def gemini_to_openai(gemini_response: Dict, request: ChatCompletionRequest, usage_info: Dict = None) -> Dict:
    choices = []
    include_thoughts = request.thinking_config and request.thinking_config.include_thoughts
    thoughts, content = extract_thoughts_and_content(gemini_response, include_thoughts)
    for i, candidate in enumerate(gemini_response.get("candidates", [])):
        message_content = f"**Thinking:**\n{thoughts}\n\n**Response:**\n{content}" if thoughts and include_thoughts else content
        choices.append({
            "index": i, "message": {"role": "assistant", "content": message_content},
            "finish_reason": map_finish_reason(candidate.get("finishReason", "STOP"))
        })
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion", "created": int(time.time()),
        "model": request.model, "choices": choices,
        "usage": usage_info or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

def map_finish_reason(gemini_reason: str) -> str:
    return {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter", "RECITATION": "content_filter", "OTHER": "stop"}.get(gemini_reason, "stop")

def validate_file_for_gemini(file_content: bytes, mime_type: str, filename: str, supported_mime_types: set, max_file_size: int, max_inline_size: int) -> Dict[str, Any]:
    file_size = len(file_content)
    if mime_type not in supported_mime_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {mime_type}")
    if file_size > max_file_size:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {max_file_size // (1024 * 1024)}MB")
    return {"size": file_size, "mime_type": mime_type, "use_inline": file_size <= max_inline_size, "filename": filename}
