"""翻译模块：调用 LLM API 翻译 SRT 字幕"""

import os
import json
import math
import time
import threading
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

from .config import (
    LANG_NAMES,
    TRANSLATE_BATCH_SIZE,
    TRANSLATE_MAX_RETRIES,
    TRANSLATE_RETRY_DELAY,
    TRANSLATE_API_TIMEOUT,
    TRANSLATE_MAX_WORKERS,
    TranslateConfig,
)
from .utils import _save_srt, _parse_srt_file, format_elapsed


# ── API 配置与连接 ────────────────────────────────────────


def get_translate_config() -> TranslateConfig:
    """从环境变量获取翻译API配置，缺失则回退到本地 Ollama"""
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "")
    model = os.environ.get("LLM_MODEL", "")

    if api_key and base_url and model:
        print(f"使用环境变量配置: {base_url} / {model}")
        return api_key, base_url, model

    print("未完整配置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL，回退到本地 Ollama...")
    return "ollama", "http://localhost:11434/v1", "qwen3:8b"


def check_translate_api(api_key: str, base_url: str, model: str) -> None:
    """检查翻译 API 是否可用，不可用则抛出 RuntimeError"""
    url = f"{base_url.rstrip('/')}/chat/completions"
    req = _build_api_request(url, api_key, {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status == 200:
                print("翻译API连接正常。")
                return
            print(f"错误: 翻译API返回异常状态码 {resp.status}。")
    except urllib.error.HTTPError as e:
        detail = _read_http_error_detail(e)
        print(f"错误: 翻译API请求失败 (HTTP {e.code}): {detail}")
    except urllib.error.URLError as e:
        print(f"错误: 无法连接翻译API: {e.reason}")
    except Exception as e:
        print(f"错误: 翻译API检查异常: {e}")

    print("请检查环境变量 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 是否正确。")
    raise RuntimeError("翻译API配置错误或不可用")


# ── 核心翻译逻辑 ──────────────────────────────────────────


def translate_batch(
        texts: list[str], target_lang: str, api_key: str, base_url: str, model: str
) -> list[str]:
    """调用大模型翻译一批字幕文本"""
    url = f"{base_url.rstrip('/')}/chat/completions"
    lang_name = LANG_NAMES.get(target_lang, target_lang)
    prompt = (
        f"Translate the following subtitle texts to {lang_name}.\n"
        f"Return ONLY a JSON array of translated strings in the same order. "
        f"Do not include any explanation.\n\n"
        f"{json.dumps(texts, ensure_ascii=False)}"
    )

    req = _build_api_request(url, api_key, {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a professional subtitle translator. "
                    "You must return ONLY a valid JSON array of translated strings, nothing else."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    })

    try:
        with urllib.request.urlopen(req, timeout=TRANSLATE_API_TIMEOUT) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = _read_http_error_detail(e)
        raise ValueError(f"翻译API请求失败 (HTTP {e.code}): {detail}") from e

    # 检查 API 是否返回了预期格式
    if "choices" not in result or not result["choices"]:
        api_error = result.get("error", {})
        if isinstance(api_error, dict):
            err_msg = api_error.get("message", json.dumps(result, ensure_ascii=False)[:500])
        else:
            err_msg = str(api_error) or json.dumps(result, ensure_ascii=False)[:500]
        raise ValueError(f"翻译API返回异常: {err_msg}")

    content = result["choices"][0]["message"]["content"]
    parsed = json.loads(_strip_markdown_code_block(content))
    if not isinstance(parsed, list):
        raise ValueError(f"翻译API返回了非数组类型: {type(parsed).__name__}")
    return parsed


@dataclass
class _BatchContext:
    """线程池批量翻译的共享上下文"""
    target_lang: str
    api_key: str
    base_url: str
    model: str
    total_batches: int
    progress_lock: threading.Lock
    progress_counter: list[int]  # 用 list 包裹以便线程内修改


def translate_srt_entries(
        srt_entries: list[str],
        target_lang: str,
        api_key: str,
        base_url: str,
        model: str,
) -> list[str]:
    """翻译 SRT 条目列表，返回翻译后的条目列表"""
    headers: list[str] = []
    texts: list[str] = []
    for entry in srt_entries:
        parts = entry.split("\n", 2)
        headers.append("\n".join(parts[:2]))
        texts.append(parts[2] if len(parts) > 2 else "")

    total_batches = math.ceil(len(texts) / TRANSLATE_BATCH_SIZE)
    workers = min(TRANSLATE_MAX_WORKERS, total_batches)

    print(f"共 {total_batches} 批，使用 {workers} 个线程并发翻译...")

    ctx = _BatchContext(
        target_lang=target_lang,
        api_key=api_key,
        base_url=base_url,
        model=model,
        total_batches=total_batches,
        progress_lock=threading.Lock(),
        progress_counter=[0],
    )

    # 按批次索引收集结果，保证顺序
    batch_results: dict[int, list[str]] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for batch_idx in range(0, len(texts), TRANSLATE_BATCH_SIZE):
            batch = texts[batch_idx: batch_idx + TRANSLATE_BATCH_SIZE]
            batch_num = batch_idx // TRANSLATE_BATCH_SIZE + 1

            future = executor.submit(_translate_single_batch, batch_num, batch, ctx)
            futures[future] = batch_num

        for future in as_completed(futures):
            batch_num = futures[future]
            try:
                batch_results[batch_num] = future.result()
            except Exception as e:
                # 兜底：线程内未捕获的异常，保留原文
                start = (batch_num - 1) * TRANSLATE_BATCH_SIZE
                end = start + TRANSLATE_BATCH_SIZE
                batch_results[batch_num] = texts[start:end]
                print(f"错误: 第 {batch_num} 批翻译线程异常: {e}，保留原文")

    # 按顺序合并结果
    translated_texts: list[str] = []
    for i in range(1, total_batches + 1):
        translated_texts.extend(batch_results[i])

    return [
        f"{header}\n{text}" for header, text in zip(headers, translated_texts)
    ]


# ── 公开的组合函数 ────────────────────────────────────────


def _translate_and_save(
        srt_entries: list[str],
        target_lang: str,
        translate_config: TranslateConfig,
        output_path: str,
) -> None:
    """翻译字幕条目并保存到文件"""
    lang_label = LANG_NAMES.get(target_lang, target_lang)
    print(f"正在将字幕翻译为 {lang_label}...")

    api_key, base_url, model_name = translate_config
    translated = translate_srt_entries(
        srt_entries, target_lang, api_key, base_url, model_name
    )

    _save_srt(translated, output_path)
    print(f"翻译完成！已保存至: {output_path}")


def translate_srt_file(
        srt: str,
        to: str,
        translate_config: TranslateConfig,
        output: Optional[str] = None,
) -> Optional[str]:
    """翻译 SRT 文件并保存。

    Args:
        srt: 输入 SRT 文件路径
        to: 翻译目标语言代码
        translate_config: 翻译 API 配置 (api_key, base_url, model_name)
        output: 输出文件路径 (None 表示自动生成)

    Returns:
        已翻译的 SRT 文件路径，失败返回 None
    """
    task_start = time.time()
    srt_path = os.path.abspath(srt)

    if not srt_path.lower().endswith(".srt"):
        print(f"警告: {srt_path} 不是 .srt 文件，继续尝试...")

    print("--- 字幕翻译任务开始 ---")
    print(f"正在读取字幕文件: {os.path.basename(srt_path)}...")

    srt_entries = _parse_srt_file(srt_path)
    if srt_entries is None:
        return None

    print(f"读取到 {len(srt_entries)} 条字幕。")

    if output is not None:
        translated_path = os.path.abspath(os.path.expanduser(output))
    else:
        base, ext = os.path.splitext(srt_path)
        translated_path = f"{base}.{to}{ext}"

    _translate_and_save(srt_entries, to, translate_config, translated_path)

    elapsed = time.time() - task_start
    print(f"\n--- 翻译任务完成 (耗时 {format_elapsed(elapsed)}) ---")
    return translated_path


# ── 内部辅助函数 ──────────────────────────────────────────


def _read_http_error_detail(e: urllib.error.HTTPError) -> str:
    """从 HTTPError 中读取响应体，提取有用的错误信息"""
    try:
        body = e.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(body)
            # OpenAI 兼容格式: {"error": {"message": "..."}}
            err_obj = data.get("error", {})
            if isinstance(err_obj, dict) and "message" in err_obj:
                return err_obj["message"]
            # 其他格式: {"message": "..."}
            if "message" in data:
                return data["message"]
            return body[:500]
        except (json.JSONDecodeError, ValueError):
            return body[:500] if body.strip() else e.reason
    except Exception:
        return e.reason


def _build_api_request(url: str, api_key: str, payload: dict) -> urllib.request.Request:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = json.dumps(payload).encode("utf-8")
    return urllib.request.Request(url, data=data, headers=headers, method="POST")


def _strip_markdown_code_block(content: str) -> str:
    content = content.strip()
    if not content.startswith("```"):
        return content
    lines = content.split("\n")
    body = "\n".join(lines[1:])
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


def _pad_or_truncate(translated: list[str], batch: list[str]) -> list[str]:
    if len(translated) < len(batch):
        translated.extend(batch[len(translated):])
    return translated[:len(batch)]


def _translate_single_batch(
        batch_num: int,
        batch: list[str],
        ctx: _BatchContext,
) -> list[str]:
    """翻译单个批次（含重试），供线程池调用"""
    translated = batch  # 默认保留原文
    success = False
    got_result = False  # 是否至少获取过一次 API 返回

    for attempt in range(TRANSLATE_MAX_RETRIES):
        try:
            result = translate_batch(batch, ctx.target_lang, ctx.api_key, ctx.base_url, ctx.model)
            translated = result
            got_result = True

            if len(translated) == len(batch):
                success = True
                break

            print(
                f"警告: 第 {batch_num} 批翻译返回数量不匹配 "
                f"(期望 {len(batch)}, 得到 {len(translated)}) "
                f"(尝试 {attempt + 1}/{TRANSLATE_MAX_RETRIES})"
            )
        except Exception as e:
            print(
                f"翻译第 {batch_num} 批请求出错 "
                f"(尝试 {attempt + 1}/{TRANSLATE_MAX_RETRIES}): {e}"
            )

        if attempt < TRANSLATE_MAX_RETRIES - 1:
            time.sleep(TRANSLATE_RETRY_DELAY)

    if not success:
        if got_result and len(translated) != len(batch):
            print(
                f"警告: 第 {batch_num} 批最终数量不匹配 "
                f"(期望 {len(batch)}, 得到 {len(translated)}), 差异部分保留原文"
            )
            translated = _pad_or_truncate(translated, batch)
        elif not got_result:
            print(f"警告: 第 {batch_num} 批翻译全部失败，保留原文")
            translated = batch
        else:
            # got_result 为 True 且数量匹配，但未 break（不应出现，防御性处理）
            print(f"警告: 第 {batch_num} 批翻译重试耗尽，使用最近一次翻译结果")

    with ctx.progress_lock:
        ctx.progress_counter[0] += 1
        print(f"翻译进度: {ctx.progress_counter[0]}/{ctx.total_batches} 批完成")

    return translated
