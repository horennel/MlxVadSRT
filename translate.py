"""翻译模块：调用 LLM API 翻译 SRT 字幕"""

import os
import sys
import json
import math
import time
import urllib.request
import urllib.error
import argparse
from typing import Optional

from config import (
    LANG_NAMES,
    TRANSLATE_BATCH_SIZE,
    TRANSLATE_MAX_RETRIES,
    TRANSLATE_RETRY_DELAY,
    TRANSLATE_API_TIMEOUT,
    TranslateConfig,
)
from utils import _save_srt, _parse_srt_file


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


def _build_api_request(url: str, api_key: str, payload: dict) -> urllib.request.Request:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = json.dumps(payload).encode("utf-8")
    return urllib.request.Request(url, data=data, headers=headers, method="POST")


def check_translate_api(api_key: str, base_url: str, model: str) -> None:
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
        print(f"错误: 翻译API请求失败 (HTTP {e.code}): {e.reason}")
    except urllib.error.URLError as e:
        print(f"错误: 无法连接翻译API: {e.reason}")
    except Exception as e:
        print(f"错误: 翻译API检查异常: {e}")

    print("请检查环境变量 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 是否正确。")
    sys.exit(1)


def _strip_markdown_code_block(content: str) -> str:
    content = content.strip()
    if not content.startswith("```"):
        return content
    lines = content.split("\n")
    body = "\n".join(lines[1:])
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


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

    with urllib.request.urlopen(req, timeout=TRANSLATE_API_TIMEOUT) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    content = result["choices"][0]["message"]["content"]
    parsed = json.loads(_strip_markdown_code_block(content))
    if not isinstance(parsed, list):
        raise ValueError(f"翻译API返回了非数组类型: {type(parsed).__name__}")
    return parsed


def _pad_or_truncate(translated: list[str], batch: list[str]) -> list[str]:
    if len(translated) < len(batch):
        translated.extend(batch[len(translated):])
    return translated[:len(batch)]


def translate_srt_entries(
        srt_entries: list[str],
        target_lang: str,
        api_key: str,
        base_url: str,
        model: str,
) -> list[str]:
    headers: list[str] = []
    texts: list[str] = []
    for entry in srt_entries:
        parts = entry.split("\n", 2)
        headers.append("\n".join(parts[:2]))
        texts.append(parts[2] if len(parts) > 2 else "")

    translated_texts: list[str] = []
    total_batches = math.ceil(len(texts) / TRANSLATE_BATCH_SIZE)

    for batch_idx in range(0, len(texts), TRANSLATE_BATCH_SIZE):
        batch = texts[batch_idx: batch_idx + TRANSLATE_BATCH_SIZE]
        batch_num = batch_idx // TRANSLATE_BATCH_SIZE + 1
        print(f"翻译进度: {batch_num}/{total_batches} 批...")

        translated = batch
        success = False

        for attempt in range(TRANSLATE_MAX_RETRIES):
            try:
                result = translate_batch(batch, target_lang, api_key, base_url, model)
                translated = result

                if len(translated) == len(batch):
                    success = True
                    break

                print(
                    f"警告: 翻译返回数量不匹配 "
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
            if len(translated) != len(batch):
                print(
                    f"警告: 第 {batch_num} 批最终数量不匹配 "
                    f"(期望 {len(batch)}, 得到 {len(translated)}), 差异部分保留原文"
                )
                translated = _pad_or_truncate(translated, batch)
            else:
                print(f"警告: 第 {batch_num} 批翻译全部失败，保留原文")

        translated_texts.extend(translated)

    return [
        f"{header}\n{text}" for header, text in zip(headers, translated_texts)
    ]


def _translate_and_save(
        srt_entries: list[str],
        target_lang: str,
        translate_config: TranslateConfig,
        output_path: str,
) -> None:
    lang_label = LANG_NAMES.get(target_lang, target_lang)
    print(f"正在将字幕翻译为 {lang_label}...")

    api_key, base_url, model_name = translate_config
    translated = translate_srt_entries(
        srt_entries, target_lang, api_key, base_url, model_name
    )

    _save_srt(translated, output_path)
    print(f"翻译完成！已保存至: {output_path}")


def translate_srt_file(args: argparse.Namespace) -> None:
    task_start = time.time()
    srt_path = os.path.abspath(args.srt)

    if not srt_path.lower().endswith(".srt"):
        print(f"警告: {srt_path} 不是 .srt 文件，继续尝试...")

    print("--- 字幕翻译任务开始 ---")
    print(f"正在读取字幕文件: {os.path.basename(srt_path)}...")

    srt_entries = _parse_srt_file(srt_path)
    if srt_entries is None:
        return

    print(f"读取到 {len(srt_entries)} 条字幕。")

    if args.output is not None:
        translated_path = os.path.abspath(args.output)
    else:
        base, ext = os.path.splitext(srt_path)
        translated_path = f"{base}.{args.to}{ext}"

    _translate_and_save(srt_entries, args.to, args.translate_config, translated_path)

    elapsed = time.time() - task_start
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"\n--- 翻译任务完成 (耗时 {minutes}分{seconds}秒) ---")
