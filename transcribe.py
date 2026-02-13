import os

# 如果本地没有模型，首次运行时会自动从 Hugging Face 下载
# 可以使用国内镜像站加速下载 (用户可通过环境变量 HF_ENDPOINT 覆盖)
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 如果已有本地缓存，可取消下面这行的注释来强制离线使用
# os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import numpy as np
import mlx_whisper
import sys
import subprocess
import argparse
import shutil
import json
import urllib.request
import gc
import time
from typing import Optional

# ── 文件类型扩展名 ─────────────────────────────────────────
AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".aiff", ".aif"
}
VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v", ".mpeg", ".mpg"
}

# ── 语言映射 ─────────────────────────────────────────────
LANG_NAMES: dict[str, str] = {
    "zh": "简体中文",
    "en": "English",
    "ja": "日本語",
    "ko": "한국어",
    "auto": "自动检测",
}

# ── VAD 参数常量 ──────────────────────────────────────────
VAD_THRESHOLD = 0.25  # 语音概率阈值，越低越敏感（0.2 可捕获轻声/背景语音）
VAD_MIN_SILENCE_MS = 600  # 最短静音时长(ms)，低于此值的停顿不切分片段
VAD_MIN_SPEECH_MS = 50  # 最短语音时长(ms)，50ms 可保留单字/语气词
VAD_SPEECH_PAD_MS = 300  # 语音片段前后填充(ms)，300ms 避免句首尾被截断

# ── 翻译参数常量 ──────────────────────────────────────────
TRANSLATE_BATCH_SIZE = 50  # 每批翻译的字幕条数
TRANSLATE_MAX_RETRIES = 3  # 翻译请求最大重试次数
TRANSLATE_RETRY_DELAY = 2  # 重试间隔(秒)
TRANSLATE_API_TIMEOUT = 200  # 翻译请求超时(秒)

# ── 其他常量 ──────────────────────────────────────────────
SAMPLE_RATE = 16000  # VAD 和 Whisper 的标准采样率
PROGRESS_INTERVAL = 5  # 每处理多少个片段打印一次进度


# ── 工具函数 ──────────────────────────────────────────────

def check_dependencies() -> None:
    """检查必要的系统依赖"""
    if not shutil.which("ffmpeg"):
        print("错误: 在系统 PATH 中找不到 'ffmpeg'。")
        print("请先安装 ffmpeg (例如: brew install ffmpeg)")
        sys.exit(1)


def _has_extension(file_path: str, extensions: set[str]) -> bool:
    """检测文件扩展名是否在给定的集合中"""
    return os.path.splitext(file_path)[1].lower() in extensions


def is_audio_file(file_path: str) -> bool:
    """检测文件是否为音频文件"""
    return _has_extension(file_path, AUDIO_EXTENSIONS)


def is_video_file(file_path: str) -> bool:
    """检测文件是否为视频文件"""
    return _has_extension(file_path, VIDEO_EXTENSIONS)


def load_audio_with_ffmpeg(file_path: str, sr: int = SAMPLE_RATE) -> Optional[torch.Tensor]:
    """使用系统 ffmpeg 命令读取音频，返回单声道 float32 Tensor

    使用 Popen 而非 subprocess.run，避免长音频在内存中被完整缓存两次。
    """
    cmd = [
        "ffmpeg",
        "-i", os.path.abspath(file_path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(sr),
        "-",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw_bytes, _ = proc.communicate()
        if proc.returncode != 0:
            return None
        return torch.from_numpy(np.frombuffer(raw_bytes, dtype=np.float32).copy())
    except Exception as e:
        print(f"调用 ffmpeg 失败: {e}")
        return None


def format_timestamp(seconds: float) -> str:
    """将秒数转换为 SRT 时间戳格式 (HH:MM:SS,mmm)"""
    total_ms = round(seconds * 1000)
    total_seconds, millis = divmod(total_ms, 1000)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _save_srt(srt_entries: list[str], output_path: str) -> None:
    """将 SRT 条目列表写入文件"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(srt_entries) + "\n")


def _parse_srt_file(srt_path: str) -> Optional[list[str]]:
    """读取并解析 SRT 文件，返回条目列表；失败时返回 None"""
    if not os.path.exists(srt_path):
        print(f"错误: 找不到字幕文件 {srt_path}")
        return None

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        print("错误: 字幕文件为空。")
        return None

    entries = [entry.strip() for entry in content.split("\n\n") if entry.strip()]
    if not entries:
        print("错误: 未解析到任何字幕条目。")
        return None

    return entries


def _translate_and_save(
        srt_entries: list[str],
        target_lang: str,
        translate_config: tuple[str, str, str],
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


# ── 翻译相关 ─────────────────────────────────────────────

def get_translate_config() -> tuple[str, str, str]:
    """从环境变量获取翻译API配置，缺失则回退到本地 Ollama"""
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "")
    model = os.environ.get("LLM_MODEL", "")

    if api_key and base_url and model:
        print(f"使用环境变量配置: {base_url} / {model}")
        return api_key, base_url, model

    print("未完整配置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL，回退到本地 Ollama...")
    return "ollama", "http://localhost:11434/v1", "qwen3:8b"


def _build_api_request(
        url: str, api_key: str, payload: dict
) -> urllib.request.Request:
    """构建翻译 API 的 HTTP 请求"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = json.dumps(payload).encode("utf-8")
    return urllib.request.Request(url, data=data, headers=headers, method="POST")


def check_translate_api(api_key: str, base_url: str, model: str) -> None:
    """检查翻译API是否可用"""
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
    """清理 LLM 响应中可能包裹的 markdown 代码块标记"""
    content = content.strip()
    if not content.startswith("```"):
        return content
    # 去掉首行 (```json / ```) 和末尾 ```
    lines = content.split("\n")
    body = "\n".join(lines[1:])
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


def translate_batch(
        texts: list[str], target_lang: str, api_key: str, base_url: str, model: str
) -> list[str]:
    """调用大模型翻译一批字幕文本（单次请求，不含重试逻辑）"""
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
    """确保翻译结果数量与原文一致：不足时用原文填充，多余时截断"""
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
    """翻译所有SRT字幕条目"""
    # 解析每条字幕：前两行(序号+时间轴)为 header，其余为 text
    headers: list[str] = []
    texts: list[str] = []
    for entry in srt_entries:
        # 用 maxsplit=2 精确拆分：序号行、时间轴行、正文
        parts = entry.split("\n", 2)
        headers.append("\n".join(parts[:2]))
        texts.append(parts[2] if len(parts) > 2 else "")

    translated_texts: list[str] = []
    total_batches = -(-len(texts) // TRANSLATE_BATCH_SIZE)  # 向上取整

    for batch_idx in range(0, len(texts), TRANSLATE_BATCH_SIZE):
        batch = texts[batch_idx: batch_idx + TRANSLATE_BATCH_SIZE]
        batch_num = batch_idx // TRANSLATE_BATCH_SIZE + 1
        print(f"翻译进度: {batch_num}/{total_batches} 批...")

        translated = batch  # 默认使用原文作为备选
        success = False

        for attempt in range(TRANSLATE_MAX_RETRIES):
            try:
                result = translate_batch(batch, target_lang, api_key, base_url, model)
                translated = result

                if len(translated) == len(batch):
                    success = True
                    break  # 数量匹配，成功

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
                # 翻译返回了结果但数量不对
                print(
                    f"警告: 第 {batch_num} 批最终数量不匹配 "
                    f"(期望 {len(batch)}, 得到 {len(translated)}), 差异部分保留原文"
                )
                translated = _pad_or_truncate(translated, batch)
            else:
                # 全部重试失败，translated 仍为原文
                print(f"警告: 第 {batch_num} 批翻译全部失败，保留原文")

        translated_texts.extend(translated)

    return [
        f"{header}\n{text}" for header, text in zip(headers, translated_texts)
    ]


# ── 核心转录流程 ─────────────────────────────────────────

def transcribe_with_vad(args: argparse.Namespace) -> None:
    task_start = time.time()
    check_dependencies()

    input_file = args.audio or args.video
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        return

    if args.video and not is_video_file(input_file):
        print(f"警告: {input_file} 看起来不像是视频文件，继续尝试...")
    elif args.audio and not is_audio_file(input_file):
        print(f"警告: {input_file} 可能不是标准音频文件，继续尝试...")

    print("--- 任务开始 ---")

    # 提前初始化关键变量，防止在 try 块中途中断时后续代码访问未绑定变量
    srt_entries: list[str] = []
    vad_model = None
    wav = None
    speech_timestamps = None

    try:
        # ── 加载 VAD 模型 ──
        print("正在加载 VAD 模型...")
        try:
            vad_model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            (get_speech_timestamps, _, _, _, _) = utils
        except Exception as e:
            print(f"加载 VAD 模型失败: {e}")
            return

        # ── 读取音频 ──
        print(f"正在读取文件: {os.path.basename(input_file)} (使用 ffmpeg 解码)...")
        wav = load_audio_with_ffmpeg(input_file, SAMPLE_RATE)
        if wav is None:
            print("读取音频失败，请检查文件权限或 ffmpeg 是否可用")
            return

        # ── 人声检测 ──
        print(
            f"正在进行人声检测 (VAD) "
            f"[threshold={VAD_THRESHOLD}, min_silence={VAD_MIN_SILENCE_MS}ms]..."
        )
        speech_timestamps = get_speech_timestamps(
            wav,
            vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=VAD_THRESHOLD,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            min_speech_duration_ms=VAD_MIN_SPEECH_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS,
        )

        if not speech_timestamps:
            print("未检测到任何有效人声片段。")
            return

        lang_display = "自动检测" if args.lang == "auto" else args.lang
        print(f"检测到 {len(speech_timestamps)} 段人声区域，开始转录 (语言: {lang_display})...")

        counter = 1
        lang_param = None if args.lang == "auto" else args.lang
        total_segments = len(speech_timestamps)

        # ── 逐段转录 ──
        for i, segment in enumerate(speech_timestamps):
            start_sec = segment["start"] / SAMPLE_RATE
            audio_chunk = wav[segment["start"]:segment["end"]].numpy()

            # 音量归一化：将音频峰值拉到 [-1, 1]，提升小声片段的识别率
            # 极小信号（peak < 1e-6）视为纯噪声，跳过以避免放大后产生幻觉
            peak = np.max(np.abs(audio_chunk))
            if peak < 1e-6:
                continue
            audio_chunk = audio_chunk / peak

            result = mlx_whisper.transcribe(
                audio_chunk,
                path_or_hf_repo=args.model,
                fp16=True,
                condition_on_previous_text=False,
                verbose=False,
                language=lang_param,
            )

            for chunk_segment in result.get("segments", []):
                text = chunk_segment.get("text", "").strip()
                if not text:
                    continue

                s_start = start_sec + chunk_segment["start"]
                s_end = start_sec + chunk_segment["end"]

                entry = (
                    f"{counter}\n"
                    f"{format_timestamp(s_start)} --> {format_timestamp(s_end)}\n"
                    f"{text}"
                )
                srt_entries.append(entry)
                counter += 1

            if (i + 1) % PROGRESS_INTERVAL == 0 or i == total_segments - 1:
                print(f"进度: {i + 1}/{total_segments} 个片段处理完成")

    except KeyboardInterrupt:
        print("\n\n检测到中断，正在保存已处理的结果...")

    if not srt_entries:
        print("\n未生成任何字幕内容。")
        return

    # ── 保存原始字幕 ──
    effective_output = args.output or "output.srt"
    output_path = os.path.abspath(effective_output)

    if args.to and args.output:
        # 用户显式指定了 --output 且需要翻译：
        # 原始字幕保存到派生路径，翻译文件使用 --output 路径
        base, ext = os.path.splitext(output_path)
        original_path = f"{base}.original{ext}"
        translated_path = output_path
    elif args.to:
        # 需要翻译但未指定 --output：原始字幕用默认路径，翻译加语言后缀
        original_path = output_path
        base, ext = os.path.splitext(output_path)
        translated_path = f"{base}.{args.to}{ext}"
    else:
        # 仅转录，不翻译
        original_path = output_path

    _save_srt(srt_entries, original_path)
    print(f"原始字幕已保存至: {original_path}")

    # 释放语音识别资源，回收内存
    if vad_model is not None or wav is not None:
        del vad_model, wav, speech_timestamps
        gc.collect()

    # ── 可选翻译 ──
    if args.to:
        print()  # 翻译前空一行分隔
        _translate_and_save(srt_entries, args.to, args.translate_config, translated_path)

    elapsed = time.time() - task_start
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"\n--- 任务完成 (耗时 {minutes}分{seconds}秒) ---")


def translate_srt_file(args: argparse.Namespace) -> None:
    """直接翻译已有的 SRT 字幕文件"""
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

    # 生成输出路径
    if args.output is not None:
        translated_path = os.path.abspath(args.output)
    else:
        base, ext = os.path.splitext(srt_path)
        translated_path = f"{base}.{args.to}{ext}"

    _translate_and_save(srt_entries, args.to, args.translate_config, translated_path)

    elapsed = time.time() - task_start
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"\n--- 翻译任务完成 (耗时 {minutes}分{seconds}秒) ---")


# ── 入口 ─────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 MLX 和 VAD 加速转录音频或视频为 SRT"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", type=str, help="输入音频文件路径")
    group.add_argument("--video", type=str, help="输入视频文件路径")
    group.add_argument("--srt", type=str, help="输入已有 SRT 字幕文件路径 (仅翻译，需配合 --to)")
    parser.add_argument(
        "--lang",
        type=str,
        default="auto",
        choices=["zh", "en", "ja", "ko", "auto"],
        help="指定语言 (默认: auto 自动检测)",
    )
    parser.add_argument(
        "--to",
        type=str,
        default=None,
        choices=["zh", "en", "ja", "ko"],
        help="将字幕翻译为指定语言 (默认: 不翻译)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/whisper-large-v3-mlx",
        help="MLX 模型路径或 HF 仓库",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 SRT 文件名 (默认: output.srt; 配合 --to 时指定翻译文件路径)",
    )

    if sys.platform != "darwin":
        print("警告: mlx-whisper 专门为 macOS 设计。")

    args = parser.parse_args()

    # --srt 模式必须指定 --to
    if args.srt and not args.to:
        print("错误: 使用 --srt 时必须同时指定 --to 目标语言。")
        sys.exit(1)

    if args.to:
        if not args.srt and args.to == args.lang:
            print("错误: --to 和 --lang 不能指定相同的语言。")
            sys.exit(1)
        api_key, base_url, model_name = get_translate_config()
        print("正在检查翻译API可用性...")
        check_translate_api(api_key, base_url, model_name)
        args.translate_config = (api_key, base_url, model_name)
    else:
        args.translate_config = None

    if args.srt:
        translate_srt_file(args)
    else:
        transcribe_with_vad(args)


if __name__ == "__main__":
    main()
