"""核心转录模块：VAD + Whisper 逐段转录"""

import os
import time
import gc
import numpy as np
import torch
import mlx_whisper
from typing import Optional

from .config import (
    SAMPLE_RATE,
    VAD_THRESHOLD,
    VAD_THRESHOLD_DENOISE,
    VAD_MIN_SILENCE_MS,
    VAD_MIN_SPEECH_MS,
    VAD_SPEECH_PAD_MS,
    PROGRESS_INTERVAL,
)
from .utils import (
    check_dependencies,
    is_audio_file,
    is_video_file,
    load_audio_with_ffmpeg,
    format_timestamp,
    format_elapsed,
    _save_srt,
)
from .denoise import extract_vocals, _cleanup_vocal_temp
from .translate import _translate_and_save


def transcribe_with_vad(
        input_file: str,
        lang: str = "auto",
        model: str = "mlx-community/whisper-large-v3-mlx",
        denoise: bool = False,
        to: Optional[str] = None,
        translate_config: Optional[tuple[str, str, str]] = None,
        output: Optional[str] = None,
        is_video: bool = False,
) -> Optional[str]:
    """使用 VAD + Whisper 转录音频/视频为 SRT 字幕。

    Args:
        input_file: 输入音频或视频文件路径
        lang: 源语言代码，"auto" 表示自动检测
        model: MLX Whisper 模型路径或 HF 仓库
        denoise: 是否先提取人声去除背景音
        to: 翻译目标语言代码 (None 表示不翻译)
        translate_config: 翻译 API 配置 (api_key, base_url, model_name)
        output: 输出 SRT 路径 (None 表示自动生成)
        is_video: 输入文件是否为视频 (用于文件类型校验提示)

    Returns:
        生成的 SRT 文件路径，失败返回 None
    """
    task_start = time.time()
    check_dependencies()

    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        return None

    _warn_file_type(input_file, is_video)

    print("--- 任务开始 ---")

    vocal_temp_path: Optional[str] = None

    try:
        # 1. 加载 VAD 模型
        vad_result = _load_vad_model()
        if vad_result is None:
            return None
        vad_model, get_speech_timestamps = vad_result

        # 2. 可选：人声提取
        audio_source = input_file
        if denoise:
            vocal_temp_path = extract_vocals(input_file)
            if vocal_temp_path is not None:
                audio_source = vocal_temp_path
                print()
            else:
                print("警告: 人声提取失败，将使用原始音频继续处理。")

        # 3. 读取音频
        print(f"正在读取文件: {os.path.basename(audio_source)} (使用 ffmpeg 解码)...")
        wav = load_audio_with_ffmpeg(audio_source, SAMPLE_RATE)
        if wav is None:
            print("读取音频失败，请检查文件权限或 ffmpeg 是否可用")
            return None

        # 4. VAD 检测
        vad_threshold = VAD_THRESHOLD_DENOISE if (denoise and vocal_temp_path) else VAD_THRESHOLD
        speech_timestamps = _run_vad(vad_model, get_speech_timestamps, wav, vad_threshold)
        if not speech_timestamps:
            print("未检测到任何有效人声片段。")
            return None

        # 5. 逐段转录
        lang_display = "自动检测" if lang == "auto" else lang
        print(f"检测到 {len(speech_timestamps)} 段人声区域，开始转录 (语言: {lang_display})...")

        srt_entries = _transcribe_segments(speech_timestamps, wav, model, lang)

        # 6. 释放大型资源
        del vad_model, wav, speech_timestamps
        gc.collect()

        if not srt_entries:
            print("\n未生成任何字幕内容。")
            return None

        # 7. 保存与翻译
        return _save_and_translate(
            srt_entries=srt_entries,
            input_file=input_file,
            output=output,
            to=to,
            translate_config=translate_config,
            task_start=task_start,
        )

    except KeyboardInterrupt:
        print("\n\n检测到中断。")
        raise

    finally:
        _cleanup_vocal_temp(vocal_temp_path)


# ── 内部辅助函数 ──────────────────────────────────────────


def _warn_file_type(input_file: str, is_video: bool) -> None:
    """文件类型不匹配时打印警告"""
    if is_video and not is_video_file(input_file):
        print(f"警告: {input_file} 看起来不像是视频文件，继续尝试...")
    elif not is_video and not is_audio_file(input_file):
        print(f"警告: {input_file} 可能不是标准音频文件，继续尝试...")


def _load_vad_model():
    """加载 Silero VAD 模型，返回 (model, get_speech_timestamps) 或 None"""
    print("正在加载 VAD 模型...")
    try:
        vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        get_speech_timestamps = utils[0]
        return vad_model, get_speech_timestamps
    except Exception as e:
        print(f"加载 VAD 模型失败: {e}")
        return None


def _run_vad(vad_model, get_speech_timestamps, wav, threshold: float):
    """执行 VAD 检测，返回语音时间戳列表"""
    print(
        f"正在进行人声检测 (VAD) "
        f"[threshold={threshold}, min_silence={VAD_MIN_SILENCE_MS}ms]..."
    )
    return get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=threshold,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        min_speech_duration_ms=VAD_MIN_SPEECH_MS,
        speech_pad_ms=VAD_SPEECH_PAD_MS,
    )


def _transcribe_segments(
        speech_timestamps: list[dict],
        wav: torch.Tensor,
        model: str,
        lang: str,
) -> list[str]:
    """逐段转录语音片段，返回 SRT 条目列表"""
    srt_entries: list[str] = []
    counter = 1
    lang_param = None if lang == "auto" else lang
    total_segments = len(speech_timestamps)

    for i, segment in enumerate(speech_timestamps):
        start_sec = segment["start"] / SAMPLE_RATE
        end_sec = segment["end"] / SAMPLE_RATE
        chunk_duration = end_sec - start_sec
        audio_chunk = wav[segment["start"]:segment["end"]].numpy()

        if np.max(np.abs(audio_chunk)) < 1e-6:
            continue

        result = mlx_whisper.transcribe(
            audio_chunk,
            path_or_hf_repo=model,
            fp16=True,
            condition_on_previous_text=False,
            verbose=False,
            language=lang_param,
        )

        for chunk_segment in result.get("segments", []):
            text = chunk_segment.get("text", "").strip()
            if not text:
                continue

            # 裁剪 Whisper 幻觉时间戳到有效范围内
            seg_start = max(0.0, chunk_segment["start"])
            seg_end = min(chunk_segment["end"], chunk_duration)
            if seg_start >= seg_end:
                continue

            s_start = start_sec + seg_start
            s_end = start_sec + seg_end

            entry = (
                f"{counter}\n"
                f"{format_timestamp(s_start)} --> {format_timestamp(s_end)}\n"
                f"{text}"
            )
            srt_entries.append(entry)
            counter += 1

        if (i + 1) % PROGRESS_INTERVAL == 0 or i == total_segments - 1:
            print(f"进度: {i + 1}/{total_segments} 个片段处理完成")

    return srt_entries


def _save_and_translate(
        srt_entries: list[str],
        input_file: str,
        output: Optional[str],
        to: Optional[str],
        translate_config: Optional[tuple[str, str, str]],
        task_start: float,
) -> str:
    """保存原始字幕，可选翻译，返回最终输出路径"""
    if output:
        output_path = os.path.abspath(os.path.expanduser(output))
    else:
        base = os.path.splitext(input_file)[0]
        output_path = os.path.abspath(f"{base}.srt")

    if to and output:
        base, ext = os.path.splitext(output_path)
        original_path = f"{base}.original{ext}"
        translated_path = output_path
    elif to:
        original_path = output_path
        base, ext = os.path.splitext(output_path)
        translated_path = f"{base}.{to}{ext}"
    else:
        original_path = output_path

    _save_srt(srt_entries, original_path)
    print(f"原始字幕已保存至: {original_path}")

    if to:
        print()
        _translate_and_save(srt_entries, to, translate_config, translated_path)
        final_output_srt = translated_path
    else:
        final_output_srt = original_path

    elapsed = time.time() - task_start
    print(f"\n--- 任务完成 (耗时 {format_elapsed(elapsed)}) ---")

    return final_output_srt
