"""核心转录模块：VAD + Whisper 逐段转录"""

import os
import time
import gc
import argparse
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
    _save_srt,
)
from .denoise import extract_vocals, _cleanup_vocal_temp
from .translate import _translate_and_save


def transcribe_with_vad(args: argparse.Namespace) -> Optional[str]:
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

    srt_entries: list[str] = []
    vad_model = None
    wav = None
    speech_timestamps = None
    vocal_temp_path = None

    try:
        try:
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

            audio_source = input_file

            if args.denoise:
                vocal_temp_path = extract_vocals(input_file)
                if vocal_temp_path is not None:
                    audio_source = vocal_temp_path
                    print()
                else:
                    print("警告: 人声提取失败，将使用原始音频继续处理。")

            print(f"正在读取文件: {os.path.basename(audio_source)} (使用 ffmpeg 解码)...")
            wav = load_audio_with_ffmpeg(audio_source, SAMPLE_RATE)
            if wav is None:
                print("读取音频失败，请检查文件权限或 ffmpeg 是否可用")
                return

            vad_threshold = VAD_THRESHOLD_DENOISE if (args.denoise and vocal_temp_path) else VAD_THRESHOLD
            print(
                f"正在进行人声检测 (VAD) "
                f"[threshold={vad_threshold}, min_silence={VAD_MIN_SILENCE_MS}ms]..."
            )
            speech_timestamps = get_speech_timestamps(
                wav,
                vad_model,
                sampling_rate=SAMPLE_RATE,
                threshold=vad_threshold,
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

            for i, segment in enumerate(speech_timestamps):
                start_sec = segment["start"] / SAMPLE_RATE
                audio_chunk = wav[segment["start"]:segment["end"]].numpy()

                if np.max(np.abs(audio_chunk)) < 1e-6:
                    continue

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

        if args.output:
            output_path = os.path.abspath(args.output)
        else:
            base = os.path.splitext(input_file)[0]
            output_path = os.path.abspath(f"{base}.srt")

        if args.to and args.output:
            base, ext = os.path.splitext(output_path)
            original_path = f"{base}.original{ext}"
            translated_path = output_path
        elif args.to:
            original_path = output_path
            base, ext = os.path.splitext(output_path)
            translated_path = f"{base}.{args.to}{ext}"
        else:
            original_path = output_path

        _save_srt(srt_entries, original_path)
        print(f"原始字幕已保存至: {original_path}")

        if vad_model is not None or wav is not None:
            del vad_model, wav, speech_timestamps
            gc.collect()

        if args.to:
            print()
            _translate_and_save(srt_entries, args.to, args.translate_config, translated_path)
            final_output_srt = translated_path
        else:
            final_output_srt = original_path

        elapsed = time.time() - task_start
        minutes, seconds = divmod(int(elapsed), 60)
        print(f"\n--- 任务完成 (耗时 {minutes}分{seconds}秒) ---")

        return final_output_srt

    finally:
        _cleanup_vocal_temp(vocal_temp_path)
