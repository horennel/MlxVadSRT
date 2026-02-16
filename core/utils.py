"""通用工具函数"""

import os
import sys
import shutil
import subprocess
import numpy as np
import torch
from typing import Optional

from .config import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, SAMPLE_RATE


def check_dependencies() -> None:
    if not shutil.which("ffmpeg"):
        print("错误: 在系统 PATH 中找不到 'ffmpeg'。")
        print("请先安装 ffmpeg (例如: brew install ffmpeg)")
        sys.exit(1)


def is_audio_file(file_path: str) -> bool:
    return os.path.splitext(file_path)[1].lower() in AUDIO_EXTENSIONS


def is_video_file(file_path: str) -> bool:
    return os.path.splitext(file_path)[1].lower() in VIDEO_EXTENSIONS


def load_audio_with_ffmpeg(file_path: str, sr: int = SAMPLE_RATE) -> Optional[torch.Tensor]:
    """使用 ffmpeg 读取音频，返回单声道 float32 Tensor"""
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
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
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
    mins, secs = divmod(total_seconds, 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def _save_srt(srt_entries: list[str], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(srt_entries) + "\n")


def _parse_srt_file(srt_path: str) -> Optional[list[str]]:
    """读取并解析 SRT 文件，返回条目列表"""
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
