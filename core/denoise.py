"""人声提取模块：MDX-NET 模型分离人声"""

import os
import sys
import shutil
import subprocess
import gc
import time
from typing import Optional

from .config import DENOISE_MODEL, DENOISE_MODEL_DIR
from .utils import format_elapsed


def _cleanup_vocal_temp(vocal_temp_path: Optional[str]) -> None:
    if vocal_temp_path and os.path.exists(vocal_temp_path):
        temp_dir = os.path.dirname(vocal_temp_path)
        shutil.rmtree(temp_dir, ignore_errors=True)


def extract_vocals(input_file: str) -> Optional[str]:
    """提取人声并返回 WAV 路径，处理后释放模型内存"""

    # py2app 兼容: py2app 会设置 sys.frozen=True，但不设置 sys._MEIPASS
    # (后者是 PyInstaller 专属属性)。audio-separator 内部的 pyrb.py 会在
    # sys.frozen 为 True 时访问 sys._MEIPASS，导致 AttributeError。
    # 预设一个合理的路径值来避免崩溃。
    if getattr(sys, "frozen", False) and not hasattr(sys, "_MEIPASS"):
        sys._MEIPASS = os.path.dirname(sys.executable)

    try:
        from audio_separator.separator import Separator
    except ImportError:
        print("错误: 人声提取需要安装 audio-separator 库。")
        print("请运行: pip install 'audio-separator[cpu]'")
        sys.exit(1)

    import logging
    import tempfile

    temp_dir = tempfile.mkdtemp(prefix="mlxvadsrt_vocals_")
    success = False

    try:
        print(f"正在加载人声提取模型: {DENOISE_MODEL}")

        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=DENOISE_MODEL_DIR,
            output_dir=temp_dir,
            output_format="WAV",
            output_single_stem="Vocals",
        )
        separator.load_model(model_filename=DENOISE_MODEL)

        print(f"正在提取人声 (模型: {DENOISE_MODEL.replace('.onnx', '')})...")
        denoise_start = time.time()

        # 预转换为 WAV 避免 codec 兼容性问题
        temp_audio_path = os.path.join(temp_dir, "input_audio.wav")
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", input_file,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                "-ac", "2",
                temp_audio_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            print("警告: 预处理音频转换失败，尝试直接处理原文件...")
            temp_audio_path = input_file

        output_files = separator.separate(temp_audio_path)

        del separator
        gc.collect()

        if temp_audio_path != input_file and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        elapsed = time.time() - denoise_start
        print(f"人声提取完成 (耗时 {format_elapsed(elapsed)})")

        if not output_files:
            print("警告: 人声提取未生成任何输出文件。")
            return None

        vocal_path = output_files[0]
        if not os.path.isabs(vocal_path):
            vocal_path = os.path.join(temp_dir, vocal_path)

        if not os.path.exists(vocal_path):
            print(f"警告: 人声文件不存在: {vocal_path}")
            return None

        print(f"人声文件: {vocal_path}")
        success = True
        return vocal_path

    except KeyboardInterrupt:
        print("\n人声提取被中断。")
        raise
    except Exception as e:
        print(f"人声提取失败: {e}")
        return None
    finally:
        if not success and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
