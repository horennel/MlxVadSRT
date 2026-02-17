"""字幕嵌入模块：FFmpeg 软字幕封装"""

import os
import subprocess
import time
import argparse

from .config import FFMPEG_LANG_CODES
from .utils import check_dependencies


def embed_subtitle(args: argparse.Namespace, auto_generated_srt: bool = False) -> None:
    """嵌入字幕到视频文件。

    Args:
        args: 包含 video, srt 等参数的命名空间
        auto_generated_srt: 若为 True 表示 SRT 由程序自动生成，嵌入后自动删除；
                            若为 False（默认）表示用户提供的文件，不删除。
    """
    task_start = time.time()
    check_dependencies()

    video_path = os.path.abspath(args.video)
    srt_path = os.path.abspath(args.srt)

    if not os.path.exists(video_path):
        print(f"错误: 找不到视频文件 {video_path}")
        return
    if not os.path.exists(srt_path):
        print(f"错误: 找不到字幕文件 {srt_path}")
        return

    print("--- 字幕嵌入任务开始 ---")
    print(f"视频文件: {os.path.basename(video_path)}")
    print(f"字幕文件: {os.path.basename(srt_path)}")

    video_ext = os.path.splitext(video_path)[1].lower()
    if video_ext in (".mkv",):
        sub_codec = "srt"
    elif video_ext in (".mp4", ".m4v", ".mov"):
        sub_codec = "mov_text"
    else:
        print(f"警告: {video_ext} 格式可能不支持软字幕，尝试使用 mov_text 编码...")
        sub_codec = "mov_text"

    # 确定字幕语言元数据
    # 优先级: 1. 翻译目标语言 (--to) 2. 源语言 (--lang, 非 auto) 3. 文件名推断
    lang_code = None
    if getattr(args, "to", None):
        lang_code = args.to
    elif getattr(args, "lang", None) and args.lang != "auto":
        lang_code = args.lang
    
    if not lang_code:
        # 从文件名推断 (如 xxx.zh.srt → zh)
        srt_base = os.path.splitext(os.path.basename(srt_path))[0]
        if "." in srt_base:
            lang_code = srt_base.rsplit(".", 1)[-1]

    ffmpeg_lang = FFMPEG_LANG_CODES.get(lang_code, "und")

    base, _ = os.path.splitext(video_path)
    temp_output = f"{base}.tmp{video_ext}"

    # 使用 ffprobe 精确探测原视频中已有的字幕轨数量
    probe_cmd = [
        "ffprobe", 
        "-v", "error", 
        "-select_streams", "s", 
        "-show_entries", "stream=index", 
        "-of", "csv=p=0", 
        video_path
    ]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        # 统计输出行数即为流数量
        existing_sub_count = len(result.stdout.strip().splitlines())
    except Exception:
        # Fallback: 如果 ffprobe 失败，假设为 0 (或者可以尝试解析 ffmpeg 输出，但风险较大)
        existing_sub_count = 0

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-i", srt_path,
        "-c", "copy",
        "-c:s", sub_codec,
        "-map", "0:v",
        "-map", "0:a?",
        "-map", "0:s?",
        "-map", "1",
        f"-metadata:s:s:{existing_sub_count}", f"language={ffmpeg_lang}",
        "-y",
        temp_output,
    ]

    if ffmpeg_lang != "und":
        print(f"正在嵌入字幕 (编码: {sub_codec}, 语言: {ffmpeg_lang})...")
    else:
        print(f"正在嵌入字幕 (编码: {sub_codec})...")

    embed_ok = False
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_lines = result.stderr.strip().split("\n")[-5:]
            print(f"字幕嵌入失败:\n" + "\n".join(error_lines))
            return
        embed_ok = True
    except Exception as e:
        print(f"调用 ffmpeg 失败: {e}")
        return
    finally:
        if not embed_ok and os.path.exists(temp_output):
            os.remove(temp_output)

    base_name, ext = os.path.splitext(video_path)
    final_output = f"{base_name}_embedded{ext}"

    try:
        os.replace(temp_output, final_output)
        print(f"字幕已嵌入至新文件: {final_output}")

        # 仅删除程序自动生成的 SRT 文件，用户提供的文件不删除
        if auto_generated_srt and os.path.exists(srt_path):
            os.remove(srt_path)
            print(f"已删除自动生成的字幕文件: {srt_path}")
    except OSError as e:
        print(f"重命名文件失败: {e}")
        print(f"带字幕的视频已保存至: {temp_output}")

    elapsed = time.time() - task_start
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"\n--- 嵌入任务完成 (耗时 {minutes}分{seconds}秒) ---")
