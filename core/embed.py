"""字幕嵌入模块：FFmpeg 软字幕封装"""

import os
import subprocess
import time
from typing import Optional

from .config import FFMPEG_LANG_CODES
from .utils import check_dependencies, format_elapsed


def embed_subtitle(
        video: str,
        srt: str,
        lang: Optional[str] = None,
        to: Optional[str] = None,
        auto_generated_srt: bool = False,
) -> bool:
    """嵌入字幕到视频文件。

    Args:
        video: 视频文件路径
        srt: SRT 字幕文件路径
        lang: 源语言代码 (如 "zh", "en")
        to: 翻译目标语言代码 (如 "zh", "en")
        auto_generated_srt: 若为 True 表示 SRT 由程序自动生成，嵌入后自动删除；
                            若为 False（默认）表示用户提供的文件，不删除。

    Returns:
        嵌入成功返回 True，失败返回 False
    """
    task_start = time.time()
    check_dependencies()

    video_path = os.path.abspath(video)
    srt_path = os.path.abspath(srt)

    if not os.path.exists(video_path):
        print(f"错误: 找不到视频文件 {video_path}")
        return False
    if not os.path.exists(srt_path):
        print(f"错误: 找不到字幕文件 {srt_path}")
        return False

    print("--- 字幕嵌入任务开始 ---")
    print(f"视频文件: {os.path.basename(video_path)}")
    print(f"字幕文件: {os.path.basename(srt_path)}")

    # 根据视频格式选择字幕编码
    video_ext = os.path.splitext(video_path)[1].lower()
    sub_codec = _select_sub_codec(video_ext)

    # 确定字幕语言元数据
    ffmpeg_lang = _resolve_ffmpeg_lang(to=to, lang=lang, srt_path=srt_path)

    base, _ = os.path.splitext(video_path)
    temp_output = f"{base}.tmp{video_ext}"

    existing_sub_count = _probe_subtitle_count(video_path)

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
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if result.returncode != 0:
            error_lines = result.stderr.strip().split("\n")[-5:]
            print(f"字幕嵌入失败:\n" + "\n".join(error_lines))
            return False
        embed_ok = True
    except Exception as e:
        print(f"调用 ffmpeg 失败: {e}")
        return False
    finally:
        if not embed_ok and os.path.exists(temp_output):
            os.remove(temp_output)

    base_name, ext = os.path.splitext(video_path)
    final_output = f"{base_name}_embed{ext}"

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
        return False

    elapsed = time.time() - task_start
    print(f"\n--- 嵌入任务完成 (耗时 {format_elapsed(elapsed)}) ---")
    return True


# ── 内部辅助函数 ──────────────────────────────────────────


def _select_sub_codec(video_ext: str) -> str:
    """根据视频容器格式选择字幕编码"""
    if video_ext == ".mkv":
        return "srt"
    if video_ext in (".mp4", ".m4v", ".mov"):
        return "mov_text"
    print(f"警告: {video_ext} 格式可能不支持软字幕，尝试使用 mov_text 编码...")
    return "mov_text"


def _resolve_ffmpeg_lang(
        to: Optional[str], lang: Optional[str], srt_path: str
) -> str:
    """确定字幕语言元数据 (ISO 639-2/B)

    优先级: 1. 翻译目标语言 (to) 2. 源语言 (lang, 非 auto) 3. 文件名推断
    """
    lang_code = to
    if not lang_code and lang and lang != "auto":
        lang_code = lang
    if not lang_code:
        srt_base = os.path.splitext(os.path.basename(srt_path))[0]
        if "." in srt_base:
            lang_code = srt_base.rsplit(".", 1)[-1]
    return FFMPEG_LANG_CODES.get(lang_code, "und")


def _probe_subtitle_count(video_path: str) -> int:
    """使用 ffprobe 精确探测原视频中已有的字幕轨数量"""
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "s",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        video_path,
    ]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, encoding="utf-8")
        return len(result.stdout.strip().splitlines())
    except Exception:
        return 0
