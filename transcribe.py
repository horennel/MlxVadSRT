import os

# 如果本地没有模型，首次运行时会自动从 Hugging Face 下载
# 可以使用国内镜像站加速下载
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
import tempfile

def check_dependencies():
    """检查必要的系统依赖"""
    if not shutil.which("ffmpeg"):
        print("错误: 在系统 PATH 中找不到 'ffmpeg'。")
        print("请先安装 ffmpeg (例如: brew install ffmpeg)")
        sys.exit(1)

def is_audio_file(file_path):
    """检测文件是否为音频文件"""
    audio_extensions = {'.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.wma', '.aiff', '.aif'}
    ext = os.path.splitext(file_path)[1].lower()
    return ext in audio_extensions

def is_video_file(file_path):
    """检测文件是否为视频文件"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpeg', '.mpg'}
    ext = os.path.splitext(file_path)[1].lower()
    return ext in video_extensions

def extract_audio_from_video(video_path):
    """从视频文件中提取音频（使用流复制，最快速）

    Args:
        video_path: 视频文件路径

    Returns:
        提取的音频文件路径
    """
    # 创建临时文件（在当前目录下）
    temp_fd, temp_path = tempfile.mkstemp(suffix='.m4a', prefix='transcribe_audio_', dir='.')
    os.close(temp_fd)

    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vn',              # 不包含视频
        '-c:a', 'copy',     # 直接复制音频流，不重新编码
        '-y',               # 覆盖输出文件
        temp_path
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return temp_path
    except subprocess.CalledProcessError as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"从视频提取音频失败: {e}")
        return None

def load_audio_with_ffmpeg(file_path, sr=16000):
    """使用系统 ffmpeg 命令读取音频，绕过库依赖问题"""
    # 使用绝对路径防止路径歧义
    file_path = os.path.abspath(file_path)

    cmd = [
        'ffmpeg',
        '-i', file_path,
        '-f', 'f32le',      # 输出 float32 格式
        '-acodec', 'pcm_f32le',
        '-ac', '1',          # 单声道
        '-ar', str(sr),     # 采样率 16000
        '-'                  # 输出到 stdout
    ]
    try:
        # stderr=subprocess.DEVNULL 隐藏 ffmpeg 的日志
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        output, _ = process.communicate()
        if process.returncode != 0:
            return None
        return torch.from_numpy(np.frombuffer(output, dtype=np.float32))
    except Exception as e:
        print(f"调用 ffmpeg 失败: {e}")
        return None

def format_timestamp(seconds: float):
    """将秒数转换为 SRT 时间戳格式 (00:00:00,000)"""
    total_seconds = int(seconds)
    millis = int((seconds - total_seconds) * 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def transcribe_with_vad(args):
    # 0. 检查 FFmpeg
    check_dependencies()

    # 1. 验证参数并检测文件类型
    if not args.audio and not args.video:
        print("错误: 必须指定 --audio 或 --video 其中之一。")
        return

    input_file = args.audio or args.video
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        return

    audio_file_to_process = input_file
    temp_audio_file = None

    # 如果指定了 --video 但传入的是音频文件，或者反之，进行验证
    if args.video:
        if not is_video_file(input_file):
            print(f"错误: {input_file} 看起来不是一个视频文件。请使用 --audio 参数。")
            return
        print(f"检测到视频文件，正在提取音频 (流复制)...")
        temp_audio_file = extract_audio_from_video(input_file)
        if temp_audio_file:
            audio_file_to_process = temp_audio_file
        else:
            print("错误: 视频提取音频失败")
            return
    elif args.audio:
        if not is_audio_file(input_file):
            # 如果是视频文件但用了 --audio，我们也可以自动处理，但按照用户要求进行提示
            if is_video_file(input_file):
                print(f"提示: {input_file} 看起来是视频文件，正在为您自动提取音频...")
                temp_audio_file = extract_audio_from_video(input_file)
                if temp_audio_file:
                    audio_file_to_process = temp_audio_file
                else:
                    print("错误: 视频提取音频失败")
                    return
            else:
                print(f"警告: {input_file} 可能不是标准音频文件，尝试直接处理。")

    # 2. 加载 Silero VAD 模型
    print(f"--- 任务开始 ---")
    print("正在加载 VAD 模型...")
    try:
        vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True)
        (get_speech_timestamps, _, _, _, _) = utils
    except Exception as e:
        print(f"加载 VAD 模型失败: {e}")
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
        return

    # 3. 读取音频
    if not os.path.exists(audio_file_to_process):
        print(f"错误: 找不到文件 {audio_file_to_process}")
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
        return

    print(f"正在读取文件: {os.path.basename(input_file)} (使用 ffmpeg 解码)...")
    wav = load_audio_with_ffmpeg(audio_file_to_process, args.sample_rate)

    if wav is None:
        print("读取音频失败，请检查文件权限或 ffmpeg 是否可用")
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
        return

    # 4. 获取人声时间戳
    print("正在进行人声检测 (VAD)...")
    speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=args.sample_rate,
        threshold=0.5,
        min_silence_duration_ms=1000
    )

    if not speech_timestamps:
        print("未检测到任何有效人声片段。")
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
        return

    print(f"检测到 {len(speech_timestamps)} 段人声区域，开始转录 (语言: {'自动检测' if args.lang == 'auto' else args.lang})...")

    srt_entries = []
    counter = 1
    lang_param = None if args.lang == "auto" else args.lang

    # 5. 循环处理转录
    try:
        for i, segment in enumerate(speech_timestamps):
            start_samples = segment['start']
            end_samples = segment['end']
            start_sec = start_samples / args.sample_rate

            # 提取片段
            audio_chunk = wav[start_samples:end_samples].numpy()

            # 使用 mlx-whisper 转录
            result = mlx_whisper.transcribe(
                audio_chunk,
                path_or_hf_repo=args.model,
                fp16=True,
                condition_on_previous_text=False,
                verbose=False,
                language=lang_param
            )

            # 处理转录结果
            for chunk_segment in result.get('segments', []):
                text = chunk_segment.get('text', '').strip()
                if not text:
                    continue

                s_start = start_sec + chunk_segment['start']
                s_end = start_sec + chunk_segment['end']

                entry = f"{counter}\n{format_timestamp(s_start)} --> {format_timestamp(s_end)}\n{text}"
                srt_entries.append(entry)
                counter += 1

            if (i + 1) % 5 == 0 or i == len(speech_timestamps) - 1:
                print(f"进度: {i+1}/{len(speech_timestamps)} 个片段处理完成")

    except KeyboardInterrupt:
        print("\n\n检测到中断，正在保存已处理的结果...")
    finally:
        # 在任何情况下都清理临时文件
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
            print("临时文件已清理")

    # 6. 保存结果
    if not srt_entries:
        print("\n未生成任何字幕内容。")
        return

    output_path = os.path.abspath(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(srt_entries) + "\n")

    print(f"\n完成！字幕已保存至: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="使用 MLX 和 VAD 加速转录音频或视频为 SRT")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", type=str, help="输入音频文件路径")
    group.add_argument("--video", type=str, help="输入视频文件路径")
    parser.add_argument("--lang", type=str, default="auto", choices=["zh", "en", "ja", "ko", "auto"], help="指定语言 (默认: auto 自动检测)")
    parser.add_argument("--model", type=str, default="mlx-community/whisper-large-v3-mlx", help="MLX 模型路径或 HF 仓库")
    parser.add_argument("--output", type=str, default="output.srt", help="输出 SRT 文件名")
    parser.add_argument("--sample_rate", type=int, default=16000, help="采样率")

    if sys.platform != "darwin":
        print("警告: mlx-whisper 专门为 macOS 设计。")

    args = parser.parse_args()
    transcribe_with_vad(args)

if __name__ == "__main__":
    main()
