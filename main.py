"""MlxVadSRT — 使用 MLX Whisper + VAD 加速转录音频/视频为 SRT 字幕"""

import os

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import argparse

from transcribe import transcribe_with_vad
from translate import get_translate_config, check_translate_api, translate_srt_file
from embed import embed_subtitle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 MLX 和 VAD 加速转录音频或视频为 SRT"
    )
    parser.add_argument("--audio", type=str, help="输入音频文件路径")
    parser.add_argument("--video", type=str, help="输入视频文件路径")
    parser.add_argument("--srt", type=str, help="输入已有 SRT 字幕文件路径")
    parser.add_argument(
        "--embed",
        action="store_true",
        help="将 --srt 字幕作为软字幕嵌入 --video 视频 (需同时指定 --video 和 --srt)",
    )
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
        help="输出 SRT 文件名 (默认: 跟随输入文件名; 配合 --to 时指定翻译文件路径)",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="转录前先用 MDX-NET 模型提取人声，去除 BGM/音效 (需安装 audio-separator)",
    )

    if sys.platform != "darwin":
        print("警告: mlx-whisper 专门为 macOS 设计。")

    args = parser.parse_args()

    if args.embed:
        if not args.video or not args.srt:
            print("错误: --embed 需要同时指定 --video (视频文件) 和 --srt (字幕文件)。")
            sys.exit(1)
        embed_subtitle(args)
        return

    input_count = sum(1 for x in [args.audio, args.video, args.srt] if x)
    if input_count == 0:
        print("错误: 请指定 --audio, --video, --srt 中的一个 (或使用 --embed 模式)。")
        sys.exit(1)
    if input_count > 1:
        print("错误: --audio, --video, --srt 不能同时使用 (嵌入字幕请用 --embed)。")
        sys.exit(1)

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
