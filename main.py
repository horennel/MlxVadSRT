"""MlxVadSRT — 使用 MLX Whisper + VAD 加速转录音频/视频为 SRT 字幕"""

import os
import sys
import argparse

from core.transcribe import transcribe_with_vad
from core.translate import get_translate_config, check_translate_api, translate_srt_file
from core.embed import embed_subtitle
from core.utils import DependencyError


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

    if args.embed and args.video and args.srt and not args.to:
        try:
            embed_subtitle(args)
        except DependencyError as e:
            print(f"\n❌ 环境依赖错误: {e}")
            sys.exit(1)
        return

    # 检查基本输入参数
    input_count = sum(1 for x in [args.audio, args.video, args.srt] if x)
    if input_count == 0:
        print("错误: 请指定 --audio, --video, --srt 中的一个 (或使用 --embed 模式)。")
        sys.exit(1)

    # 允许 --embed 与视频同时使用，由后续逻辑生成 srt
    if args.embed and not args.video:
        print("错误: --embed 必须配合 --video 使用。")
        sys.exit(1)

    if input_count > 1:
        # 特例：允许 --embed + --video + --srt 组合（用于翻译现有字幕并嵌入）
        is_translate_embed_combo = args.embed and args.video and args.srt and not args.audio

        if not is_translate_embed_combo:
            print("错误: --audio, --video, --srt 不能同时使用。")
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

    try:
        final_srt_path = None
        if args.srt:
            final_srt_path = translate_srt_file(args)
        else:
            final_srt_path = transcribe_with_vad(args)

        # 自动嵌入字幕步骤
        if args.embed and final_srt_path and os.path.exists(final_srt_path):
            print(f"\n--- 正在执行字幕嵌入 (SRT: {os.path.basename(final_srt_path)}) ---")
            args.srt = final_srt_path  # 更新 srt 参数为生成的路径
            embed_subtitle(args, auto_generated_srt=True)
            
    except DependencyError as e:
        print(f"\n❌ 环境依赖错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
