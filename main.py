"""MlxVadSRT — 使用 MLX Whisper + VAD 加速转录音频/视频为 SRT 字幕"""

import sys
import argparse

from core.pipeline import TaskParams, validate_params, prepare_translate_config, run_task
from core.utils import DependencyError


def _parse_args() -> argparse.Namespace:
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
        help="输出路径 (默认自动生成在输入文件旁; 配合 --embed 时指定视频输出路径，否则指定 SRT 路径)",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="转录前先用 MDX-NET 模型提取人声，去除 BGM/音效 (需安装 audio-separator)",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="启动 Web UI",
    )
    return parser.parse_args()


def main() -> None:
    if sys.platform != "darwin":
        print("警告: mlx-whisper 专门为 macOS 设计。")

    args = _parse_args()

    # Web UI 模式
    if args.web:
        import web.app
        print("启动 Web 界面...")
        web.app.run()
        return

    # CLI 模式：构建 TaskParams
    params = TaskParams(
        audio=args.audio,
        video=args.video,
        srt=args.srt,
        lang=args.lang,
        to=args.to,
        model=args.model,
        output=args.output,
        denoise=args.denoise,
        embed=args.embed,
    )

    # 参数校验
    errors = validate_params(params)
    if errors:
        for e in errors:
            print(f"错误: {e}")
        sys.exit(1)

    # 翻译配置
    err = prepare_translate_config(params)
    if err:
        print(f"错误: {err}")
        sys.exit(1)

    # 执行任务
    try:
        result = run_task(params)
        if not result.success:
            sys.exit(1)
    except DependencyError as e:
        print(f"\n环境依赖错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
