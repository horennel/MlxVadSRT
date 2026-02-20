"""统一的任务执行管道

将参数校验和任务编排逻辑从 main.py / web/app.py 中抽取出来，
避免 CLI 和 Web 两端重复实现相同逻辑。
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from .config import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS


# ── 数据类定义 ──────────────────────────────────────────────


@dataclass
class TaskParams:
    """统一的任务参数，CLI 和 Web 端共用"""
    audio: Optional[str] = None
    video: Optional[str] = None
    srt: Optional[str] = None
    lang: str = "auto"
    to: Optional[str] = None
    model: str = "mlx-community/whisper-large-v3-mlx"
    output: Optional[str] = None
    denoise: bool = False
    embed: bool = False
    translate_config: Optional[tuple[str, str, str]] = None


@dataclass
class TaskResult:
    """任务执行结果"""
    success: bool = False
    output_path: Optional[str] = None
    errors: list[str] = field(default_factory=list)


# ── 参数校验 ──────────────────────────────────────────────


def validate_params(params: TaskParams) -> list[str]:
    """校验任务参数，返回错误消息列表（空列表 = 无错误）"""
    errors: list[str] = []

    input_count = sum(1 for x in [params.audio, params.video, params.srt] if x)

    # 嵌入模式: --embed + --video + --srt（无需转录）
    if params.embed and params.video and params.srt and not params.to:
        if not params.audio:
            return errors  # 合法的纯嵌入模式
        errors.append("嵌入模式下不能同时指定音频文件。")
        return errors

    if input_count == 0:
        errors.append("请指定音频、视频或 SRT 文件。")
        return errors

    if params.embed and not params.video:
        errors.append("嵌入字幕需要视频文件（--embed 必须配合视频使用）。")

    if input_count > 1:
        is_translate_embed = params.embed and params.video and params.srt and not params.audio
        if not is_translate_embed:
            errors.append("音频、视频、SRT 不能同时使用。")

    if params.srt and not params.to:
        errors.append("使用 SRT 文件时必须指定翻译目标语言。")

    if params.to and not params.srt and params.to == params.lang:
        errors.append("源语言和目标语言不能相同。")

    return errors


# ── 翻译配置检查 ──────────────────────────────────────────


def prepare_translate_config(params: TaskParams) -> Optional[str]:
    """准备翻译配置。成功返回 None，失败返回错误消息。"""
    if not params.to:
        params.translate_config = None
        return None

    from .translate import get_translate_config, check_translate_api

    try:
        api_key, base_url, model_name = get_translate_config()
        print("正在检查翻译API可用性...")
        check_translate_api(api_key, base_url, model_name)
        params.translate_config = (api_key, base_url, model_name)
        return None
    except RuntimeError as e:
        return str(e)
    except Exception as e:
        return f"翻译API配置错误: {e}"


# ── 任务执行 ──────────────────────────────────────────────


def detect_file_type(file_path: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """根据文件扩展名检测类型，返回 (audio, video, srt)"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".srt":
        return None, None, file_path
    elif ext in VIDEO_EXTENSIONS:
        return None, file_path, None
    elif ext in AUDIO_EXTENSIONS:
        return file_path, None, None
    else:
        # 未知扩展名默认当音频处理
        return file_path, None, None


def run_task(params: TaskParams) -> TaskResult:
    """执行完整的任务管道（校验 → 执行 → 嵌入）

    此函数是 CLI 和 Web 端共用的核心执行逻辑。
    所有 print 输出会原样写到 stdout，由调用者决定如何呈现。

    output 语义：
      - 未勾选嵌入: output → SRT 文件路径
      - 勾选嵌入:   output → 嵌入后的视频路径，SRT 作为中间产物自动生成
    """
    from .utils import DependencyError

    result = TaskResult()

    # 1. 参数校验
    errors = validate_params(params)
    if errors:
        result.errors = errors
        for e in errors:
            print(f"❌ 错误: {e}")
        return result

    # 2. 准备翻译配置（跳过已配置的情况，避免重复检查）
    if params.to and not params.translate_config:
        err = prepare_translate_config(params)
        if err:
            result.errors = [err]
            print(f"❌ {err}")
            return result

    # 3. 区分 output 语义：嵌入模式下 output 指视频路径，SRT 自动生成
    embed_output = None
    srt_output = params.output
    if params.embed and params.output:
        embed_output = os.path.abspath(os.path.expanduser(params.output))
        srt_output = None  # SRT 使用默认路径

    try:
        # 4a. 纯嵌入模式: --embed + --video + --srt（无翻译）
        if params.embed and params.video and params.srt and not params.to:
            print("开始字幕嵌入流程...")
            from .embed import embed_subtitle
            embed_subtitle(
                video=params.video,
                srt=params.srt,
                lang=params.lang if params.lang != "auto" else None,
            )

            base_name, ext = os.path.splitext(params.video)
            default_embed = f"{base_name}_embed{ext}"
            result.output_path = _rename_if_needed(default_embed, embed_output)
            result.success = True
            print("字幕嵌入完成！")
            return result

        # 4b. 主任务: 转录或翻译
        final_srt_path: Optional[str] = None
        if params.srt:
            print("开始字幕翻译流程...")
            from .translate import translate_srt_file
            final_srt_path = translate_srt_file(
                srt=params.srt,
                to=params.to,
                translate_config=params.translate_config,
                output=srt_output,
            )
        else:
            print("开始音视频转录流程...")
            from .transcribe import transcribe_with_vad
            final_srt_path = transcribe_with_vad(
                input_file=params.audio or params.video,
                lang=params.lang,
                model=params.model,
                denoise=params.denoise,
                to=params.to,
                translate_config=params.translate_config,
                output=srt_output,
                is_video=bool(params.video),
            )

        if final_srt_path:
            print(f"生成字幕文件: {final_srt_path}")
            result.output_path = final_srt_path

        # 4c. 自动嵌入
        if params.embed and final_srt_path and os.path.exists(final_srt_path):
            print(f"\n--- 正在执行字幕嵌入 (SRT: {os.path.basename(final_srt_path)}) ---")
            from .embed import embed_subtitle
            embed_subtitle(
                video=params.video,
                srt=final_srt_path,
                lang=params.lang if params.lang != "auto" else None,
                to=params.to,
                auto_generated_srt=True,
            )

            base_name, ext = os.path.splitext(params.video)
            default_embed = f"{base_name}_embed{ext}"
            result.output_path = _rename_if_needed(default_embed, embed_output)
            print("字幕嵌入完成！")

        result.success = True

    except DependencyError as e:
        result.errors = [str(e)]
        print(f"❌ 环境依赖错误: {e}")
    except Exception as e:
        import traceback
        result.errors = [str(e)]
        print(f"❌ 任务出错: {e}\n{traceback.format_exc()}")

    return result


def _rename_if_needed(default_path: str, target_path: Optional[str]) -> str:
    """如果用户指定了输出路径，将默认输出文件重命名到目标路径"""
    if not target_path:
        return default_path
    if os.path.exists(default_path):
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        os.replace(default_path, target_path)
        print(f"输出文件已移动至: {target_path}")
        return target_path
    return default_path
